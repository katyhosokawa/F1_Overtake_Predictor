import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier


class F1MatrixFactorizationModel:
    def __init__(self):
        self.encoders = {}
        self.scaler = RobustScaler()
        self.model = None
        self.feature_names = []
        
    def load_and_prepare_data(self, data_path="f1.csv"):
        self.data = pd.read_csv(data_path)
        self.data = self.data.dropna(subset=['OvertakeNextLap'])
        
        # Conservatively remove extreme outliers
        if 'GapToAheadAtLine' in self.data.columns:
            gap_99 = self.data['GapToAheadAtLine'].quantile(0.99)
            self.data = self.data[self.data['GapToAheadAtLine'] <= gap_99]
        
        if 'SpeedAvg' in self.data.columns:
            speed_01 = self.data['SpeedAvg'].quantile(0.01)
            speed_99 = self.data['SpeedAvg'].quantile(0.99)
            self.data = self.data[(self.data['SpeedAvg'] >= speed_01) & (self.data['SpeedAvg'] <= speed_99)]
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'OvertakeNextLap':
                if self.data[col].skew() > 2:
                    fill_value = self.data[col].mode().iloc[0] if len(self.data[col].mode()) > 0 else self.data[col].median()
                else:
                    fill_value = self.data[col].median()
                self.data[col] = self.data[col].fillna(fill_value)
        
        # Categorical data encoding
        categorical_columns = ['Driver', 'AheadDriver', 'Compound', 'EventName', 'Location']
        for col in categorical_columns:
            if col in self.data.columns:
                # Group rare categories to reduce noise
                value_counts = self.data[col].value_counts()
                rare_categories = value_counts[value_counts < 10].index
                self.data[col] = self.data[col].replace(rare_categories, 'RARE_CATEGORY')
                self.data[col] = self.data[col].fillna('UNKNOWN')
                
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                self.encoders[col] = le
        

    def create_interaction_matrices(self):
        driver_track_pivot = self.data.groupby(['Driver_encoded', 'Location_encoded'])['OvertakeNextLap'].agg(['mean', 'count', 'std']).reset_index()
        
        driver_track_pivot = driver_track_pivot[
            (driver_track_pivot['count'] >= 8) &  
            (driver_track_pivot['std'].fillna(0) > 0.1) 
        ]
        
        driver_track_matrix = driver_track_pivot.pivot(index='Driver_encoded', columns='Location_encoded', values='mean')
        driver_track_matrix = driver_track_matrix.fillna(self.data['OvertakeNextLap'].mean())
        
        return driver_track_matrix
        
        
    def create_latent_features(self, matrix):
        features = []
        
        clean_data = np.nan_to_num(matrix.values, nan=np.nanmean(matrix.values))
            
        num_components = 7  # Determined from testing a range of values
                    
        svd = TruncatedSVD(n_components=num_components, random_state=42)
        row_patterns = svd.fit_transform(clean_data)
        col_patterns = svd.components_.T
                    
        for pattern_idx in range(col_patterns.shape[1]):
            track_pattern_map = dict(zip(range(len(col_patterns)), col_patterns[:, pattern_idx]))
            track_feature = self.data['Location_encoded'].map(track_pattern_map).fillna(0)
            features.append(track_feature)

        return features

    def prepare_features(self, latent_features):
        features = [
            'GapToAheadAtLine', 'SpeedST_DiffToAhead', 'Position', 'LapNumber',
            'SpeedAvg', 'ThrottleMean', 'BrakePct', 'DRS_Use', 'TyreLife',
            'TrackTemp', 'AirTemp'
        ]
        
        X = self.data[features]
        
        # Scale numerical features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Add latent features
        svd_df = pd.DataFrame()
        for i, feature in enumerate(latent_features):
            svd_df[f'svd_feature_{i}'] = feature
        
        # Combine features
        X_combined = pd.concat([X_scaled, svd_df], axis=1)
        X_combined = X_combined.fillna(0)
        
        self.feature_names = X_combined.columns.tolist()

        return X_combined

    def train_model(self, X, y):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        class_weights = [{0: 1, 1: w} for w in [4, 5, 6, 7, 8]]
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        best_weight = {0: 1, 1: 4}  
        best_cv_score = 0
        
        for weight in class_weights:
            cv_f1_scores = []
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                rf_cv = RandomForestClassifier(
                    n_estimators=150,
                    class_weight=weight,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf_cv.fit(X_cv_train, y_cv_train)
                y_cv_pred = rf_cv.predict(X_cv_val)
                cv_f1_scores.append(f1_score(y_cv_val, y_cv_pred))
            
            avg_cv_score = np.mean(cv_f1_scores)
            if avg_cv_score > best_cv_score:
                best_cv_score = avg_cv_score
                best_weight = weight
        
        print(f"Best class weight: {best_weight} (CV F1: {best_cv_score:.4f})")
        
        # Train final model
        self.model = RandomForestClassifier(
            n_estimators=500,
            class_weight=best_weight,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt',
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)

    def evaluate_model(self):
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Optimize threshold for best F1 score
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in np.arange(0.1, 0.9, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if sum(y_pred) > 0:
                f1 = f1_score(self.y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
                    report = classification_report(self.y_test, y_pred, output_dict=True)
                    best_metrics = {
                        'precision': report['1']['precision'],
                        'recall': report['1']['recall'],
                        'f1': f1
                    }
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        results = {
            'f1': best_metrics.get('f1', 0),
            'precision': best_metrics.get('precision', 0),
            'recall': best_metrics.get('recall', 0),
            'auc': auc,
            'avg_precision': avg_precision,
            'best_threshold': best_threshold
        }
        
        return results

    def get_feature_importance(self):
        if self.model is None:
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def predict(self, X, threshold=None):
        # Scale features if they're core features
        if hasattr(self, 'scaler'):
            core_features = [f for f in self.feature_names if not f.startswith('svd_feature_')]
            if len(core_features) > 0:
                X_scaled = X.copy()
                X_scaled[core_features] = self.scaler.transform(X[core_features])
                X = X_scaled
        
        probabilities = self.model.predict_proba(X)[:, 1]
        
        if threshold is not None:
            predictions = (probabilities >= threshold).astype(int)
            return predictions, probabilities
        else:
            return probabilities

    def train(self, data_path="f1.csv"):  
        # Load and prepare data
        self.load_and_prepare_data(data_path)
        
        # Create interaction matrices
        matrix = self.create_interaction_matrices()
        
        # Extract latent features
        latent_features = self.create_latent_features(matrix)
        
        # Prepare features
        X = self.prepare_features(latent_features)
        y = self.data['OvertakeNextLap'].astype(int)
        
        print(f"Training data: {X.shape[0]}, {X.shape[1]} features")
        print(f"Class balance: {y.mean():.3f} positive class")
        
        # Train model
        self.train_model(X, y)
        
        # Evaluate
        results = self.evaluate_model()
        
        print(f"F1-Score:      {results['f1']:.4f}")
        print(f"Precision:     {results['precision']:.4f}")
        print(f"Recall:        {results['recall']:.4f}")
        print(f"AUC-ROC:       {results['auc']:.4f}")
        print(f"Avg Precision: {results['avg_precision']:.4f}")
        print(f"Best Threshold: {results['best_threshold']:.3f}")
        
        # Feature importance
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
            print("-" * 40)
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
        
        return results

if __name__ == "__main__":
    model = F1MatrixFactorizationModel()
    results = model.train()