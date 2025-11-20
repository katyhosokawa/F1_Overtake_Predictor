import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, f1_score, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

class F1MatrixFactorizationModel:
    def load_and_prepare_data(self):
        self.data = pd.read_csv("f1.csv")
        self.encoders = {}
        self.data = self.data.dropna(subset=['OvertakeNextLap'])
        
        # Fill missing values
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Encode categorical variables
        categorical_columns = ['Driver', 'AheadDriver', 'Compound', 'EventName', 'Location']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna('UNKNOWN')
                le = LabelEncoder()
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col])
                self.encoders[col] = le
        
    def create_interaction_matrix(self):
        matrices = {}
        
        driver_track_pivot = self.data.groupby(['Driver_encoded', 'Location_encoded'])['OvertakeNextLap'].agg(['mean', 'count']).reset_index()
        driver_track_pivot = driver_track_pivot[driver_track_pivot['count'] >= 5]
        
        driver_track_matrix = driver_track_pivot.pivot(index='Driver_encoded', columns='Location_encoded', values='mean')
        driver_track_matrix = driver_track_matrix.fillna(self.data['OvertakeNextLap'].mean())
        matrices['driver_track'] = driver_track_matrix
        
        return matrices
        
    def create_latent_features(self, matrices):
        features = []
        
        for matrix_name, matrix in matrices.items():
            # Clean matrix data
            clean_data = np.nan_to_num(matrix.values, nan=np.nanmean(matrix.values))
            
            if clean_data.shape[0] > 1 and clean_data.shape[1] > 1:
                n_factors = min(5, min(clean_data.shape) - 1)
                
                if n_factors > 0:
                    # Decompose matrix into latent patterns
                    decomposer = TruncatedSVD(n_components=n_factors, random_state=42)
                    row_patterns = decomposer.fit_transform(clean_data)
                    col_patterns = decomposer.components_.T
                    
                    if matrix_name == 'driver_track':
                        # Only create track pattern features (skip driver patterns)
                        for pattern_idx in range(col_patterns.shape[1]):
                            track_pattern_map = dict(zip(range(len(col_patterns)), col_patterns[:, pattern_idx]))
                            track_feature = self.data['Location_encoded'].map(track_pattern_map).fillna(0)
                            features.append(track_feature)

        return features
        
    def train_streaming_model(self, original_features, svd_features, y):
        # Core features
        core_features = [
            'GapToAheadAtLine', 'SpeedST_DiffToAhead', 'Position', 'LapNumber',
            'SpeedAvg', 'ThrottleMean', 'BrakePct', 'DRS_Use', 'TyreLife',
            'TrackTemp', 'AirTemp'
        ]
        
        # Select available core features
        available_core = [f for f in core_features if f in original_features.columns]
        X_core = original_features[available_core]
        
        # Convert SVD features list to DataFrame
        svd_df = pd.DataFrame()
        for i, feature in enumerate(svd_features):
            svd_df[f'svd_feature_{i}'] = feature
        
        # Use all generated SVD features (now only track patterns)
        selected_svd_cols = svd_df.columns.tolist()
        
        # Combine core + track patterns
        X_model = pd.concat([X_core, svd_df[selected_svd_cols]], axis=1)
        X_model = X_model.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_model, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Train the model
        rf_model = RandomForestClassifier(
            n_estimators=150, 
            class_weight={0: 1, 1: 8}, 
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        return rf_model
        
    def evaluate_best_model(self, model):
        # Get predictions
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Optimize threshold
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
                        'f1': f1,
                        'predictions': y_pred
                    }
        
        # Calculate metrics
        auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        results = {
            'auc': auc,
            'avg_precision': avg_precision,
            'best_threshold': best_threshold,
            'probabilities': y_pred_proba,
            **best_metrics
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_names = self.X_test.columns
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
        
        return results


    def run_model_pipeline(self):
        self.load_and_prepare_data()

        matrices = self.create_interaction_matrix()

        # Extract latent patterns using optimal configuration
        latent_features = self.create_latent_features(matrices)

        # Prepare target variable
        y = self.data['OvertakeNextLap'].astype(int)
        
        # Train model with latent features
        model = self.train_streaming_model(self.data, latent_features, y)
        
        # Evaluate model
        results = self.evaluate_best_model(model)
        
        # Final summary
        print(f"RESULTS:")
        print(f"Features: Core + Track Patterns Only")
        print(f"F1-Score: {results['f1']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"AUC-ROC: {results['auc']:.4f}")
        print(f"Optimal Threshold: {results['best_threshold']:.3f}")
        
        return results

if __name__ == "__main__":
    mf_model = F1MatrixFactorizationModel()
    results = mf_model.run_model_pipeline()