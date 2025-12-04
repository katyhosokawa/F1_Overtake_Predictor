# Formula 1 Overtaking Predictor

Author: Katy Hosokawa  

---

## Overview

This project is aimed at building a machine learning model to predict when an F1 driver is likely to overtake another car in the following lap. My approach was permutative experimentation with different algorithms, features, and data preprocessing methods. Through this, the model with the highest predictive power was a hybrid Random Forest and Matrix Factorization classifier that achieves an F1-score of 0.4961. With an AUC-ROC of 0.868, the model demonstrates strong discriminative ability while maintaining relative simplicity and practical interpretability for real-world applications. This model reveals that traditional racing metrics combined with latent features derived from historical driver-track interactions are signficant predictors of next lap overtaking.


## Motivation

Formula 1 racing proves itself to be an exceedingly data-rich and strategically complex sports, where real-time team decisions outside the car can significantly impact race outcomes. Overtaking is a decisive aspect of racing, so whether or not an individual is likely to overtake another represents an interesting predicition problem that has real applications for strategists, viewers, and drivers. For teams, specifically, it is crucial that these prediction models are highly optimized as the ability to predict when an overtaking attempt is likely to succeed has direct implications for pit strategy, tire management, and other important race decisions. This project seeks to guide and potentially automate decisions by developing a robust machine learning algorithm specifically designed to utilize open-sourced telemetry and other historical lap data to predict overtaking.


## Data

The model utilizes a dataset derived from the official Formula 1 FIA timing and telemetry data, accessed through the open-source FastF1 Python library. While the comprehensive dataset is much more comprehensive, race laps from the 2021 to 2024 seasons were used for the training data. This data time frame was chosen in order to balance having a large dataset with maintaining relevance, especially considering livery capabilities and FIA regulations have evolved over the time. These laps were further filtered down to exclude pit-in laps, pit-out laps, and laps completed under non-green flag conditions, since they represent abnormal conditions that may effectively serve as outliers that likely create unnecessary noise to the model. The final training set consists of 88,267 lap records, featuring 33 variables, which consist of telemetry data, positional information, tire conditions, and environmental factors. The target variable, `OvertakeNextLap`, represents a binary outcome indicating whether a successful overtake occurred / will occur in the subsequent lap.

Given the data and the relative rarity of overtakes, it was evident that cases of the positive class, where an overtake occured in the next lap, would be underrepresented. This can be confirmed by the fact that overtakes consist of only 11.7% of samples. Such class imbalance was accounted for by implementing class weights, as it proved a more effective and straightforward approach than simulative resampling. 

In terms of preprocessing, the FastF1 dataset performs some initial data cleaning, leaving no unexpected missingness and filling in data with median or mode values when appropriate. To further enhance data quality, some presumable outliers were further removed through the use of 99th percentile thresholds, leaving 85,637 usable laps. Additionally, when categorically encoding, rare category grouping with frequency less than 10 indicating rarity was used to reduce noise. 


## Feature Engineering 

When selecting features, the strategy was to try incorporate diverse features within a wide range of combinations of latent and explicit features. These features related to telemetry, weather conditions, track details, driver patterns, and temporal data. 

The best performing set of features consisted of the following:
  - `GapToAheadAtLine`: Distance to car ahead 
  - `SpeedST_DiffToAhead`: Difference in speed between cars
  - `Position`: Driver position
  - `DRS_Use`: Percent of samples with Drag Reduction System activated thoughout the lap
  - `TyreLife`: Age of active tyres
  - `TrackTemp`: Temperature of race track
  - `AirTemp`: Temperature of the air
  - `SpeedAvg`: Average car speed throughout the lap
  - `ThrottleMean`: Average throttle throughout the lap
  - `BrakePct`: Fraction of samples with braking in use
  - `LapNumber`: Race lap number

It was also oberseved that over-engineering approaches with upwards of 30+ features performed poorly and consistently lowered the F1 score ~10-15%. 

In addition to the previously mentioned features, Truncated Singular Value Decomposition (SVD) was used to extract 7 latent features from driver-track interaction matrices. This approach effectively captured complex relationships between driver behavior and track characteristics that are not otherwise apparent in raw features. The selected quantity of latent features was hyperparameter tuned for further optimization. Other matrices, such as a weather matrix and driver to driver interaction matrix, were also tested, but did not add enough meaningful signa to justify the added complexity.


## The Model

Apart from the feature engineering, further experimentation was completed across multiple modeling approaches. Some of the key algorithms that were tested and evaluated included XGBoost, Linear Regression, Collaborative Filtering, and Random Forests. Through comprehensive experimentation, it was determined that a Random Forest classifier with the aforementioned feature set of explicit paired with the SVD gathered latent features provided the best performance-complexity trade-off.

Each iteration of the model underwent cross-validation with 3 k-folds and train / test splits for consistent evaluation. Additionally, a random baseline with an F1 score of 0.101 and a model that always predicts overtake with an F1 score of 0.2069 were used for a basic comparision. This evaluation process was also used for hyperparameter tuning several values in the selected model. The optimal threshold value of 0.28 was determined by testing values from 0.1 - 0.9 at 0.02 iterative steps. For class weights, the overtaking class was tested with a weights of 3 - 8 to every 1 non-overtake lap. The optimized weight was { 0:1, 1:4 }. For the number of trees in the random forest, 500 was chosen to balance performance and speed, with the number of max features being the squareroot of the number of features. This decision was made to prevent overfitting and ensure a level of randomness and variety amongst the trees. Finally, 2 minimum samples were required per leaf to prevent overfitting.


## Results and Analysis

The optimized model achieved the following performance metrics:
           F1 Matrix Factorization  Random Baseline  Always Positive
F1-Score                    0.4961           0.1012           0.2088
Precision                   0.4389           0.1017           0.1166
Recall                      0.5704           0.1007           1.0000
AUC-ROC                     0.8682           0.5000           0.5000

The model reveals actionable insights for racing strategy with the following features proving the most influencial:

| Rank | Feature | Importance | Strategic Insight |
|------|---------|------------|-------------------|
| 1 | GapToAheadAtLine | 16.01% | Primary overtaking indicator |
| 2 | LapNumber | 12.56% | Race temporal context is signifcant |
| 3 | Position | 8.85% | Track position is meaningful; overtaking is more common within lower positons |
| 4 | SpeedAvg | 7.63% | Overall performance indicator |
| 5 | TrackTemp | 7.44% | Track conditions are meaningful |

Furthermore, the SVD-derived features collectively contributed about 15.2% of model importance, demonstrating that the hidden driver-track interaction patterns do provide substantial predictive value beyond observable metrics and successfully improve the model. In fact, these added features improved the model from a F1 score of 0.4644 to 0.4961. This further indicates why a model, such as this one, could be insightful for a team; the model provides further rubustness that may not be otherwise available with standard data features and metrics. 

The reliability of this model can be further supported by its consistency across the cross-validation attempts. The consistent performance across different data splits reaffirms that the evaluation metrics are likely reliable and the model is not simply overfitting to the training data.

In terms of result significance and application, the 57% recall enables teams to identify most overtaking opportunities and supports the idea that the cost of failing to identify a future overtake may be paramount. With that being said, a 43.8% precision still provides reasonable confidence in predictions. The model also shines in its lack of complexity and low-latency prediction, which is key in making split-second decisions. 

Additionally, the ranked feature importance, which came out-of-the-box with Random Forest, can be used to guide priorities for driver development and raising overtaking likelihood. Clear feature importance and general interprebability enables the model findings to more easily translate to strategic decision-making and enhances overall human comprehension. 

All in all, comprehensive experimentation revealed several key insights when approaching this problem space. The first is that over-engineered approaches consistently underperformed to a refined, more simplistic design. That being said, keeping some representation of diverse types of features was still helpful. Furthermore, domain knowledge served very useful in targeting and engineering features that made intuitive sense, as opposed to a more randomized approach. Finally, this prediction problem was much more difficult than anticipated due to the fast paced nature of motor racing. So much can change within a lap making predicting overtakes a lap in advanced difficult. Future work may find more success by further breaking down a race into sectors, as opposed to laps. This would support more granular temporal sequencing and help ensure only the most relevant, updated data is used to make informed predictions.


## Conclusion

This research successfully developed a machine learning model for predicting next lap Formula 1 overtakes, achieving an F1-score of 0.4961 with strong AUC-ROC of 0.86482. The model demonstrates that combining traditional racing metrics with latent driver-track interaction patterns provides optimal predictive performance while maintaining practical interpretability.

At its core, the model was built on a systematic approach that iteratively tested different models and features across a standardized evaluation approach and compared against simple baselines. This resulted in a highly-optimized model that was further optimized with hyperparameter tuning and is suitable for real-time racing strategy.

The model's 57.04% recall capability enables racing teams to identify most overtaking opportunities, while 43.89% precision provides reasonable confidence for strategic decision-making. This balance makes the model particularly valuable for pit and tyre strategy optimization and other real-time race decisions. Thus, the model and extracted features establish a foundation for data-driven racing strategy optimization and demonstrates how automated machine learning approaches have the opportunity to minimize human-error in decision making and all-together enhance team performance in Formula 1. 
