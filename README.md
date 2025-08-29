Customer Churn Prediction (XGBoost)
End-to-end churn modeling that turns click/visit behavior into retention insights. Includes data prep, feature engineering, tuned XGBoost, and business takeaways. Key result: ~0.75 AUC on validation. 
Overview
We analyze online activity and purchase patterns to identify at-risk customers and recommend targeted re-engagement strategies (offers, loyalty nudges, outreach timing). 
Data
train.csv, test.csv (tabular behavioral features)
No missing values; outliers kept to preserve signal. Target churn stored as numeric. 
Feature Engineering
Scaled core features for equitable contribution.
Custom interaction_rate (engagement per visit).
Considered PCA during exploration but excluded for final model to retain interpretability. 
Modeling
Algorithm: XGBoost (binary:logistic), optimized for AUC.
75/25 stratified train/validation split; ~100 boosting rounds; tuned depth, eta, subsample. 
Results
Validation AUC ≈ 0.75 with clear churn signals from inactivity windows and declining visit frequency.