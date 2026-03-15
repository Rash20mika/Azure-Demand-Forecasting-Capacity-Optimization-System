Azure Demand Forecasting & Capacity Optimization System
Project Overview

This project focuses on building a predictive system to forecast Azure Compute and Storage demand accurately. 
The objective is to support Azure Supply Chain teams in making informed capacity provisioning decisions, reducing both over-investment and under-investment in infrastructure.
The solution leverages data science, feature engineering, and machine learning techniques to improve forecasting accuracy and optimize regional capacity allocation.

Milestone1
Expected Outcomes

- Improved accuracy in Azure service demand forecasting
- Optimized regional capacity provisioning
- Reduction in infrastructure CAPEX waste
- Actionable insights for Azure Supply Chain teams
- Data-driven capacity planning strategy

Tasks Completed:

✔ Collected Azure Compute and Storage usage data  
✔ Incorporated regional and seasonal dimensions  
✔ Integrated external variables 
✔ Cleaned and validated datasets:
- Handled missing values
- Standardized formats
- Ensured logical consistency
- Removed duplicates
- Validated business rules (Demand ≤ Capacity, Availability within limits)

Output:
A clean, validated dataset ready for feature engineering and modeling.

Technologies Used (Milestone 1)

- Python
- Pandas
- CSV datasets
- Data validation techniques

Dataset Attributes

- Date_of_usage
- Azure_region
- Service_Type (Compute / Storage)
- Demand_units
- Capacity_allocated
- Cost_USD
- Service_availability
- Market_Trend_Index
- Holiday_Flag

Milestone 2 – Feature Engineering & Time-Series Preparation

In this phase, the dataset was transformed into a machine learning-ready format.

Time-Series Processing

Converted Date_of_usage to datetime format
Sorted dataset chronologically by:
Azure_region
Service_Type
Date_of_usage

Feature Engineering

The following features were created:
Lag Features
lag_1 → Previous day demand
lag_2 → Demand from two days ago
Purpose: Helps model learn temporal dependency and usage patterns.

Rolling Mean

rolling_mean_3 → 3-day moving average
Purpose: Reduces noise and captures demand trend.

Spike Detection

usage_spike (Binary Feature)
1 → Unusual high demand
0 → Normal demand
Threshold = Mean + Standard Deviation
Purpose: Supports anomaly detection and spike prediction modeling.

Dataset Enhancement Output

This dataset is:
Chronologically ordered
Feature engineered
Validated

Technology used:

Pandas (groupby, rolling, shift)
NumPy
Time-series feature engineering
Data preprocessing

Milestone3:

Models used
ARIMA
ARIMA is used as a time series forecasting model to predict `Demand_units` based on historical demand values.

XGBoost Regressor
XGBoost Regressor is used to predict `Demand_units` using multiple input features such as lag values, rolling mean, market trend, service availability, and other attributes.

Performance metrics
The following metrics were used to evaluate model performance:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 Score

Results
ARIMA
- RMSE: 18.65
- MAE: 14.62
- R2 Score: 0.00058

XGBoost Regressor
- RMSE: 14.68
- MAE: 11.26
- R2 Score: 0.99157

Inference
From the comparison, XGBoost Regressor performed better than ARIMA for this dataset. It produced lower error values and a much higher R2 score, showing that it captured the relationship between input features and demand more effectively.

Technologies used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Statsmodels
- XGBoost

Hyperparameters for XGBoost regressor

1. n_estimators=200
Number of trees in the model.
2. learning_rate=0.05
Controls how fast the model learns.
3. max_depth=6
Maximum depth of each decision tree.
4. subsample=0.8
Model uses only 80% of training rows for building each tree.
5. colsample_bytree=0.8
Model uses only 80% of features for each tree.
6. random_state=42
Fixes randomness so you get the same result each time.
