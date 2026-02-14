Azure Demand Forecasting & Capacity Optimization System
Project Overview

This project focuses on building a predictive system to forecast Azure Compute and Storage demand accurately. 
The objective is to support Azure Supply Chain teams in making informed capacity provisioning decisions, reducing both over-investment and under-investment in infrastructure.
The solution leverages data science, feature engineering, and machine learning techniques to improve forecasting accuracy and optimize regional capacity allocation.

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

