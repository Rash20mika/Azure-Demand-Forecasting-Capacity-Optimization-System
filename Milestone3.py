import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Milestone2.csv")

# -----------------------------
# 2. Convert Date Column
# -----------------------------
df['Date_of_usage'] = pd.to_datetime(df['Date_of_usage'])
df = df.sort_values(by='Date_of_usage')

# =========================================================
# PART A: ARIMA MODEL
# =========================================================
# ARIMA needs one value per date, so aggregate Demand_units by date
daily_df = df.groupby('Date_of_usage')['Demand_units'].mean().reset_index()

# Set date as index
daily_df.set_index('Date_of_usage', inplace=True)

# Target series for ARIMA
arima_data = daily_df['Demand_units']

# Train-test split
train_size_arima = int(len(arima_data) * 0.8)
train_arima = arima_data.iloc[:train_size_arima]
test_arima = arima_data.iloc[train_size_arima:]

# Train ARIMA
arima_model = ARIMA(train_arima, order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Forecast
arima_pred = arima_model_fit.forecast(steps=len(test_arima))

# ARIMA Metrics
arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_pred))
arima_mae = mean_absolute_error(test_arima, arima_pred)
arima_r2 = r2_score(test_arima, arima_pred)

# =========================================================
# PART B: XGBOOST REGRESSOR MODEL
# =========================================================
xgb_df = df.copy()

# Create time features
xgb_df['year'] = xgb_df['Date_of_usage'].dt.year
xgb_df['month'] = xgb_df['Date_of_usage'].dt.month
xgb_df['day'] = xgb_df['Date_of_usage'].dt.day
xgb_df['day_of_week'] = xgb_df['Date_of_usage'].dt.dayofweek
xgb_df['week_of_year'] = xgb_df['Date_of_usage'].dt.isocalendar().week.astype(int)
xgb_df['is_weekend'] = (xgb_df['day_of_week'] >= 5).astype(int)

# Encode categorical columns
le_region = LabelEncoder()
le_service = LabelEncoder()

xgb_df['Azure_region'] = le_region.fit_transform(xgb_df['Azure_region'])
xgb_df['Service_Type'] = le_service.fit_transform(xgb_df['Service_Type'])

# Define features and target
X = xgb_df.drop(['Demand_units', 'Date_of_usage', 'usage_spike'], axis=1, errors='ignore')
y = xgb_df['Demand_units']

# Time-based split
train_size_xgb = int(len(xgb_df) * 0.8)

X_train = X.iloc[:train_size_xgb]
X_test = X.iloc[train_size_xgb:]
y_train = y.iloc[:train_size_xgb]
y_test = y.iloc[train_size_xgb:]

# Train XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# XGBoost Metrics
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# =========================================================
# 3. Print Results
# =========================================================
print("\n========== MODEL PERFORMANCE COMPARISON ==========\n")

print("ARIMA Performance:")
print("RMSE :", arima_rmse)
print("MAE  :", arima_mae)
print("R2   :", arima_r2)

print("\nXGBoost Regressor Performance:")
print("RMSE :", xgb_rmse)
print("MAE  :", xgb_mae)
print("R2   :", xgb_r2)

# =========================================================
# 4. Plot ARIMA Forecast vs Actual
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(test_arima.index, test_arima, label='Actual Demand (ARIMA)')
plt.plot(test_arima.index, arima_pred, label='Predicted Demand (ARIMA)')
plt.title("ARIMA: Actual vs Predicted Demand")
plt.xlabel("Date")
plt.ylabel("Demand Units")
plt.legend()
plt.show()

# =========================================================
# 5. Plot XGBoost Predictions vs Actual
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Demand (XGBoost)')
plt.plot(xgb_pred, label='Predicted Demand (XGBoost)')
plt.title("XGBoost Regressor: Actual vs Predicted Demand")
plt.xlabel("Test Records")
plt.ylabel("Demand Units")
plt.legend()
plt.show()