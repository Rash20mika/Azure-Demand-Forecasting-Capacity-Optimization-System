import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# Convert date
df["Date_of_usage"] = pd.to_datetime(df["Date_of_usage"], dayfirst=True)

# Sort properly
df = df.sort_values(
    ["Azure_region", "Service_Type", "Date_of_usage"]
).reset_index(drop=True)

# Lag features
df["lag_1"] = df.groupby(
    ["Azure_region", "Service_Type"]
)["Demand_units"].shift(1)

df["lag_2"] = df.groupby(
    ["Azure_region", "Service_Type"]
)["Demand_units"].shift(2)

# Rolling mean
df["rolling_mean_3"] = df.groupby(
    ["Azure_region", "Service_Type"]
)["Demand_units"].rolling(3).mean().reset_index(level=[0,1], drop=True)

# Spike detection
threshold = df["Demand_units"].mean() + df["Demand_units"].std()
df["usage_spike"] = np.where(df["Demand_units"] > threshold, 1, 0)

# Drop NaN from lag & rolling
df = df.dropna().reset_index(drop=True)

# Save modified dataset
df.to_csv("Milestone2.csv", index=False)

