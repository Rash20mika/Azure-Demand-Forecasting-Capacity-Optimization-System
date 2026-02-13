import pandas as pd
df = pd.read_csv("azure_dataset.csv")
df['Date_of_usage'] = pd.to_datetime(
    df['Date_of_usage'],
    format="%d-%m-%Y",   # Explicit format
    errors="coerce"
)
df['Azure_region'] = df['Azure_region'].str.strip().str.title()
df['Service_Type'] = df['Service_Type'].str.strip().str.title()
numeric_cols = [
    'Demand_units',
    'Capacity_allocated',
    'Cost_USD',
    'Service_availability',
    'Market_Trend_Index',
    'Holiday_Flag'
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Demand_units'] = df['Demand_units'].fillna(df['Demand_units'].median())
df['Capacity_allocated'] = df['Capacity_allocated'].fillna(df['Capacity_allocated'].median())
df['Cost_USD'] = df['Cost_USD'].fillna(df['Cost_USD'].median())
df['Service_availability'] = df['Service_availability'].fillna(df['Service_availability'].mean())
df = df.dropna(subset=['Date_of_usage'])
df['Holiday_Flag'] = df['Holiday_Flag'].apply(lambda x: 1 if x == 1 else 0)
df = df.drop_duplicates()
df = df[df['Demand_units'] <= df['Capacity_allocated']]
df = df[(df['Service_availability'] >= 0) & (df['Service_availability'] <= 100)]
df.to_csv("cleaned_dataset1.csv", index=False)
print("Data cleaning and validation completed successfully!")
