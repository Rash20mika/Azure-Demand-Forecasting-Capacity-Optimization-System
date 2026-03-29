import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Azure Demand Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# FULL CUSTOM CSS (VISIBLE TEXT)
# -----------------------------
st.markdown("""
    <style>
    /* -----------------------------
       GLOBAL BACKGROUND
    ----------------------------- */
    .stApp {
        background: linear-gradient(135deg, #020617, #0f172a, #111827);
        color: white !important;
    }

    .main {
        background: transparent !important;
        color: white !important;
    }

    html, body, [class*="css"] {
        color: white !important;
    }

    /* -----------------------------
       SIDEBAR
    ----------------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120, #111827);
        color: white !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: white !important;
        font-weight: 600 !important;
    }

    /* -----------------------------
       LABELS
    ----------------------------- */
    .stMultiSelect label,
    .stSelectbox label,
    .stTextInput label,
    .stDateInput label,
    .stNumberInput label,
    .stSlider label,
    .stRadio label {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    /* -----------------------------
       MULTISELECT / SELECT BOX
    ----------------------------- */
    div[data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        color: #111827 !important;
        border-radius: 12px !important;
        border: 2px solid #38bdf8 !important;
        min-height: 52px !important;
        box-shadow: 0 0 10px rgba(56,189,248,0.15);
    }

    /* Selected tags */
    div[data-baseweb="tag"] {
        background-color: #ef4444 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 4px 8px !important;
    }

    div[data-baseweb="tag"] span {
        color: white !important;
    }

    /* Input text inside dropdown */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] div {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    /* Dropdown list */
    ul[role="listbox"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        color: black !important;
    }

    ul[role="listbox"] li {
        color: black !important;
        font-weight: 700 !important;
    }

    /* -----------------------------
       RADIO BUTTONS
    ----------------------------- */
    div[role="radiogroup"] label {
        color: white !important;
        font-size: 17px !important;
        font-weight: 600 !important;
    }

    /* -----------------------------
       METRIC CARDS
    ----------------------------- */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        padding: 18px !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    div[data-testid="metric-container"] label {
        color: #cbd5e1 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 34px !important;
        font-weight: 800 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #38bdf8 !important;
    }

    /* -----------------------------
       HEADINGS / TEXT
    ----------------------------- */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    p, span, div {
        color: #f8fafc;
    }

    /* -----------------------------
       BUTTONS
    ----------------------------- */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #06b6d4) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 18px rgba(37,99,235,0.35);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #1d4ed8, #0891b2) !important;
        color: white !important;
    }

    /* -----------------------------
       DATAFRAME / TABLE
    ----------------------------- */
    .stDataFrame, .stTable {
        background-color: #111827 !important;
        color: white !important;
        border-radius: 12px !important;
    }

    .stDataFrame div {
        color: white !important;
    }

    table {
        color: white !important;
    }

    thead tr th {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    tbody tr td {
        background-color: #111827 !important;
        color: #f8fafc !important;
    }

    /* -----------------------------
       ALERT BOXES
    ----------------------------- */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }

    /* -----------------------------
       EXPANDERS / CONTAINERS
    ----------------------------- */
    .stExpander {
        background-color: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
    }

    /* -----------------------------
       PLOTLY CHART AREA
    ----------------------------- */
    .js-plotly-plot, .plotly, .plot-container {
        border-radius: 18px !important;
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("Milestone2.csv")
df['Date_of_usage'] = pd.to_datetime(df['Date_of_usage'])
df = df.sort_values(by='Date_of_usage')

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.title("⚙️ Dashboard Filters")

selected_region = st.sidebar.multiselect(
    "Select Azure Region",
    options=df['Azure_region'].unique(),
    default=df['Azure_region'].unique()
)

selected_service = st.sidebar.multiselect(
    "Select Service Type",
    options=df['Service_Type'].unique(),
    default=df['Service_Type'].unique()
)

page = st.sidebar.radio("📂 Navigation", [
    "Home",
    "Dataset",
    "Visualizations",
    "Model Comparison"
])

# -----------------------------
# APPLY FILTERS
# -----------------------------
filtered_df = df[
    (df['Azure_region'].isin(selected_region)) &
    (df['Service_Type'].isin(selected_service))
]

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("☁️ Azure Demand Forecasting & Capacity Optimization Dashboard")
    st.markdown("### Real-Time Monitoring Style Dashboard for ARIMA and XGBoost")

    st.write("""
    This dashboard helps analyze and forecast **Demand Units** in Azure cloud environments.
    
    ### Models Used:
    - **ARIMA** → Time Series Forecasting
    - **XGBoost Regressor** → Machine Learning Regression
    
    ### Objective:
    - Forecast demand
    - Optimize cloud capacity
    - Compare model performance
    """)

    avg_demand = round(filtered_df['Demand_units'].mean(), 2)
    avg_cost = round(filtered_df['Cost_USD'].mean(), 2)
    avg_availability = round(filtered_df['Service_availability'].mean(), 2)
    total_records = filtered_df.shape[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Avg Demand", avg_demand)
    with col2:
        st.metric("💲 Avg Cost (USD)", avg_cost)
    with col3:
        st.metric("⚡ Service Availability", f"{avg_availability}%")
    with col4:
        st.metric("📁 Total Records", total_records)

# -----------------------------
# DATASET PAGE
# -----------------------------
elif page == "Dataset":
    st.title("📊 Filtered Dataset Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {filtered_df.shape[0]}, Columns: {filtered_df.shape[1]}")

# -----------------------------
# VISUALIZATIONS PAGE
# -----------------------------
elif page == "Visualizations":
    st.title("📈 Interactive Visualizations")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Demand", round(filtered_df['Demand_units'].mean(), 2))
        with col2:
            st.metric("Avg Capacity", round(filtered_df['Capacity_allocated'].mean(), 2))
        with col3:
            st.metric("Avg Cost", round(filtered_df['Cost_USD'].mean(), 2))
        with col4:
            st.metric("Availability", f"{round(filtered_df['Service_availability'].mean(), 2)}%")

        st.markdown("---")

        st.subheader("📉 Demand Units Over Time")
        fig1 = px.line(
            filtered_df, x='Date_of_usage', y='Demand_units',
            title='Demand Units Over Time', template='plotly_dark'
        )
        fig1.update_layout(
            font=dict(color="white", size=14),
            title_font=dict(size=20, color="white"),
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117"
        )
        st.plotly_chart(fig1, use_container_width=True)

        col5, col6 = st.columns(2)

        with col5:
            st.subheader("🌍 Average Demand by Azure Region")
            region_df = filtered_df.groupby('Azure_region')['Demand_units'].mean().reset_index()
            fig2 = px.bar(
                region_df, x='Azure_region', y='Demand_units',
                title='Average Demand by Region', template='plotly_dark'
            )
            fig2.update_layout(font=dict(color="white"), paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
            st.plotly_chart(fig2, use_container_width=True)

        with col6:
            st.subheader("🛠️ Average Demand by Service Type")
            service_df = filtered_df.groupby('Service_Type')['Demand_units'].mean().reset_index()
            fig3 = px.bar(
                service_df, x='Service_Type', y='Demand_units',
                title='Average Demand by Service Type', template='plotly_dark'
            )
            fig3.update_layout(font=dict(color="white"), paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📈 Demand Trend with 7-Day Rolling Mean")
        daily_df = filtered_df.groupby('Date_of_usage')['Demand_units'].mean().reset_index()
        daily_df['rolling_mean_7'] = daily_df['Demand_units'].rolling(7).mean()

        fig4 = px.line(
            daily_df, x='Date_of_usage', y=['Demand_units', 'rolling_mean_7'],
            title='Demand Trend and Rolling Mean', template='plotly_dark'
        )
        fig4.update_layout(font=dict(color="white"), paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
        st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# MODEL COMPARISON PAGE
# -----------------------------
elif page == "Model Comparison":
    st.title("🤖 Model Comparison")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    elif st.button("Run ARIMA and XGBoost Models"):

        st.write("⏳ Running models... please wait")

        daily_df = filtered_df.groupby('Date_of_usage')['Demand_units'].mean().reset_index()
        daily_df.set_index('Date_of_usage', inplace=True)

        arima_data = daily_df['Demand_units']

        if len(arima_data) < 20:
            st.error("Not enough time-series data for ARIMA after filtering. Please select more data.")
        else:
            train_size_arima = int(len(arima_data) * 0.8)
            train_arima = arima_data.iloc[:train_size_arima]
            test_arima = arima_data.iloc[train_size_arima:]

            arima_model = ARIMA(train_arima, order=(5, 1, 0))
            arima_model_fit = arima_model.fit()
            arima_pred = arima_model_fit.forecast(steps=len(test_arima))

            arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_pred))
            arima_mae = mean_absolute_error(test_arima, arima_pred)
            arima_r2 = r2_score(test_arima, arima_pred)

            xgb_df = filtered_df.copy()

            xgb_df['year'] = xgb_df['Date_of_usage'].dt.year
            xgb_df['month'] = xgb_df['Date_of_usage'].dt.month
            xgb_df['day'] = xgb_df['Date_of_usage'].dt.day
            xgb_df['day_of_week'] = xgb_df['Date_of_usage'].dt.dayofweek
            xgb_df['week_of_year'] = xgb_df['Date_of_usage'].dt.isocalendar().week.astype(int)
            xgb_df['is_weekend'] = (xgb_df['day_of_week'] >= 5).astype(int)

            le_region = LabelEncoder()
            le_service = LabelEncoder()

            xgb_df['Azure_region'] = le_region.fit_transform(xgb_df['Azure_region'])
            xgb_df['Service_Type'] = le_service.fit_transform(xgb_df['Service_Type'])

            X = xgb_df.drop(['Demand_units', 'Date_of_usage', 'usage_spike'], axis=1, errors='ignore')
            y = xgb_df['Demand_units']

            train_size_xgb = int(len(xgb_df) * 0.8)

            X_train = X.iloc[:train_size_xgb]
            X_test = X.iloc[train_size_xgb:]
            y_train = y.iloc[:train_size_xgb]
            y_test = y.iloc[train_size_xgb:]

            xgb_model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)

            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)

            st.subheader("📌 Performance Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("ARIMA")
                st.metric("RMSE", f"{arima_rmse:.2f}")
                st.metric("MAE", f"{arima_mae:.2f}")
                st.metric("R²", f"{arima_r2:.4f}")

            with col2:
                st.subheader("XGBoost")
                st.metric("RMSE", f"{xgb_rmse:.2f}")
                st.metric("MAE", f"{xgb_mae:.2f}")
                st.metric("R²", f"{xgb_r2:.4f}")

            comparison_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R²'],
                'ARIMA': [arima_rmse, arima_mae, arima_r2],
                'XGBoost': [xgb_rmse, xgb_mae, xgb_r2]
            })

            fig5 = px.bar(
                comparison_df, x='Metric', y=['ARIMA', 'XGBoost'],
                barmode='group', title='ARIMA vs XGBoost Performance',
                template='plotly_dark'
            )
            fig5.update_layout(font=dict(color="white"), paper_bgcolor="#0E1117", plot_bgcolor="#0E1117")
            st.plotly_chart(fig5, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                st.subheader("📉 ARIMA: Actual vs Predicted")
                fig6, ax1 = plt.subplots(figsize=(8, 4))
                fig6.patch.set_facecolor('#0E1117')
                ax1.set_facecolor('#0E1117')
                ax1.plot(test_arima.index, test_arima, label="Actual", color='cyan')
                ax1.plot(test_arima.index, arima_pred, label="Predicted", color='orange')
                ax1.legend()
                ax1.tick_params(colors='white')
                ax1.set_title("ARIMA Forecast", color='white')
                for spine in ax1.spines.values():
                    spine.set_color('white')
                st.pyplot(fig6)

            with col4:
                st.subheader("📉 XGBoost: Actual vs Predicted")
                fig7, ax2 = plt.subplots(figsize=(8, 4))
                fig7.patch.set_facecolor('#0E1117')
                ax2.set_facecolor('#0E1117')
                ax2.plot(y_test.values, label="Actual", color='cyan')
                ax2.plot(xgb_pred, label="Predicted", color='orange')
                ax2.legend()
                ax2.tick_params(colors='white')
                ax2.set_title("XGBoost Forecast", color='white')
                for spine in ax2.spines.values():
                    spine.set_color('white')
                st.pyplot(fig7)