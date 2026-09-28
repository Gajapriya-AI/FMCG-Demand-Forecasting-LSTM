import streamlit as st
import pandas as pd
import os

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FMCG Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# =========================================================
# CUSTOM COLORFUL DASHBOARD STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #fff0f6,
        #eef6ff,
        #f3efff
    );
}

/* Main Title */
h1 {
    color: #7c3aed !important;
    font-weight: 900 !important;
}

/* Section Headings */
h2, h3 {
    color: #1d4ed8 !important;
    font-weight: 800 !important;
}

/* Description Text */
.stApp p {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #ec4899,
        #8b5cf6,
        #2563eb
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar Header */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* KPI Cards */
[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #ffffff,
        #fdf2f8
    );

    border-radius: 18px;
    padding: 20px;

    box-shadow:
        0px 6px 20px rgba(124, 58, 237, 0.20);

    border-top: 5px solid #ec4899;
    border-bottom: 5px solid #3b82f6;
}

/* KPI Values */
[data-testid="stMetricValue"] {
    color: #7c3aed !important;
    font-weight: 900 !important;
}

/* KPI Labels */
[data-testid="stMetricLabel"] {
    color: #2563eb !important;
    font-weight: 700 !important;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(
        90deg,
        #ec4899,
        #8b5cf6,
        #2563eb
    );

    color: white !important;
    border: none;
    border-radius: 12px;
    font-weight: 700;
}

/* Data Tables */
[data-testid="stDataFrame"] {
    border-radius: 15px;

    box-shadow:
        0px 4px 15px rgba(37, 99, 235, 0.15);
}

/* Caption / Footer */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #be185d !important;
    font-weight: 700 !important;
}

/* Divider */
hr {
    border: none;
    height: 3px;

    background: linear-gradient(
        90deg,
        #ec4899,
        #8b5cf6,
        #2563eb
    );
}

/* Alert Box */
[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# ARTIFACT FOLDER
# =========================================================

ARTIFACT_DIR = "fmcg_lstm_artifacts"


# =========================================================
# FILE PATHS
# =========================================================

forecast_file = os.path.join(
    ARTIFACT_DIR,
    "forecast_7day_all.csv"
)

metrics_file = os.path.join(
    ARTIFACT_DIR,
    "evaluation_metrics.csv"
)

alerts_file = os.path.join(
    ARTIFACT_DIR,
    "inventory_alerts.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    forecast = pd.read_csv(forecast_file)

    metrics = pd.read_csv(metrics_file)

    alerts = pd.read_csv(alerts_file)

    forecast["date"] = pd.to_datetime(
        forecast["date"]
    )

    return forecast, metrics, alerts


forecast, metrics, alerts = load_data()


# =========================================================
# TITLE
# =========================================================

st.title("📦 FMCG Demand Forecasting System")

st.write(
    "LSTM-based 7-day FMCG demand forecasting "
    "and inventory management dashboard"
)

st.divider()


# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔎 Forecast Filters")

sku_list = sorted(
    forecast["sku"].unique()
)

region_list = sorted(
    forecast["region"].unique()
)

selected_sku = st.sidebar.selectbox(
    "Select SKU",
    ["All"] + sku_list
)

selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + region_list
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_forecast = forecast.copy()

if selected_sku != "All":

    filtered_forecast = filtered_forecast[
        filtered_forecast["sku"] == selected_sku
    ]


if selected_region != "All":

    filtered_forecast = filtered_forecast[
        filtered_forecast["region"] == selected_region
    ]


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📊 Forecast Records",
        len(filtered_forecast)
    )


with col2:

    st.metric(
        "📦 Forecast Units",
        f"{filtered_forecast['forecast_units'].sum():,.0f}"
    )


with col3:

    st.metric(
        "🛒 SKUs",
        filtered_forecast["sku"].nunique()
    )


with col4:

    st.metric(
        "🌍 Regions",
        filtered_forecast["region"].nunique()
    )


st.divider()


# =========================================================
# FORECAST CHART
# =========================================================

st.subheader("📈 7-Day Demand Forecast")

daily_forecast = (
    filtered_forecast
    .groupby("date")["forecast_units"]
    .sum()
)

st.line_chart(
    daily_forecast,
    height=400
)


# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader("📋 Forecast Details")

display_forecast = filtered_forecast.copy()

display_forecast["date"] = (
    display_forecast["date"]
    .dt.strftime("%Y-%m-%d")
)

display_forecast["forecast_units"] = (
    display_forecast["forecast_units"]
    .round(2)
)

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =========================================================
# MODEL EVALUATION
# =========================================================

st.subheader("🎯 Model Evaluation")

metric_dict = dict(
    zip(
        metrics["Metric"],
        metrics["Value"]
    )
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📉 MAE",
        f"{metric_dict.get('MAE', 0):.2f}"
    )


with col2:

    st.metric(
        "📊 RMSE",
        f"{metric_dict.get('RMSE', 0):.2f}"
    )


with col3:

    st.metric(
        "📈 MAPE",
        f"{metric_dict.get('MAPE (%)', 0):.2f}%"
    )


st.divider()


# =========================================================
# INVENTORY ALERTS
# =========================================================

st.subheader("⚠️ Inventory Alerts")

if alerts.empty:

    st.success(
        "✅ No inventory alerts available."
    )

else:

    alert_data = alerts.copy()

    if (
        selected_sku != "All"
        and "sku" in alert_data.columns
    ):

        alert_data = alert_data[
            alert_data["sku"] == selected_sku
        ]


    if (
        selected_region != "All"
        and "region" in alert_data.columns
    ):

        alert_data = alert_data[
            alert_data["region"] == selected_region
        ]


    st.dataframe(
        alert_data,
        use_container_width=True,
        hide_index=True
    )


st.divider()


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "FMCG Demand Forecasting | LSTM-based Machine Learning Project"
)