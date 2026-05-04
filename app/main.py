import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

from app.config import STORE_ID, ITEM_ID, FEATURE_COLS, MODEL_PATH
from data.data_utils import load_timeseries, make_forecast_row
from model.model_utils import load_champion_model

st.set_page_config(
    page_title="Corporacion Favorita | Sales Forecast",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = (
    "<style>"
    ".metric-card{background:linear-gradient(135deg,#1e2130,#2a2f45);"
    "border-radius:12px;padding:1.2rem 1.5rem;"
    "border-left:4px solid #4c9be8;margin-bottom:0.5rem;}"
    ".metric-label{color:#8b9dc3;font-size:0.78rem;text-transform:uppercase;}"
    ".metric-value{color:#e8eaf6;font-size:1.8rem;font-weight:700;}"
    ".metric-sub{color:#4c9be8;font-size:0.75rem;margin-top:0.2rem;}"
    ".section-header{color:#4c9be8;font-size:1.05rem;font-weight:600;"
    "text-transform:uppercase;letter-spacing:0.06em;"
    "border-bottom:1px solid #2a2f45;padding-bottom:0.3rem;margin-bottom:0.5rem;}"
    "</style>"
)
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading champion model...")
def get_model():
    return load_champion_model()

@st.cache_data(show_spinner="Loading historical data...")
def get_timeseries():
    return load_timeseries()

ts = get_timeseries()
last_date = ts.index[-1].date()
min_forecast = last_date + timedelta(days=1)
max_forecast = last_date + timedelta(days=90)

with st.sidebar:
    st.markdown("## Corporacion Favorita")
    st.markdown("---")
    st.markdown("**Store ID:** " + str(STORE_ID))
    st.markdown("**Item ID:** " + str(ITEM_ID))
    st.markdown("**Model:** XGBoost (HyperOpt tuned)")
    st.markdown("**Test MAE:** 132.67 units")
    st.markdown("**Test RMSE:** 156.32 units")
    st.markdown("---")
    forecast_start = st.date_input(
        "Forecast Start Date",
        value=min_forecast,
        min_value=min_forecast,
        max_value=max_forecast,
    )
    n_days = st.slider("Forecast Horizon (days)", 1, 30, 14)
    history_window = st.slider("Historical window (days)", 30, 180, 90, 30)
    st.caption("Sprint 4 | Store 44 / Item 1047679")

st.markdown("# Corporacion Favorita — Sales Forecast")
st.markdown(
    "**Champion model:** XGBoost (HyperOpt, 50 trials) | "
    "**Store:** 44 | **Item:** 1047679 | "
    "**Last historical date:** " + str(last_date)
)
st.markdown("---")

model = get_model()

forecast_dates = pd.date_range(start=pd.Timestamp(forecast_start), periods=n_days, freq="D")
history = ts.copy()
predictions = []
for fdate in forecast_dates:
    row = make_forecast_row(fdate, history)
    pred = max(0.0, float(model.predict(row)[0]))
    predictions.append(pred)
    history = pd.concat([history, pd.Series([pred], index=[fdate])])

forecast_df = pd.DataFrame({"date": forecast_dates.strftime("%Y-%m-%d"), "forecast_sales": [round(p,2) for p in predictions]})

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Forecast Horizon", str(n_days) + " days")
with col2:
    st.metric("Total Forecasted Sales", f"{sum(predictions):,.0f} units")
with col3:
    st.metric("Avg Daily Forecast", f"{np.mean(predictions):,.1f} units")
with col4:
    peak_idx = int(np.argmax(predictions))
    st.metric("Peak Day", f"{predictions[peak_idx]:,.0f} units", forecast_dates[peak_idx].strftime("%b %d"))

st.markdown("---")
st.markdown("### Sales History + Forecast")

hist_window = ts.iloc[-history_window:]
mae = 132.67
upper = [p + mae for p in predictions]
lower = [max(0, p - mae) for p in predictions]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_window.index, y=hist_window.values, mode="lines", name="Historical Sales", line=dict(color="#4c9be8", width=1.5)))
fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, mode="lines+markers", name="Forecast (XGBoost)", line=dict(color="#f97316", width=2.5, dash="dash"), marker=dict(size=6)))
fig.add_trace(go.Scatter(
    x=list(forecast_dates) + list(reversed(forecast_dates)),
    y=upper + list(reversed(lower)),
    fill="toself", fillcolor="rgba(249,115,22,0.12)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Band (+/- MAE)"
))
fig.add_trace(go.Scatter(
    x=[forecast_dates[0], forecast_dates[0]],
    y=[0, hist_window.max() * 1.2],
    mode="lines",
    name="Forecast Start",
    line=dict(color="#a0aec0", width=1.5, dash="dot"),
    showlegend=False,
))
fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=420,
    margin=dict(l=10,r=10,t=30,b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#1e2130"), yaxis=dict(gridcolor="#1e2130", title="Unit Sales"),
    hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns([2, 1])
with col_left:
    st.markdown("### Forecast Table")
    display_df = forecast_df.copy()
    display_df.index = range(1, len(display_df)+1)
    display_df.columns = ["Date", "Forecasted Sales (units)"]
    st.dataframe(display_df, use_container_width=True)
with col_right:
    st.markdown("### Download")
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast as CSV", csv_bytes,
        file_name="forecast_store44_item1047679.csv", mime="text/csv")
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: XGBoost")
    st.markdown("- Tuning: HyperOpt (50 trials)")
    st.markdown("- Test MAE: **132.67 units**")
    st.markdown("- Test RMSE: **156.32 units**")
    st.markdown("- Features: " + str(len(FEATURE_COLS)) + " engineered features")

st.markdown("---")
st.caption("Corporacion Favorita Sales Forecast | Sprint 4 | XGBoost Champion | Store 44 / Item 1047679")
