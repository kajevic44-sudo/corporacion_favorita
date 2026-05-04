import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import TS_CSV_PATH, FEATURE_COLS, LAG_DAYS


def load_timeseries() -> pd.Series:
    df = pd.read_csv(TS_CSV_PATH, index_col="date", parse_dates=True)
    df = df.asfreq("D")
    return df["unit_sales"]


def create_features(series: pd.Series) -> pd.DataFrame:
    df = series.to_frame(name="unit_sales")
    df["day_of_week"]     = df.index.dayofweek
    df["month"]           = df.index.month
    df["day_of_month"]    = df.index.day
    df["week_of_year"]    = df.index.isocalendar().week.astype(int)
    df["is_weekend"]      = (df.index.dayofweek >= 5).astype(int)
    df["quarter"]         = df.index.quarter
    df["lag_1"]           = df["unit_sales"].shift(1)
    df["lag_7"]           = df["unit_sales"].shift(7)
    df["lag_14"]          = df["unit_sales"].shift(14)
    df["lag_30"]          = df["unit_sales"].shift(30)
    df["rolling_mean_7"]  = df["unit_sales"].shift(1).rolling(7).mean()
    df["rolling_std_7"]   = df["unit_sales"].shift(1).rolling(7).std()
    df["rolling_mean_14"] = df["unit_sales"].shift(1).rolling(14).mean()
    return df


def make_forecast_row(date: pd.Timestamp, history: pd.Series) -> pd.DataFrame:
    row = {
        "day_of_week":     date.dayofweek,
        "month":           date.month,
        "day_of_month":    date.day,
        "week_of_year":    date.isocalendar()[1],
        "is_weekend":      int(date.dayofweek >= 5),
        "quarter":         date.quarter,
        "lag_1":           history.iloc[-1]  if len(history) >= 1  else np.nan,
        "lag_7":           history.iloc[-7]  if len(history) >= 7  else np.nan,
        "lag_14":          history.iloc[-14] if len(history) >= 14 else np.nan,
        "lag_30":          history.iloc[-30] if len(history) >= 30 else np.nan,
        "rolling_mean_7":  history.iloc[-7:].mean()  if len(history) >= 7  else np.nan,
        "rolling_std_7":   history.iloc[-7:].std()   if len(history) >= 7  else np.nan,
        "rolling_mean_14": history.iloc[-14:].mean() if len(history) >= 14 else np.nan,
    }
    return pd.DataFrame([row])[FEATURE_COLS]
