import os

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MLflow tracking store (local copy from Sprint 3)
# Must be an absolute file:/// URI
MLFLOW_TRACKING_URI = "file:///" + os.path.join(BASE_DIR, "mlflow_results").replace("\\", "/")

# Champion model URI (XGBoost - best model from Sprint 3)
CHAMPION_RUN_ID = "e98efe56e2984d0aa6f0907458eca43f"
MODEL_URI = f"runs:/{CHAMPION_RUN_ID}/xgb_best_model"

# Local fallback paths (used if MLflow load fails)
MODEL_PATH    = os.path.join(BASE_DIR, "models", "champion_xgboost.json")
SCALER_PATH   = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_JSON = os.path.join(BASE_DIR, "models", "champion_features.json")
TS_CSV_PATH   = os.path.join(BASE_DIR, "models", "ts_store44_item1047679.csv")

# Model and series constants
STORE_ID  = 44
ITEM_ID   = 1047679
LAG_DAYS  = 30

# Feature columns (same order as training)
FEATURE_COLS = [
    "day_of_week", "month", "day_of_month", "week_of_year",
    "is_weekend", "quarter",
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
]
