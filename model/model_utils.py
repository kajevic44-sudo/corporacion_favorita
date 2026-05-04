import os
import sys
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import MODEL_PATH

def load_champion_model():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model

def predict_one(model, feature_row: pd.DataFrame) -> float:
    pred = model.predict(feature_row)
    return float(pred[0])
