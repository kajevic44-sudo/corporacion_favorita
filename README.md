# Corporacion Favorita - Sales Forecast

**Store 44 / Item 1047679** | Sprint 4 - Streamlit Deployment

---

## Project Structure

    corporacion_favorita/
    app/config.py          - Central configuration
    app/main.py            - Streamlit application
    data/data_utils.py     - Data loading and feature engineering
    model/model_utils.py   - Model loading and prediction
    models/                - Saved model artifacts
    mlflow_results/        - MLflow tracking store

---

## Model Comparison

| Model | MAE | RMSE |
|---|---|---|
| ARIMA(1,0,1) | 181.18 | 211.01 |
| SARIMA(1,0,1)(1,1,1,7) | 196.75 | 224.05 |
| XGBoost (default) | 135.85 | 163.68 |
| LSTM Baseline | 142.21 | 166.05 |
| **XGBoost (HyperOpt)** | **132.67** | **156.32** |
| LSTM (HyperOpt) | 155.29 | 188.34 |

---

## Why XGBoost Was Chosen

1. Best test performance: MAE=132.67, RMSE=156.32
2. Walk-forward backtesting: Mean MAE=105.98 +/- 9.49 (5 folds)
3. Fast inference - ideal for real-time web app
4. Interpretable feature importance
5. HyperOpt tuned: n_estimators=300, max_depth=3, lr=0.01

---

## How to Run

    pip install -r requirements.txt
    streamlit run app/main.py

App opens at http://localhost:8501

---

## MLflow UI

    mlflow ui --backend-store-uri mlflow_results/

Open http://127.0.0.1:5000
Champion Run ID: e98efe56e2984d0aa6f0907458eca43f
