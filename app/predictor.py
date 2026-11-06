"""
predictor.py - Model Loader & Inference Engine
E-Commerce Product Return Prediction System
"""

import json
import os
import sys
import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.preprocessor import (
    ALL_INPUT_FEATURES,
    engineer_features,
)

MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best_model.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "model", "metrics.json")

_model = None
_model_name = "Unknown"


def get_model():
    """
    Lazy loads the trained model pipeline.
    """
    global _model, _model_name
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH, "r") as f:
                    metrics_data = json.load(f)
                    _model_name = metrics_data.get("best_model", "Loaded Model")
            except Exception:
                _model_name = "Trained Pipeline"
        else:
            _model_name = "Trained Pipeline"

    return _model, _model_name


def predict_single(input_dict: dict) -> dict:
    """
    Takes a single input dictionary, applies feature engineering, runs inference,
    and returns prediction and return probability.
    """
    model, model_name = get_model()

    df = pd.DataFrame([input_dict])

    # Feature engineering
    engineered_df = engineer_features(df)

    delay = float(engineered_df["Delivery_Delay_Days"].iloc[0])
    freq = float(engineered_df["Customer_Return_Frequency"].iloc[0])

    features_to_pass = ALL_INPUT_FEATURES + ["Delivery_Delay_Days", "Customer_Return_Frequency"]
    features_df = engineered_df[features_to_pass]

    # Predict probability and class
    prob = float(model.predict_proba(features_df)[0][1])
    pred_code = int(model.predict(features_df)[0])
    prediction_label = "Returned" if pred_code == 1 else "Not Returned"

    return {
        "prediction": prediction_label,
        "return_probability": round(prob, 4),
        "return_status_code": pred_code,
        "model_used": model_name,
        "engineered_features": {
            "Delivery_Delay_Days": round(delay, 2),
            "Customer_Return_Frequency": round(freq, 4),
        },
    }
