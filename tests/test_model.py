"""
test_model.py - Unit Tests for Trained ML Model & Evaluation Metrics
"""

import json
import os
import joblib
import pytest

from app.predictor import get_model, predict_single

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best_model.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "model", "metrics.json")


def test_model_artifact_exists():
    """
    Ensure the best_model.joblib file was saved and is readable.
    """
    assert os.path.exists(MODEL_PATH), "best_model.joblib must exist in model/ folder"
    model, model_name = get_model()
    assert model is not None
    assert hasattr(model, "predict"), "Loaded model must implement predict()"
    assert hasattr(model, "predict_proba"), "Loaded model must implement predict_proba()"


def test_metrics_json_integrity():
    """
    Verify evaluation metrics JSON structure and contents.
    Must contain accuracy, precision, recall, f1_score, and roc_auc for both models.
    """
    assert os.path.exists(METRICS_PATH), "metrics.json must exist in model/ folder"

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "best_model" in metrics
    assert "selection_metric" in metrics
    assert metrics["selection_metric"] == "f1_score"
    assert "models_evaluated" in metrics

    models_eval = metrics["models_evaluated"]
    assert "LogisticRegression" in models_eval
    assert "RandomForestClassifier" in models_eval

    for model_name, scores in models_eval.items():
        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert metric in scores, f"{metric} missing in {model_name}"
            assert 0.0 <= scores[metric] <= 1.0, f"{metric} for {model_name} out of bounds: {scores[metric]}"

    # Verify best_model selection matches highest F1-score
    best = metrics["best_model"]
    other = "RandomForestClassifier" if best == "LogisticRegression" else "LogisticRegression"
    assert models_eval[best]["f1_score"] >= models_eval[other]["f1_score"]


def test_single_prediction_output():
    """
    Test direct prediction call through inference engine.
    """
    sample_payload = {
        "Product_Category": "Fashion",
        "Product_Price_INR": 12825.0,
        "Discount_Percent": 25.0,
        "Expected_Delivery_Days": 6,
        "Actual_Delivery_Days": 8.0,
        "Customer_Rating_PreOrder": 4.5,
        "Previous_Returns": 4,
        "Previous_Orders": 12,
        "Payment_Method": "Card",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Medium",
        "Promotion_Active": "Yes",
    }

    result = predict_single(sample_payload)

    assert "prediction" in result
    assert result["prediction"] in ["Returned", "Not Returned"]
    assert "return_probability" in result
    assert 0.0 <= result["return_probability"] <= 1.0
    assert result["return_status_code"] in [0, 1]
    assert "engineered_features" in result
    assert result["engineered_features"]["Delivery_Delay_Days"] == 2.0
    assert result["engineered_features"]["Customer_Return_Frequency"] == pytest.approx(0.3333, abs=1e-2)
