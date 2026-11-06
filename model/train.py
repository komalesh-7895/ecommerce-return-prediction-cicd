"""
train.py - Machine Learning Model Training & Evaluation Pipeline
E-Commerce Product Return Prediction System
"""

import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.preprocessor import (
    ALL_INPUT_FEATURES,
    create_preprocessor_pipeline,
    engineer_features,
)


def load_and_preprocess_data(csv_path: str):
    """
    Loads raw CSV dataset, performs feature engineering, and splits X and y.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"Loading dataset from: {csv_path}")
    raw_df = pd.read_csv(csv_path)
    print(f"Initial raw shape: {raw_df.shape}")

    # Identify target column Return_Status
    if "Return_Status" not in raw_df.columns:
        raise ValueError("Target column 'Return_Status' not found in dataset")

    # Apply data cleaning and feature engineering
    processed_df = engineer_features(raw_df)

    # Encode target: 'Returned' -> 1, 'Not Returned' -> 0
    y = (processed_df["Return_Status"].astype(str).str.strip().str.lower() == "returned").astype(int)

    # Features
    X = processed_df[ALL_INPUT_FEATURES + ["Delivery_Delay_Days", "Customer_Return_Frequency"]]

    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    return X, y


def train_and_evaluate(csv_path: str = None, output_dir: str = None):
    """
    Trains Logistic Regression and Random Forest models,
    evaluates both on test split, selects the best model using F1-score,
    and saves the model and metrics JSON.
    """
    if csv_path is None:
        csv_path = os.path.join(PROJECT_ROOT, "data", "Section7_Q4_Ecommerce_Product_Return_CICD.csv")

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "model")

    os.makedirs(output_dir, exist_ok=True)

    X, y = load_and_preprocess_data(csv_path)

    # 80/20 train/test stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    candidate_models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            max_depth=10,
            min_samples_leaf=2,
        ),
    }

    metrics_results = {}
    fitted_pipelines = {}

    for model_name, classifier in candidate_models.items():
        print(f"\n--- Training {model_name} ---")
        preprocessor = create_preprocessor_pipeline()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))

        metrics_results[model_name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        }

        fitted_pipelines[model_name] = pipeline

        print(f"Metrics for {model_name}:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")

    # Select best model based on F1-score
    best_model_name = max(
        metrics_results, key=lambda k: metrics_results[k]["f1_score"]
    )
    best_f1 = metrics_results[best_model_name]["f1_score"]
    print(f"\n>>> Best model selected by F1-score: {best_model_name} (F1 = {best_f1}) <<<")

    best_pipeline = fitted_pipelines[best_model_name]

    # Save model using joblib
    model_save_path = os.path.join(output_dir, "best_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved best model pipeline to: {model_save_path}")

    # Save metrics JSON
    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "models_evaluated": metrics_results,
    }
    metrics_save_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_save_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    return best_model_name, metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
