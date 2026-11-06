"""
preprocessor.py - Data Preprocessing & Feature Engineering Module
E-Commerce Product Return Prediction System
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numerical and categorical feature definitions
NUMERICAL_FEATURES = [
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Delivery_Delay_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
    "Customer_Return_Frequency",
]

CATEGORICAL_FEATURES = [
    "Product_Category",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]

ALL_INPUT_FEATURES = [
    "Product_Category",
    "Product_Price_INR",
    "Discount_Percent",
    "Expected_Delivery_Days",
    "Actual_Delivery_Days",
    "Customer_Rating_PreOrder",
    "Previous_Returns",
    "Previous_Orders",
    "Payment_Method",
    "Damaged_Package_Flag",
    "Package_Size",
    "Promotion_Active",
]


def validate_discount_percent(val):
    """
    Validates discount percent. Values outside [0, 100] are flagged as invalid (np.nan).
    """
    if pd.isna(val):
        return np.nan
    try:
        f_val = float(val)
        if 0.0 <= f_val <= 100.0:
            return f_val
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def validate_product_price(val):
    """
    Validates product price. Values <= 0 are flagged as invalid (np.nan).
    """
    if pd.isna(val):
        return np.nan
    try:
        f_val = float(val)
        if f_val > 0:
            return f_val
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data cleaning, validation, and feature engineering:
    1. Product_Price_INR validation (> 0)
    2. Discount_Percent validation (0 - 100)
    3. Delivery_Delay_Days = Actual_Delivery_Days - Expected_Delivery_Days
    4. Customer_Return_Frequency = Previous_Returns / Previous_Orders
    """
    data = df.copy()

    # Validate numeric values
    if "Product_Price_INR" in data.columns:
        data["Product_Price_INR"] = data["Product_Price_INR"].apply(validate_product_price)

    if "Discount_Percent" in data.columns:
        data["Discount_Percent"] = data["Discount_Percent"].apply(validate_discount_percent)

    # Feature 1: Delivery_Delay_Days
    if "Actual_Delivery_Days" in data.columns and "Expected_Delivery_Days" in data.columns:
        data["Delivery_Delay_Days"] = data["Actual_Delivery_Days"] - data["Expected_Delivery_Days"]

    # Feature 2: Customer_Return_Frequency
    if "Previous_Returns" in data.columns and "Previous_Orders" in data.columns:
        safe_orders = data["Previous_Orders"].replace(0, 1)
        data["Customer_Return_Frequency"] = (data["Previous_Returns"] / safe_orders).clip(lower=0.0, upper=1.0)

    return data


def create_preprocessor_pipeline() -> ColumnTransformer:
    """
    Builds the ColumnTransformer pipeline with imputation, scaling, and one-hot encoding.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor
