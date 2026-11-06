"""
test_preprocessing.py - Unit Tests for Data Preprocessing & Feature Engineering
"""

import math
import numpy as np
import pandas as pd
import pytest

from model.preprocessor import (
    engineer_features,
    validate_discount_percent,
    validate_product_price,
)


def test_validate_discount_percent():
    """
    Test validation of Discount_Percent (valid range: [0, 100]).
    """
    assert validate_discount_percent(25.0) == 25.0
    assert validate_discount_percent(0.0) == 0.0
    assert validate_discount_percent(100.0) == 100.0
    assert np.isnan(validate_discount_percent(-5.0))
    assert np.isnan(validate_discount_percent(125.0))
    assert np.isnan(validate_discount_percent("invalid"))
    assert np.isnan(validate_discount_percent(None))


def test_validate_product_price():
    """
    Test validation of Product_Price_INR (valid range: > 0).
    """
    assert validate_product_price(1500.50) == 1500.50
    assert np.isnan(validate_product_price(0.0))
    assert np.isnan(validate_product_price(-200.0))
    assert np.isnan(validate_product_price(None))


def test_feature_engineering_calculations():
    """
    Test calculations of Delivery_Delay_Days and Customer_Return_Frequency.
    """
    test_df = pd.DataFrame([
        {
            "Expected_Delivery_Days": 5,
            "Actual_Delivery_Days": 8.0,
            "Previous_Returns": 4,
            "Previous_Orders": 10,
            "Discount_Percent": 15.0,
            "Product_Price_INR": 2500.0,
        },
        {
            "Expected_Delivery_Days": 7,
            "Actual_Delivery_Days": 5.0,
            "Previous_Returns": 0,
            "Previous_Orders": 0,  # Zero order boundary
            "Discount_Percent": 120.0,  # Invalid discount
            "Product_Price_INR": -50.0,  # Invalid price
        },
    ])

    result = engineer_features(test_df)

    # Row 0: Delay = 8 - 5 = 3
    assert result.loc[0, "Delivery_Delay_Days"] == 3.0
    # Row 0: Frequency = 4 / 10 = 0.4
    assert result.loc[0, "Customer_Return_Frequency"] == pytest.approx(0.4)
    assert result.loc[0, "Discount_Percent"] == 15.0
    assert result.loc[0, "Product_Price_INR"] == 2500.0

    # Row 1: Delay = 5 - 7 = -2
    assert result.loc[1, "Delivery_Delay_Days"] == -2.0
    # Row 1: Frequency with 0 orders handled safely without ZeroDivisionError
    assert result.loc[1, "Customer_Return_Frequency"] == 0.0
    # Invalid discount and price converted to NaN
    assert np.isnan(result.loc[1, "Discount_Percent"])
    assert np.isnan(result.loc[1, "Product_Price_INR"])
