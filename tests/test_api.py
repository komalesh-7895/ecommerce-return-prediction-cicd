"""
test_api.py - Integration Tests for FastAPI Endpoints
E-Commerce Product Return Prediction System
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """
    Test GET / returns service info.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "E-Commerce Product Return Prediction API"
    assert data["status"] == "online"


def test_health_endpoint():
    """
    Test GET /health returns 200 OK and model status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "best_model" in data
    assert data["version"] in ["1.0.0", "1.0.1"]


def test_predict_endpoint_valid():
    """
    Test POST /predict with a valid order payload.
    """
    payload = {
        "Product_Category": "Beauty",
        "Product_Price_INR": 8772.58,
        "Discount_Percent": 20.1,
        "Expected_Delivery_Days": 10,
        "Actual_Delivery_Days": 12.0,
        "Customer_Rating_PreOrder": 4.1,
        "Previous_Returns": 3,
        "Previous_Orders": 18,
        "Payment_Method": "Card",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Large",
        "Promotion_Active": "No",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in ["Returned", "Not Returned"]
    assert "return_probability" in data
    assert 0.0 <= data["return_probability"] <= 1.0
    assert data["return_status_code"] in [0, 1]
    assert data["model_used"] != ""
    assert "engineered_features" in data
    assert data["engineered_features"]["Delivery_Delay_Days"] == 2.0
    assert data["engineered_features"]["Customer_Return_Frequency"] == pytest.approx(0.1667, abs=1e-2)


def test_predict_endpoint_high_risk_order():
    """
    Test POST /predict with high-risk attributes (damaged package, high return frequency, delivery delay).
    """
    payload = {
        "Product_Category": "Fashion",
        "Product_Price_INR": 22055.97,
        "Discount_Percent": 45.0,
        "Expected_Delivery_Days": 5,
        "Actual_Delivery_Days": 12.0,  # 7 days delay
        "Customer_Rating_PreOrder": 2.2,
        "Previous_Returns": 7,
        "Previous_Orders": 8,  # Very high return rate 7/8 = 87.5%
        "Payment_Method": "COD",
        "Damaged_Package_Flag": "Yes",  # Damaged package
        "Package_Size": "Small",
        "Promotion_Active": "Yes",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["return_probability"] <= 1.0
    assert data["engineered_features"]["Delivery_Delay_Days"] == 7.0


def test_predict_endpoint_validation_error():
    """
    Test POST /predict with invalid data types and out-of-range discount percent (> 100).
    """
    payload = {
        "Product_Category": "Beauty",
        "Product_Price_INR": 8772.58,
        "Discount_Percent": 150.0,  # Invalid: > 100
        "Expected_Delivery_Days": 10,
        "Actual_Delivery_Days": 12.0,
        "Customer_Rating_PreOrder": 4.1,
        "Previous_Returns": 3,
        "Previous_Orders": 18,
        "Payment_Method": "Card",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Large",
        "Promotion_Active": "No",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error
