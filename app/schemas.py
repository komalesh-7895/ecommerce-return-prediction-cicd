"""
schemas.py - Pydantic Request & Response Models for FastAPI
E-Commerce Product Return Prediction System
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class OrderPredictionInput(BaseModel):
    Product_Category: str = Field(..., example="Beauty", description="Category of product")
    Product_Price_INR: float = Field(..., example=8772.58, description="Product price in INR")
    Discount_Percent: float = Field(..., ge=0.0, le=100.0, example=20.1, description="Discount percent (0-100)")
    Expected_Delivery_Days: int = Field(..., ge=1, example=10, description="Expected delivery days")
    Actual_Delivery_Days: float = Field(..., ge=0.0, example=12.0, description="Actual delivery days")
    Customer_Rating_PreOrder: float = Field(..., ge=1.0, le=5.0, example=4.1, description="Customer rating before order")
    Previous_Returns: int = Field(..., ge=0, example=3, description="Number of previous returns by customer")
    Previous_Orders: int = Field(..., ge=0, example=18, description="Number of previous orders by customer")
    Payment_Method: str = Field(..., example="Card", description="Payment method: Card, COD, UPI, Wallet")
    Damaged_Package_Flag: str = Field(..., example="No", description="Damaged package indicator: Yes or No")
    Package_Size: str = Field(..., example="Large", description="Size of package: Small, Medium, Large")
    Promotion_Active: str = Field(..., example="No", description="Promotion active indicator: Yes or No")

    class Config:
        json_schema_extra = {
            "example": {
                "Product_Category": "Electronics",
                "Product_Price_INR": 22055.97,
                "Discount_Percent": 21.6,
                "Expected_Delivery_Days": 9,
                "Actual_Delivery_Days": 12.0,
                "Customer_Rating_PreOrder": 4.4,
                "Previous_Returns": 0,
                "Previous_Orders": 33,
                "Payment_Method": "Card",
                "Damaged_Package_Flag": "Yes",
                "Package_Size": "Large",
                "Promotion_Active": "Yes",
            }
        }


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="'Returned' or 'Not Returned'")
    return_probability: float = Field(..., description="Probability of product return (0.0 to 1.0)")
    return_status_code: int = Field(..., description="1 for Returned, 0 for Not Returned")
    model_used: str = Field(..., description="Name of the best ML model in production")
    engineered_features: Dict[str, Any] = Field(..., description="Calculated feature engineering metrics")


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    best_model: str = Field(..., example="LogisticRegression")
    version: str = Field(..., example="1.0.0")
