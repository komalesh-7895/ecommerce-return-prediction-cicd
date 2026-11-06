"""
main.py - FastAPI Application
E-Commerce Product Return Prediction System using CI/CD
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import get_model, predict_single
from app.schemas import HealthResponse, OrderPredictionInput, PredictionResponse

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description="Production-grade ML API to predict e-commerce product returns with CI/CD integration.",
    version="1.0.0",
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def read_root():
    """
    Root endpoint providing service overview.
    """
    return {
        "service": "E-Commerce Product Return Prediction API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
        "prediction_endpoint": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    GET /health - Verifies service health, model loading status, and metadata.
    """
    try:
        _, model_name = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            best_model=model_name,
            version="1.0.1",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_return(order_data: OrderPredictionInput):
    """
    POST /predict - Accepts order details, computes engineered features,
    and returns return prediction ('Returned' or 'Not Returned') with probability.
    """
    try:
        result = predict_single(order_data.model_dump())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
