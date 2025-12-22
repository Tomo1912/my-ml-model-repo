"""
House Price Prediction API
A production-ready ML service for predicting California house prices.
"""

import pickle
import json
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered API for predicting California house prices using Random Forest",
    version="2.0.0"
)

# Load model and metadata at startup
MODEL_PATH = Path("model.pkl")
METADATA_PATH = Path("model_metadata.json")

model = None
metadata = None


@app.on_event("startup")
async def load_model():
    """Load the trained model and metadata on startup."""
    global model, metadata

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

    print("✅ Model loaded successfully!")
    print(f"   Model type: {metadata.get('model_type', 'Unknown')}")
    print(f"   R² Score: {metadata.get('r2_score', 'N/A')}")
    print(f"   MAE: ${metadata.get('mae', 'N/A'):.2f}")


# Pydantic models for request/response validation
class HouseFeaturesInput(BaseModel):
    """Input features for house price prediction."""
    median_income: float = Field(..., description="Median income in block group (in $10,000s)", ge=0, le=15)
    house_age: float = Field(..., description="Median house age in block group", ge=1, le=52)
    avg_rooms: float = Field(..., description="Average number of rooms per household", ge=1, le=20)
    avg_bedrooms: float = Field(..., description="Average number of bedrooms per household", ge=0.5, le=10)
    population: float = Field(..., description="Block group population", ge=3, le=10000)
    avg_occupancy: float = Field(..., description="Average number of household members", ge=0.5, le=15)
    latitude: float = Field(..., description="Block group latitude", ge=32.5, le=42)
    longitude: float = Field(..., description="Block group longitude", ge=-124.5, le=-114)

    class Config:
        json_schema_extra = {
            "example": {
                "median_income": 8.3252,
                "house_age": 41.0,
                "avg_rooms": 6.98,
                "avg_bedrooms": 1.02,
                "population": 322.0,
                "avg_occupancy": 2.55,
                "latitude": 37.88,
                "longitude": -122.23
            }
        }


class PredictionResponse(BaseModel):
    """Response model for price prediction."""
    predicted_price: float = Field(..., description="Predicted house price in USD")
    model_version: str = Field(..., description="Model version used for prediction")
    features_used: dict = Field(..., description="Input features used for prediction")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_type: str
    version: str
    features: List[str]
    mae: float
    r2_score: float
    training_samples: int
    test_samples: int


# API Endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict_house_price(features: HouseFeaturesInput):
    """
    Predict house price based on input features.

    Returns the predicted price in USD (median house value for the block group).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Prepare input features in correct order
    input_array = np.array([[
        features.median_income,
        features.house_age,
        features.avg_rooms,
        features.avg_bedrooms,
        features.population,
        features.avg_occupancy,
        features.latitude,
        features.longitude
    ]])

    # Make prediction
    prediction = model.predict(input_array)[0]

    # Convert from $100,000s to USD
    predicted_price = float(prediction * 100000)

    return PredictionResponse(
        predicted_price=predicted_price,
        model_version=metadata.get("version", "2.0.0"),
        features_used=features.dict()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the deployed model."""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return ModelInfoResponse(
        model_type=metadata.get("model_type", "Unknown"),
        version=metadata.get("version", "2.0.0"),
        features=metadata.get("features", []),
        mae=metadata.get("mae", 0.0),
        r2_score=metadata.get("r2_score", 0.0),
        training_samples=metadata.get("training_samples", 0),
        test_samples=metadata.get("test_samples", 0)
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "House Price Prediction API"
    }


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "House Price Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "model_info": "/model/info (GET)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
