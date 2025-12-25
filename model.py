"""
House Price Prediction API
A production-ready ML service for predicting California house prices.

Security Features:
- ONNX model format (safe deserialization - no arbitrary code execution)
- Rate limiting to prevent DoS attacks
- Input validation with strict bounds
- CORS configuration
- Security headers
- Non-root Docker user
- Request size limits
- Model integrity verification (SHA256)
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import numpy as np
import onnxruntime as ort

# Configure logging (don't expose sensitive info)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app with security settings
app = FastAPI(
    title="House Price Prediction API",
    description="ML-powered API for predicting California house prices using Random Forest",
    version="2.0.0",
    docs_url="/docs",  # Can be set to None in production
    redoc_url=None,    # Disable redoc
    openapi_url="/openapi.json"  # Can be set to None in production
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration - restrict to specific origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers["Cache-Control"] = "no-store"
    return response

# Request size limit middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    MAX_REQUEST_SIZE = 10 * 1024  # 10KB max
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": "Request too large"}
        )
    return await call_next(request)

# Load model and metadata at startup
MODEL_PATH = Path("model.onnx")
METADATA_PATH = Path("model_metadata.json")
MODEL_HASH_PATH = Path("model_hash.txt")

# ONNX Runtime session (thread-safe)
ort_session = None
metadata = None


def verify_model_integrity(model_path: Path) -> bool:
    """Verify model file integrity using SHA256 hash."""
    if not MODEL_HASH_PATH.exists():
        logger.warning("Model hash file not found - skipping integrity check")
        return True

    with open(model_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    with open(MODEL_HASH_PATH, 'r') as f:
        expected_hash = f.read().strip()

    if file_hash != expected_hash:
        logger.error(f"Model integrity check FAILED! Expected: {expected_hash[:16]}..., Got: {file_hash[:16]}...")
        return False

    logger.info(f"Model integrity check passed (SHA256: {file_hash[:16]}...)")
    return True


@app.on_event("startup")
async def load_model():
    """Load the ONNX model and metadata on startup."""
    global ort_session, metadata

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # Verify model integrity before loading
    if not verify_model_integrity(MODEL_PATH):
        raise ValueError("Model integrity verification failed")

    try:
        # ONNX Runtime is SAFE - it only loads model weights, not arbitrary code
        # This prevents Remote Code Execution (RCE) attacks via malicious model files
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(str(MODEL_PATH), sess_options)
        logger.info("ONNX Runtime session created successfully")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise

    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

    logger.info("✅ Model loaded successfully!")
    logger.info(f"   Model format: ONNX (secure)")
    logger.info(f"   Model type: {metadata.get('model_type', 'Unknown')}")
    logger.info(f"   R² Score: {metadata.get('r2_score', 'N/A')}")
    logger.info(f"   MAE: ${metadata.get('mae', 'N/A'):.2f}")


# Pydantic models for request/response validation with relaxed validation
class HouseFeaturesInput(BaseModel):
    """Input features for house price prediction with relaxed validation."""
    median_income: float = Field(..., description="Median income in block group (in $10,000s)", ge=0, le=50)
    house_age: float = Field(..., description="Median house age in block group", ge=0, le=200)
    avg_rooms: float = Field(..., description="Average number of rooms per household", ge=0, le=100)
    avg_bedrooms: float = Field(..., description="Average number of bedrooms per household", ge=0, le=50)
    population: float = Field(..., description="Block group population", ge=0, le=100000)
    avg_occupancy: float = Field(..., description="Average number of household members", ge=0, le=100)
    latitude: float = Field(..., description="Block group latitude", ge=-90, le=90)
    longitude: float = Field(..., description="Block group longitude", ge=-180, le=180)

    @field_validator('*', mode='before')
    @classmethod
    def check_not_nan_or_inf(cls, v):
        """Prevent NaN and Inf values that could cause model issues."""
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                raise ValueError("NaN or Inf values are not allowed")
        return v

    model_config = {
        "json_schema_extra": {
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
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute per IP
async def predict_house_price(request: Request, features: HouseFeaturesInput):
    """
    Predict house price based on input features.

    Returns the predicted price in USD (median house value for the block group).
    Rate limited to 30 requests per minute.
    Uses ONNX Runtime for secure inference.
    """
    if ort_session is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare input features in correct order (ONNX expects float32)
        input_array = np.array([[
            features.median_income,
            features.house_age,
            features.avg_rooms,
            features.avg_bedrooms,
            features.population,
            features.avg_occupancy,
            features.latitude,
            features.longitude
        ]], dtype=np.float32)

        # Validate array doesn't contain NaN/Inf after conversion
        if np.any(np.isnan(input_array)) or np.any(np.isinf(input_array)):
            raise HTTPException(status_code=400, detail="Invalid numeric values")

        # Make prediction using ONNX Runtime (secure - no arbitrary code execution)
        input_name = ort_session.get_inputs()[0].name
        prediction = ort_session.run(None, {input_name: input_array})[0][0]

        # Validate prediction is reasonable
        if prediction < 0 or prediction > 100:  # $0 to $10M range
            logger.warning(f"Unusual prediction value: {prediction}")

        # Convert from $100,000s to USD
        predicted_price = float(prediction * 100000)

        # Log prediction (without sensitive data)
        logger.info(f"Prediction made: ${predicted_price:.2f}")

        return PredictionResponse(
            predicted_price=predicted_price,
            model_version=metadata.get("version", "2.0.0"),
            features_used=features.model_dump()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/model/info", response_model=ModelInfoResponse)
@limiter.limit("60/minute")
async def get_model_info(request: Request):
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
@limiter.limit("120/minute")
def health_check(request: Request):
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "model_loaded": ort_session is not None,
        "model_format": "ONNX",
        "service": "House Price Prediction API"
    }


@app.get("/", response_class=HTMLResponse)
@limiter.limit("60/minute")
def root(request: Request):
    """Root endpoint - HTML frontend for house price prediction."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏠 House Price Prediction</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; }
        .card { background: white; border-radius: 20px; padding: 30px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }
        h1 { text-align: center; color: #333; margin-bottom: 10px; font-size: 2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #555; font-weight: 600; font-size: 0.9em; }
        input { width: 100%; padding: 12px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 1em; transition: border-color 0.3s; }
        input:focus { outline: none; border-color: #667eea; }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
        button { width: 100%; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 10px; font-size: 1.1em; font-weight: 600; cursor: pointer; margin-top: 20px; transition: transform 0.2s, box-shadow 0.2s; }
        button:hover { transform: translateY(-2px); box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4); }
        button:disabled { opacity: 0.7; cursor: not-allowed; transform: none; }
        .result { margin-top: 25px; padding: 20px; border-radius: 15px; text-align: center; display: none; }
        .result.success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
        .result.error { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); color: white; }
        .price { font-size: 2.5em; font-weight: bold; margin: 10px 0; }
        .links { text-align: center; margin-top: 20px; }
        .links a { color: #667eea; margin: 0 10px; text-decoration: none; font-weight: 600; }
        .links a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>🏠 House Price Prediction</h1>
            <p class="subtitle">Enter house features to get a price estimate</p>
            <form id="predictForm">
                <div class="row">
                    <div class="form-group">
                        <label>💰 Median Income ($10k)</label>
                        <input type="number" id="median_income" step="0.01" value="8.32" required>
                    </div>
                    <div class="form-group">
                        <label>📅 House Age (years)</label>
                        <input type="number" id="house_age" step="1" value="41" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>🛋️ Avg Rooms</label>
                        <input type="number" id="avg_rooms" step="0.01" value="6.98" required>
                    </div>
                    <div class="form-group">
                        <label>🛏️ Avg Bedrooms</label>
                        <input type="number" id="avg_bedrooms" step="0.01" value="1.02" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>👥 Population</label>
                        <input type="number" id="population" step="1" value="322" required>
                    </div>
                    <div class="form-group">
                        <label>🏠 Avg Occupancy</label>
                        <input type="number" id="avg_occupancy" step="0.01" value="2.55" required>
                    </div>
                </div>
                <div class="row">
                    <div class="form-group">
                        <label>📍 Latitude</label>
                        <input type="number" id="latitude" step="0.01" value="37.88" required>
                    </div>
                    <div class="form-group">
                        <label>📍 Longitude</label>
                        <input type="number" id="longitude" step="0.01" value="-122.23" required>
                    </div>
                </div>
                <button type="submit" id="submitBtn">🔮 Predict Price</button>
            </form>
            <div id="result" class="result"></div>
        </div>
    </div>
    <script>
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('submitBtn');
            const result = document.getElementById('result');
            btn.disabled = true;
            btn.textContent = '⏳ Predicting...';
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        median_income: parseFloat(document.getElementById('median_income').value),
                        house_age: parseFloat(document.getElementById('house_age').value),
                        avg_rooms: parseFloat(document.getElementById('avg_rooms').value),
                        avg_bedrooms: parseFloat(document.getElementById('avg_bedrooms').value),
                        population: parseFloat(document.getElementById('population').value),
                        avg_occupancy: parseFloat(document.getElementById('avg_occupancy').value),
                        latitude: parseFloat(document.getElementById('latitude').value),
                        longitude: parseFloat(document.getElementById('longitude').value)
                    })
                });
                const data = await response.json();
                if (response.ok) {
                    result.className = 'result success';
                    result.innerHTML = '<div>Predicted Price</div><div class="price">$' + data.predicted_price.toLocaleString('en-US', {maximumFractionDigits: 0}) + '</div><div>Model v' + data.model_version + '</div>';
                } else {
                    result.className = 'result error';
                    result.innerHTML = '❌ ' + (data.detail || 'Prediction failed');
                }
            } catch (err) {
                result.className = 'result error';
                result.innerHTML = '❌ Connection error';
            }
            result.style.display = 'block';
            btn.disabled = false;
            btn.textContent = '🔮 Predict Price';
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    # In production, use gunicorn with multiple workers
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
