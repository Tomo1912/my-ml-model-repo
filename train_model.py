"""
Train House Price Prediction Model
This script trains a Random Forest model on California Housing dataset
and saves it in ONNX format for secure production use.

SECURITY: ONNX format is safer than pickle because:
- It only stores model weights and structure, not arbitrary Python code
- Cannot execute arbitrary code during deserialization
- Industry standard for ML model interchange
"""

import hashlib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json


def train_model():
    """Train and save the house price prediction model in ONNX format."""

    print("📊 Loading California Housing dataset...")
    # Load dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target  # Median house value in $100,000s

    # Feature names (convert to list for JSON serialization)
    feature_names = list(housing.feature_names)
    print(f"Features: {feature_names}")
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🤖 Training Random Forest model...")
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: ${mae * 100000:.2f}")
    print(f"R² Score: {r2:.4f}")

    # Convert to ONNX format (SECURE - no arbitrary code execution)
    print("\n💾 Converting to ONNX format (secure serialization)...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save ONNX model
    with open('model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())

    # Generate SHA256 hash for integrity verification
    with open('model.onnx', 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    with open('model_hash.txt', 'w') as f:
        f.write(model_hash)

    # Save metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "model_format": "ONNX",
        "n_estimators": 100,
        "features": feature_names,
        "mae": float(mae * 100000),
        "r2_score": float(r2),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "version": "2.0.0",
        "model_hash": model_hash
    }

    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("✅ Model saved successfully!")
    print(f"   - model.onnx (ONNX format - secure)")
    print(f"   - model_hash.txt (SHA256: {model_hash[:16]}...)")
    print(f"   - model_metadata.json")

    return model, metadata


if __name__ == "__main__":
    train_model()

