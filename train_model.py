"""
Train House Price Prediction Model
This script trains a Random Forest model on California Housing dataset
and saves it for production use.
"""

import pickle
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json


def train_model():
    """Train and save the house price prediction model."""
    
    print("📊 Loading California Housing dataset...")
    # Load dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target  # Median house value in $100,000s
    
    # Feature names
    feature_names = housing.feature_names
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
    
    # Save model
    print("\n💾 Saving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "n_estimators": 100,
        "features": feature_names,
        "mae": float(mae * 100000),
        "r2_score": float(r2),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "version": "1.0.0"
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Model saved successfully!")
    print(f"   - model.pkl ({model.__sizeof__()} bytes)")
    print(f"   - model_metadata.json")
    
    return model, metadata


if __name__ == "__main__":
    train_model()

