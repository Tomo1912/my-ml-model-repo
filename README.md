# 🏠 House Price Prediction API

A production-ready Machine Learning API for predicting California house prices, deployed using modern MLOps practices.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Kubernetes](https://img.shields.io/badge/Kubernetes-K3s-blue)
![ArgoCD](https://img.shields.io/badge/ArgoCD-GitOps-red)

## 🎯 Project Overview

This project demonstrates a complete **end-to-end MLOps pipeline** including:

- ✅ Machine Learning model training and evaluation
- ✅ REST API with FastAPI
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions
- ✅ GitOps deployment with ArgoCD
- ✅ Kubernetes orchestration (K3s)
- ✅ Cloud hosting (Hetzner)

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Developer     │────▶│   GitHub Repo    │────▶│ GitHub Actions  │
│   (Push Code)   │     │  (my-ml-model)   │     │   (CI/CD)       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Kubernetes    │◀────│     ArgoCD       │◀────│  GitOps Repo    │
│   (K3s Cloud)   │     │   (Auto-Sync)    │     │ (my-mlops-gitops)│
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│   ML API Pod    │
│  (FastAPI)      │
└─────────────────┘
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Algorithm** | Random Forest Regressor |
| **R² Score** | 0.80 (80% variance explained) |
| **MAE** | $33,291 |
| **Training Samples** | 16,512 |
| **Test Samples** | 4,128 |

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| **API** | http://37.27.8.233:32199 |
| **Swagger Docs** | http://37.27.8.233:32199/docs |
| **ArgoCD Dashboard** | http://37.27.8.233:30082 |

## 📡 API Endpoints

### Health Check
```bash
curl http://37.27.8.233:32199/health
```
Response:
```json
{"status": "healthy", "model_loaded": true, "service": "House Price Prediction API"}
```

### Model Info
```bash
curl http://37.27.8.233:32199/model/info
```

### Predict House Price
```bash
curl -X POST http://37.27.8.233:32199/predict \
  -H "Content-Type: application/json" \
  -d '{
    "median_income": 8.3252,
    "house_age": 41,
    "avg_rooms": 6.98,
    "avg_bedrooms": 1.02,
    "population": 322,
    "avg_occupancy": 2.55,
    "latitude": 37.88,
    "longitude": -122.23
  }'
```
Response:
```json
{
  "predicted_price": 423567.01,
  "model_version": "1.0.0",
  "features_used": {...}
}
```

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | scikit-learn, pandas, numpy |
| **API Framework** | FastAPI, Pydantic, Uvicorn |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Container Registry** | GitHub Container Registry (GHCR) |
| **Orchestration** | Kubernetes (K3s) |
| **GitOps** | ArgoCD |
| **Cloud Provider** | Hetzner Cloud |

## 📁 Project Structure

```
my-ml-model-repo/
├── model.py              # FastAPI application
├── train_model.py        # Model training script
├── model.pkl             # Trained ML model
├── model_metadata.json   # Model metrics & metadata
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
└── .github/
    └── workflows/
        └── main.yml      # CI/CD pipeline
```

## 🔄 CI/CD Pipeline

The pipeline automatically triggers on push to `main` branch:

1. **Build** - Docker image built with model and API
2. **Push** - Image pushed to GitHub Container Registry
3. **Update** - GitOps repo updated with new image tag
4. **Deploy** - ArgoCD detects change and deploys to Kubernetes

## 🏃 Local Development

```bash
# Clone repository
git clone https://github.com/Tomo1912/my-ml-model-repo.git
cd my-ml-model-repo

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (optional - pre-trained model included)
python train_model.py

# Run API locally
uvicorn model:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Dataset

The model is trained on the **California Housing Dataset** from scikit-learn, which contains:
- 20,640 samples of California housing data
- 8 features including median income, house age, location, etc.
- Target: Median house value (in $100,000s)

