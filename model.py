# model.py
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="ML Deployment Service")


# Dummy model: y = 2x + 5
# Ovo je nova, nebitna linija koda koja okida pipeline...
# Dodajemo razmak da forsiramo trigger...
@app.get("/predict/{number}")
def predict(number: int):
    """Simple linear prediction."""
    result = 2 * number + 6
    return {"input": number, "prediction": result, "model_version": "v1.0.0"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
