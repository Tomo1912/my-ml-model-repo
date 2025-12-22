FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY model.py .
COPY model.pkl .
COPY model_metadata.json .

EXPOSE 8000
CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000"]
