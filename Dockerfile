FROM python:3.11-slim

# Security: Don't run as root
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model (ONNX format - secure)
COPY model.py .
COPY model.onnx .
COPY model_metadata.json .
COPY model_hash.txt .

# Security: Change ownership and switch to non-root user
RUN chown -R appuser:appgroup /app
USER appuser

# Security: Don't expose unnecessary info
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
