# Base image — slim Python to keep image size small
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by Pillow and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Copy trained model artifact
COPY model.h5 .

# Environment variable so app.py knows where the model lives
ENV MODEL_PATH=/app/model.h5

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
