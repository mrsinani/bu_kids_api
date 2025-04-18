FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models if they don't exist
RUN mkdir -p models/onnx

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["python", "main.py", "api", "--host", "0.0.0.0", "--port", "8000"] 