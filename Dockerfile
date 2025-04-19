FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_railway.txt .

# Install specific dependencies that might need extra attention
RUN pip install --no-cache-dir numpy>=1.20.0 opencv-python-headless>=4.5.4
RUN pip install --no-cache-dir albumentations>=1.0.0
RUN pip install --no-cache-dir -r requirements_railway.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8080

# Start gunicorn
CMD ["gunicorn", "ocr_api:app", "--bind", "0.0.0.0:8080", "--log-file", "-"] 