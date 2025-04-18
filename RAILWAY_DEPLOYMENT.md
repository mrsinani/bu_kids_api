# Railway Deployment Guide for PaddleOCR API

This guide provides step-by-step instructions for deploying the PaddleOCR API to Railway.

## Prerequisites

1. A [Railway](https://railway.app/) account
2. The [Railway CLI](https://docs.railway.app/develop/cli) installed (optional, for local development)
3. Git installed on your machine

## Deployment Options

There are multiple ways to deploy this application to Railway:

### Option 1: Deploy with Nixpacks (Recommended)

1. Ensure your repository has these files:

   - `ocr_api.py` - The Flask API application
   - `requirements_railway.txt` - Python dependencies with headless OpenCV
   - `nixpacks.toml` - System dependencies configuration
   - `.railway.json` - Railway configuration

2. Railway will automatically use Nixpacks to build your application.

### Option 2: Deploy with Dockerfile

1. Ensure your repository has the Dockerfile we've provided
2. When creating a new project in Railway, select "Deploy from Dockerfile"

## Step-by-Step Deployment

### 1. Prepare your repository

Ensure your repository has the necessary files based on your chosen deployment option.

### 2. Deploy to Railway via GitHub

1. Log in to [Railway Dashboard](https://railway.app/dashboard)
2. Click on "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Railway will automatically detect the configuration

### 3. Set Environment Variables

In the Railway dashboard, go to your project and set the following environment variables:

- Click on "Variables" tab
- Add any necessary environment variables (if needed)

### 4. Configure the Domain

1. Go to the "Settings" tab
2. In the "Domains" section, click "Generate Domain"
3. Railway will provide a public domain for your API

### 5. Verify Deployment

1. Once deployed, check the logs to ensure everything is running correctly
2. Test your API using the provided Railway domain:
   ```
   curl -X POST -F "image=@path/to/image.jpg" https://your-railway-domain.railway.app/ocr
   ```

## Troubleshooting OpenCV Issues

If you encounter issues with OpenCV dependencies:

1. We've switched to `opencv-python-headless` which doesn't require GUI libraries
2. The Dockerfile approach explicitly installs all necessary system dependencies
3. The nixpacks.toml file specifies system packages needed for building

## Model Files

Ensure that all the ONNX model files are properly included in your repository:

- `./inference/det_onnx/model.onnx` - Text detection model
- `./inference/rec_onnx/model.onnx` - Text recognition model
- `./inference/cls_onnx/model.onnx` - Text classification model
- `./ppocr/utils/en_dict.txt` - Character dictionary

## Additional Notes

- The `uploads` directory is created automatically at startup
- Railway automatically handles HTTPS certificates
- Railway will restart your application if it crashes
