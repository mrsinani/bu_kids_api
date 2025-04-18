# Railway Deployment Guide for PaddleOCR API

This guide provides step-by-step instructions for deploying the PaddleOCR API to Railway.

## Prerequisites

1. A [Railway](https://railway.app/) account
2. The [Railway CLI](https://docs.railway.app/develop/cli) installed (optional, for local development)
3. Git installed on your machine

## Deployment Steps

### 1. Prepare your repository

Ensure your repository has the following files:

- `ocr_api.py` - The Flask API application
- `Procfile` - Tells Railway how to run your application
- `runtime.txt` - Specifies the Python version
- `requirements_railway.txt` - Lists all the dependencies
- All necessary model files in the expected directories

### 2. Deploy to Railway via GitHub

1. Log in to [Railway Dashboard](https://railway.app/dashboard)
2. Click on "New Project"
3. Select "Deploy from GitHub repo"
4. Select your repository
5. Configure the deployment:
   - Set the environment to "Python"
   - Set the build command to `pip install -r requirements_railway.txt`
   - Set the start command to `gunicorn ocr_api:app --log-file -`

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

## Model Files

Ensure that all the ONNX model files are properly included in your repository:

- `./inference/det_onnx/model.onnx` - Text detection model
- `./inference/rec_onnx/model.onnx` - Text recognition model
- `./inference/cls_onnx/model.onnx` - Text classification model
- `./ppocr/utils/en_dict.txt` - Character dictionary

## Troubleshooting

1. If deployment fails, check the logs in the Railway dashboard
2. Ensure all required model files are in the correct locations
3. Verify that the `uploads` directory is created at startup

## Additional Notes

- The `uploads` directory is created automatically at startup
- Railway automatically handles HTTPS certificates
- Railway will restart your application if it crashes
