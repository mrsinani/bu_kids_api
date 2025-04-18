#!/usr/bin/env python3

import io
import os
import sys
import base64
import uvicorn
import tempfile
import numpy as np
import cv2
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ocr_pipeline import OCRPipeline, get_model_paths


# Define data models
class OcrResponse(BaseModel):
    text: str
    box: List[List[float]]
    confidence: float


class OcrRequest(BaseModel):
    image_base64: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="BU Kids OCR API",
    description="Text detection and recognition API using PaddleOCR models in ONNX format.",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR pipeline
ocr_pipeline = None


@app.on_event("startup")
def startup_event():
    """Initialize resources on startup"""
    global ocr_pipeline
    try:
        # Get model paths
        det_model, cls_model, rec_model, dict_path = get_model_paths()
        
        # Check if models exist
        for model_path in [det_model, cls_model, rec_model]:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize OCR pipeline
        ocr_pipeline = OCRPipeline(det_model, cls_model, rec_model, dict_path)
    except Exception as e:
        print(f"Error initializing OCR pipeline: {e}")
        sys.exit(1)


@app.post("/ocr", response_model=List[OcrResponse])
async def perform_ocr(file: Optional[UploadFile] = File(None), base64_image: Optional[str] = Form(None)):
    """
    Perform OCR on an image
    
    The image can be provided either as a file upload or as a base64-encoded string.
    """
    if not ocr_pipeline:
        raise HTTPException(status_code=500, detail="OCR pipeline not initialized")
    
    try:
        # Read image
        if file:
            # Read from file upload
            img_bytes = await file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        elif base64_image:
            # Read from base64 string
            img_bytes = base64.b64decode(base64_image.split(',')[-1])
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, img)
        
        try:
            # Run OCR pipeline
            results = ocr_pipeline(temp_path)
            return results
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/ocr/base64", response_model=List[OcrResponse])
async def ocr_from_base64(request: OcrRequest):
    """
    Perform OCR on a base64-encoded image
    
    This endpoint is useful for JavaScript clients that want to send base64 encoded images.
    """
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="No image provided")
    
    # Strip data URL prefix if present
    if ',' in request.image_base64:
        base64_str = request.image_base64.split(',', 1)[1]
    else:
        base64_str = request.image_base64
    
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode base64 image")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, img)
        
        try:
            # Run OCR pipeline
            results = ocr_pipeline(temp_path)
            return results
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """Check if the API is healthy"""
    if ocr_pipeline:
        return {"status": "healthy"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "OCR pipeline not initialized"}
        )


def start_server(host="0.0.0.0", port=8000):
    """Start the API server"""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server() 