# BU Kids OCR API

A streamlined API for Optical Character Recognition (OCR) using PaddleOCR models converted to ONNX format.

## Features

- Text detection, recognition, and orientation classification
- Fast inference with ONNX Runtime
- Clean and simple API with FastAPI
- Multiple usage modes: API, CLI, or direct library integration

## Setup

### Prerequisites

- Python 3.7 or higher
- Models in ONNX format (see "Preparing Models" section)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/bu-kids-ocr.git
cd bu-kids-ocr
```

2. Install dependencies:

```bash
pip install -e .
```

### Preparing Models

The API requires three ONNX models for OCR:

- Text detection model
- Text classification model
- Text recognition model

You can convert PaddleOCR models to ONNX format using the built-in conversion tool:

```bash
# From the PaddleOCR directory
cd /path/to/PaddleOCR

# Download PaddleOCR models if needed
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
cd ./inference && tar xf en_PP-OCRv3_det_infer.tar && cd ..

wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf en_PP-OCRv3_rec_infer.tar && cd ..

wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..

# Install paddle2onnx if needed
pip install paddle2onnx

# Convert models
cd bu_kids_api
python main.py convert --paddle-dir ../inference
```

After conversion, the ONNX models will be in the `models/onnx` directory.

## Usage

### As an API Server

```bash
# Start the API server
python main.py api --port 8000
```

The API will be available at http://localhost:8000. You can test it using:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### As a CLI Tool

```bash
# Process a single image
python main.py cli --image path/to/image.jpg --output result.jpg

# Save results as JSON
python main.py cli --image path/to/image.jpg --json results.json
```

### As a Python Library

```python
from bu_kids_api.src.ocr_pipeline import OCRPipeline, get_model_paths

# Get model paths
det_model, cls_model, rec_model, dict_path = get_model_paths()

# Initialize OCR pipeline
ocr = OCRPipeline(det_model, cls_model, rec_model, dict_path)

# Process an image
results = ocr("path/to/image.jpg")

# Print detected text
for result in results:
    print(f"Text: {result['text']}, Confidence: {result['confidence']}")
```

## API Endpoints

- `POST /ocr`: Perform OCR on an uploaded image file
- `POST /ocr/base64`: Perform OCR on a base64-encoded image
- `GET /health`: Check if the API is healthy

## Example Test

```bash
# Test with a sample image
python main.py cli --image samples/test.jpg --output result.jpg
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This API is based on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models.
