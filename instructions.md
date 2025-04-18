# PaddleOCR ONNX API Instructions

This document provides instructions for setting up and using the OCR API that extracts text from images.

## Prerequisites

- Python 3.7+
- Required Python packages:
  - Flask
  - OpenCV (cv2)
  - NumPy
  - PaddleOCR
  - ONNX Runtime

## Setup

1. **Install dependencies**

```bash
pip install flask opencv-python numpy paddlepaddle paddleocr onnxruntime
```

2. **Download ONNX models**

Make sure you have the following model files in the correct locations:

- `./inference/det_onnx/model.onnx` - Text detection model
- `./inference/rec_onnx/model.onnx` - Text recognition model
- `./inference/cls_onnx/model.onnx` - Text classification model
- `./ppocr/utils/en_dict.txt` - Character dictionary for recognition

You can download these models from the [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR).

3. **Create the uploads directory**

```bash
mkdir uploads
```

## Running the API

Start the API server:

```bash
python ocr_api.py
```

The server will run on `http://0.0.0.0:5001`.

## API Usage

### General OCR Endpoint

**Endpoint**: `POST /ocr`

**Request**:

- Content-Type: `multipart/form-data`
- Form parameter: `image` (file)

**Sample Request (using curl)**:

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5001/ocr
```

**Sample Request (using Python)**:

```python
import requests

url = "http://localhost:5001/ocr"
files = {"image": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Sample Response**:

```json
{
  "status": "success",
  "processing_time": {
    "detection": 0.125,
    "classification": 0.075,
    "recognition": 0.3,
    "total": 0.5
  },
  "results": [
    {
      "id": 1,
      "text": "Sample Text",
      "confidence": 0.98,
      "bounding_box": [
        [10, 10],
        [100, 10],
        [100, 30],
        [10, 30]
      ]
    },
    {
      "id": 2,
      "text": "Another Text",
      "confidence": 0.95,
      "bounding_box": [
        [10, 40],
        [120, 40],
        [120, 60],
        [10, 60]
      ]
    }
  ]
}
```

### Lottery Ticket Information Endpoint

**Endpoint**: `POST /lottery`

This specialized endpoint extracts specific information from lottery tickets:

- Ticket number
- Game type (Mega Millions, Powerball, etc.)
- Date
- Megaball number

**Request**:

- Content-Type: `multipart/form-data`
- Form parameter: `image` (file)

**Sample Request (using curl)**:

```bash
curl -X POST -F "image=@path/to/your/lottery_ticket.jpg" http://localhost:5001/lottery
```

**Sample Request (using Python)**:

```python
import requests

url = "http://localhost:5001/lottery"
files = {"image": open("path/to/your/lottery_ticket.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Sample Response**:

```json
{
  "status": "success",
  "processing_time": {
    "detection": 0.135,
    "classification": 0.089,
    "recognition": 0.312,
    "total": 0.536
  },
  "lottery_info": {
    "ticket_number": "1234567890123456",
    "game_type": "megamillions",
    "date": "04/18/2023",
    "megaball_number": "25"
  },
  "ocr_results": [
    {
      "id": 1,
      "text": "MEGA MILLIONS",
      "confidence": 0.95,
      "bounding_box": [
        [10, 10],
        [150, 10],
        [150, 30],
        [10, 30]
      ]
    },
    {
      "id": 2,
      "text": "Ticket #: 1234567890123456",
      "confidence": 0.97,
      "bounding_box": [
        [10, 40],
        [200, 40],
        [200, 60],
        [10, 60]
      ]
    },
    {
      "id": 3,
      "text": "04/18/2023",
      "confidence": 0.98,
      "bounding_box": [
        [10, 70],
        [100, 70],
        [100, 90],
        [10, 90]
      ]
    },
    {
      "id": 4,
      "text": "MB: 25",
      "confidence": 0.94,
      "bounding_box": [
        [10, 100],
        [60, 100],
        [60, 120],
        [10, 120]
      ]
    }
  ]
}
```

## Testing the API

To test the API with a sample image:

1. Find an image with text (a screenshot, document scan, photo of text, etc.)
2. Use the curl command or Python script examples above with your image file
3. The API will return the detected text in JSON format

Example test using curl:

```bash
# General OCR test
curl -X POST -F "image=@test_images/sample.jpg" http://localhost:5001/ocr

# Lottery ticket test
curl -X POST -F "image=@test_images/lottery_ticket.jpg" http://localhost:5001/lottery
```

### Testing with the Test Client

You can also use the provided test client script:

```bash
# General OCR test
python test_client.py test_images/sample.jpg

# Lottery ticket test
python test_client.py test_images/lottery_ticket.jpg --url http://localhost:5001/lottery
```

## Performance Considerations

- The OCR API uses ONNX models which are optimized for CPU inference
- For larger images, processing might take longer
- The API implements a confidence threshold (0.5 by default) to filter out low-confidence text detections
- Processing time is included in the response for performance monitoring

## Troubleshooting

- If you encounter a "No module named..." error, make sure all dependencies are installed.
- If model loading fails, verify that the model files are in the correct locations.
- The API has a 16MB upload size limit for images.
- If the API returns no results, try with a clearer image or adjust the confidence threshold in the code.
- For lottery tickets, make sure the image is clear and text is readable. The lottery parser works best with clear, well-lit images.

## Security Notes

- This API is meant for development/internal use. For production deployment, add appropriate authentication.
- The API automatically deletes uploaded images after processing to save space.
