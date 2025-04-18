# PaddleOCR ONNX REST API

A simple REST API that exposes PaddleOCR ONNX inference capabilities.

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements_onnx.txt
```

2. Start the API server:

```bash
python ocr_api.py
```

The server will start and listen on port 5001 by default.

## API Usage

### OCR Endpoint

**URL**: `/ocr`

**Method**: `POST`

**Content-Type**: `multipart/form-data`

**Form Parameters**:

- `image`: The image file to be processed

### Example Request

Using cURL:

```bash
curl -X POST -F "image=@./images/IMG_1538.png" http://localhost:5001/ocr
```

Using Python with requests:

```python
import requests

url = "http://localhost:5001/ocr"
files = {"image": open("./images/IMG_1538.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Example Response

```json
{
  "status": "success",
  "processing_time": {
    "detection": 0.0708,
    "classification": 0.0326,
    "recognition": 0.5622,
    "total": 0.8503
  },
  "results": [
    {
      "id": 1,
      "text": "THE LOTTERY",
      "confidence": 0.948,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    },
    {
      "id": 2,
      "text": "MEGA",
      "confidence": 0.968,
      "bounding_box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    },
    ...
  ]
}
```

## Error Responses

In case of an error, the API will return a JSON response with an error message:

```json
{
  "error": "Error message description"
}
```

Common error codes:

- `400 Bad Request`: No image file provided or unable to read the image
- `500 Internal Server Error`: Server-side error processing the image

## Performance

The API uses Method 3 (custom pipeline) from the PaddleOCR ONNX demo, which is the most efficient approach with processing times around 0.8-0.9 seconds for typical images.

## Customization

You can modify the following parameters in the `init_ocr_components` function:

- `args.drop_score`: Confidence threshold (default: 0.5)
- `args.use_gpu`: Enable GPU (default: False)
- `args.rec_char_dict_path`: Character dictionary path
