# PaddleOCR with ONNX Inference

This repository contains a simplified version of PaddleOCR focused on ONNX inference for optical character recognition.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements_onnx.txt
```

2. Run the demo on the sample image:

```bash
python paddleocr_onnx_demo.py --image_path=./images/IMG_1538.png --methods=1
```

## Demo Script Options

The `paddleocr_onnx_demo.py` script demonstrates three methods for OCR inference:

- Method 1: Using the high-level PaddleOCR class
- Method 2: Using the predict_system.py command-line tool
- Method 3: Using a custom OCR pipeline

You can specify which methods to run:

```bash
python paddleocr_onnx_demo.py --image_path=./images/IMG_1538.png --methods=1,2,3
```

## Comprehensive Documentation

For detailed instructions on using PaddleOCR with ONNX models, refer to:
[README_ONNX_INFERENCE.md](README_ONNX_INFERENCE.md)

## Model Information

This repository includes ONNX models for:

- Text Detection (`inference/det_onnx/model.onnx`)
- Text Recognition (`inference/rec_onnx/model.onnx`)
- Text Direction Classification (`inference/cls_onnx/model.onnx`)

## License

PaddleOCR is released under the Apache 2.0 license.
