# PaddleOCR with ONNX Inference

This guide explains how to use PaddleOCR with ONNX models for efficient inference across platforms.

## Installation

1. Install PaddleOCR and dependencies:

```bash
# Install PaddlePaddle
pip install paddlepaddle

# Install ONNX Runtime
pip install onnxruntime

# Install PaddleOCR (from repository root)
pip install -e .

# Install paddle2onnx (for model conversion)
pip install paddle2onnx
```

## Model Directory Structure

The ONNX models should be placed in the following directory structure:

```
inference/
├── det_onnx/
│   └── model.onnx     # Text detection model
├── rec_onnx/
│   └── model.onnx     # Text recognition model
└── cls_onnx/
    └── model.onnx     # Text direction classification model
```

## Converting PaddleOCR Models to ONNX

If you don't have the ONNX models yet, you can convert PaddleOCR models using paddle2onnx:

```bash
# Download Paddle models
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
cd ./inference && tar xf en_PP-OCRv3_det_infer.tar && cd ..

wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf en_PP-OCRv3_rec_infer.tar && cd ..

wget -nc -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..

# Convert to ONNX
mkdir -p inference/det_onnx inference/rec_onnx inference/cls_onnx

paddle2onnx --model_dir ./inference/en_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/en_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/cls_onnx/model.onnx \
--opset_version 11 \
--enable_onnx_checker True
```

## Usage Methods

### Method 1: Using the PaddleOCR High-Level API

The simplest way to use PaddleOCR with ONNX is through the high-level PaddleOCR class:

```python
from paddleocr import PaddleOCR

# Initialize PaddleOCR with ONNX support
ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=False,
    use_onnx=True,
    det_model_dir='./inference/det_onnx/model.onnx',
    rec_model_dir='./inference/rec_onnx/model.onnx',
    cls_model_dir='./inference/cls_onnx/model.onnx',
    rec_char_dict_path='ppocr/utils/en_dict.txt',
    show_log=False
)

# Run OCR on an image
result = ocr.ocr('path/to/your/image.jpg', cls=True)

# Process results
for idx, line in enumerate(result[0]):
    print(f"Text #{idx+1}: {line[1][0]}, Confidence: {line[1][1]:.3f}")
```

### Method 2: Using the Command-Line Tool

PaddleOCR provides a command-line tool for OCR inference:

```bash
python tools/infer/predict_system.py \
  --use_gpu=False \
  --use_onnx=True \
  --det_model_dir=./inference/det_onnx/model.onnx \
  --rec_model_dir=./inference/rec_onnx/model.onnx \
  --cls_model_dir=./inference/cls_onnx/model.onnx \
  --image_dir=path/to/your/image.jpg \
  --rec_char_dict_path=ppocr/utils/en_dict.txt
```

The results will be saved to `./inference_results/`.

### Method 3: Using the Custom OCR Pipeline

For more control over the OCR process, you can use the individual components directly:

```python
import tools.infer.utility as utility
from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_cls import TextClassifier

# Initialize parser and set arguments
parser = utility.init_args()
args = parser.parse_args([])

# Set parameters
args.use_gpu = False
args.use_onnx = True
args.det_model_dir = './inference/det_onnx/model.onnx'
args.rec_model_dir = './inference/rec_onnx/model.onnx'
args.cls_model_dir = './inference/cls_onnx/model.onnx'
args.rec_char_dict_path = './ppocr/utils/en_dict.txt'
args.use_angle_cls = True
args.rec_image_shape = '3, 48, 320'
args.rec_algorithm = 'SVTR_LCNet'

# Initialize components
text_detector = TextDetector(args)
text_recognizer = TextRecognizer(args)
text_classifier = TextClassifier(args)

# Process image
img = cv2.imread('path/to/your/image.jpg')
dt_boxes, _ = text_detector(img)
img_crop_list = [get_rotate_crop_image(img, box) for box in dt_boxes]

if args.use_angle_cls:
    img_crop_list, _, _ = text_classifier(img_crop_list)

rec_res, _ = text_recognizer(img_crop_list)

# Process results
for idx, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
    if score >= args.drop_score:
        print(f"Text #{idx+1}: {text} (Confidence: {score:.3f})")
```

## Performance Comparison

Performance comparison between the three methods on a sample image:

| Method            | Processing Time | Benefits                                 |
| ----------------- | --------------- | ---------------------------------------- |
| PaddleOCR Class   | ~1.8s           | Simplest API, easy to use                |
| Command-Line Tool | ~2.8s           | No coding required, good for quick tests |
| Custom Pipeline   | ~0.8s           | Most efficient, highest customization    |

## Demo Script

A comprehensive demo script showing all three methods is available:

```bash
python paddleocr_onnx_demo.py --image_path=path/to/your/image.jpg --methods=1,2,3
```

## Supported ONNX Providers

You can specify different ONNX providers based on your hardware:

- For CPU: `CPUExecutionProvider`
- For CUDA: `CUDAExecutionProvider`
- For TensorRT: `TensorrtExecutionProvider`

Example:

```python
ocr = PaddleOCR(
    use_onnx=True,
    det_model_dir='./inference/det_onnx/model.onnx',
    rec_model_dir='./inference/rec_onnx/model.onnx',
    cls_model_dir='./inference/cls_onnx/model.onnx',
    use_gpu=True,
    onnx_providers=['CUDAExecutionProvider']
)
```

## Troubleshooting

If you encounter issues with character encoding or recognition:

1. Make sure you're using the correct dictionary file for your language:

   - English: `ppocr/utils/en_dict.txt`
   - Chinese: `ppocr/utils/ppocr_keys_v1.txt`

2. Set the correct recognition algorithm and image shape:

   - For PP-OCRv3: `rec_image_shape='3, 48, 320'`
   - For PP-OCRv2: `rec_image_shape='3, 32, 320'`

3. If ONNX inference is slow, try optimizing your model:
   ```bash
   python -m paddle2onnx.optimize --input_model inference/det_onnx/model.onnx \
   --output_model inference/det_onnx/model.onnx \
   --input_shape_dict "{'x': [-1,3,-1,-1]}"
   ```
