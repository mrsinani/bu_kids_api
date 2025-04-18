#!/usr/bin/env python
"""
PaddleOCR with ONNX Demo Script

This script demonstrates three different ways to use PaddleOCR with ONNX models:
1. Using the high-level PaddleOCR class
2. Using the predict_system.py script through subprocess
3. Using a custom OCR pipeline that directly uses the model components
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Method 1: Using the PaddleOCR class
def ocr_using_paddleocr_class(image_path, output_path):
    """
    Use the high-level PaddleOCR API to perform OCR
    """
    from paddleocr import PaddleOCR

    print("\n=== Method 1: Using PaddleOCR class ===")
    start_time = time.time()

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

    # Run OCR
    result = ocr.ocr(image_path, cls=True)

    # Process results
    if result and len(result) > 0:
        for idx, line in enumerate(result[0]):
            print(f"Text #{idx+1}: {line[1][0]}, Confidence: {line[1][1]:.3f}")
    
    # Visualize results
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes and text
    output_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(output_img)
    font_path = './doc/fonts/simfang.ttf'
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, 20)
    else:
        font = ImageFont.load_default()

    if result and len(result) > 0:
        for line in result[0]:
            # Draw bounding box
            box = line[0]
            draw.polygon([
                (box[0][0], box[0][1]),
                (box[1][0], box[1][1]),
                (box[2][0], box[2][1]),
                (box[3][0], box[3][1])
            ], outline=(0, 255, 0), width=2)
            
            # Draw text near the box
            text, confidence = line[1][0], line[1][1]
            draw.text((box[0][0], box[0][1] - 20), f"{text}", fill=(255, 0, 0), font=font)

    # Save result image
    output_path_method1 = f"{os.path.splitext(output_path)[0]}_method1.png"
    output_img.save(output_path_method1)

    end_time = time.time()
    print(f"Method 1 took {end_time - start_time:.4f} seconds")
    print(f"Result saved to {output_path_method1}")
    
    return result

# Method 2: Using predict_system.py via subprocess
def ocr_using_predict_system(image_path, output_path):
    """
    Use the predict_system.py script to perform OCR via subprocess
    """
    print("\n=== Method 2: Using predict_system.py via subprocess ===")
    start_time = time.time()
    
    # Build command
    cmd = [
        "python", "tools/infer/predict_system.py",
        "--use_gpu=False",
        "--use_onnx=True",
        "--det_model_dir=./inference/det_onnx/model.onnx",
        "--rec_model_dir=./inference/rec_onnx/model.onnx",
        "--cls_model_dir=./inference/cls_onnx/model.onnx",
        f"--image_dir={image_path}",
        "--rec_char_dict_path=ppocr/utils/en_dict.txt",
        "--draw_img_save_dir=./inference_results"
    ]
    
    # Execute command
    print(f"Running command: {' '.join(cmd)}")
    subprocess_result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print(subprocess_result.stdout)
    
    # Copy the result to our output path
    output_path_method2 = f"{os.path.splitext(output_path)[0]}_method2.png"
    img_name = os.path.basename(image_path)
    result_path = os.path.join("./inference_results", img_name)
    
    if os.path.exists(result_path):
        import shutil
        shutil.copy(result_path, output_path_method2)
        print(f"Result saved to {output_path_method2}")
    else:
        print(f"Result image not found at {result_path}")
    
    end_time = time.time()
    print(f"Method 2 took {end_time - start_time:.4f} seconds")
    
    # Read system_results.txt to get the detailed results
    system_results_path = os.path.join("./inference_results", "system_results.txt")
    if os.path.exists(system_results_path):
        with open(system_results_path, 'r', encoding='utf-8') as f:
            results_line = f.readline().strip()
            if img_name in results_line:
                import json
                results_json = results_line.split('\t')[1]
                results = json.loads(results_json)
                return results
    
    return None

# Method 3: Using custom OCR pipeline
def ocr_using_custom_pipeline(image_path, output_path):
    """
    Use a custom OCR pipeline that directly uses the model components
    """
    print("\n=== Method 3: Using custom OCR pipeline ===")
    start_time = time.time()
    
    # Import PaddleOCR components
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
    args.rec_batch_num = 6
    args.max_text_length = 25
    args.use_space_char = True
    
    # Initialize components
    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)
    text_classifier = TextClassifier(args)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Store original image
    ori_img = img.copy()
    
    # Step 1: Text detection
    dt_boxes, det_time = text_detector(img)
    print(f"Detection took {det_time:.4f}s, found {len(dt_boxes)} text regions")
    
    # Sort text boxes
    def sort_boxes(dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes
    
    dt_boxes = sort_boxes(dt_boxes)
    
    # Function to crop text regions
    def get_rotate_crop_image(img, points):
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), 
                                     borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
    
    # Prepare cropped text regions
    img_crop_list = []
    for box in dt_boxes:
        img_crop = get_rotate_crop_image(ori_img, box)
        img_crop_list.append(img_crop)
    
    # Step 2: Text direction classification
    img_crop_list, angle_list, cls_time = text_classifier(img_crop_list)
    print(f"Classification took {cls_time:.4f}s")
    
    # Step 3: Text recognition
    rec_res, rec_time = text_recognizer(img_crop_list)
    print(f"Recognition took {rec_time:.4f}s")
    
    # Filter by confidence score
    drop_score = args.drop_score
    filter_boxes, filter_rec_res = [], []
    for box, rec_result in zip(dt_boxes, rec_res):
        text, score = rec_result[0], rec_result[1]
        if score >= drop_score:
            filter_boxes.append(box)
            filter_rec_res.append(rec_result)
    
    # Print results
    print("\nOCR Results:")
    for idx, (box, (text, score)) in enumerate(zip(filter_boxes, filter_rec_res)):
        print(f"Text #{idx+1}: {text} (Confidence: {score:.3f})")
    
    # Draw results
    result_img = ori_img.copy()
    for box, (text, score) in zip(filter_boxes, filter_rec_res):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(result_img, [box], True, (0, 255, 0), 2)
        
        # Put text near the top-left corner of the box
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result_img, f"{text} ({score:.2f})", 
                   (box[0][0], box[0][1] - 10), 
                   font, 0.5, (0, 0, 255), 1)
    
    # Save result
    output_path_method3 = f"{os.path.splitext(output_path)[0]}_method3.png"
    cv2.imwrite(output_path_method3, result_img)
    
    end_time = time.time()
    print(f"Method 3 took {end_time - start_time:.4f} seconds")
    print(f"Result saved to {output_path_method3}")
    
    # Format results like the other methods
    results = []
    for box, (text, score) in zip(filter_boxes, filter_rec_res):
        results.append({
            "transcription": text,
            "points": box.tolist(),
            "confidence": float(score)
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR with ONNX Demo")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_path", type=str, default="ocr_result.png", help="Path to save output image")
    parser.add_argument("--methods", type=str, default="1,2,3", help="Methods to run (comma-separated: 1,2,3)")
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Parse methods to run
    methods = [int(m) for m in args.methods.split(",")]
    
    # Run selected methods
    results = []
    
    if 1 in methods:
        result1 = ocr_using_paddleocr_class(args.image_path, args.output_path)
        results.append(("PaddleOCR class", result1))
    
    if 2 in methods:
        result2 = ocr_using_predict_system(args.image_path, args.output_path)
        results.append(("predict_system.py", result2))
    
    if 3 in methods:
        result3 = ocr_using_custom_pipeline(args.image_path, args.output_path)
        results.append(("Custom pipeline", result3))
    
    print("\n=== Summary ===")
    print(f"Image: {args.image_path}")
    for method_name, result in results:
        if result:
            text_count = len(result) if isinstance(result, list) else len(result[0]) if result and len(result) > 0 else 0
            print(f"{method_name}: Detected {text_count} text regions")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 