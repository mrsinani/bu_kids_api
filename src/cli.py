#!/usr/bin/env python3

import argparse
import cv2
import json
import os
import sys
from pathlib import Path

from .ocr_pipeline import OCRPipeline, get_model_paths


def visualize_results(image_path, results, output_path=None):
    """Visualize OCR results on the image"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # Draw bounding boxes and text
    for result in results:
        box = result["box"]
        text = result["text"]
        
        # Convert box to contour format
        box = [tuple(map(int, point)) for point in box]
        
        # Draw contour
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        
        # Get text position
        text_pos = (min(p[0] for p in box), min(p[1] for p in box) - 10)
        
        # Draw text
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to {output_path}")
    else:
        # Display in a window
        cv2.imshow("OCR Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="OCR Pipeline for Text Detection and Recognition")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to save visualization output (if not provided, will display)")
    parser.add_argument("--json", "-j", help="Path to save results as JSON")
    parser.add_argument("--det-model", help="Path to detection model")
    parser.add_argument("--cls-model", help="Path to classification model")
    parser.add_argument("--rec-model", help="Path to recognition model")
    parser.add_argument("--dict", help="Path to dictionary file")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return 1
    
    # Get model paths
    det_model, cls_model, rec_model, dict_path = get_model_paths()
    
    # Override with user-provided paths if any
    if args.det_model:
        det_model = args.det_model
    if args.cls_model:
        cls_model = args.cls_model
    if args.rec_model:
        rec_model = args.rec_model
    if args.dict:
        dict_path = args.dict
    
    # Check if models exist
    for model_path in [det_model, cls_model, rec_model]:
        if not os.path.isfile(model_path):
            print(f"Error: Model file '{model_path}' not found")
            return 1
    
    # Initialize pipeline
    try:
        ocr_pipeline = OCRPipeline(det_model, cls_model, rec_model, dict_path)
    except Exception as e:
        print(f"Error initializing OCR pipeline: {e}")
        return 1
    
    # Process image
    try:
        results = ocr_pipeline(args.image)
        print(f"Detected {len(results)} text regions")
        
        # Print results
        for i, result in enumerate(results):
            print(f"Text {i+1}: {result['text']} (Confidence: {result['confidence']:.2f})")
        
        # Save results as JSON if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.json}")
        
        # Visualize results
        visualize_results(args.image, results, args.output)
        
        return 0
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 