#!/usr/bin/env python
"""
PaddleOCR ONNX API

A simple Flask API that accepts image uploads and returns OCR results as JSON.
Uses the efficient custom OCR pipeline (Method 3) from the demo script.
"""

import os
import time
import uuid
import cv2
import numpy as np
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import PaddleOCR components
import tools.infer.utility as utility
from tools.infer.predict_det import TextDetector
from tools.infer.predict_rec import TextRecognizer
from tools.infer.predict_cls import TextClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OCR components
def init_ocr_components():
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
    args.drop_score = 0.5  # Confidence threshold
    
    # Initialize components
    text_detector = TextDetector(args)
    text_recognizer = TextRecognizer(args)
    text_classifier = TextClassifier(args)
    
    return args, text_detector, text_recognizer, text_classifier

# Initialize once at startup
args, text_detector, text_recognizer, text_classifier = init_ocr_components()

# Helper function to crop text regions
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

# Sort text boxes based on position (top to bottom, left to right)
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

# Function to parse lottery ticket information from OCR results
def parse_lottery_info(ocr_results):
    ticket_number = None
    game_type = None
    date = None
    megaball_number = None
    
    # Common game types
    game_patterns = {
        'megamillions': ['mega millions', 'megamillions'],
        'powerball': ['powerball', 'power ball'],
        'lotto': ['lotto', 'lottery']
    }
    
    # Extract information from OCR results
    for item in ocr_results:
        text = item['text'].lower()
        
        # Look for ticket number (typically 12-24 digits)
        ticket_match = re.search(r'\b\d{10,24}\b', text)
        if ticket_match and not ticket_number:
            ticket_number = ticket_match.group(0)
        
        # Look for game type
        for game, patterns in game_patterns.items():
            if any(pattern in text for pattern in patterns) and not game_type:
                game_type = game
        
        # Look for date (in various formats)
        date_match = re.search(r'\b\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}\b', text)
        if date_match and not date:
            date = date_match.group(0)
        
        # Look for Megaball number (often prefixed with "MB" or similar)
        megaball_match = re.search(r'(?:mb|megaball|mega ball)[^\d]*(\d+)', text)
        if megaball_match and not megaball_number:
            megaball_number = megaball_match.group(1)
    
    return {
        'ticket_number': ticket_number,
        'game_type': game_type,
        'date': date,
        'megaball_number': megaball_number
    }

@app.route('/ocr', methods=['POST'])
def ocr():
    # Check if request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser also submits an empty part
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate a unique filename
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the file
    file.save(filepath)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({'error': 'Unable to read image'}), 400
        
        # Store original image
        ori_img = img.copy()
        
        # Step 1: Text detection
        dt_boxes, det_time = text_detector(img)
        
        # Sort text boxes
        dt_boxes = sort_boxes(dt_boxes)
        
        # Prepare cropped text regions
        img_crop_list = []
        for box in dt_boxes:
            img_crop = get_rotate_crop_image(ori_img, box)
            img_crop_list.append(img_crop)
        
        # Step 2: Text direction classification
        img_crop_list, angle_list, cls_time = text_classifier(img_crop_list)
        
        # Step 3: Text recognition
        rec_res, rec_time = text_recognizer(img_crop_list)
        
        # Filter by confidence score
        results = []
        for idx, (box, rec_result) in enumerate(zip(dt_boxes, rec_res)):
            text, confidence = rec_result[0], rec_result[1]
            if confidence >= args.drop_score:
                # Convert box to a list of coordinates for JSON serialization
                box_coords = box.tolist()
                results.append({
                    'id': idx + 1,
                    'text': text,
                    'confidence': float(confidence),
                    'bounding_box': box_coords
                })
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Delete the uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'status': 'success',
            'processing_time': {
                'detection': float(det_time),
                'classification': float(cls_time),
                'recognition': float(rec_time),
                'total': float(total_time)
            },
            'results': results
        })
        
    except Exception as e:
        # Delete the uploaded file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/lottery', methods=['POST'])
def lottery_ocr():
    # Check if request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # If user does not select file, browser also submits an empty part
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate a unique filename
    filename = str(uuid.uuid4()) + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the file
    file.save(filepath)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Read image
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            return jsonify({'error': 'Unable to read image'}), 400
        
        # Store original image
        ori_img = img.copy()
        
        # Step 1: Text detection
        dt_boxes, det_time = text_detector(img)
        
        # Sort text boxes
        dt_boxes = sort_boxes(dt_boxes)
        
        # Prepare cropped text regions
        img_crop_list = []
        for box in dt_boxes:
            img_crop = get_rotate_crop_image(ori_img, box)
            img_crop_list.append(img_crop)
        
        # Step 2: Text direction classification
        img_crop_list, angle_list, cls_time = text_classifier(img_crop_list)
        
        # Step 3: Text recognition
        rec_res, rec_time = text_recognizer(img_crop_list)
        
        # Filter by confidence score and prepare for lottery parsing
        ocr_results = []
        for idx, (box, rec_result) in enumerate(zip(dt_boxes, rec_res)):
            text, confidence = rec_result[0], rec_result[1]
            if confidence >= args.drop_score:
                # Convert box to a list of coordinates for JSON serialization
                box_coords = box.tolist()
                ocr_results.append({
                    'id': idx + 1,
                    'text': text,
                    'confidence': float(confidence),
                    'bounding_box': box_coords
                })
        
        # Parse lottery ticket information
        lottery_info = parse_lottery_info(ocr_results)
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Delete the uploaded file
        os.remove(filepath)
        
        # Return results
        return jsonify({
            'status': 'success',
            'processing_time': {
                'detection': float(det_time),
                'classification': float(cls_time),
                'recognition': float(rec_time),
                'total': float(total_time)
            },
            'lottery_info': lottery_info,
            'ocr_results': ocr_results  # Include the raw OCR results for debugging
        })
        
    except Exception as e:
        # Delete the uploaded file in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False) 