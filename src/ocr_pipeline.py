import os
import cv2
import math
import numpy as np
import onnxruntime as ort
from PIL import Image
from shapely.geometry import Polygon
import pyclipper

class TextDetector:
    def __init__(self, model_path, max_side_len=960):
        """
        Args:
            model_path: ONNX model path for text detection
            max_side_len: maximum image side length for resizing
        """
        self.max_side_len = max_side_len
        # Load ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def resize_image(self, img):
        """Resize image while maintaining aspect ratio"""
        h, w = img.shape[:2]
        ratio = float(self.max_side_len) / max(h, w)
        if max(h, w) > self.max_side_len:
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
        else:
            resize_h = h
            resize_w = w
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        img = cv2.resize(img, (resize_w, resize_h))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def preprocess(self, img):
        """Preprocess image for network input"""
        img, ratio_list = self.resize_image(img)
        img = img.astype('float32')
        img = img / 255.0
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = img[np.newaxis, :]  # NCHW
        return img, ratio_list

    def postprocess(self, pred, ratio_list):
        """Process prediction map to get text boxes"""
        pred = pred[:, 0, :, :]
        segmentation = pred > 0.3  # binary map
        
        boxes_list = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w = pred.shape[1:]
            mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            
            boxes_list.append({'points': boxes, 'scores': scores})
        
        # Adjust box coordinates by image ratio
        for i in range(len(boxes_list)):
            boxes = boxes_list[i]['points']
            if len(boxes) > 0:
                boxes_list[i]['points'] = boxes / np.array([ratio_list[1], ratio_list[0]])
        
        return boxes_list

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        """Convert binary map to boxes"""
        height, width = bitmap.shape
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        scores = []
        
        for contour in contours:
            if len(contour) < 4:
                continue
            # Calculate polygon area
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            
            # Filter small boxes
            if len(points) < 4 or cv2.contourArea(points.reshape((-1, 1, 2))) < 10:
                continue
                
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            
            # Calculate box score
            box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
            mask_roi = bitmap[box_y:box_y + box_h, box_x:box_x + box_w]
            roi_h, roi_w = mask_roi.shape
            if roi_h * roi_w == 0:
                continue
            pred_roi = pred[box_y:box_y + box_h, box_x:box_x + box_w]
            score = np.mean(pred_roi[mask_roi])
            
            # Add to results
            boxes.append(box)
            scores.append(score)
        
        return np.array(boxes), np.array(scores)

    def detect(self, img):
        """Detect text in an image"""
        ori_im = img.copy()
        img, ratio_list = self.preprocess(img)
        
        # Forward pass through the model
        outputs = self.session.run(self.output_names, {self.input_name: img})
        boxes_list = self.postprocess(outputs[0], ratio_list)
        
        if len(boxes_list) > 0:
            boxes = boxes_list[0]['points']
            scores = boxes_list[0]['scores']
            
            # Filter boxes by score
            inds = scores > 0.5
            boxes = boxes[inds]
            scores = scores[inds]
            
            # Sort boxes by y-coordinate
            boxes = sorted(boxes, key=lambda x: min(p[1] for p in x))
        else:
            boxes = []
        
        return boxes


class TextClassifier:
    def __init__(self, model_path):
        """
        Args:
            model_path: ONNX model path for text orientation classification
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.class_list = ['0', '180']
        
    def preprocess(self, img):
        """Preprocess image for network input"""
        h, w = img.shape[:2]
        
        # Resize to model input shape
        img = cv2.resize(img, (192, 48))
        
        # Normalize and convert to NCHW format
        img = img.astype('float32')
        img = img / 255.0
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :]
        
        return img
        
    def predict(self, img):
        """Classify text orientation"""
        img = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img})
        
        # Get prediction
        pred = outputs[0]
        pred_idx = np.argmax(pred, axis=1)[0]
        pred_class = self.class_list[pred_idx]
        pred_score = pred[0][pred_idx]
        
        return pred_class, pred_score


class TextRecognizer:
    def __init__(self, model_path, dict_path=None):
        """
        Args:
            model_path: ONNX model path for text recognition
            dict_path: Path to character dictionary file
        """
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load dictionary
        if dict_path is None:
            # Default English dictionary
            self.character_dict = "0123456789abcdefghijklmnopqrstuvwxyz"
        else:
            with open(dict_path, 'r') as f:
                self.character_dict = f.readlines()[0].strip()
        
        # Add blank character for CTC decoder
        self.character_dict = ["blank"] + list(self.character_dict)
        
    def preprocess(self, img):
        """Preprocess image for network input"""
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Resize to fixed height and variable width
        h, w = img.shape
        target_h = 48
        scale = target_h / h
        target_w = max(int(w * scale), 16)
        target_w = min(target_w, 320)  # Limit max width
        
        resized_img = cv2.resize(img, (target_w, target_h))
        
        # Normalize and convert to NCHW format
        resized_img = resized_img.astype('float32') / 255.0
        resized_img = resized_img[np.newaxis, np.newaxis, :, :]
        
        return resized_img
    
    def decode(self, pred):
        """CTC greedy decoder"""
        # Get prediction indexes
        pred_idx = np.argmax(pred, axis=1)
        
        # Remove duplicate indexes
        char_list = []
        for i, idx in enumerate(pred_idx):
            # Skip blank character
            if idx > 0 and (i == 0 or idx != pred_idx[i - 1]):
                char_list.append(self.character_dict[idx])
        
        # Convert to text
        text = ''.join(char_list)
        
        return text
    
    def recognize(self, img):
        """Recognize text in an image"""
        img = self.preprocess(img)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: img})
        
        # Decode prediction
        pred = outputs[0]
        text = self.decode(pred)
        
        return text


class OCRPipeline:
    def __init__(self, det_model_path, cls_model_path, rec_model_path, dict_path=None):
        """
        Args:
            det_model_path: Path to text detection ONNX model
            cls_model_path: Path to text classification ONNX model
            rec_model_path: Path to text recognition ONNX model
            dict_path: Path to character dictionary file
        """
        self.text_detector = TextDetector(det_model_path)
        self.text_classifier = TextClassifier(cls_model_path)
        self.text_recognizer = TextRecognizer(rec_model_path, dict_path)
    
    def extract_text_regions(self, img, boxes):
        """Extract text regions from image based on detected boxes"""
        regions = []
        for box in boxes:
            # Get rotated rectangle
            rect = cv2.minAreaRect(box)
            center, (width, height), angle = rect
            
            # Get transformation matrix
            if width < height:
                angle += 90
                width, height = height, width
            
            # Prepare transformation matrix
            angle_rad = angle * np.pi / 180
            cos_theta = np.cos(angle_rad)
            sin_theta = np.sin(angle_rad)
            
            # New width and height with some padding
            new_w = int(width + 10)
            new_h = int(height + 10)
            
            # Get transformation matrix
            m = np.array([
                [cos_theta, sin_theta, -center[0] * cos_theta - center[1] * sin_theta + new_w / 2],
                [-sin_theta, cos_theta, center[0] * sin_theta - center[1] * cos_theta + new_h / 2]
            ])
            
            # Warp image
            region = cv2.warpAffine(img, m, (new_w, new_h))
            regions.append(region)
        
        return regions
    
    def __call__(self, img_path):
        """
        Run OCR pipeline on an image
        
        Args:
            img_path: Path to input image
            
        Returns:
            List of detected text and their bounding boxes
        """
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        
        # Detect text regions
        boxes = self.text_detector.detect(img)
        
        if len(boxes) == 0:
            return []
        
        # Extract text regions
        text_regions = self.extract_text_regions(img, boxes)
        
        # Process each region
        results = []
        for i, (box, region) in enumerate(zip(boxes, text_regions)):
            # Classify orientation
            cls, cls_score = self.text_classifier.predict(region)
            
            # Rotate if needed
            if cls == "180":
                region = cv2.rotate(region, cv2.ROTATE_180)
            
            # Recognize text
            text = self.text_recognizer.recognize(region)
            
            # Add to results
            results.append({
                "text": text,
                "box": box.tolist(),
                "confidence": float(cls_score)
            })
        
        return results


def get_model_paths():
    """Get paths to ONNX models"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_model = os.path.join(base_dir, "models", "onnx", "det_model.onnx")
    cls_model = os.path.join(base_dir, "models", "onnx", "cls_model.onnx")
    rec_model = os.path.join(base_dir, "models", "onnx", "rec_model.onnx")
    dict_path = os.path.join(base_dir, "models", "en_dict.txt")
    
    return det_model, cls_model, rec_model, dict_path 