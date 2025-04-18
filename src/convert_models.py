#!/usr/bin/env python3

import os
import argparse
import sys
import shutil


def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def convert_paddle_to_onnx(paddle_model_dir, params_filename, model_filename, save_path, opset_version=11):
    """
    Convert PaddlePaddle model to ONNX format
    
    Args:
        paddle_model_dir: Directory containing Paddle model
        params_filename: Filename of model parameters
        model_filename: Filename of model structure
        save_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    # Ensure output directory exists
    ensure_dir(os.path.dirname(save_path))
    
    # Import paddle2onnx here to avoid dependency if not converting
    try:
        import paddle
        import paddle2onnx
    except ImportError:
        print("Error: paddle and paddle2onnx packages are required for conversion")
        print("Please install using: pip install paddle2onnx")
        return False
    
    try:
        # Run conversion
        paddle2onnx_cmd = [
            'paddle2onnx',
            f'--model_dir={paddle_model_dir}',
            f'--model_filename={model_filename}',
            f'--params_filename={params_filename}',
            f'--save_file={save_path}',
            f'--opset_version={opset_version}',
            '--enable_onnx_checker=True'
        ]
        
        # Join command for display
        cmd_str = ' '.join(paddle2onnx_cmd)
        print(f"Running: {cmd_str}")
        
        # Use paddle2onnx API directly
        result = paddle2onnx.command.convert(
            model_dir=paddle_model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            save_file=save_path,
            opset_version=opset_version,
            enable_onnx_checker=True
        )
        
        if os.path.exists(save_path):
            print(f"Conversion successful: {save_path}")
            return True
        else:
            print(f"Error: Conversion failed, output file not created")
            return False
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False


def prepare_dictionary(source_dict_path, target_dict_path):
    """Copy dictionary file to target location"""
    try:
        shutil.copy2(source_dict_path, target_dict_path)
        print(f"Dictionary copied to {target_dict_path}")
        return True
    except Exception as e:
        print(f"Error copying dictionary: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PaddleOCR models to ONNX format")
    parser.add_argument("--paddle-dir", default="./inference", help="Directory containing Paddle models")
    parser.add_argument("--output-dir", default="./models/onnx", help="Directory to save ONNX models")
    parser.add_argument("--det-model", default="en_PP-OCRv3_det_infer", help="Detection model directory name")
    parser.add_argument("--cls-model", default="ch_ppocr_mobile_v2.0_cls_infer", help="Classification model directory name")
    parser.add_argument("--rec-model", default="en_PP-OCRv3_rec_infer", help="Recognition model directory name")
    parser.add_argument("--dict-path", default="ppocr/utils/en_dict.txt", help="Path to character dictionary")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    
    args = parser.parse_args()
    
    # Check if paddle directory exists
    if not os.path.isdir(args.paddle_dir):
        print(f"Error: Paddle model directory '{args.paddle_dir}' not found")
        return 1
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Process detection model
    det_model_dir = os.path.join(args.paddle_dir, args.det_model)
    det_model_output = os.path.join(args.output_dir, "det_model.onnx")
    if not os.path.isdir(det_model_dir):
        print(f"Warning: Detection model directory '{det_model_dir}' not found")
    else:
        print(f"Converting detection model...")
        convert_paddle_to_onnx(
            det_model_dir, 
            "inference.pdiparams", 
            "inference.pdmodel", 
            det_model_output,
            args.opset
        )
    
    # Process classification model
    cls_model_dir = os.path.join(args.paddle_dir, args.cls_model)
    cls_model_output = os.path.join(args.output_dir, "cls_model.onnx")
    if not os.path.isdir(cls_model_dir):
        print(f"Warning: Classification model directory '{cls_model_dir}' not found")
    else:
        print(f"Converting classification model...")
        convert_paddle_to_onnx(
            cls_model_dir, 
            "inference.pdiparams", 
            "inference.pdmodel", 
            cls_model_output,
            args.opset
        )
    
    # Process recognition model
    rec_model_dir = os.path.join(args.paddle_dir, args.rec_model)
    rec_model_output = os.path.join(args.output_dir, "rec_model.onnx")
    if not os.path.isdir(rec_model_dir):
        print(f"Warning: Recognition model directory '{rec_model_dir}' not found")
    else:
        print(f"Converting recognition model...")
        convert_paddle_to_onnx(
            rec_model_dir, 
            "inference.pdiparams", 
            "inference.pdmodel", 
            rec_model_output,
            args.opset
        )
    
    # Copy dictionary file
    dict_output = os.path.join(os.path.dirname(args.output_dir), "en_dict.txt")
    if not os.path.isfile(args.dict_path):
        print(f"Warning: Dictionary file '{args.dict_path}' not found")
    else:
        prepare_dictionary(args.dict_path, dict_output)
    
    print("\nModel conversion completed. Please check the output directory for results.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 