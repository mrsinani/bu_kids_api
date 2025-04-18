#!/usr/bin/env python
"""
Test script for the PaddleOCR ONNX API

Sends a sample image to the API and prints the results
"""

import sys
import json
import requests
from pprint import pprint

def test_ocr_api(image_path, api_url="http://localhost:5001/ocr"):
    """Test the OCR API with an image"""
    print(f"Sending image: {image_path} to {api_url}")
    
    try:
        # Open the image file
        with open(image_path, "rb") as img_file:
            # Prepare the files for the request
            files = {"image": img_file}
            
            # Send POST request to the API
            response = requests.post(api_url, files=files)
            
            # Check if request was successful
            if response.status_code == 200:
                # Parse and print the JSON response
                result = response.json()
                
                # Pretty print processing time
                print("\nProcessing Time:")
                pprint(result["processing_time"])
                
                # Print detected text items
                print("\nDetected Text:")
                for item in result["results"]:
                    print(f"ID: {item['id']} | Text: {item['text']} | Confidence: {item['confidence']:.3f}")
                
                # Print total items detected
                print(f"\nTotal items detected: {len(result['results'])}")
                
                return result
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
                return None
                
    except FileNotFoundError:
        print(f"Error: Could not find image file {image_path}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    return None

if __name__ == "__main__":
    # Use command line argument for image path if provided, otherwise use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "./images/IMG_1538.png"
    test_ocr_api(image_path) 