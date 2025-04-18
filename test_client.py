#!/usr/bin/env python
"""
OCR API Test Client

A simple script to test the OCR API by sending an image and displaying the detected text.
"""

import argparse
import json
import requests
from pprint import pprint

def test_ocr_api(image_path, url="http://localhost:5001/ocr"):
    """
    Send an image to the OCR API and display the results.
    
    Args:
        image_path (str): Path to the image file
        url (str): URL of the OCR API endpoint
    """
    print(f"Sending image: {image_path}")
    print(f"To API: {url}")
    
    try:
        # Open the image file
        with open(image_path, 'rb') as f:
            # Create the files dictionary for the multipart/form-data request
            files = {'image': f}
            
            # Send the POST request
            response = requests.post(url, files=files)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                result = response.json()
                
                # Print processing time
                print("\nProcessing Time:")
                print(f"  Detection: {result['processing_time']['detection']:.3f}s")
                print(f"  Classification: {result['processing_time']['classification']:.3f}s")
                print(f"  Recognition: {result['processing_time']['recognition']:.3f}s")
                print(f"  Total: {result['processing_time']['total']:.3f}s")
                
                # Check if this is a lottery endpoint response
                if 'lottery_info' in result:
                    print("\n╔══════════════════════════════════╗")
                    print("║        LOTTERY INFORMATION        ║")
                    print("╚══════════════════════════════════╝")
                    
                    lottery_info = result['lottery_info']
                    
                    # Format and display lottery information
                    print(f"Ticket Number  : {lottery_info['ticket_number'] or 'Not detected'}")
                    print(f"Game Type      : {lottery_info['game_type'] or 'Not detected'}")
                    print(f"Date           : {lottery_info['date'] or 'Not detected'}")
                    print(f"Megaball Number: {lottery_info['megaball_number'] or 'Not detected'}")
                    
                    print("\nDetected Text:")
                    if result['ocr_results']:
                        for item in result['ocr_results']:
                            confidence = item['confidence'] * 100
                            print(f"  [{item['id']}] {item['text']} (Confidence: {confidence:.1f}%)")
                    else:
                        print("  No text detected in the image.")
                else:
                    # Print detected text (standard OCR response)
                    print("\nDetected Text:")
                    if result['results']:
                        for item in result['results']:
                            confidence = item['confidence'] * 100
                            print(f"  [{item['id']}] {item['text']} (Confidence: {confidence:.1f}%)")
                    else:
                        print("  No text detected in the image.")
                
                # Optional: print the full JSON response if requested
                if args.verbose:
                    print("\nFull JSON Response:")
                    pprint(result)
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found.")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from the API.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the OCR API with an image")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--url", default="http://localhost:5001/ocr", 
                        help="URL of the OCR API endpoint. Use http://localhost:5001/lottery for lottery tickets.")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Display the full JSON response")
    
    args = parser.parse_args()
    
    # Test the API
    test_ocr_api(args.image_path, args.url) 