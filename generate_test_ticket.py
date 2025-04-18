#!/usr/bin/env python
"""
Generate Test Lottery Ticket

Creates a simple test image of a lottery ticket with key information for testing the OCR API.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
from datetime import datetime, timedelta

def generate_lottery_ticket():
    """
    Generate a test lottery ticket image with Mega Millions information
    """
    # Create a white image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to load a font
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Mac OS path
        if not os.path.exists(font_path):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Linux path
        
        # If neither path works, we'll fall back to default
        title_font = ImageFont.truetype(font_path, 36) if os.path.exists(font_path) else ImageFont.load_default()
        body_font = ImageFont.truetype(font_path, 24) if os.path.exists(font_path) else ImageFont.load_default()
    except Exception:
        # Fall back to default if any error occurs
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # Generate random ticket data
    ticket_number = ''.join([str(random.randint(0, 9)) for _ in range(16)])
    
    # Random date within the last month
    today = datetime.now()
    days_ago = random.randint(1, 30)
    ticket_date = today - timedelta(days=days_ago)
    date_str = ticket_date.strftime("%m/%d/%Y")
    
    # Random megaball number
    megaball = random.randint(1, 25)
    
    # Random picked numbers
    picked_numbers = sorted([random.randint(1, 70) for _ in range(5)])
    
    # Draw lottery information
    draw.text((50, 50), "MEGA MILLIONS", font=title_font, fill=(0, 0, 0))
    draw.text((50, 120), f"Ticket #: {ticket_number}", font=body_font, fill=(0, 0, 0))
    draw.text((50, 170), f"Date: {date_str}", font=body_font, fill=(0, 0, 0))
    
    # Draw picked numbers
    draw.text((50, 220), "Your Numbers:", font=body_font, fill=(0, 0, 0))
    for i, number in enumerate(picked_numbers):
        draw.ellipse([50 + i*60, 260, 100 + i*60, 310], outline=(0, 0, 0), width=2)
        num_text = str(number)
        # Center the text in the circle (using getbbox instead of textsize for newer Pillow versions)
        bbox = body_font.getbbox(num_text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((50 + i*60 + (50-w)/2, 260 + (50-h)/2), num_text, font=body_font, fill=(0, 0, 0))
    
    # Draw megaball
    draw.text((50, 330), "Mega Ball:", font=body_font, fill=(0, 0, 0))
    draw.ellipse([50, 370, 100, 420], outline=(0, 0, 0), width=2, fill=(255, 215, 0))
    mb_text = str(megaball)
    # Center the text in the circle
    bbox = body_font.getbbox(mb_text)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((50 + (50-w)/2, 370 + (50-h)/2), mb_text, font=body_font, fill=(0, 0, 0))
    
    # Add a clear text for the megaball number for OCR to pick up
    draw.text((150, 385), f"MB: {megaball}", font=body_font, fill=(0, 0, 0))
    
    # Save image
    os.makedirs('test_images', exist_ok=True)
    image_path = os.path.join('test_images', 'lottery_ticket.jpg')
    image.save(image_path)
    
    print(f"Generated test lottery ticket at: {image_path}")
    print(f"Ticket data for verification:")
    print(f"  Game Type: Mega Millions")
    print(f"  Ticket #: {ticket_number}")
    print(f"  Date: {date_str}")
    print(f"  Megaball: {megaball}")
    
    return image_path

if __name__ == "__main__":
    generate_lottery_ticket() 