#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="bu_kids_api",
    version="1.0.0",
    description="OCR API using PaddleOCR models in ONNX format",
    author="Boston University Kids Team",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.3",
        "onnxruntime>=1.8.0",
        "Pillow>=8.2.0",
        "pyclipper>=1.3.0",
        "shapely>=1.7.1",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
    ],
    entry_points={
        "console_scripts": [
            "bu-kids-ocr=main:main",
        ],
    },
) 