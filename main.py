#!/usr/bin/env python3

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="BU Kids OCR API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run the CLI tool")
    cli_parser.add_argument("--image", "-i", required=True, help="Path to input image")
    cli_parser.add_argument("--output", "-o", help="Path to output visualization")
    cli_parser.add_argument("--json", "-j", help="Path to output JSON results")
    
    # Conversion command
    convert_parser = subparsers.add_parser("convert", help="Convert Paddle models to ONNX")
    convert_parser.add_argument("--paddle-dir", default="../inference", help="Directory containing Paddle models")
    convert_parser.add_argument("--output-dir", default="./models/onnx", help="Directory to save ONNX models")
    convert_parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    
    args = parser.parse_args()
    
    if args.command == "api":
        from src.api import start_server
        start_server(host=args.host, port=args.port)
    
    elif args.command == "cli":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.cli import main as cli_main
        sys.argv = [sys.argv[0]] + ["--image", args.image]
        if args.output:
            sys.argv += ["--output", args.output]
        if args.json:
            sys.argv += ["--json", args.json]
        return cli_main()
    
    elif args.command == "convert":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from src.convert_models import main as convert_main
        sys.argv = [sys.argv[0]]
        if args.paddle_dir:
            sys.argv += ["--paddle-dir", args.paddle_dir]
        if args.output_dir:
            sys.argv += ["--output-dir", args.output_dir]
        if args.opset:
            sys.argv += ["--opset", str(args.opset)]
        return convert_main()
    
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 