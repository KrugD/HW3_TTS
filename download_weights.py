#!/usr/bin/env python3
"""
Download Model Weights Script.

This script downloads pre-trained HiFi-GAN model weights from Google Drive.

Usage:
    python download_weights.py
    python download_weights.py --output_dir models_weights
"""

import os
import argparse
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Please install gdown: pip install gdown")
    exit(1)


# Google Drive file ID for generator_best.pt
WEIGHTS_FILE_ID = "1LTl3m5ZScfefGvQUXTBAYfBeoSKhDZR2"
WEIGHTS_URL = f"https://drive.google.com/uc?id={WEIGHTS_FILE_ID}"


def download_weights(output_dir: str = "models_weights") -> bool:
    """
    Download pre-trained model weights from Google Drive.
    
    Args:
        output_dir: Directory to save weights
        
    Returns:
        True if successful
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / "generator_best.pt"
    
    if output_path.exists():
        print(f"Weights already exist at {output_path}")
        return True
    
    print(f"Downloading HiFi-GAN generator weights...")
    print(f"Source: https://drive.google.com/file/d/{WEIGHTS_FILE_ID}/view")
    print(f"Destination: {output_path}")
    
    try:
        gdown.download(WEIGHTS_URL, str(output_path), quiet=False)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"\nDownload complete! File size: {file_size:.2f} MB")
            return True
        else:
            print("Download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://drive.google.com/file/d/{WEIGHTS_FILE_ID}/view")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained HiFi-GAN weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models_weights",
        help="Output directory for weights (default: models_weights)"
    )
    
    args = parser.parse_args()
    
    success = download_weights(args.output_dir)
    
    if success:
        print("\nDone! You can now run synthesis:")
        print("  python synthesize.py checkpoint=models_weights/generator_best.pt")
    else:
        print("\nDownload failed!")
        exit(1)


if __name__ == "__main__":
    main()
