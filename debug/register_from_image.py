"""
Register a new face from an image file (not camera).
Usage:
    uv run src/register_from_image.py --image path/to/image.jpg --name NAME

This script loads an image, detects the face, extracts embeddings, and adds it to the database.
"""
import argparse
import cv2
import os
from src.recognition.recognizer import FaceRecognizer
from config import FACE_DB_PATH, ENHANCED_DIR

def main():
    parser = argparse.ArgumentParser(description="Register a face from an image file.")
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--name', required=True, help='Name to register')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Image not found: {args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}")
        return

    recognizer = FaceRecognizer()
    success = recognizer.enroll(args.name, img)
    if success:
        print(f"Registered {args.name} from {args.image}")
    else:
        print(f"Failed to register {args.name} from {args.image}")

if __name__ == "__main__":
    main()
