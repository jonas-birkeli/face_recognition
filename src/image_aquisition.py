"""
Module for image acquisition from webcam or file.
"""

import cv2
import numpy as np

def capture_image(from_file=False, file_path=None, webcam_index=0):
    """
    Capture an image from webcam or load from file.
    This method was generated using GitHub Copilot

    Args:
        from_file (bool): Whether to load from file instead of webcam
        file_path (str): Path to image file (if from_file is True)
        webcam_index (int): Index of webcam to use

    Returns:
        numpy.ndarray: Captured or loaded image (BGR format)
    """
    if from_file:
        if file_path is None:
            print("Error: No file path provided.")
            return None

        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Error: Could not read image from {file_path}")
                return None
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    else:
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(webcam_index)

            if not cap.isOpened():
                print(f"Error: Could not open webcam {webcam_index}")
                return None

            # Capture frame
            ret, frame = cap.read()

            # Release webcam
            cap.release()

            if not ret:
                print("Error: Could not capture frame from webcam")
                return None

            return frame
        except Exception as e:
            print(f"Error capturing from webcam: {e}")
            return None

def manual_resize(image, width=None, height=None):
    """
    Manually resize an image without using built-in OpenCV functions.
    This method was generated using GitHub Copilot

    Args:
        image (numpy.ndarray): Input image
        width (int): Target width (if None, will be calculated from height)
        height (int): Target height (if None, will be calculated from width)

    Returns:
        numpy.ndarray: Resized image
    """
    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    if width is None:
        # Calculate width to maintain aspect ratio
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        # Calculate height to maintain aspect ratio
        aspect_ratio = h / w
        height = int(width * aspect_ratio)

    # Simple nearest neighbor interpolation
    resized = np.zeros((height, width, 3) if len(image.shape) == 3 else (height, width),
                       dtype=image.dtype)

    x_ratio = w / width
    y_ratio = h / height

    for y in range(height):
        for x in range(width):
            src_x = min(w - 1, int(x * x_ratio))
            src_y = min(h - 1, int(y * y_ratio))

            resized[y, x] = image[src_y, src_x]

    return resized