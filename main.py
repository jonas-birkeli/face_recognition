#!/usr/bin/env python3

"""
Face distance detection system

This script captures an image from a webcam
and determines
if a face is too close to the screen
using basic image processing techniques
without relying on built-in face detection.
"""

from src.preprocessing import grayscale_conversion, histogram_equalization, noise_reduction

def main():
  original_image = capture_image()

  if original_image is None:
    print("Failed to acquire image. Exiting.")
    return

  # Preprocessing
  gray_image = grayscale_conversion(original_image)
  equalized_image = histogram_equalization(gray_image)
  filtered_image = noise_reduction(equalized_image)

  # Feature detection
  edge_image = edge_detection(filtered_image)
  morphed_image = morphological_operations(edge_image)
  eye_regions, eye_center = detect_eyes(filtered_image)
