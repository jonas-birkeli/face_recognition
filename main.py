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
from src.feature_detection import edge_detection, morphological_operations, detect_eyes
from src.distance_analysis import measure_distance, is_too_close


def main():
  original_image = capture_image()

  if original_image is None:
    print('Failed to acquire image. Exiting.')
    return

  # Preprocessing
  gray_image = grayscale_conversion(original_image)
  equalized_image = histogram_equalization(gray_image)
  filtered_image = noise_reduction(equalized_image)

  # Feature detection
  edge_image = edge_detection(filtered_image)
  morphed_image = morphological_operations(edge_image)
  eye_regions, eye_centers = detect_eyes(filtered_image)

  if eye_centers is not None and len(eye_centers) >= 2:
    eye_distance = measure_distance(eye_centers)
    too_close = is_too_close(eye_distance, original_image.shape[1], threshold=500)

    pipeline_images = {
      'Original Image': original_image,
      'Grayscale Image': gray_image,
      'Histogram Equalized': equalized_image,
      'Noise Reduced': filtered_image,
      'Edge Detection': edge_image,
      'Morphological Operations': morphed_image,
      'Upper Face Region': eye_regions
    }

    visualize_pipeline(pipelinse_images, save=False, save_dir='')

    display_result(original_image, eye_centers, eye_distance, too_close, save=False, save_dir='')

    if too_close:
      print('Face is too close to the screen!')
    else:
      print('Face is at an acceptable distance from the screen.')
    print(f'Eye distance: {eye_distance:.2f} pixels')
    print(f'Distance ratio: {eye_distance / original_image.shape[1]:.2f}')
  else:
    print('Could not detect two distinct eye regions.')

    pipeline_images = {
      'Original Image': original_image,
      'Grayscale Image': gray_image,
      'Histogram Equalized': equalized_image,
      'Noise Reduced': filtered_image,
      'Edge Detection': edge_image,
      'Morphological Operations': morphed_image,
    }

    visualize_pipeline(pipelinse_images, save=False, save_dir='')