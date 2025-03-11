#!/usr/bin/env python3

"""
Face distance detection system

This script captures an image from a webcam
and determines
if a face is too close to the screen
using basic image processing techniques
without relying on built-in face detection.
"""
import cv2
import numpy as np

from src.preprocessing import rgb_to_grayscale_conversion, rgb_to_hsv_conversion, histogram_equalization, noise_reduction
from src.feature_detection import edge_detection, morphological_operations, detect_eyes
from src.distance_analysis import measure_distance, is_too_close
from src.visualization import visualize_pipeline, display_result
from src.image_aquisition import capture_image


def main():
  original_image = capture_image()

  if original_image is None:
    print('Failed to acquire image. Exiting.')
    return

  # Preprocessing

  hsv_image = rgb_to_hsv_conversion(original_image)
  print("Converted to HSV")

  cv2.imshow('Grayscale', hsv_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  #gray_image = rgb_to_grayscale_conversion(original_image)
  gray_image = hsv_image[:, :, 2] * 255
  gray_image = gray_image.astype(np.uint8)
  print("Converted to grayscale.")

  cv2.imshow('Grayscale', gray_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  equalized_image = histogram_equalization(gray_image)
  print("Equalized using histogram")

  cv2.imshow('Equalized', equalized_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  filtered_image = noise_reduction(equalized_image)
  print("Reduced noise")

  cv2.imshow('Reduced noise', filtered_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  # Feature detection
  edge_image = edge_detection(filtered_image)
  print("Finding edges")

  cv2.imshow('Edged', edge_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  morphed_image = morphological_operations(edge_image)
  print("Morphing image")

  cv2.imshow('Morphed', morphed_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  eye_regions, eye_centers = detect_eyes(filtered_image)
  print("Detecting eyes")

  cv2.imshow('Eye regions', eye_regions)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  """
  if eye_centers is not None and len(eye_centers) >= 2:
    eye_distance = measure_distance(eye_centers)
    too_close = is_too_close(eye_distance, original_image.shape[1], threshold=0.25)

    pipeline_images = {
      'Original Image': original_image,
      'Grayscale Image': gray_image,
      'Histogram Equalized': equalized_image,
      'Noise Reduced': filtered_image,
      'Edge Detection': edge_image,
      'Morphological Operations': morphed_image,
      'Upper Face Region': eye_regions
    }

    visualize_pipeline(pipeline_images, save=False, save_dir='')

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

    visualize_pipeline(pipeline_images, save=False, save_dir='')
  """


if __name__ == '__main__':
  main()