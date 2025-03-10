"""
Module for analyzing face distance
"""

import numpy as np

def measure_distance(eye_centers):
  """
  Measure the distance between detected eye centroids

  :param eye_centers: List of (y, x) coordinates of eye centers

  :return: Distance between eyes in pixels
  """
  if len(eye_centers) < 2:
    return 0

  y1, x1 = eye_centers[0]
  y2, x2 = eye_centers[1]

  # Calculate distance, any angle is supported
  distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
  return distance


def is_too_close(eye_distance, image_width, threshold=0.25):
  """
  Determine if the face is too close to the screen based on eye distance

  :param eye_distance: Distance between eyes in pixel
  :param image_width: Width of the image in pixels
  :param threshold: Threshold ratio for determination
  :return:
  """
  if image_width == 0:
    return True

  # Calculate the ratio between eye distance to image width
  distance_ratio = eye_distance / image_width

  return distance_ratio > threshold