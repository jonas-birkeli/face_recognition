"""
Module for feature detection functions, including edge detection and eye detection.
"""

import numpy as np

def edge_detection(image):
  """
  Apply Sobel edge detection

  :param image: Input grayscale image

  :return: Binary edge image
  """

  sobel_x = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

  sobel_y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float32)

  height, width = image.shape

  # Pad the image to allow Sobel kernel to fit anywhere
  padded = np.zeros((height + 2, width + 2), dtype=np.float32)
  padded[1:height+1, 1:width+1] = image

  # Apply Sobel kernel
  gradient_x = np.zeros((height, width), dtype=np.float32)
  gradient_y = np.zeros((height, width), dtype=np.float32)

  for i in range(height):
    for j in range(width):
      roi = padded[i:i+3, j:j+3]
      gradient_x[i, j] = np.sum(roi * sobel_x)
      gradient_y[i, j] = np.sum(roi * sobel_y)

  gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

  # Normalize within 0-255 range
  if np.max(gradient_magnitude) > 0:
    gradient_magnitude = gradient_magnitude * 255 / np.max(gradient_x)

  threshold = 50 # TODO Adjustable!
  edge_image = (gradient_magnitude > threshold).astype(np.uint8) * 255
  # Cool math, taking bool value times 255 to create either 0 or 255 :)

  return edge_image