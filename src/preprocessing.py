"""
Module for image preprocessing functions
"""

import numpy as np

def grayscale_conversion(image):
  """
  Converts RGB image to grayscale using weighted method.

  :param image: Input image in RGB format

  :return: Grayscale image
  """
  if len(image.shape) < 3:
    # Already grayscale
    return image

  gray = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      gray[i, j] = int(0.299 * image[i, j, 0] +
                       0.587 * image[i, j, 1] +
                       0.114 * image[i, j, 2])

  return gray

