"""
Module for image preprocessing functions
"""

import numpy as np
from numpy.conftest import dtype


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

  # gray = 0.299R + 0.587G + 0.114B
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      gray[i, j] = int(0.299 * image[i, j, 0] +
                       0.587 * image[i, j, 1] +
                       0.114 * image[i, j, 2])

  return gray


def histogram_equalization(gray_image):
  """
  Performs histogram equalization on a grayscale image

  :param gray_image: Grayscale input image

  :return: Histogram equalized image
  """

  # Create hisotgram
  hist = np.zeros(256, dtype=np.int32)

  for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
      hist[gray_image[i, j]] += 1

  # Cumulative distribution function (CDF=
  cdf = np.zeros(256, dtype=np.int32)
  cdf[0] = hist[0]

  for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist[i]

  # Normalize CDF
  cdf_min = cdf[np.nonzero(hist)[0][0]] if np.any(hist) else 0
  cdf_normalized = np.zeros(256, dtype=np.uint8)

  for i in range(256):
    if cdf[i] > cdf_min:
      cdf_normalized[i] = int(((cdf[i] - cdf_min) * 256) / (gray_image.shape[0] * gray_image.shape[1] - cdf_min))
    else:
      cdf_normalized[i] = 0

  # Apply mapping to equalize histogram
  equalized = np.zeros_like(gray_image)

  for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
      equalized[i, j] = cdf_normalized[gray_image[i, j]]

  return equalized

def noise_reduction(image, kernel_size=3):
  """
  Reduce noise using simple averaging filter

  :param image: Input image
  :param kernel_size: Size of averaging kernel (must be odd)

  :return: Filtered image
  """
  if kernel_size % 2 == 0:
    kernel_size += 1

  pad = kernel_size // 2
  height, width = image.shape

  # Create pad to image
  padded_img = np.zeros((height + 2*pad, width + 2*pad), dtype=np.float32)
  padded_img[pad:pad+height, pad:pad+width] = image  # Put image in the middle

  kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

  # Apply convolution
  filtered = np.zeros_like(image)

  for i in range(height):
    for j in range(width):
      roi = padded_img[i:i+kernel_size, j:j+kernel_size] # Region of interest
      filtered[i, j] = np.sum(roi * kernel)

  return filtered.astype(np.uint8)