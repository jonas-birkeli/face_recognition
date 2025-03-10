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

def create_structuring_element(size=3, shape='disk'):
  """
  Create a structuring element for morphological operations.

  :param size: Radius of the structuring element
  :param shape: Shape type ('disk', 'square', 'cross')

  :return: Binary structuring element
  """
  element_size = 2 * size + 1  # Radius * 2 plus the middle part
  element = np.zeros((element_size, element_size), dtype=np.uint8)

  if shape == 'disk':
    center = size
    for i in range(element_size):
      for j in range(element_size):
        if np.sqrt((i - center)**2 + (j - center)**2) <= size:
          element[i, j] = 1
  elif shape == 'square':
    element.fill(1)
  elif shape == 'cross':
    element[size, :] = 1
    element[:, size] = 1

  return element

def dilate(binary_image, element):
  """
  Apply dilation operation

  :param binary_image: Binary image input
  :param element: Structuring element

  :return: Dilated image
  """
  binary = binary_image.copy()
  if np.max(binary) > 1:
    binary = (binary > 0).astype(np.uint8)

  height, width = binary.shape
  element_size = element.shape[0]
  pad = element_size // 2

  # Create padded image
  padded = np.zeros((height + 2*pad, width + 2*pad), dtype=np.uint8)
  padded[pad:pad+height, pad:pad+width] = binary  # Put image in the center

  dilated = np.zeros_like(binary)

  for i in range(height):
    for j in range(width):
      roi = padded[i:i+element_size, j:j+element_size]
      if np.any(roi & element):
        dilated[i, j] = 1

  # Asure it is not above 255
  return dilated * 255 if np.max(binary_image) > 1 else dilated

def erode(binary_image, element):
  """
  Apply erosion to image

  :param binary_image: Binary image input
  :param element: Structuring element

  :return: Eroded image
  """
  binary = binary_image.copy()
  if np.max(binary) > 1:
    binary = (binary > 0).astype(np.uint8)

  height, width = binary.shape
  element_size = element.shape[0]
  pad = element_size // 2

  # Create padded image
  padded = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.uint8)
  padded[pad:pad + height, pad:pad + width] = binary  # Put image in the center

  eroded = np.zeros_like(binary)

  for i in range(height):
    for j in range(width):
      roi = padded[i:i + element_size, j:j + element_size]
      if np.all(roi[element == 1] == 1):
        eroded[i, j] = 1

  # Asure it is not above 255
  return eroded * 255 if np.max(binary_image) > 1 else eroded

def morphological_operations(edge_image):
  """
  Applies morphological operations to enhance features.
  Dilates with disk size 2, erodes with disk size 1 (Almost closing)

  :param edge_image: Binary edge image

  :return: Processed image
  """

  se_dilation = create_structuring_element(size=2, shape='disk')
  se_erosion = create_structuring_element(size=1, shape='disk')

  dilated = dilate(edge_image, se_dilation)

  eroded = erode(dilated, se_erosion)

  return eroded