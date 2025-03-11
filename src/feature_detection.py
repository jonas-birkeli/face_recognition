"""
Module for feature detection functions, including edge detection and eye detection.
"""
from operator import truediv

import numpy as np

def edge_detection(image):
  """
  Apply Sobel-edge detection

  :param image: Input grayscale image

  :return: Binary-edge image
  """

  sobel_x = np.array([[-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]], dtype=np.float32)

  sobel_y = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)

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

  threshold = 25 # TODO Adjustable!
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

def connected_components(binary_image, min_size=50):
  """
  Finds connected regions of a morphed binary image.
  This method was generated using GitHub Copilot

  :param binary_image: Input binary image
  :param min_size: Minimum component size to keep

  :return: Labeled components in a list
  """

  if np.max(binary_image) > 1:
    binary = (binary_image > 0).astype(np.uint8)
  else:
    binary = binary_image.copy()

  height, width = binary.shape
  labels = np.zeros((height, width), dtype=np.int32)

  # Initialize component properties list
  components = []

  current_label = 1

  # First pass: label components
  for i in range(height):
    for j in range(width):
      if binary[i, j] == 1 and labels[i, j] == 0:
        # Start a new component
        stack = [(i, j)]
        pixels_in_component = 0
        sum_i, sum_j = 0, 0  # For calculating center

        while stack:
          y, x = stack.pop()

          if (y < 0 or y >= height or x < 0 or x >= width or
              binary[y, x] == 0 or labels[y, x] != 0):
            continue

          labels[y, x] = current_label
          pixels_in_component += 1
          sum_i += y
          sum_j += x

          # Add neighbors to stack
          stack.append((y - 1, x))  # Up
          stack.append((y + 1, x))  # Down
          stack.append((y, x - 1))  # Left
          stack.append((y, x + 1))  # Right

        # Calculate component properties
        if pixels_in_component >= min_size:
          components.append({
            'label': current_label,
            'size': pixels_in_component,
            'center': (
            sum_i / pixels_in_component, sum_j / pixels_in_component)
          })
          current_label += 1
        else:
          # If component is too small, remove it
          labels[labels == current_label] = 0

  # Create final labeled image (only components that meet the size requirement)
  output = np.zeros_like(labels)
  for comp in components:
    output[labels == comp['label']] = comp['label']

  return output, components


def detect_eyes(image):
  """
  Detect eye regions in the image
  This method was generated using GitHub Copilot

  :param image: Preprocessed greyscale image

  :return: Image showing eye regions
  """

  height, _ = image.shape

  # Focus on upper half of image
  upper_face = image[:height // 2, :]

  # Apply adaptive thresholding to highlight potential eye regions
  window_size = 15
  eye_candidates = np.zeros_like(upper_face)

  # Create padded image for local thresholding
  pad = window_size // 2
  padded = np.zeros(
      (upper_face.shape[0] + 2 * pad, upper_face.shape[1] + 2 * pad),
      dtype=np.float32)
  padded[pad:pad + upper_face.shape[0],
  pad:pad + upper_face.shape[1]] = upper_face

  for i in range(upper_face.shape[0]):
    for j in range(upper_face.shape[1]):
      # Get a local neighborhood
      neighborhood = padded[i:i + window_size, j:j + window_size]
      local_mean = np.mean(neighborhood)

      # Apply a threshold
      if upper_face[i, j] < local_mean * 0.8:  # Threshold parameter
        eye_candidates[i, j] = 1

  # Apply connected components to identify eye regions
  labeled, components = connected_components(eye_candidates, min_size=50)

  # Sort components by size and get top candidates
  components.sort(key=lambda x: x['size'], reverse=True)

  # Create visualization of eye regions
  eye_regions = np.zeros_like(upper_face)
  for comp in components[:min(len(components), 5)]:  # Show top 5 candidates
    mask = labeled == comp['label']
    eye_regions[mask] = 255

  # Get eye center (assuming the two largest components are eyes)
  if len(components) >= 2:
    eye_center = [components[0]['center'], components[1]['center']]

    # Adjust y-coordinates to account for using only upper half
    eye_center = [(y, x) for y, x in eye_center]

    # Filter eyes based on horizontal arrangement (eyes should be roughly at same height)
    if abs(eye_center[0][0] - eye_center[1][0]) > height * 0.1:
      # Eyes are not at similar heights, might not be reliable
      return eye_regions, None

    # Ensure left-to-right ordering
    if eye_center[0][1] > eye_center[1][1]:
      eye_center = [eye_center[1], eye_center[0]]

    return eye_regions, eye_center
  else:
    return eye_regions, None