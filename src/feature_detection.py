"""
Module for feature detection functions, including edge detection and eye detection.
"""
from operator import truediv

import numpy as np

from src.color_space_functions import rgb_to_hsv, rgb_to_ycbcr, \
  detect_skin_ycbcr


def edge_detection_sobel(image):
  """
  Apply Sobel-edge detection

  :param image: Input grayscale image

  :return: Binary-edge image
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

  threshold = 25 # TODO Adjustable!
  edge_image = (gradient_magnitude > threshold).astype(np.uint8) * 255
  # Cool math, taking bool value times 255 to create either 0 or 255 :)

  return edge_image

def edge_detection_canny(image):
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


def detect_eyes_improved(image):
  """
  Detect eye regions using multiple color spaces.
  This method was created using GitHub Copilot.

  :param image: Input color image

  :return: Image showing eye regions, list of eye centers
  """
  height, width = image.shape[:2]

  # Convert to HSV and YCbCr color spaces
  hsv_image = rgb_to_hsv(image)
  ycbcr_image = rgb_to_ycbcr(image)

  # Detect skin regions using YCbCr
  skin_mask = detect_skin_ycbcr(ycbcr_image)

  # Focus on upper half of the image for eye detection
  upper_half = int(height * 0.6)  # Consider upper 60% of the image
  face_upper = np.zeros_like(skin_mask)
  face_upper[:upper_half, :] = skin_mask[:upper_half, :]

  # Detect eye regions using HSV
  hsv_eye_mask = detect_eyes_hsv(hsv_image[:upper_half, :])

  # Pad the hsv_eye_mask to match the original image size
  padded_eye_mask = np.zeros_like(skin_mask)
  padded_eye_mask[:upper_half, :] = hsv_eye_mask

  # Combine skin and eye detection
  refined_eye_regions = refine_eye_regions(padded_eye_mask, face_upper)

  # Apply morphological operations to clean up the eye mask
  se_open = create_structuring_element(size=1, shape='disk')
  se_close = create_structuring_element(size=3, shape='disk')

  # Open operation (erosion followed by dilation) to remove noise
  eroded = erode(refined_eye_regions, se_open)
  opened = dilate(eroded, se_open)

  # Close operation (dilation followed by erosion) to fill holes
  dilated = dilate(opened, se_close)
  closed = erode(dilated, se_close)

  # Apply connected components to identify eye regions
  labeled, components = connected_components(closed, min_size=30)

  # Create visualization of eye regions
  eye_regions = np.zeros_like(closed)
  for comp in components[:min(len(components), 5)]:  # Show top 5 candidates
    mask = labeled == comp['label']
    eye_regions[mask] = 255

  # Get eye centroids (assuming the two largest components are eyes)
  if len(components) >= 2:
    # Sort components by size
    components.sort(key=lambda x: x['size'], reverse=True)

    # Get the two largest components
    eye_centroids = [components[0]['centroid'], components[1]['centroid']]

    # Filter eyes based on horizontal arrangement (eyes should be roughly at same height)
    if abs(eye_centroids[0][0] - eye_centroids[1][0]) > height * 0.1:
      # Eyes are not at similar heights, might not be reliable
      return eye_regions, None

    # Ensure left-to-right ordering
    if eye_centroids[0][1] > eye_centroids[1][1]:
      eye_centroids = [eye_centroids[1], eye_centroids[0]]

    return eye_regions, eye_centroids
  else:
    return eye_regions, None

def detect_eyes(image):
  """
  Detect eye regions in image.
  A wrapper around improved eye detection
  This method was generated using GitHub Copilot

  :param image: Input image

  :return: Image showing eye regions, list of eye centers
  """
  # Check if input is grayscale or color
  if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
    # If grayscale, convert to 3-channel grayscale
    if len(image.shape) == 2:
      image_3ch = np.stack([image] * 3, axis=2)
    else:
      image_3ch = np.concatenate([image] * 3, axis=2)

    return detect_eyes_improved(image_3ch)
  else:
    # Already color image
    return detect_eyes_improved(image)