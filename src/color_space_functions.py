"""
Module for color space operations to improve facial feature detection
"""

from feature_detection import dilate, create_structuring_element

import numpy as np

def rgb_to_hsv(rgb_image):
  """
  Convert RGB image to HSV color space without using built-in functions.

  :param rgb_image: RGB image with values in range [0, 255]

  :return: HSV image with H in range [0, 179], S and V in range [0, 255]
  """
  # Ensure input is in correct format
  rgb = rgb_image.astype(np.float32) / 255.0

  # Get individual color channels
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

  # Calculate Value (V)
  v = np.max(rgb, axis=2)

  # Calculate delta = max - min
  min_rgb = np.min(rgb, axis=2)
  delta = v - min_rgb

  # Initialize H and S arrays
  h = np.zeros_like(r)
  s = np.zeros_like(r)

  # Calculate Saturation (S)
  # If V is 0, S is 0. Otherwise, S = delta/V
  non_zero_v_mask = v > 0
  s[non_zero_v_mask] = delta[non_zero_v_mask] / v[non_zero_v_mask]

  # Calculate Hue (H)
  # If delta is 0, H is 0
  non_zero_delta_mask = delta > 0

  # For pixels where R is max
  r_max_mask = (v == r) & non_zero_delta_mask
  h[r_max_mask] = (60 * (
        (g[r_max_mask] - b[r_max_mask]) / delta[r_max_mask])) % 360

  # For pixels where G is max
  g_max_mask = (v == g) & non_zero_delta_mask
  h[g_max_mask] = 60 * (
        (b[g_max_mask] - r[g_max_mask]) / delta[g_max_mask]) + 120

  # For pixels where B is max
  b_max_mask = (v == b) & non_zero_delta_mask
  h[b_max_mask] = 60 * (
        (r[b_max_mask] - g[b_max_mask]) / delta[b_max_mask]) + 240

  # Normalize to OpenCV HSV range
  h = h / 2.0  # Convert [0, 360] to [0, 180]
  s = s * 255.0
  v = v * 255.0

  # Stack channels and convert to uint8
  hsv = np.stack([h, s, v], axis=2).astype(np.uint8)

  return hsv


def rgb_to_ycbcr(rgb_image):
  """
  Convert RGB image to YCbCr color space without using built-in functions.

  :param rgb_image: RGB image with values in range [0, 255]

  :return: YCbCr image with values in range [0, 255]
  """
  # Ensure input is in correct format
  rgb = rgb_image.astype(np.float32)

  # Get individual color channels
  r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

  # RGB to YCbCr conversion
  y = 0.299 * r + 0.587 * g + 0.114 * b
  cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
  cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b

  # Stack channels and convert to uint8
  ycbcr = np.stack([y, cb, cr], axis=2).astype(np.uint8)

  return ycbcr


def detect_skin_ycbcr(ycbcr_image):
  """
  Detect skin regions using YCvCr color space.

  :param ycbcr_image: YCbCr image

  :return: Binary mask where 1 represents skin regions
  """
  # Extract Cb and Cr channels
  cb = ycbcr_image[:, :, 1]
  cr = ycbcr_image[:, :, 2]

  # Define skin color range in YCbCr
  # These ranges are common for skin detection across different ethnicities
  skin_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)

  return skin_mask.astype(np.uint8) * 255


def detect_eyes_hsv(hsv_image):
  """
  Detect eye regions using HSV color space

  :param hsv_image: HSV image

  :return: Binary mask where 1 represents potential eye regions
  """
  # Extract H, S, V channels
  h = hsv_image[:, :, 0]
  s = hsv_image[:, :, 1]
  v = hsv_image[:, :, 2]

  # Eye regions typically have:
  # 1. Low saturation (dark pupils, white sclera)
  # 2. Low to medium value (especially pupils)
  # 3. Hue in certain ranges (depends on eye color)

  # Create masks for potential eye regions
  # Low saturation for white parts of eyes
  low_s_mask = s < 70

  # Low value for pupils
  low_v_mask = v < 80

  # Medium value for iris
  med_v_mask = (v >= 80) & (v <= 160)

  # Combine masks for eye regions
  # This combines pupil detection (dark regions) with sclera detection (white regions)
  eye_mask = (low_s_mask & med_v_mask) | low_v_mask

  return eye_mask.astype(np.uint8) * 255

def refine_eye_regions(eye_mask, skin_mask):
  """
  Regine eye regions by combining eye detection and skin detection

  :param eye_mask: Binary mask of potential eye regions
  :param skin_mask: Binary mask of skin regions

  :return: Refined binary mask for eye regions
  """
  # Convert to binary
  if np.max(eye_mask) > 1:
    eye_mask = (eye_mask > 0).astype(np.uint8)

  if np.max(skin_mask) > 1:
    skin_mask = (skin_mask > 0).astype(np.uint8)

  # Dilate skin mask slightly to ensure it covers facial features
  skin_dilated = dilate(skin_mask, create_structuring_element(1, 'disk'))

  # Find eye regions within the face area (skin)
  # We're looking for eyes that are surrounded by skin but are not skin themselves
  refined_eye_mask = eye_mask & skin_dilated

  return refined_eye_mask * 255