"""
Module for visualizing images
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def visualize_pipeline(image_dict, save=False, save_dir=None):
  """
  Visualize the processing pipeline.
  This method was generated using GitHub Copilot

  :param image_dict: Dictionary of named images to display
  :param save: TODO Whether to save the visualization
  :param save_dir: TODO Directory to save the visualization

  :return: None
  """
  num_images = len(image_dict)

  if num_images == 0:
    return

  # Determine grid size
  if num_images <= 3:
    rows, cols = 1, num_images
  elif num_images <= 6:
    rows, cols = 2, 3
  else:
    rows, cols = 3, 3

  # Create figure
  plt.figure(figsize=(4 * cols, 4 * rows))

  # Plot each image
  for i, (name, img) in enumerate(image_dict.items()):
    if i >= rows * cols:
      break

    # Create subplot without using plt.subplot
    ax = plt.gcf().add_subplot(rows, cols, i + 1)

    # Handle different image types
    if len(img.shape) == 3:  # Color image
      ax.imshow(img[:, :, [2, 1, 0]])  # Convert BGR to RGB for display
    else:  # Grayscale image
      ax.imshow(img, cmap='gray')

    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

  # Add spacing between subplots manually instead of using tight_layout
  plt.subplots_adjust(wspace=0.3, hspace=0.3)

  if save and save_dir is not None:
    # TODO
    pass

  plt.show()

def display_result(image, eye_centers, eye_distance, too_close, save=False, save_dir=None):
  """
  Display the final result with eye detection and distance measurement.
  This method was generated using GitHub Copilot

  :param image: Original image
  :param eye_centers: List of (y, x) coordinates for eye centers
  :param eye_distance: Distance between eyes
  :param too_close: Whether the face is too close
  :param save: Whether to save the result
  :param save_dir: Dictionary to save the result

  :return: None
  """
  # Create a copy of the image for visualization
  result_img = image.copy()

  # Convert to RGB for display
  if len(result_img.shape) == 3:
    display_img = result_img[:, :, [2, 1, 0]]
  else:
    # Convert grayscale to RGB
    display_img = np.stack([result_img] * 3, axis=2)

  plt.figure(figsize=(10, 8))

  # Display the image
  plt.imshow(display_img)

  # Plot eye positions and line
  if eye_centers and len(eye_centers) >= 2:
    y1, x1 = eye_centers[0]
    y2, x2 = eye_centers[1]

    plt.plot(x1, y1, 'ro', markersize=10)
    plt.plot(x2, y2, 'ro', markersize=10)
    plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)

    # Add distance information
    plt.text(10, 30, f"Eye Distance: {eye_distance:.2f} pixels",
             color='yellow', fontsize=12,
             bbox=dict(facecolor='black', alpha=0.5))

    plt.text(10, 60, f"Distance Ratio: {eye_distance / image.shape[1]:.2f}",
             color='yellow', fontsize=12,
             bbox=dict(facecolor='black', alpha=0.5))

  # Add result text
  if too_close:
    plt.text(image.shape[1] // 2 - 100, 40, "TOO CLOSE!",
             color='red', fontsize=24, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7))
  else:
    plt.text(image.shape[1] // 2 - 100, 40, "Distance OK",
             color='green', fontsize=24, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7))

  plt.axis('off')

  if save and save_dir is not None:
    # TODO
    pass

  plt.show()