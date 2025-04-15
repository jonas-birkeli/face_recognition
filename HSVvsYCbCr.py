import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_skin_hsv(image):
    """Detect skin using HSV color space"""
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for skin
    lower_skin_hsv = np.array([0, 30, 80], dtype=np.uint8)
    upper_skin_hsv = np.array([20, 150, 255], dtype=np.uint8)

    # Create binary mask
    skin_mask_hsv = cv2.inRange(hsv_image, lower_skin_hsv, upper_skin_hsv)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask_hsv = cv2.morphologyEx(skin_mask_hsv, cv2.MORPH_OPEN, kernel)
    skin_mask_hsv = cv2.morphologyEx(skin_mask_hsv, cv2.MORPH_CLOSE, kernel)

    return skin_mask_hsv


def detect_skin_ycrcb(image):
    """Detect skin using YCrCb color space"""
    # Convert to YCrCb
    ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Define YCrCb range for skin
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])

    # Create binary mask
    skin_mask_ycrcb = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask_ycrcb = cv2.morphologyEx(skin_mask_ycrcb, cv2.MORPH_OPEN, kernel)
    skin_mask_ycrcb = cv2.morphologyEx(skin_mask_ycrcb, cv2.MORPH_CLOSE, kernel)

    return skin_mask_ycrcb


# Example usage
if __name__ == "__main__":
    # Load an image
    image_path = "images/Lighting-Samples.jpg"  # Replace with your test image path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit(1)

    # Resize if the image is too large
    max_dim = 800
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Method 1: HSV skin detection
    skin_mask_hsv = detect_skin_hsv(image)

    # Method 2: YCrCb skin detection
    skin_mask_ycrcb = detect_skin_ycrcb(image)

    # Display results
    plt.figure(figsize=(15, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(skin_mask_hsv, cmap='gray')
    plt.title("HSV Skin Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(skin_mask_ycrcb, cmap='gray')
    plt.title("YCrCb Skin Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()