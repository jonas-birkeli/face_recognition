import cv2
import numpy as np
import os
import collections


def detect_face_width(frame):
    """Detect face width focusing on skin color detection"""
    height, width = frame.shape[:2]

    # Process the entire frame with YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Define skin color range in YCrCb
    lower_ycrcb = np.array([0, 135, 85])
    upper_ycrcb = np.array([255, 180, 135])

    # Create binary mask
    skin_mask = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    faces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Minimum face size
            x, y, w, h = cv2.boundingRect(contour)

            # Face aspect ratio check
            if 0.5 < w / h < 1.6 and h > 80:
                # Calculate center of the face
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate distance from center of frame
                center_x, center_y = width // 2, height // 2
                distance_from_center = np.sqrt(
                    (face_center_x - center_x) ** 2 +
                    (face_center_y - center_y) ** 2)

                # Store face info
                faces.append({
                    'rect': (x, y, w, h),
                    'width': w,
                    'area': area,
                    'distance_from_center': distance_from_center,
                    'contour': contour
                })

    # Sort faces by distance from center and take the closest
    if faces:
        faces.sort(key=lambda x: x['distance_from_center'])
        faces = [faces[0]]  # Keep only the face closest to center

    return skin_mask, faces


def main():
    # Get the directory with images
    image_dir = "thispersondoesnotexist"

    # Check if directory exists
    if not os.path.isdir(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist")
        return

    # Get output directory for saving processed images
    output_dir = "result"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) if
                   os.path.splitext(f.lower())[1] in image_extensions]

    if not image_files:
        print(f"No image files found in '{image_dir}'")
        return

    print(f"Found {len(image_files)} image(s)")

    # Set display window properties
    cv2.namedWindow('Face Width Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Width Detection', 1280, 720)

    print("Press 'q' to go to the next image, any other key to quit")
    print(f"Processed images will be saved to: {output_dir}")

    # Queue for smoothing width measurements
    width_history = collections.deque(maxlen=10)

    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"Processing image {i + 1}/{len(image_files)}: {image_file}")

        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image '{image_path}'")
            continue

        # Detect faces and their widths
        skin_mask, faces = detect_face_width(frame)

        # Create visualization image
        display_frame = frame.copy()

        # Draw skin mask in corner (resized)
        mask_height, mask_width = skin_mask.shape
        mask_small = cv2.resize(skin_mask, (mask_width // 4, mask_height // 4))
        mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        mask_height_small, mask_width_small = mask_small.shape
        display_frame[0:mask_height_small, 0:mask_width_small] = mask_color

        # Draw face measurements on the image
        if faces:
            face = faces[0]  # Single face detection
            x, y, w, h = face['rect']

            # Add current width to history for smoothing
            width_history.append(w)

            # Calculate smoothed width
            smoothed_width = int(sum(width_history) / len(width_history))

            # Draw rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw width line
            mid_y = y + h // 2
            cv2.line(display_frame, (x, mid_y), (x + w, mid_y), (255, 0, 0), 2)

            # Add width text
            width_text = f"Width: {smoothed_width}px"
            cv2.putText(display_frame, width_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Distance estimation (rough approximation)
            estimated_distance = 500 / (
                    smoothed_width + 1e-5)  # Avoid division by zero
            dist_text = f"~{estimated_distance:.1f}m"
            cv2.putText(display_frame, dist_text, (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # No face detected
            cv2.putText(display_frame, "No face detected",
                        (display_frame.shape[1] // 2 - 100,
                         display_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Clear width history when no face is detected
            width_history.clear()

        # Add image name and count
        cv2.putText(display_frame,
                    f"Image: {i + 1}/{len(image_files)} - {image_file}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Save the processed image
        output_filename = f"processed_{image_file}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, display_frame)
        print(f"Saved: {output_path}")

        # Display the resulting frame
        cv2.imshow('Face Width Detection', display_frame)

        # Wait for 'q' key to move to next image, any other key to exit
        key = cv2.waitKey(0)
        if key != ord('q'):
            break

    cv2.destroyAllWindows()
    print("Processing complete")


if __name__ == '__main__':
    main()