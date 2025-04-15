import cv2
import numpy as np
import time
import collections


def detect_face_width(frame):
    """Detect face width focusing on a single person in the center of the frame"""
    # Get frame dimensions

    """Gaussian smoothing?"""
    """Histogram equalization?"""
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Define center region of interest (ROI)
    roi_size = min(width, height) * 3 // 4  # Use 75% of the smaller dimension
    roi_x = max(0, center_x - roi_size // 2)
    roi_y = max(0, center_y - roi_size // 2)
    roi_width = min(roi_size, width - roi_x)
    roi_height = min(roi_size, height - roi_y)

    # Extract center ROI
    roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Draw ROI indicator on frame (optional)
    cv2.rectangle(frame, (roi_x, roi_y),
                  (roi_x + roi_width, roi_y + roi_height),
                  (255, 255, 0), 2)

    # Process center region with YCrCb
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

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

    # Create a full-frame mask for visualization
    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = skin_mask

    faces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:  # Minimum face size
            x, y, w, h = cv2.boundingRect(contour)

            # Face aspect ratio check
            """0.8 to 1.3?"""
            if 0.5 < w / h < 1.6 and h > 60:
                # Calculate center of the face
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate distance from center of ROI
                roi_center_x = roi_width // 2
                roi_center_y = roi_height // 2
                distance_from_center = np.sqrt(
                    (face_center_x - roi_center_x) ** 2 +
                    (face_center_y - roi_center_y) ** 2)

                # Adjust coordinates to original frame
                x_global = x + roi_x
                y_global = y + roi_y

                # Store face info
                faces.append({
                    'rect': (x_global, y_global, w, h),
                    'width': w,
                    'area': area,
                    'distance_from_center': distance_from_center,
                    'contour': contour
                })

    # If we found faces in the ROI, sort by distance from center and take the closest
    if faces:
        faces.sort(key=lambda x: x['distance_from_center'])
        faces = [faces[0]]  # Keep only the face closest to center
    else:
        # If no faces in ROI, try full frame as fallback
        ycrcb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask_full = cv2.inRange(ycrcb_full, lower_ycrcb, upper_ycrcb)
        skin_mask_full = cv2.morphologyEx(skin_mask_full, cv2.MORPH_OPEN,
                                          kernel)
        skin_mask_full = cv2.morphologyEx(skin_mask_full, cv2.MORPH_CLOSE,
                                          kernel)

        contours_full, _ = cv2.findContours(skin_mask_full, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

        full_mask = skin_mask_full  # Update full mask for visualization

        for contour in contours_full:
            area = cv2.contourArea(contour)
            if area > 5000:  # Slightly higher threshold for full frame
                x, y, w, h = cv2.boundingRect(contour)

                if 0.5 < w / h < 1.6 and h > 80:
                    # Calculate distance from center of frame
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    distance_from_center = np.sqrt(
                        (face_center_x - center_x) ** 2 +
                        (face_center_y - center_y) ** 2)

                    faces.append({
                        'rect': (x, y, w, h),
                        'width': w,
                        'area': area,
                        'distance_from_center': distance_from_center,
                        'contour': contour
                    })

        # Take center-most face if any found
        if faces:
            faces.sort(key=lambda x: x['distance_from_center'])
            faces = [faces[0]]

    return full_mask, faces


def main():
    # Initialize webcam (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set display window properties
    cv2.namedWindow('Face Width Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Width Detection', 1280, 720)

    print("Press 'q' to quit")

    # FPS calculation variables
    prev_time = time.time()
    fps = 0

    # Queue for smoothing width measurements
    width_history = collections.deque(maxlen=10)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #ret = True
        #frame = cv2.imread('images/Lighting-Samples.jpg')


        if not ret:
            print("Error: Failed to capture image from webcam")
            break

        # Mirror image for more intuitive self-viewing
        frame = cv2.flip(frame, 1)

        # Current time for FPS calculation
        current_time = time.time()
        time_diff = current_time - prev_time

        # Process frame
        if time_diff > 1 / 30:  # Limit processing to ~30fps
            # Calculate FPS
            fps = 1.0 / time_diff
            prev_time = current_time

            # Detect faces and their widths
            skin_mask, faces = detect_face_width(frame)

            # Create visualization image
            display_frame = frame.copy()

            # Draw skin mask in corner (resized)
            mask_height, mask_width = skin_mask.shape
            mask_small = cv2.resize(skin_mask,
                                    (mask_width // 4, mask_height // 4))
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
                cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                # Draw width line
                mid_y = y + h // 2
                cv2.line(display_frame, (x, mid_y), (x + w, mid_y), (255, 0, 0),
                         2)

                # Add width text
                width_text = f"Width: {smoothed_width}px"
                cv2.putText(display_frame, width_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Distance estimation (rough approximation)
                # This would need proper calibration for accurate results
                estimated_distance = 500 / (
                        smoothed_width + 1e-5)  # Avoid division by zero
                dist_text = f"~{estimated_distance:.1f}m"
                cv2.putText(display_frame, dist_text, (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw "sweet spot" indicator
                frame_center_x = display_frame.shape[1] // 2
                frame_center_y = display_frame.shape[0] // 2
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Calculate distance from ideal center position
                center_distance = np.sqrt(
                    (face_center_x - frame_center_x) ** 2 +
                    (face_center_y - frame_center_y) ** 2)

                # Draw positioning guide
                if center_distance < 50:
                    position_text = "Perfect Position"
                    text_color = (0, 255, 0)  # Green
                elif center_distance < 100:
                    position_text = "Good Position"
                    text_color = (0, 255, 255)  # Yellow
                else:
                    position_text = "Move to Center"
                    text_color = (0, 0, 255)  # Red

                cv2.putText(display_frame, position_text,
                            (frame_center_x - 100, frame_center_y + 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            else:
                # No face detected
                cv2.putText(display_frame, "No face detected",
                            (display_frame.shape[1] // 2 - 100,
                             display_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Clear width history when no face is detected
                width_history.clear()

            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the resulting frame
            cv2.imshow('Face Width Detection', display_frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()