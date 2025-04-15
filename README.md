# Face Distance Detection System

This project implements a face distance detection system using webcam input, without relying on pre-trained facial recognition models or built-in face detection functions. Instead, it uses basic image processing techniques, with a special focus on image segmentation approaches.

## Overview

The system detects a face in a webcam image, locates the eyes within the face, measures the distance between them, and determines if the person is sitting too close to the screen based on the ratio of eye distance to image width.

## Features

- **Multiple segmentation methods:**
  - Threshold-based segmentation (using Otsu's method)
  - Skin color segmentation (using HSV and YCrCb color spaces)
  - Region growing segmentation
  - Watershed segmentation
  - Mean shift segmentation

- **Processing pipeline:**
  - Image acquisition from webcam
  - Image preprocessing (noise reduction, contrast enhancement)
  - Face segmentation
  - Eye detection within the face region
  - Distance measurement and analysis
  - Visual feedback

- **Interactive controls:**
  - Cycle between segmentation methods
  - Toggle debug visualizations
  - Compare all segmentation methods side-by-side
  - Save frames to disk

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/face-distance-detection.git
   cd face-distance-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install opencv-python numpy
   ```

## Usage

### Running the Main Program

```
python main.py [--method METHOD] [--threshold THRESHOLD] [--debug] [--compare]
```

Options:
- `--method`: Segmentation method to use (choices: threshold, skin_color, region_growing, watershed, mean_shift)
- `--threshold`: Eye distance to image width ratio threshold (default: 0.25)
- `--debug`: Enable debug visualizations
- `--compare`: Show comparison of all segmentation methods

### Controls During Execution

- `q`: Quit the program
- `m`: Cycle through segmentation methods
- `d`: Toggle debug mode
- `c`: Toggle comparison mode
- `s`: Save the current frame to disk

## Project Structure

```
face_distance_detection/
│
├── main.py                     # Main script
├── README.md                   # This file
│
├── modules/
│   ├── __init__.py
│   ├── acquisition.py          # Webcam image acquisition
│   ├── preprocessing.py        # Image preprocessing functions
│   ├── color_spaces.py         # Color space conversions
│   ├── segmentation.py         # Image segmentation techniques
│   ├── feature_detection.py    # Face and eye detection
│   ├── distance_analysis.py    # Eye distance measurement
│   └── visualization.py        # Result visualization
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_segmentation.py
    └── test_feature_detection.py
```

## Technical Details

### Image Segmentation Techniques

1. **Threshold-based segmentation:**
   - Uses Otsu's method for optimal thresholding
   - Applies morphological operations to refine the mask
   - Identifies the largest connected component as the face

2. **Skin color segmentation:**
   - Combines skin detection in both HSV and YCrCb color spaces
   - Uses color ranges optimized for human skin tones
   - Applies morphological operations to reduce noise

3. **Region growing segmentation:**
   - Starts from a seed point in the center of the image
   - Expands the region based on pixel similarity
   - Uses a threshold parameter to control growth

4. **Watershed segmentation:**
   - Uses distance transform to identify foreground markers
   - Applies watershed algorithm to separate face from background
   - Handles complex backgrounds better than simple thresholding

5. **Mean shift segmentation:**
   - Applies mean shift filtering to create homogeneous regions
   - Performs thresholding on the filtered image
   - Preserves edges while reducing noise

### Eye Detection

The system detects eyes within the face region using:
- Region of interest selection based on face location
- Contrast enhancement
- Adaptive thresholding to find dark regions (pupils)
- Contour analysis with filtering based on shape characteristics
- Anthropometric proportions as a fallback method

### Distance Analysis

- Calculates the Euclidean distance between detected eye centers
- Computes the ratio of eye distance to image width
- Compares the ratio to a threshold to determine if the face is too close
- Provides visual feedback when the threshold is exceeded

## Performance Considerations

- Processing time is displayed on each frame
- The choice of segmentation method affects both accuracy and performance
- Debug visualizations can slow down processing but provide insights into each step
- Comparison mode allows evaluating all methods simultaneously

## Challenges and Solutions

1. **Lighting conditions:**
   - Multiple color spaces (HSV, YCrCb) used to handle varying lighting
   - Contrast enhancement applied to improve feature detection

2. **Segmentation accuracy:**
   - Multiple methods implemented to compare effectiveness
   - Morphological operations used to refine segmentation results

3. **Eye detection reliability:**
   - Adaptive thresholding tuned for eye detection
   - Fallback to anthropometric proportions when direct detection fails

4. **Performance optimization:**
   - Region of interest selection to reduce processing area
   - Early stage filtering of unlikely eye candidates

## Limitations

- Performance depends on lighting conditions
- Does not handle multiple faces in the frame
- May struggle with highly complex backgrounds
- Requires a reasonable starting face position for initial detection

## Future Improvements

- Implement tracking to improve frame-to-frame stability
- Add face orientation estimation
- Incorporate temporal filtering to reduce jitter
- Support multiple faces simultaneously
- Optimize for performance on resource-constrained devices

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenCV for the computer vision functionality
- The image processing community for segmentation algorithms