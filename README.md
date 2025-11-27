# Image Processing Application - DIP Midterm Project

This is a image processing application I made for Digital Image Processing course. I used PyQt5 to make a simple GUI where you can load images and do different image processing operations. The source code files are in src folder and executable file is in dist folder.

## App Overview

The application has two parts. Left side has control panel with operation buttons. Right side has two image panels showing original and result images.

## Implemented Features

File Operations: Load Image, Save Image, Reset

Basic Operations: Grayscale, Flip Horizontal/Vertical, Rotate +90/-90 degrees

Affine Transformations: Rotate (custom angle), Scale (uniform or X/Y), Translate (dx, dy), Shear (X/Y/Both)

Intensity Adjustments: Contrast (slider 50-200%), Contrast Stretch, Gamma (slider 0.1-3.0), Negative

Spatial Filters: Mean/Box (5x5), Gaussian (15x15), Median (5x5), Laplacian (3x3), Sobel X/Y (3x3)

Histogram Operations: Show Histogram, Histogram Equalization

Morphological Operations: Global threshold (0-255), Otsu threshold (automatic), Erode, Dilate, Open, Close

## Key Parameters

Threshold: Global uses manual value (0-255, default 127). Otsu is automatic.

Morphological Operations: Uses 3x3 ellipse kernel, 1 iteration. Need to apply threshold first.

Spatial Filters: Kernel sizes are fixed (Mean 5x5, Gaussian 15x15, Median 5x5, Laplacian/Sobel 3x3). All convert to grayscale.

Scale: Positive numbers only (0.5 = half size, 2.0 = double size).

Contrast: Slider 50-200% (100% = no change).

Gamma: Slider 0.1-3.0 (1.0 = no change).

## How to Use

1. Click Load Image to select file
2. Click operation button from left panel
3. See result in right panel
4. For morphology: Apply Threshold first, then use Erode/Dilate/Open/Close
5. Click Save Image to save result
6. Click Reset to go back to original

## Known Limitations

- Filters and morphology work on grayscale only (color images converted automatically)
- Large images might be slow
- Scale must be positive number
- Morphology needs threshold first
- Large scale values (over 10x) might cause problems

## Installation

```bash
pip install -r requirements.txt
python app_gui_qt.py
```

## Requirements

Python 3.9+, PyQt5, OpenCV, NumPy, Pillow

Course: Digital Image Processing (DIP) - Midterm Project
