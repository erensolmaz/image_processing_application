# Image Processing Application - DIP Midterm Project

A PyQt5-based image processing application for Digital Image Processing course.

## Features

- **Basic Operations**: Grayscale, Flip, Rotate
- **Affine Transformations**: Rotation, Scaling, Translation, Shear (X/Y/Both)
- **Intensity Adjustments**: Contrast, Gamma Correction, Negative
- **Spatial Filters**: Mean, Gaussian, Median, Laplacian, Sobel
- **Histogram Operations**: Display and Equalization
- **Morphology**: Thresholding (Global/Otsu), Erosion, Dilation, Opening, Closing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app_gui_qt.py
```

## Build Executable

**Mac/Linux:**
```bash
./build.sh
```

**Windows:**
```cmd
build.bat
```

Executable will be in `dist/` folder as `DIP_Midterm`.

## Requirements

- Python 3.9+
- PyQt5
- OpenCV
- NumPy
- Pillow

## Project Structure

```
├── app_gui_qt.py       # Main application
├── src/
│   ├── processors/     # Image processing modules
│   └── utils/          # Utilities
├── build.sh            # Build script (Mac/Linux)
├── build.bat           # Build script (Windows)
└── requirements.txt    # Dependencies
```

## Usage Tips

1. Load an image using "Load Image" button
2. Apply operations from the left panel
3. For morphology: Apply threshold first, then morphological operations
4. Save results with "Save Image" button
5. Use "Reset" to restore original image

---

**Course**: Digital Image Processing (DIP) - Midterm Project
