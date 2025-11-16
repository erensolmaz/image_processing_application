# Image Processing Application

İki panelli (Original/Result) PyQt5 uygulaması. 
## Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt
python app_gui_qt.py
```

## Tek EXE
- Windows: `DIP_Midterm.bat` → `dist/ImageProcessingApplication.exe`
- macOS/Linux: `bash DIP_Midterm.sh` → `dist/ImageProcessingApplication`

## Özellikler (20)
- Basics: Grayscale, Negative, Flip H/V, Rotate ±90°, Reset
- Affine: Rotate (angle), Scale (Uniform/X,Y), Translate (dx,dy), Shear (X,Y)
- Intensity: Contrast Stretching, Gamma (slider), Histogram Equalization
- Spatial: Mean/Box, Gaussian, Median, Laplacian, Sobel X, Sobel Y
- Histogram: Show Histogram
- Morphology: Global/Otsu Threshold, Erode, Dilate, Open, Close

## Yapı
```
app_gui_qt.py
DIP_Midterm.bat / DIP_Midterm.sh
build_exe.spec
src/ (gui, processors, utils)
images/, outputs/
```

