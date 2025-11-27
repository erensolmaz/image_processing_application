
import sys
from pathlib import Path

# PyQt5
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QLabel, QFileDialog, QMessageBox, QGroupBox, QScrollArea, QGridLayout,
        QDialog, QFormLayout, QLineEdit, QCheckBox, QSlider, QRadioButton, QSpinBox
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QImage, QFont
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    print("PyQt5 not found. pip install PyQt5")

if PYQT5_AVAILABLE:
    import cv2
    import numpy as np
    from src.processors import (
        FilterProcessor, TransformationProcessor, SegmentationProcessor,
        EnhancementProcessor, EdgeDetectionProcessor, MorphologyProcessor
    )
    from src.utils import ImageIO, ImageVisualizer

    class ImageProcessingGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Image Processing Application")
            self.resize(1280, 800)

            self.current_image = None
            self.original_image = None
            self.thresh_method = 'otsu'    # 'otsu' or 'global'
            self.global_thresh = 127
            self.morph_kernel = 3
            self.morph_iters = 1
            self.morph_shape = 'ellipse'  # Default shape
            self.io = ImageIO()
            self.visualizer = ImageVisualizer()
            self.filters = FilterProcessor()
            self.transforms = TransformationProcessor()
            self.seg = SegmentationProcessor()
            self.enh = EnhancementProcessor()
            self.edges = EdgeDetectionProcessor()
            self.morphology = MorphologyProcessor()

            self._apply_styles()
            self._build_ui()

        # ---------- Styles ----------
        def _apply_styles(self):
            """Apply modern, beautiful styling to the application"""
            self.setStyleSheet("""
                /* Main Window - Dark Theme */
                QMainWindow {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #0f0f1e, stop:1 #1a1a2e);
                }
                
                /* Scroll Area - Dark Glass Effect */
                QScrollArea {
                    background: rgba(0, 0, 0, 0.3);
                    border: none;
                    border-radius: 0px;
                }
                
                QScrollBar:vertical {
                    background: rgba(0, 0, 0, 0.3);
                    width: 8px;
                    border: none;
                    border-radius: 4px;
                }
                
                QScrollBar::handle:vertical {
                    background: rgba(100, 100, 120, 0.5);
                    border-radius: 4px;
                    min-height: 30px;
                }
                
                QScrollBar::handle:vertical:hover {
                    background: rgba(120, 120, 140, 0.7);
                }
                
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                
                /* Group Boxes - Dark Glassmorphism */
                QGroupBox {
                    font-weight: 600;
                    font-size: 10pt;
                    color: #e0e0e0;
                    border: 1px solid rgba(100, 100, 120, 0.3);
                    border-radius: 12px;
                    margin-top: 10px;
                    padding-top: 10px;
                    padding-left: 5px;
                    padding-right: 5px;
                    padding-bottom: 5px;
                    background: rgba(30, 30, 45, 0.6);
                }
                
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 15px;
                    padding: 5px 15px;
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #5a5a7a, stop:1 #4a4a6a);
                    border: 1px solid rgba(120, 120, 140, 0.5);
                    border-radius: 10px;
                    color: #ffffff;
                    font-weight: 700;
                    font-size: 9pt;
                }
                
                /* Buttons - Dark Modern Design */
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3a3a4a, stop:1 #2a2a3a);
                    color: #e0e0e0;
                    border: 1px solid rgba(100, 100, 120, 0.5);
                    border-radius: 8px;
                    padding: 6px 12px;
                    font-size: 8pt;
                    font-weight: 600;
                    min-height: 28px;
                    max-width: 190px;
                }
                
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4a4a5a, stop:1 #3a3a4a);
                    border: 1px solid rgba(120, 120, 140, 0.7);
                    color: #ffffff;
                }
                
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2a2a3a, stop:1 #1a1a2a);
                    border: 1px solid rgba(80, 80, 100, 0.5);
                }
                
                /* Special button styles - Dark Theme */
                QPushButton[class="file"] {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #2d7a4e, stop:1 #1f5a3a);
                    color: #a0e0b0;
                    border: 1px solid rgba(60, 150, 100, 0.5);
                }
                
                QPushButton[class="file"]:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3d8a5e, stop:1 #2d7a4e);
                    border: 1px solid rgba(80, 170, 120, 0.7);
                    color: #c0f0d0;
                }
                
                QPushButton[class="file"]:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #1f5a3a, stop:1 #153a2a);
                    border: 1px solid rgba(40, 120, 80, 0.5);
                }
                
                QPushButton[class="danger"] {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #8a2a2a, stop:1 #6a1a1a);
                    color: #f0a0a0;
                    border: 1px solid rgba(150, 60, 60, 0.5);
                }
                
                QPushButton[class="danger"]:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #9a3a3a, stop:1 #8a2a2a);
                    border: 1px solid rgba(170, 80, 80, 0.7);
                    color: #ffb0b0;
                }
                
                QPushButton[class="danger"]:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6a1a1a, stop:1 #4a0a0a);
                    border: 1px solid rgba(120, 40, 40, 0.5);
                }
                
                /* Labels */
                QLabel {
                    color: #e0e0e0;
                    background: transparent;
                    border: none;
                    padding: 5px;
                }
                
                /* Image Display Labels */
                QLabel[class="image-display"] {
                    background: rgba(0, 0, 0, 0.5);
                    border: 2px solid rgba(100, 100, 120, 0.4);
                    border-radius: 12px;
                }
                
                /* Radio Buttons */
                QRadioButton {
                    color: #e0e0e0;
                    font-size: 9pt;
                    spacing: 6px;
                }
                
                QRadioButton::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 9px;
                    border: 2px solid rgba(120, 120, 140, 0.6);
                    background: rgba(40, 40, 50, 0.8);
                }
                
                QRadioButton::indicator:hover {
                    border-color: rgba(140, 140, 160, 0.8);
                    background: rgba(50, 50, 60, 0.9);
                }
                
                QRadioButton::indicator:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6a6a7a, stop:1 #5a5a6a);
                    border-color: #8a8a9a;
                }
                
                /* Spin Boxes */
                QSpinBox {
                    background: rgba(40, 40, 50, 0.8);
                    color: #e0e0e0;
                    border: 1px solid rgba(100, 100, 120, 0.5);
                    border-radius: 6px;
                    padding: 4px;
                    font-size: 9pt;
                }
                
                QSpinBox:hover {
                    border-color: rgba(120, 120, 140, 0.7);
                    background: rgba(50, 50, 60, 0.9);
                }
                
                QSpinBox:focus {
                    border-color: rgba(140, 140, 160, 0.9);
                    background: rgba(60, 60, 70, 1.0);
                }
                
                /* Line Edits */
                QLineEdit {
                    background: rgba(40, 40, 50, 0.8);
                    color: #e0e0e0;
                    border: 1px solid rgba(100, 100, 120, 0.5);
                    border-radius: 6px;
                    padding: 6px;
                    font-size: 9pt;
                }
                
                QLineEdit:hover {
                    border-color: rgba(120, 120, 140, 0.7);
                    background: rgba(50, 50, 60, 0.9);
                }
                
                QLineEdit:focus {
                    border-color: rgba(140, 140, 160, 0.9);
                    background: rgba(60, 60, 70, 1.0);
                }
                
                /* Sliders */
                QSlider::groove:horizontal {
                    background: rgba(40, 40, 50, 0.8);
                    height: 6px;
                    border-radius: 3px;
                }
                
                QSlider::handle:horizontal {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6a6a7a, stop:1 #5a5a6a);
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    margin: -7px 0;
                }
                
                QSlider::handle:horizontal:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #7a7a8a, stop:1 #6a6a7a);
                }
                
                QSlider::sub-page:horizontal {
                    background: rgba(100, 100, 120, 0.6);
                    border-radius: 3px;
                }
                
                /* Check Boxes */
                QCheckBox {
                    color: #e0e0e0;
                    font-size: 9pt;
                    spacing: 6px;
                }
                
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border: 2px solid rgba(120, 120, 140, 0.6);
                    border-radius: 4px;
                    background: rgba(40, 40, 50, 0.8);
                }
                
                QCheckBox::indicator:hover {
                    border-color: rgba(140, 140, 160, 0.8);
                    background: rgba(50, 50, 60, 0.9);
                }
                
                QCheckBox::indicator:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #6a6a7a, stop:1 #5a5a6a);
                    border-color: #8a8a9a;
                }
                
                /* Dialogs */
                QDialog {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #0f0f1e, stop:1 #1a1a2e);
                }
                
                /* Form Layout Labels */
                QFormLayout QLabel {
                    color: #e0e0e0;
                    background-color: transparent;
                    border: none;
                    padding: 0;
                }
            """)

        # ---------- UI ----------
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QHBoxLayout(central)

            # Left: controls (with vertical scroll)
            scroll = QScrollArea()
            scroll.setFixedWidth(270)
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            ctrl_host = QWidget()
            ctrl = QVBoxLayout(ctrl_host)
            ctrl.setSpacing(5)
            ctrl.setContentsMargins(6, 6, 6, 6)
            scroll.setWidget(ctrl_host)
            root.addWidget(scroll)

            # File
            file_g = QGroupBox("📁 File")
            f = QVBoxLayout(file_g)
            f.setSpacing(3)
            b = QPushButton("📂 Load Image"); b.setProperty("class", "file"); b.clicked.connect(self.load_image); f.addWidget(b)
            b = QPushButton("💾 Save Image"); b.setProperty("class", "file"); b.clicked.connect(self.save_image); f.addWidget(b)
            b = QPushButton("🔄 Reset"); b.setProperty("class", "danger"); b.clicked.connect(self.reset); f.addWidget(b)
            ctrl.addWidget(file_g)

            # Basics
            basic_g = QGroupBox("🔧 Basic Operations")
            f = QVBoxLayout(basic_g)
            f.setSpacing(3)
            self._btn(f, "🎨 Grayscale", self.to_gray)
            self._btn(f, "↔️ Flip Horizontal", lambda: self.flip('horizontal'))
            self._btn(f, "↕️ Flip Vertical", lambda: self.flip('vertical'))
            self._btn(f, "↻ Rotate +90°", lambda: self.rotate_quick(90))
            self._btn(f, "↺ Rotate -90°", lambda: self.rotate_quick(-90))
            ctrl.addWidget(basic_g)

            # Affine
            aff_g = QGroupBox("🔄 Affine Transformations")
            f = QVBoxLayout(aff_g)
            f.setSpacing(3)
            self._btn(f, "🔄 Rotate (angle)", self.rotate_dialog)
            self._btn(f, "📏 Scale (Uniform/X,Y)", self.scale_dialog)
            self._btn(f, "➡️ Translate (dx,dy)", self.translate_dialog)
            self._btn(f, "✂️ Shear (X/Y/Both)", self.shear_dialog)
            ctrl.addWidget(aff_g)

            # Intensity
            int_g = QGroupBox("💡 Intensity Adjustments")
            f = QVBoxLayout(int_g)
            f.setSpacing(3)
            self._btn(f, "⚡ Contrast (slider)", self.contrast_dialog)
            self._btn(f, "📈 Contrast Stretch", self.contrast_stretch)
            self._btn(f, "🔆 Gamma (slider)", self.gamma_dialog)
            self._btn(f, "🔄 Negative (Invert)", self.negative)
            ctrl.addWidget(int_g)

            # Spatial
            spa_g = QGroupBox("🔍 Spatial Filters")
            f = QVBoxLayout(spa_g)
            f.setSpacing(3)
            self._btn(f, "📦 Mean/Box", lambda: self.apply_filter('box'))
            self._btn(f, "🌊 Gaussian", lambda: self.apply_filter('gaussian'))
            self._btn(f, "📊 Median", lambda: self.apply_filter('median'))
            self._btn(f, "⚡ Laplacian", self.laplacian)
            self._btn(f, "➡️ Sobel X", lambda: self.sobel('x'))
            self._btn(f, "⬇️ Sobel Y", lambda: self.sobel('y'))
            ctrl.addWidget(spa_g)

            # Histogram
            his_g = QGroupBox("📊 Histogram")
            f = QVBoxLayout(his_g)
            f.setSpacing(3)
            self._btn(f, "📈 Show Histogram", self.show_hist)
            self._btn(f, "⚖️ Histogram Equalization", self.hist_eq)
            ctrl.addWidget(his_g)

            # Morphological Operations
            mor_g = QGroupBox("🔬 Morphological Operations")
            fm = QVBoxLayout(mor_g)
            fm.setSpacing(6)
            fm.setContentsMargins(10, 10, 10, 10)
            
            # Threshold Method Selection
            th_row = QHBoxLayout()
            th_row.setSpacing(10)
            self.rb_global = QRadioButton("Global")
            self.rb_otsu = QRadioButton("Otsu")
            self.rb_otsu.setChecked(True)
            self.thresh_method = 'otsu'
            self.rb_global.toggled.connect(lambda: setattr(self, 'thresh_method', 'global') if self.rb_global.isChecked() else None)
            self.rb_otsu.toggled.connect(lambda: setattr(self, 'thresh_method', 'otsu') if self.rb_otsu.isChecked() else None)
            self.rb_global.toggled.connect(lambda _: self._update_thresh_ui())
            self.rb_otsu.toggled.connect(lambda _: self._update_thresh_ui())
            th_row.addWidget(self.rb_global)
            th_row.addWidget(self.rb_otsu)
            th_row.addStretch(1)
            fm.addLayout(th_row)
            
            # Threshold Value (only for Global)
            thresh_val_row = QHBoxLayout()
            thresh_val_row.addWidget(QLabel("Threshold Value:"))
            self.sp_thresh = QSpinBox()
            self.sp_thresh.setRange(0, 255)
            self.sp_thresh.setValue(self.global_thresh)
            self.sp_thresh.setEnabled(False)
            self.sp_thresh.setMaximumWidth(80)
            thresh_val_row.addWidget(self.sp_thresh)
            thresh_val_row.addStretch(1)
            fm.addLayout(thresh_val_row)
            
            # Apply Threshold Button
            btn_apply_th = QPushButton("🎯 Apply Threshold (Binary)")
            btn_apply_th.clicked.connect(self.apply_threshold_action)
            fm.addWidget(btn_apply_th)
            
            # Morphological Operations Buttons
            morph_btns = QGridLayout()
            morph_btns.setSpacing(5)
            
            btn_erode = QPushButton("🔽 Erode")
            btn_erode.clicked.connect(lambda: self.morph_action('erode'))
            morph_btns.addWidget(btn_erode, 0, 0)
            
            btn_dilate = QPushButton("🔼 Dilate")
            btn_dilate.clicked.connect(lambda: self.morph_action('dilate'))
            morph_btns.addWidget(btn_dilate, 0, 1)
            
            btn_open = QPushButton("🔓 Open")
            btn_open.clicked.connect(lambda: self.morph_action('open'))
            morph_btns.addWidget(btn_open, 1, 0)
            
            btn_close = QPushButton("🔒 Close")
            btn_close.clicked.connect(lambda: self.morph_action('close'))
            morph_btns.addWidget(btn_close, 1, 1)
            
            fm.addLayout(morph_btns)
            ctrl.addWidget(mor_g)

            ctrl.addStretch(1)

            # Right: two panels
            panels = QGridLayout()
            panels.setSpacing(20)
            panels.setContentsMargins(20, 20, 20, 20)
            right = QWidget()
            right.setStyleSheet("background: transparent;")
            right.setLayout(panels)
            root.addWidget(right, 1)

            self.lbl_orig = QLabel("Original")
            self.lbl_res = QLabel("Result")
            for lbl in (self.lbl_orig, self.lbl_res):
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setProperty("class", "image-display")
                lbl.setMinimumSize(520, 520)
            panels.addWidget(self._group("🖼️ Original", self.lbl_orig), 0, 0)
            panels.addWidget(self._group("✨ Result", self.lbl_res), 0, 1)

        def _group(self, title, widget):
            g = QGroupBox(title)
            l = QVBoxLayout(g); l.addWidget(widget)
            return g

        def _btn(self, layout, text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            layout.addWidget(b)

        # ---------- File / Display ----------
        def load_image(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if not path: return
            try:
                self.original_image = self.io.load_image(Path(path))
                self.current_image = self.original_image.copy()
                self._refresh()
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

        def save_image(self):
            if self.current_image is None:
                QMessageBox.warning(self, "Warning", "No image to save!")
                return
            
            path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
            )
            if not path: 
                return
            
            # Add extension if not present
            if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
                if 'png' in selected_filter.lower():
                    path += '.png'
                elif 'jpg' in selected_filter.lower() or 'jpeg' in selected_filter.lower():
                    path += '.jpg'
            
            try:
                # Ensure image is contiguous and in correct format
                img_to_save = np.ascontiguousarray(self.current_image.copy())
                
                # Ensure image is uint8
                if img_to_save.dtype != np.uint8:
                    img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
                
                # Check if image is valid
                if img_to_save.size == 0:
                    QMessageBox.critical(self, "Error", "Image is empty!")
                    return
                
                # Save using cv2 directly with error checking
                success = cv2.imwrite(str(path), img_to_save)
                
                if success:
                    QMessageBox.information(self, "Success", f"Image saved to:\n{path}")
                else:
                    QMessageBox.critical(self, "Error", 
                        f"Failed to save image to:\n{path}\n\n"
                        f"Please check:\n"
                        f"- File path is valid\n"
                        f"- You have write permissions\n"
                        f"- Disk has enough space")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving image:\n{str(e)}")

        def _to_qpix(self, img):
            # Make sure image is contiguous and copy it
            img = np.ascontiguousarray(img)
            
            if img.ndim == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            
            # Create QImage from array data
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        def _fit_pix(self, pix, target_lbl):
            if pix.isNull(): return pix
            return pix.scaled(target_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        def _refresh(self):
            if self.original_image is not None:
                po = self._fit_pix(self._to_qpix(self.original_image), self.lbl_orig)
                self.lbl_orig.setPixmap(po)
            if self.current_image is not None:
                pr = self._fit_pix(self._to_qpix(self.current_image), self.lbl_res)
                self.lbl_res.setPixmap(pr)
                self.lbl_res.update()
            QApplication.processEvents()

        # ---------- Basics ----------
        def to_gray(self):
            if self.current_image is None: return
            if self.current_image.ndim == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self._refresh()

        def negative(self):
            if self.current_image is None: return
            self.current_image = cv2.bitwise_not(self.current_image)
            self._refresh()

        def flip(self, direction):
            if self.current_image is None: return
            code = 1 if direction == 'horizontal' else 0
            self.current_image = cv2.flip(self.current_image, code)
            self._refresh()

        def rotate_quick(self, angle):
            if self.current_image is None: return
            self.current_image = self.transforms.rotate(self.current_image, angle)
            self._refresh()

        def reset(self):
            if self.original_image is None: return
            self.current_image = self.original_image.copy()
            self._refresh()

        # ---------- Affine dialogs ----------
        def rotate_dialog(self):
            if self.current_image is None: return
            angle, ok = self._simple_input("Rotate", "Angle (deg):", "45")
            if not ok: return
            try:
                self.current_image = self.transforms.rotate(self.current_image, float(angle))
                self._refresh()
            except: pass

        def scale_dialog(self):
            if self.current_image is None:
                QMessageBox.warning(self, "Warning", "Please load an image first!")
                return
            
            dlg = QDialog(self)
            dlg.setWindowTitle("Scale")
            layout = QVBoxLayout(dlg)
            
            sx_edit = QLineEdit("1.0")
            sy_edit = QLineEdit("1.0")
            uniform_cb = QCheckBox("Uniform (X=Y)")
            uniform_cb.setChecked(True)
            
            form = QFormLayout()
            form.addRow("Scale X:", sx_edit)
            form.addRow("Scale Y:", sy_edit)
            form.addRow("", uniform_cb)
            layout.addLayout(form)
            
            def apply_scale():
                try:
                    fx = float(sx_edit.text())
                    fy = fx if uniform_cb.isChecked() else float(sy_edit.text())
                    
                    if fx <= 0 or fy <= 0:
                        QMessageBox.warning(dlg, "Error", "Scale must be greater than 0!")
                        return
                    
                    if self.current_image is None:
                        QMessageBox.warning(dlg, "Error", "No image loaded!")
                        return
                    
                    # Get current image and ensure it's a copy
                    img = np.ascontiguousarray(self.current_image.copy())
                    h, w = img.shape[:2]
                    
                    # Calculate new size
                    new_w = max(1, int(w * fx))
                    new_h = max(1, int(h * fy))
                    
                    # Resize image
                    result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Ensure result is contiguous
                    result = np.ascontiguousarray(result)
                    
                    # Update image
                    self.current_image = result.copy()
                    
                    # Close dialog first
                    dlg.accept()
                    
                    # Force update display after dialog closes
                    self._refresh()
                    QApplication.processEvents()
                    
                except ValueError:
                    QMessageBox.warning(dlg, "Error", "Please enter valid numbers!")
                except Exception as e:
                    QMessageBox.critical(dlg, "Error", f"Scale failed: {str(e)}")
            
            ok_btn = QPushButton("✅ Apply")
            ok_btn.clicked.connect(apply_scale)
            layout.addWidget(ok_btn)
            
            cancel_btn = QPushButton("❌ Cancel")
            cancel_btn.clicked.connect(dlg.reject)
            layout.addWidget(cancel_btn)
            
            dlg.exec_()

        def translate_dialog(self):
            if self.current_image is None: return
            dx, ok1 = self._simple_input("Translate", "dx:", "0")
            if not ok1: return
            dy, ok2 = self._simple_input("Translate", "dy:", "0")
            if not ok2: return
            try:
                self.current_image = self.transforms.translate(self.current_image, int(dx), int(dy))
                self._refresh()
            except: pass

        def shear_dialog(self):
            if self.current_image is None: return
            dlg = QDialog(self)
            dlg.setWindowTitle("Shear")
            form = QFormLayout(dlg)
            
            shx_edit = QLineEdit("0.0")
            shy_edit = QLineEdit("0.0")
            mode_group = QHBoxLayout()
            rb_x = QRadioButton("X only")
            rb_y = QRadioButton("Y only")
            rb_both = QRadioButton("Both")
            rb_both.setChecked(True)
            mode_group.addWidget(rb_x)
            mode_group.addWidget(rb_y)
            mode_group.addWidget(rb_both)
            
            form.addRow("Shear X:", shx_edit)
            form.addRow("Shear Y:", shy_edit)
            form.addRow("Mode:", mode_group)
            
            ok_btn = QPushButton("✅ Apply")
            ok_btn.clicked.connect(dlg.accept)
            form.addRow(ok_btn)
            
            if dlg.exec_():
                try:
                    shx = float(shx_edit.text())
                    shy = float(shy_edit.text())
                    h, w = self.current_image.shape[:2]
                    
                    if rb_x.isChecked():
                        # X only
                        M = np.float32([[1, shx, 0], [0, 1, 0]])
                        new_w = int(w + abs(shx) * h)
                        new_h = h
                    elif rb_y.isChecked():
                        # Y only
                        M = np.float32([[1, 0, 0], [shy, 1, 0]])
                        new_w = w
                        new_h = int(h + abs(shy) * w)
                    else:
                        # Both
                        M = np.float32([[1, shx, 0], [shy, 1, 0]])
                        new_w = int(w + abs(shx) * h)
                        new_h = int(h + abs(shy) * w)
                    
                    self.current_image = cv2.warpAffine(self.current_image, M, (new_w, new_h))
                    self._refresh()
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Invalid input: {e}")

        def _simple_input(self, title, label, default):
            dlg = QDialog(self); dlg.setWindowTitle(title)
            form = QFormLayout(dlg); edit = QLineEdit(default); form.addRow(label, edit)
            ok = QPushButton("✅ OK"); ok.clicked.connect(dlg.accept); form.addRow(ok)
            if dlg.exec_(): return edit.text(), True
            return None, False

        # ---------- Intensity ----------
        def contrast_stretch(self):
            if self.current_image is None:
                QMessageBox.warning(self, "Warning", "Please load an image first!")
                return
            
            try:
                # Get current image
                img = self.current_image.copy()
                
                # Use percentile clipping for more aggressive contrast (clip 2% from each end)
                clip_percent = 2.0
                
                if img.ndim == 2:
                    # Grayscale - aggressive contrast stretching with percentile clipping
                    img_float = img.astype(np.float32)
                    min_val = np.percentile(img_float, clip_percent)
                    max_val = np.percentile(img_float, 100 - clip_percent)
                    
                    if max_val > min_val:
                        # Stretch: (pixel - min) * 255 / (max - min)
                        stretched = ((img_float - min_val) * 255.0) / (max_val - min_val)
                        result = np.clip(stretched, 0, 255).astype(np.uint8)
                    else:
                        result = img.copy()
                else:
                    # Color - process each channel separately with percentile clipping
                    result = np.zeros_like(img, dtype=np.uint8)
                    for i in range(3):
                        ch = img[:, :, i].astype(np.float32)
                        ch_min = np.percentile(ch, clip_percent)
                        ch_max = np.percentile(ch, 100 - clip_percent)
                        if ch_max > ch_min:
                            stretched = ((ch - ch_min) * 255.0) / (ch_max - ch_min)
                            result[:, :, i] = np.clip(stretched, 0, 255).astype(np.uint8)
                        else:
                            result[:, :, i] = img[:, :, i]
                
                # Update image
                self.current_image = result.copy()
                
                # Force refresh
                self._refresh()
                QApplication.processEvents()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error: {str(e)}")

        def gamma_dialog(self):
            if self.current_image is None: return
            dlg = QDialog(self); dlg.setWindowTitle("Gamma")
            layout = QVBoxLayout(dlg)
            # Base image snapshot (prevent cumulative changes)
            base_img = self.current_image.copy()
            base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY) if base_img.ndim==3 else base_img
            s = QSlider(Qt.Horizontal); s.setMinimum(1); s.setMaximum(300); s.setValue(100)
            layout.addWidget(s)
            preview = {'img': base_img}
            def on_change():
                gamma = max(0.1, s.value()/100.0)
                preview_img = self.enh.gamma_correction(base_gray, gamma)
                preview['img'] = preview_img
                # live preview (commit to UI), will be reverted on Cancel
                self.current_image = preview_img
                self._refresh()
            s.valueChanged.connect(on_change)
            def do_apply():
                self.current_image = preview['img']
                self._refresh()
                dlg.accept()
            btn_apply = QPushButton("✅ Apply"); btn_apply.clicked.connect(do_apply)
            layout.addWidget(btn_apply)
            def do_cancel():
                self.current_image = base_img
                self._refresh()
                dlg.reject()
            btn_close = QPushButton("❌ Cancel"); btn_close.setProperty("class", "danger"); btn_close.clicked.connect(do_cancel)
            layout.addWidget(btn_close)
            dlg.exec_()

        def contrast_dialog(self):
            if self.current_image is None: return
            dlg = QDialog(self); dlg.setWindowTitle("Contrast")
            layout = QVBoxLayout(dlg)
            base_img = self.current_image.copy()
            s = QSlider(Qt.Horizontal); s.setMinimum(50); s.setMaximum(200); s.setValue(100)
            layout.addWidget(s)
            preview = {'img': base_img}
            def apply_alpha(alpha):
                # alpha: 0.5 .. 2.0
                img = base_img
                if img.ndim == 2:
                    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                else:
                    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            def on_change():
                alpha = s.value() / 100.0
                preview_img = apply_alpha(alpha)
                preview['img'] = preview_img
                self.current_image = preview_img
                self._refresh()
            s.valueChanged.connect(on_change)
            def do_apply():
                self.current_image = preview['img']
                self._refresh()
                dlg.accept()
            btn_apply = QPushButton("✅ Apply"); btn_apply.clicked.connect(do_apply)
            layout.addWidget(btn_apply)
            def do_cancel():
                self.current_image = base_img
                self._refresh()
                dlg.reject()
            btn_close = QPushButton("❌ Cancel"); btn_close.setProperty("class", "danger"); btn_close.clicked.connect(do_cancel)
            layout.addWidget(btn_close)
            dlg.exec_()

        def hist_eq(self):
            if self.current_image is None: return
            img = self.current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
            self.current_image = self.enh.histogram_equalization(gray)
            self._refresh()

        # ---------- Spatial ----------
        def apply_filter(self, kind):
            if self.current_image is None: return
            img = self.current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
            if kind == 'box':
                self.current_image = self.filters.box_filter(gray, (5,5), True)
            elif kind == 'gaussian':
                self.current_image = self.filters.gaussian_blur(gray, (15,15))
            elif kind == 'median':
                self.current_image = self.filters.median_blur(gray, 5)
            self._refresh()

        def laplacian(self):
            if self.current_image is None: return
            img = self.current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
            lap = self.edges.laplacian(gray, 3)
            self.current_image = np.uint8(np.clip(np.abs(lap),0,255))
            self._refresh()

        def sobel(self, direction):
            if self.current_image is None: return
            img = self.current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
            sb = self.edges.sobel(gray, direction, 3)
            self.current_image = np.uint8(np.clip(np.abs(sb),0,255))
            self._refresh()

        # ---------- Histogram ----------
        def show_hist(self):
            if self.current_image is None: return
            img = self.current_image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
            hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
            h_img = np.zeros((200,256), np.uint8)
            hist = (hist/hist.max()*200).astype(np.int32)
            for x,v in enumerate(hist): h_img[200-v:, x] = 255
            self.current_image = h_img
            self._refresh()

        def _update_thresh_ui(self):
            if self.rb_global.isChecked():
                self.thresh_method = 'global'
            elif self.rb_otsu.isChecked():
                self.thresh_method = 'otsu'
            elif self.rb_adaptive.isChecked():
                self.thresh_method = 'adaptive'
            self.sp_thresh.setEnabled(self.thresh_method == 'global')

        def _prepare_grayscale(self):
            if self.current_image is None:
                return None
            if self.current_image.ndim == 3:
                return cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            return self.current_image.copy()

        def _apply_threshold(self):
            gray = self._prepare_grayscale()
            if gray is None:
                return None
            
            try:
                if self.thresh_method == 'global':
                    threshold_value = int(self.sp_thresh.value())
                    threshold_value = max(0, min(255, threshold_value))
                    self.global_thresh = threshold_value
                    
                    binary, used_thresh = self.morphology.threshold(
                        gray, 
                        method='binary',
                        threshold_value=threshold_value,
                        max_value=255
                    )
                    return binary
                else:  # Otsu
                    binary, used_thresh = self.morphology.threshold(
                        gray,
                        method='otsu',
                        max_value=255
                    )
                    return binary
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Threshold operation failed: {e}")
                return None

        def apply_threshold_action(self):
            binary = self._apply_threshold()
            if binary is not None:
                self.current_image = binary
                self._refresh()

        def _get_morph_params(self):
            # Use default values for simplicity
            kernel_size = 3
            iterations = 1
            
            self.morph_kernel = kernel_size
            self.morph_iters = iterations
            
            return kernel_size, iterations

        def morph_action(self, operation: str):
            # Apply threshold first to get binary image
            binary = self._apply_threshold()
            if binary is None:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Please apply threshold first to create binary image!"
                )
                return
            
            try:
                kernel_size, iterations = self._get_morph_params()
                
                if operation == 'erode':
                    result = self.morphology.erode(
                        binary,
                        kernel_shape='ellipse',
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'dilate':
                    result = self.morphology.dilate(
                        binary,
                        kernel_shape='ellipse',
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'open':
                    result = self.morphology.opening(
                        binary,
                        kernel_shape='ellipse',
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'close':
                    result = self.morphology.closing(
                        binary,
                        kernel_shape='ellipse',
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                else:
                    QMessageBox.warning(self, "Warning", f"Unknown operation: {operation}")
                    return
                
                self.current_image = result
                self._refresh()
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Morphological operation failed: {e}"
                )

    def main():
        if not PYQT5_AVAILABLE:
            print("PyQt5 not found.")
            sys.exit(1)
        app = QApplication(sys.argv)
        w = ImageProcessingGUI()
        w.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()

