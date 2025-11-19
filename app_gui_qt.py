
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
    from PyQt5.QtGui import QPixmap, QImage
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
            self.thresh_mode = 'otsu'    # 'otsu' or 'global'
            self.global_thresh = 127
            self.morph_kernel = 3
            self.morph_iters = 1
            self.io = ImageIO()
            self.visualizer = ImageVisualizer()
            self.filters = FilterProcessor()
            self.transforms = TransformationProcessor()
            self.seg = SegmentationProcessor()
            self.enh = EnhancementProcessor()
            self.edges = EdgeDetectionProcessor()
            self.morphology = MorphologyProcessor()

            self._build_ui()

        # ---------- UI ----------
        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            root = QHBoxLayout(central)

            # Left: controls
            scroll = QScrollArea()
            scroll.setFixedWidth(280)
            scroll.setWidgetResizable(True)
            ctrl_host = QWidget()
            ctrl = QVBoxLayout(ctrl_host)
            scroll.setWidget(ctrl_host)
            root.addWidget(scroll)

            # File
            file_g = QGroupBox("File")
            f = QVBoxLayout(file_g)
            b = QPushButton("Load Image"); b.clicked.connect(self.load_image); f.addWidget(b)
            b = QPushButton("Save Image"); b.clicked.connect(self.save_image); f.addWidget(b)
            b = QPushButton("Reset"); b.clicked.connect(self.reset); f.addWidget(b)
            ctrl.addWidget(file_g)

            # Basics
            basic_g = QGroupBox("Basic")
            f = QVBoxLayout(basic_g)
            self._btn(f, "Grayscale", self.to_gray)
            self._btn(f, "Flip Horizontal", lambda: self.flip('horizontal'))
            self._btn(f, "Flip Vertical", lambda: self.flip('vertical'))
            self._btn(f, "Rotate +90°", lambda: self.rotate_quick(90))
            self._btn(f, "Rotate -90°", lambda: self.rotate_quick(-90))
            ctrl.addWidget(basic_g)

            # Affine
            aff_g = QGroupBox("Affine")
            f = QVBoxLayout(aff_g)
            self._btn(f, "Rotate (angle)", self.rotate_dialog)
            self._btn(f, "Scale (Uniform/X,Y)", self.scale_dialog)
            self._btn(f, "Translate (dx,dy)", self.translate_dialog)
            self._btn(f, "Shear (X/Y/Both)", self.shear_dialog)
            ctrl.addWidget(aff_g)

            # Intensity
            int_g = QGroupBox("Intensity")
            f = QVBoxLayout(int_g)
            self._btn(f, "Contrast (slider)", self.contrast_dialog)
            self._btn(f, "Contrast Stretch", self.contrast_stretch)
            self._btn(f, "Gamma (slider)", self.gamma_dialog)
            self._btn(f, "Negative (Invert)", self.negative)
            ctrl.addWidget(int_g)

            # Spatial
            spa_g = QGroupBox("Spatial Filters")
            f = QVBoxLayout(spa_g)
            self._btn(f, "Mean/Box", lambda: self.apply_filter('box'))
            self._btn(f, "Gaussian", lambda: self.apply_filter('gaussian'))
            self._btn(f, "Median", lambda: self.apply_filter('median'))
            self._btn(f, "Laplacian", self.laplacian)
            self._btn(f, "Sobel X", lambda: self.sobel('x'))
            self._btn(f, "Sobel Y", lambda: self.sobel('y'))
            ctrl.addWidget(spa_g)

            # Histogram
            his_g = QGroupBox("Histogram")
            f = QVBoxLayout(his_g)
            self._btn(f, "Show Histogram", self.show_hist)
            self._btn(f, "Histogram Equalization", self.hist_eq)
            ctrl.addWidget(his_g)

            # Morphology (improved)
            mor_g = QGroupBox("Morphology")
            fm = QVBoxLayout(mor_g)
            # Threshold mode
            th_row = QHBoxLayout()
            th_row.addWidget(QLabel("Threshold:"))
            self.rb_global = QRadioButton("Global")
            self.rb_otsu = QRadioButton("Otsu")
            self.rb_otsu.setChecked(True)
            self.rb_global.toggled.connect(lambda _: self._update_thresh_ui())
            self.rb_otsu.toggled.connect(lambda _: self._update_thresh_ui())
            th_row.addWidget(self.rb_global); th_row.addWidget(self.rb_otsu); th_row.addStretch(1)
            fm.addLayout(th_row)
            # Params
            grid = QGridLayout()
            grid.addWidget(QLabel("T:"), 0, 0)
            self.sp_thresh = QSpinBox(); self.sp_thresh.setRange(0,255); self.sp_thresh.setValue(self.global_thresh); grid.addWidget(self.sp_thresh, 0, 1)
            grid.addWidget(QLabel("Kernel:"), 0, 2)
            self.sp_ksize = QSpinBox(); self.sp_ksize.setRange(1, 99); self.sp_ksize.setSingleStep(2); self.sp_ksize.setValue(3); grid.addWidget(self.sp_ksize, 0, 3)
            grid.addWidget(QLabel("Iter:"), 0, 4)
            self.sp_iters = QSpinBox(); self.sp_iters.setRange(1, 20); self.sp_iters.setValue(1); grid.addWidget(self.sp_iters, 0, 5)
            fm.addLayout(grid)
            # Kernel shape selection
            kernel_row = QHBoxLayout()
            kernel_row.addWidget(QLabel("Shape:"))
            self.rb_rect = QRadioButton("Rect")
            self.rb_ellipse = QRadioButton("Ellipse")
            self.rb_cross = QRadioButton("Cross")
            self.rb_ellipse.setChecked(True)
            self.morph_shape = 'ellipse'
            self.rb_rect.toggled.connect(lambda: setattr(self, 'morph_shape', 'rect') if self.rb_rect.isChecked() else None)
            self.rb_ellipse.toggled.connect(lambda: setattr(self, 'morph_shape', 'ellipse') if self.rb_ellipse.isChecked() else None)
            self.rb_cross.toggled.connect(lambda: setattr(self, 'morph_shape', 'cross') if self.rb_cross.isChecked() else None)
            kernel_row.addWidget(self.rb_rect)
            kernel_row.addWidget(self.rb_ellipse)
            kernel_row.addWidget(self.rb_cross)
            kernel_row.addStretch(1)
            fm.addLayout(kernel_row)
            # Buttons
            btns = QGridLayout()
            btn_apply_th = QPushButton("Apply Threshold"); btn_apply_th.clicked.connect(self.apply_threshold_action); btns.addWidget(btn_apply_th, 0, 0, 1, 2)
            btn_erode  = QPushButton("Erode");  btn_erode.clicked.connect(lambda: self.morph_action('erode'));  btns.addWidget(btn_erode, 1, 0)
            btn_dilate = QPushButton("Dilate"); btn_dilate.clicked.connect(lambda: self.morph_action('dilate')); btns.addWidget(btn_dilate, 1, 1)
            btn_open   = QPushButton("Open");   btn_open.clicked.connect(lambda: self.morph_action('open'));   btns.addWidget(btn_open, 2, 0)
            btn_close  = QPushButton("Close");  btn_close.clicked.connect(lambda: self.morph_action('close'));  btns.addWidget(btn_close, 2, 1)
            fm.addLayout(btns)
            ctrl.addWidget(mor_g)

            ctrl.addStretch(1)

            # Right: two panels
            panels = QGridLayout()
            right = QWidget(); right.setLayout(panels)
            root.addWidget(right, 1)

            self.lbl_orig = QLabel("Original")
            self.lbl_res = QLabel("Result")
            for lbl in (self.lbl_orig, self.lbl_res):
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("border:1px solid #aaa; background:#f7f7f7;")
                lbl.setMinimumSize(520, 520)
            panels.addWidget(self._group("Original", self.lbl_orig), 0, 0)
            panels.addWidget(self._group("Result", self.lbl_res), 0, 1)

        def _group(self, title, widget):
            g = QGroupBox(title)
            l = QVBoxLayout(g); l.addWidget(widget)
            return g

        def _btn(self, layout, text, slot):
            b = QPushButton(text); b.clicked.connect(slot); layout.addWidget(b)

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
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)")
            if not path: return
            self.io.save_image(self.current_image, Path(path))
            QMessageBox.information(self, "Information", "Image saved.")

        def _to_qpix(self, img):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
            if self.current_image is None: return
            dlg = QDialog(self); dlg.setWindowTitle("Scale")
            form = QFormLayout(dlg)
            sx = QLineEdit("1.0"); sy = QLineEdit("1.0"); uniform = QCheckBox("Uniform (X=Y)"); uniform.setChecked(True)
            form.addRow("Scale X:", sx); form.addRow("Scale Y:", sy); form.addRow("", uniform)
            ok = QPushButton("Apply"); ok.clicked.connect(dlg.accept); form.addRow(ok)
            if dlg.exec_():
                try:
                    fx = float(sx.text()); fy = fx if uniform.isChecked() else float(sy.text())
                    h, w = self.current_image.shape[:2]
                    self.current_image = cv2.resize(self.current_image, (int(w*fx), int(h*fy)))
                    self._refresh()
                except: pass

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
            
            ok_btn = QPushButton("Apply")
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
            ok = QPushButton("OK"); ok.clicked.connect(dlg.accept); form.addRow(ok)
            if dlg.exec_(): return edit.text(), True
            return None, False

        # ---------- Intensity ----------
        def contrast_stretch(self):
            if self.current_image is None:
                return
            img = self.current_image.copy()
            
            # Improved contrast stretching implementation
            if img.ndim == 2:
                # Grayscale: linear contrast stretching
                min_val = float(img.min())
                max_val = float(img.max())
                
                if max_val > min_val:
                    # Linear stretch: s = (r - r_min) * (255 / (r_max - r_min))
                    stretched = ((img.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                else:
                    stretched = img.copy()
            else:
                # Color: stretch each channel independently for better contrast
                stretched = np.zeros_like(img, dtype=np.float32)
                for i in range(3):
                    channel = img[:, :, i].astype(np.float32)
                    min_val = float(channel.min())
                    max_val = float(channel.max())
                    
                    if max_val > min_val:
                        stretched[:, :, i] = ((channel - min_val) * (255.0 / (max_val - min_val)))
                    else:
                        stretched[:, :, i] = channel
                
                stretched = np.clip(stretched, 0, 255).astype(np.uint8)
            
            self.current_image = stretched
            self._refresh()

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
            btn_apply = QPushButton("Apply"); btn_apply.clicked.connect(do_apply)
            layout.addWidget(btn_apply)
            def do_cancel():
                self.current_image = base_img
                self._refresh()
                dlg.reject()
            btn_close = QPushButton("Cancel"); btn_close.clicked.connect(do_cancel)
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
            btn_apply = QPushButton("Apply"); btn_apply.clicked.connect(do_apply)
            layout.addWidget(btn_apply)
            def do_cancel():
                self.current_image = base_img
                self._refresh()
                dlg.reject()
            btn_close = QPushButton("Cancel"); btn_close.clicked.connect(do_cancel)
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
            self.thresh_mode = 'global' if self.rb_global.isChecked() else 'otsu'
            self.sp_thresh.setEnabled(self.thresh_mode == 'global')

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
                if self.thresh_mode == 'global':
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
                else:
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
            kernel_size = int(self.sp_ksize.value())
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(1, min(99, kernel_size))
            
            iterations = int(self.sp_iters.value())
            iterations = max(1, min(20, iterations))
            
            self.morph_kernel = kernel_size
            self.morph_iters = iterations
            
            return kernel_size, iterations

        def morph_action(self, operation: str):
            binary = self._apply_threshold()
            if binary is None:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    "Please apply threshold first!"
                )
                return
            
            try:
                kernel_size, iterations = self._get_morph_params()
                
                if operation == 'erode':
                    result = self.morphology.erode(
                        binary,
                        kernel_shape=self.morph_shape,
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'dilate':
                    result = self.morphology.dilate(
                        binary,
                        kernel_shape=self.morph_shape,
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'open':
                    result = self.morphology.opening(
                        binary,
                        kernel_shape=self.morph_shape,
                        kernel_size=kernel_size,
                        iterations=iterations
                    )
                elif operation == 'close':
                    result = self.morphology.closing(
                        binary,
                        kernel_shape=self.morph_shape,
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
                    f"Morphology operation failed: {e}"
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

