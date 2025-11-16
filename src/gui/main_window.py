"""
Main GUI Window
Ana Grafik Arayüz Penceresi
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

from ..pipeline import ImageProcessingPipeline
from ..processors import (
    FilterProcessor, TransformationProcessor,
    SegmentationProcessor, EnhancementProcessor,
    EdgeDetectionProcessor
)
from ..utils import ImageIO, ImageVisualizer


class ImageProcessingGUI:
    """Main GUI application for image processing"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Application - Görüntü İşleme Uygulaması")
        self.root.geometry("1200x800")
        
        # Variables
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.pipeline = None
        
        # Initialize processors
        self.filters = FilterProcessor()
        self.transformations = TransformationProcessor()
        self.segmentation = SegmentationProcessor()
        self.enhancement = EnhancementProcessor()
        self.edges = EdgeDetectionProcessor()
        self.io = ImageIO()
        self.visualizer = ImageVisualizer()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(left_panel, text="Dosya İşlemleri", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Görüntü Yükle", 
                  command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Görüntü Kaydet", 
                  command=self.save_image).pack(fill=tk.X, pady=2)
        
        # Basics
        basics_frame = ttk.LabelFrame(left_panel, text="Temel İşlemler", padding="10")
        basics_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(basics_frame, text="Convert to Grayscale", 
                  command=self.convert_to_grayscale).pack(fill=tk.X, pady=2)
        ttk.Button(basics_frame, text="Negative (Invert)", 
                  command=self.apply_negative).pack(fill=tk.X, pady=2)
        ttk.Button(basics_frame, text="Show Histogram", 
                  command=self.show_histogram).pack(fill=tk.X, pady=2)
        
        # Filters
        filter_frame = ttk.LabelFrame(left_panel, text="Spatial Filters", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(filter_frame, text="Mean/Box Filter", 
                  command=lambda: self.apply_filter('box')).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Gaussian Blur", 
                  command=lambda: self.apply_filter('gaussian')).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Median Filter", 
                  command=lambda: self.apply_filter('median')).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Laplacian", 
                  command=self.apply_laplacian).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Sobel X", 
                  command=lambda: self.apply_sobel('x')).pack(fill=tk.X, pady=2)
        ttk.Button(filter_frame, text="Sobel Y", 
                  command=lambda: self.apply_sobel('y')).pack(fill=tk.X, pady=2)
        
        # Enhancement
        enhance_frame = ttk.LabelFrame(left_panel, text="Intensity Transformations", padding="10")
        enhance_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(enhance_frame, text="Contrast Stretch", 
                  command=self.apply_contrast_stretch).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Gamma Correction (Slider)", 
                  command=self.apply_gamma_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Histogram Equalization", 
                  command=self.apply_histogram_eq).pack(fill=tk.X, pady=2)
        ttk.Button(enhance_frame, text="Negative (Invert)", 
                  command=self.apply_negative).pack(fill=tk.X, pady=2)
        
        # Segmentation & Morphology
        seg_frame = ttk.LabelFrame(left_panel, text="Morphological Operations", padding="10")
        seg_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(seg_frame, text="Global/Otsu Threshold", 
                  command=self.apply_otsu).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Erode", 
                  command=lambda: self.apply_morphology('erode')).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Dilate", 
                  command=lambda: self.apply_morphology('dilate')).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Open", 
                  command=lambda: self.apply_morphology('open')).pack(fill=tk.X, pady=2)
        ttk.Button(seg_frame, text="Close", 
                  command=lambda: self.apply_morphology('close')).pack(fill=tk.X, pady=2)
        
        # Transformations
        trans_frame = ttk.LabelFrame(left_panel, text="Affine Transforms", padding="10")
        trans_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(trans_frame, text="Rotate (Angle Input)", 
                  command=self.apply_rotate_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Rotate 90°", 
                  command=lambda: self.apply_transform('rotate', 90)).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Rotate -90°", 
                  command=lambda: self.apply_transform('rotate', -90)).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Scale (Uniform/X/Y)", 
                  command=self.apply_scale_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Translate (dx, dy)", 
                  command=self.apply_translate_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Shear (X/Y/Both)", 
                  command=self.apply_shear_dialog).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Flip Horizontal", 
                  command=lambda: self.apply_transform('flip', 'horizontal')).pack(fill=tk.X, pady=2)
        ttk.Button(trans_frame, text="Flip Vertical", 
                  command=lambda: self.apply_transform('flip', 'vertical')).pack(fill=tk.X, pady=2)
        
        # Reset and Pipeline
        action_frame = ttk.LabelFrame(left_panel, text="İşlemler", padding="10")
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Orijinali Yükle", 
                  command=self.reset_image).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Pipeline Çalıştır", 
                  command=self.run_pipeline).pack(fill=tk.X, pady=2)
        
        # Right panel - Image display (Two panels: Original vs Result)
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(1, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Original image panel
        original_frame = ttk.LabelFrame(right_panel, text="Original (Orijinal)", padding="10")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        
        self.canvas_original = tk.Canvas(original_frame, bg="gray", width=400, height=600)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Result image panel
        result_frame = ttk.LabelFrame(right_panel, text="Result (Sonuç)", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.canvas_result = tk.Canvas(result_frame, bg="gray", width=400, height=600)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Hazır - Görüntü yükleyin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.image_path = Path(file_path)
                self.original_image = self.io.load_image(self.image_path)
                self.current_image = self.original_image.copy()
                self.update_display()
                self.status_var.set(f"Görüntü yüklendi: {self.image_path.name}")
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü yüklenemedi: {str(e)}")
    
    def save_image(self):
        """Save current image"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek görüntü yok!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Görüntüyü Kaydet",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.io.save_image(self.current_image, Path(file_path))
                self.status_var.set(f"Görüntü kaydedildi: {Path(file_path).name}")
                messagebox.showinfo("Başarılı", "Görüntü başarıyla kaydedildi!")
            except Exception as e:
                messagebox.showerror("Hata", f"Görüntü kaydedilemedi: {str(e)}")
    
    def update_display(self):
        """Update image display on both canvases (Original and Result)"""
        # Update original panel
        if self.original_image is not None:
            self._update_canvas(self.canvas_original, self.original_image)
        
        # Update result panel
        if self.current_image is not None:
            self._update_canvas(self.canvas_result, self.current_image)
    
    def _update_canvas(self, canvas, image):
        """Helper method to update a single canvas"""
        try:
            # Convert to RGB if needed
            if image.ndim == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize to fit canvas
            canvas.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = display_image.shape[:2]
                scale = min(canvas_width / w, canvas_height / h, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                if new_w > 0 and new_h > 0:
                    display_image = cv2.resize(display_image, (new_w, new_h))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(display_image)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update canvas
            canvas.delete("all")
            canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                image=photo, anchor=tk.CENTER
            )
            # Keep reference to prevent garbage collection
            canvas.image = photo
        except Exception as e:
            print(f"Display error: {e}")
    
    def apply_filter(self, filter_type):
        """Apply filter to image"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            # Convert to grayscale if needed
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            if filter_type == 'gaussian':
                self.current_image = self.filters.gaussian_blur(gray, (15, 15))
            elif filter_type == 'median':
                self.current_image = self.filters.median_blur(gray, 5)
            elif filter_type == 'bilateral':
                self.current_image = self.filters.bilateral_filter(gray, 9, 75, 75)
            elif filter_type == 'box':
                self.current_image = self.filters.box_filter(gray, (5, 5), normalize=True)
            
            self.update_display()
            self.status_var.set(f"{filter_type.capitalize()} filtresi uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Filtre uygulanamadı: {str(e)}")
    
    def apply_clahe(self):
        """Apply CLAHE enhancement"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.apply_clahe(gray, 2.0, (8, 8))
            self.update_display()
            self.status_var.set("CLAHE iyileştirmesi uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"CLAHE uygulanamadı: {str(e)}")
    
    def apply_histogram_eq(self):
        """Apply histogram equalization"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.histogram_equalization(gray)
            self.update_display()
            self.status_var.set("Histogram eşitleme uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Histogram eşitleme uygulanamadı: {str(e)}")
    
    def apply_gamma(self):
        """Apply gamma correction"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.gamma_correction(gray, 1.5)
            self.update_display()
            self.status_var.set("Gamma düzeltmesi uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Gamma düzeltme uygulanamadı: {str(e)}")
    
    def apply_sharpen(self):
        """Apply sharpening filter"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.sharpen(gray, 1.0)
            self.update_display()
            self.status_var.set("Keskinleştirme filtresi uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Keskinleştirme uygulanamadı: {str(e)}")
    
    def apply_log_transform(self):
        """Apply logarithmic transformation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.log_transform(gray)
            self.update_display()
            self.status_var.set("Logaritmik dönüşüm uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Log dönüşümü uygulanamadı: {str(e)}")
    
    def apply_contrast_stretch(self):
        """Apply contrast stretching"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            self.current_image = self.enhancement.contrast_stretch(gray)
            self.update_display()
            self.status_var.set("Kontrast germe uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Kontrast germe uygulanamadı: {str(e)}")
    
    def apply_otsu(self):
        """Apply Otsu thresholding"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            binary, thresh_val = self.segmentation.threshold(gray, 'otsu')
            self.current_image = binary
            self.update_display()
            self.status_var.set(f"Otsu threshold uygulandı (değer: {thresh_val:.1f})")
        except Exception as e:
            messagebox.showerror("Hata", f"Otsu threshold uygulanamadı: {str(e)}")
    
    def apply_adaptive_thresh(self):
        """Apply adaptive thresholding"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            binary, _ = self.segmentation.threshold(gray, 'adaptive_mean')
            self.current_image = binary
            self.update_display()
            self.status_var.set("Adaptive threshold uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Adaptive threshold uygulanamadı: {str(e)}")
    
    def find_contours(self):
        """Find and draw contours"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            binary, _ = self.segmentation.threshold(gray, 'otsu')
            contours = self.segmentation.find_contours(binary)
            
            # Draw contours on original
            if self.original_image is not None:
                if self.original_image.ndim == 3:
                    display_img = self.original_image.copy()
                else:
                    display_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            else:
                display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            self.current_image = self.visualizer.draw_contours(display_img, contours, (0, 255, 0), 2)
            self.update_display()
            self.status_var.set(f"{len(contours)} kontur bulundu")
        except Exception as e:
            messagebox.showerror("Hata", f"Kontur bulunamadı: {str(e)}")
    
    def apply_transform(self, transform_type, param=None):
        """Apply transformation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if transform_type == 'rotate':
                self.current_image = self.transformations.rotate(self.current_image, param)
            elif transform_type == 'flip':
                self.current_image = self.transformations.flip(self.current_image, param)
            
            self.update_display()
            self.status_var.set(f"{transform_type.capitalize()} dönüşümü uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Dönüşüm uygulanamadı: {str(e)}")
    
    def apply_resize(self):
        """Apply resize with dialog"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Resize")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="Yeni Genişlik:").pack(pady=5)
        width_var = tk.StringVar(value="800")
        ttk.Entry(dialog, textvariable=width_var, width=20).pack(pady=5)
        
        ttk.Label(dialog, text="Yeni Yükseklik:").pack(pady=5)
        height_var = tk.StringVar(value="600")
        ttk.Entry(dialog, textvariable=height_var, width=20).pack(pady=5)
        
        def do_resize():
            try:
                w = int(width_var.get())
                h = int(height_var.get())
                self.current_image = self.transformations.resize(self.current_image, (w, h))
                self.update_display()
                self.status_var.set(f"Görüntü yeniden boyutlandırıldı: {w}x{h}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_resize).pack(pady=10)
    
    def reset_image(self):
        """Reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_display()
            self.status_var.set("Orijinal görüntüye dönüldü")
        else:
            messagebox.showwarning("Uyarı", "Yüklenmiş görüntü yok!")
    
    def run_pipeline(self):
        """Run example pipeline"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        def pipeline_thread():
            try:
                self.status_var.set("Pipeline çalışıyor...")
                output_dir = Path('outputs/gui_pipeline')
                pipeline = ImageProcessingPipeline(output_dir=output_dir)
                
                # Save current image temporarily
                temp_path = Path('outputs/temp_input.png')
                self.io.save_image(self.current_image, temp_path)
                
                pipeline.load_image(temp_path, grayscale=True)
                
                enhancement = EnhancementProcessor()
                filters = FilterProcessor()
                
                pipeline.add_step('Original', lambda img: img, {}) \
                        .add_step('CLAHE', enhancement.apply_clahe, 
                                 {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}) \
                        .add_step('Gaussian Blur', filters.gaussian_blur, 
                                 {'kernel_size': (5, 5)})
                
                result = pipeline.execute()
                self.current_image = result
                self.update_display()
                self.status_var.set("Pipeline tamamlandı!")
                messagebox.showinfo("Başarılı", "Pipeline başarıyla çalıştırıldı!")
            except Exception as e:
                messagebox.showerror("Hata", f"Pipeline hatası: {str(e)}")
        
        threading.Thread(target=pipeline_thread, daemon=True).start()
    
    def find_connected_components(self):
        """Find and visualize connected components"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            binary, _ = self.segmentation.threshold(gray, 'otsu')
            num_labels, labels, stats, centroids = self.segmentation.connected_components(binary)
            
            # Draw bounding boxes
            if self.original_image is not None:
                if self.original_image.ndim == 3:
                    display_img = self.original_image.copy()
                else:
                    display_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
            else:
                display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            self.current_image = self.visualizer.draw_bounding_boxes(display_img, stats, (0, 0, 255), 2)
            self.update_display()
            self.status_var.set(f"{num_labels - 1} bağlı bileşen bulundu")
        except Exception as e:
            messagebox.showerror("Hata", f"Bağlı bileşenler bulunamadı: {str(e)}")
    
    def apply_morphology(self, operation):
        """Apply morphological operation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            # First threshold to get binary
            binary, _ = self.segmentation.threshold(gray, 'otsu')
            
            # Apply morphological operation
            self.current_image = self.filters.morphological_operations(
                binary, operation, (3, 3), 'ellipse', 1
            )
            self.update_display()
            self.status_var.set(f"Morphological {operation} uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Morphological işlem uygulanamadı: {str(e)}")
    
    def apply_crop(self):
        """Apply crop (ROI selection)"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Crop (ROI)")
        dialog.geometry("350x200")
        
        h, w = self.current_image.shape[:2]
        
        ttk.Label(dialog, text=f"Mevcut boyut: {w}x{h}").pack(pady=5)
        
        ttk.Label(dialog, text="X koordinatı:").pack(pady=2)
        x_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=x_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Y koordinatı:").pack(pady=2)
        y_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=y_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Genişlik:").pack(pady=2)
        width_var = tk.StringVar(value=str(w))
        ttk.Entry(dialog, textvariable=width_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Yükseklik:").pack(pady=2)
        height_var = tk.StringVar(value=str(h))
        ttk.Entry(dialog, textvariable=height_var, width=20).pack(pady=2)
        
        def do_crop():
            try:
                x = int(x_var.get())
                y = int(y_var.get())
                width = int(width_var.get())
                height = int(height_var.get())
                
                self.current_image = self.transformations.crop(
                    self.current_image, x, y, width, height
                )
                self.update_display()
                self.status_var.set(f"Görüntü kırpıldı: {width}x{height}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_crop).pack(pady=10)
    
    def convert_to_grayscale(self):
        """Convert image to grayscale"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            self.update_display()
            self.status_var.set("Görüntü gri tonlamaya dönüştürüldü")
        except Exception as e:
            messagebox.showerror("Hata", f"Dönüşüm uygulanamadı: {str(e)}")
    
    def apply_negative(self):
        """Apply negative (invert) transformation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            self.current_image = cv2.bitwise_not(self.current_image)
            self.update_display()
            self.status_var.set("Negative (invert) uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Negative uygulanamadı: {str(e)}")
    
    def show_histogram(self):
        """Show histogram of current image"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Create histogram visualization
            hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
            hist_normalized = hist.astype(np.float32)
            hist_normalized = hist_normalized / np.max(hist_normalized) * 300
            
            for i in range(256):
                cv2.line(hist_img, (i*2, 300), (i*2, 300-int(hist_normalized[i])), (255, 255, 255), 2)
            
            # Show in new window
            hist_window = tk.Toplevel(self.root)
            hist_window.title("Histogram")
            hist_canvas = tk.Canvas(hist_window, width=512, height=300)
            hist_canvas.pack()
            
            pil_hist = Image.fromarray(hist_img)
            photo_hist = ImageTk.PhotoImage(image=pil_hist)
            hist_canvas.create_image(256, 150, image=photo_hist, anchor=tk.CENTER)
            hist_canvas.image = photo_hist
            
            self.status_var.set("Histogram gösterildi")
        except Exception as e:
            messagebox.showerror("Hata", f"Histogram oluşturulamadı: {str(e)}")
    
    def apply_laplacian(self):
        """Apply Laplacian edge detection"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            laplacian = self.edges.laplacian(gray, 3)
            laplacian = np.abs(laplacian)
            self.current_image = np.uint8(np.clip(laplacian, 0, 255))
            self.update_display()
            self.status_var.set("Laplacian edge detection uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Laplacian uygulanamadı: {str(e)}")
    
    def apply_sobel(self, direction):
        """Apply Sobel edge detection"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        try:
            if self.current_image.ndim == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image.copy()
            
            sobel = self.edges.sobel(gray, direction, 3)
            sobel = np.abs(sobel)
            self.current_image = np.uint8(np.clip(sobel, 0, 255))
            self.update_display()
            self.status_var.set(f"Sobel {direction.upper()} uygulandı")
        except Exception as e:
            messagebox.showerror("Hata", f"Sobel uygulanamadı: {str(e)}")
    
    def apply_canny(self):
        """Apply Canny edge detection with dialog"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Canny Edge Detection")
        dialog.geometry("300x120")
        
        ttk.Label(dialog, text="Threshold 1:").pack(pady=2)
        thresh1_var = tk.StringVar(value="100")
        ttk.Entry(dialog, textvariable=thresh1_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Threshold 2:").pack(pady=2)
        thresh2_var = tk.StringVar(value="200")
        ttk.Entry(dialog, textvariable=thresh2_var, width=20).pack(pady=2)
        
        def do_canny():
            try:
                thresh1 = float(thresh1_var.get())
                thresh2 = float(thresh2_var.get())
                
                if self.current_image.ndim == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image.copy()
                
                self.current_image = self.edges.canny(gray, thresh1, thresh2)
                self.update_display()
                self.status_var.set(f"Canny edge detection uygulandı ({thresh1}, {thresh2})")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_canny).pack(pady=10)
    
    def apply_gamma_dialog(self):
        """Apply gamma correction with slider dialog"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Gamma Correction")
        dialog.geometry("400x150")
        
        gamma_var = tk.DoubleVar(value=1.0)
        gamma_label = ttk.Label(dialog, text=f"Gamma: {gamma_var.get():.2f}")
        gamma_label.pack(pady=5)
        
        def update_gamma(val):
            gamma_label.config(text=f"Gamma: {float(val):.2f}")
            try:
                if self.current_image.ndim == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image.copy()
                
                temp_result = self.enhancement.gamma_correction(gray, float(val))
                # Show preview (you can enhance this)
                self.current_image = temp_result
                self.update_display()
            except:
                pass
        
        gamma_scale = ttk.Scale(dialog, from_=0.1, to=3.0, variable=gamma_var, 
                                orient=tk.HORIZONTAL, command=update_gamma)
        gamma_scale.pack(fill=tk.X, padx=20, pady=10)
        
        def apply_gamma():
            gamma_val = gamma_var.get()
            if self.original_image is not None:
                if self.original_image.ndim == 3:
                    gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.original_image.copy()
                self.current_image = self.enhancement.gamma_correction(gray, gamma_val)
            else:
                if self.current_image.ndim == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image.copy()
                self.current_image = self.enhancement.gamma_correction(gray, gamma_val)
            
            self.update_display()
            self.status_var.set(f"Gamma correction uygulandı (γ={gamma_val:.2f})")
            dialog.destroy()
        
        ttk.Button(dialog, text="Uygula", command=apply_gamma).pack(pady=5)
    
    def apply_rotate_dialog(self):
        """Apply rotation with angle input"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Rotate")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text="Açı (derece):").pack(pady=5)
        angle_var = tk.StringVar(value="45")
        ttk.Entry(dialog, textvariable=angle_var, width=20).pack(pady=5)
        
        def do_rotate():
            try:
                angle = float(angle_var.get())
                self.current_image = self.transformations.rotate(self.current_image, angle)
                self.update_display()
                self.status_var.set(f"Görüntü {angle}° döndürüldü")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli bir açı girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_rotate).pack(pady=10)
    
    def apply_scale_dialog(self):
        """Apply scale transformation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Scale")
        dialog.geometry("350x180")
        
        ttk.Label(dialog, text="Scale X:").pack(pady=2)
        scale_x_var = tk.StringVar(value="1.0")
        ttk.Entry(dialog, textvariable=scale_x_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Scale Y:").pack(pady=2)
        scale_y_var = tk.StringVar(value="1.0")
        ttk.Entry(dialog, textvariable=scale_y_var, width=20).pack(pady=2)
        
        uniform_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Uniform (X=Y)", variable=uniform_var).pack(pady=5)
        
        def do_scale():
            try:
                sx = float(scale_x_var.get())
                if uniform_var.get():
                    sy = sx
                else:
                    sy = float(scale_y_var.get())
                
                h, w = self.current_image.shape[:2]
                new_w = int(w * sx)
                new_h = int(h * sy)
                self.current_image = self.transformations.resize(self.current_image, (new_w, new_h))
                self.update_display()
                self.status_var.set(f"Görüntü ölçeklendi: {sx}x{sy}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_scale).pack(pady=10)
    
    def apply_translate_dialog(self):
        """Apply translation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Translate")
        dialog.geometry("300x120")
        
        ttk.Label(dialog, text="dx (X translation):").pack(pady=2)
        dx_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=dx_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="dy (Y translation):").pack(pady=2)
        dy_var = tk.StringVar(value="0")
        ttk.Entry(dialog, textvariable=dy_var, width=20).pack(pady=2)
        
        def do_translate():
            try:
                dx = int(dx_var.get())
                dy = int(dy_var.get())
                self.current_image = self.transformations.translate(self.current_image, dx, dy)
                self.update_display()
                self.status_var.set(f"Görüntü ötelenmiş: dx={dx}, dy={dy}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_translate).pack(pady=10)
    
    def apply_shear_dialog(self):
        """Apply shear transformation"""
        if self.current_image is None:
            messagebox.showwarning("Uyarı", "Önce bir görüntü yükleyin!")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Shear")
        dialog.geometry("350x180")
        
        ttk.Label(dialog, text="Shear X:").pack(pady=2)
        shear_x_var = tk.StringVar(value="0.0")
        ttk.Entry(dialog, textvariable=shear_x_var, width=20).pack(pady=2)
        
        ttk.Label(dialog, text="Shear Y:").pack(pady=2)
        shear_y_var = tk.StringVar(value="0.0")
        ttk.Entry(dialog, textvariable=shear_y_var, width=20).pack(pady=2)
        
        def do_shear():
            try:
                shx = float(shear_x_var.get())
                shy = float(shear_y_var.get())
                
                h, w = self.current_image.shape[:2]
                M = np.float32([[1, shx, 0], [shy, 1, 0]])
                new_w = int(w + shx * h)
                new_h = int(h + shy * w)
                self.current_image = cv2.warpAffine(self.current_image, M, (new_w, new_h))
                self.update_display()
                self.status_var.set(f"Shear uygulandı: X={shx}, Y={shy}")
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Hata", "Geçerli sayılar girin!")
        
        ttk.Button(dialog, text="Uygula", command=do_shear).pack(pady=10)

