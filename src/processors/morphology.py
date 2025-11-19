import cv2
import numpy as np
from typing import Tuple, Optional, Union


class MorphologyProcessor:
    
    @staticmethod
    def threshold(image: np.ndarray, 
                  method: str = 'binary',
                  threshold_value: int = 127,
                  max_value: int = 255,
                  block_size: int = 11,
                  c_value: float = 2.0) -> Tuple[np.ndarray, float]:
        if image.ndim != 2:
            raise ValueError("Image must be grayscale for threshold operation.")
        
        if method == 'otsu':
            threshold_value, binary = cv2.threshold(
                image, 0, max_value, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary, float(threshold_value)
        
        elif method == 'adaptive_mean':
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                image, max_value, 
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 
                block_size, c_value
            )
            return binary, threshold_value
        
        elif method == 'adaptive_gaussian':
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                image, max_value,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size, c_value
            )
            return binary, threshold_value
        
        else:
            thresh_map = {
                'binary': cv2.THRESH_BINARY,
                'binary_inv': cv2.THRESH_BINARY_INV,
                'trunc': cv2.THRESH_TRUNC,
                'tozero': cv2.THRESH_TOZERO,
                'tozero_inv': cv2.THRESH_TOZERO_INV
            }
            
            thresh_type = thresh_map.get(method, cv2.THRESH_BINARY)
            threshold_value, binary = cv2.threshold(
                image, threshold_value, max_value, thresh_type
            )
            return binary, float(threshold_value)
    
    @staticmethod
    def multi_threshold(image: np.ndarray, 
                       thresholds: list,
                       max_value: int = 255) -> np.ndarray:
        if image.ndim != 2:
            raise ValueError("Image must be grayscale for multi-threshold operation.")
        
        result = np.zeros_like(image)
        sorted_thresh = sorted(thresholds)
        
        mask = image < sorted_thresh[0]
        result[mask] = 0
        
        for i in range(len(sorted_thresh) - 1):
            mask = (image >= sorted_thresh[i]) & (image < sorted_thresh[i + 1])
            result[mask] = int(max_value * (i + 1) / (len(sorted_thresh) + 1))
        
        mask = image >= sorted_thresh[-1]
        result[mask] = max_value
        
        return result
    
    @staticmethod
    def get_kernel(shape: str = 'rect', 
                  size: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
        shape_map = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        
        if isinstance(size, int):
            ksize = (size, size)
        else:
            ksize = size
        
        if ksize[0] % 2 == 0:
            ksize = (ksize[0] + 1, ksize[1])
        if ksize[1] % 2 == 0:
            ksize = (ksize[0], ksize[1] + 1)
        
        morph_shape = shape_map.get(shape.lower(), cv2.MORPH_ELLIPSE)
        return cv2.getStructuringElement(morph_shape, ksize)
    
    @staticmethod
    def erode(image: np.ndarray,
              kernel: Optional[np.ndarray] = None,
              kernel_shape: str = 'rect',
              kernel_size: Union[int, Tuple[int, int]] = 3,
              iterations: int = 1,
              border_type: int = cv2.BORDER_CONSTANT) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.erode(image, kernel, iterations=iterations, 
                        borderType=border_type)
    
    @staticmethod
    def dilate(image: np.ndarray,
               kernel: Optional[np.ndarray] = None,
               kernel_shape: str = 'rect',
               kernel_size: Union[int, Tuple[int, int]] = 3,
               iterations: int = 1,
               border_type: int = cv2.BORDER_CONSTANT) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.dilate(image, kernel, iterations=iterations,
                         borderType=border_type)
    
    @staticmethod
    def opening(image: np.ndarray,
               kernel: Optional[np.ndarray] = None,
               kernel_shape: str = 'rect',
               kernel_size: Union[int, Tuple[int, int]] = 3,
               iterations: int = 1) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, 
                               iterations=iterations)
    
    @staticmethod
    def closing(image: np.ndarray,
               kernel: Optional[np.ndarray] = None,
               kernel_shape: str = 'rect',
               kernel_size: Union[int, Tuple[int, int]] = 3,
               iterations: int = 1) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel,
                               iterations=iterations)
    
    @staticmethod
    def gradient(image: np.ndarray,
                kernel: Optional[np.ndarray] = None,
                kernel_shape: str = 'rect',
                kernel_size: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    @staticmethod
    def tophat(image: np.ndarray,
              kernel: Optional[np.ndarray] = None,
              kernel_shape: str = 'rect',
              kernel_size: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    @staticmethod
    def blackhat(image: np.ndarray,
                kernel: Optional[np.ndarray] = None,
                kernel_shape: str = 'rect',
                kernel_size: Union[int, Tuple[int, int]] = 3) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    @staticmethod
    def hit_or_miss(image: np.ndarray,
                   kernel: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    
    @staticmethod
    def morphology_ex(image: np.ndarray,
                     operation: str,
                     kernel: Optional[np.ndarray] = None,
                     kernel_shape: str = 'rect',
                     kernel_size: Union[int, Tuple[int, int]] = 3,
                     iterations: int = 1) -> np.ndarray:
        if kernel is None:
            kernel = MorphologyProcessor.get_kernel(kernel_shape, kernel_size)
        
        op_map = {
            'erode': cv2.MORPH_ERODE,
            'dilate': cv2.MORPH_DILATE,
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT
        }
        
        op_type = op_map.get(operation.lower(), cv2.MORPH_OPEN)
        return cv2.morphologyEx(image, op_type, kernel, iterations=iterations)
