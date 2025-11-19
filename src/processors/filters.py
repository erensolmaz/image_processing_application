import cv2
import numpy as np
from typing import Tuple

class FilterProcessor:
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                     sigma: float = 0) -> np.ndarray:
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def box_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                  normalize: bool = True) -> np.ndarray:
        ddepth = -1 if normalize else cv2.CV_8U
        return cv2.boxFilter(image, ddepth, kernel_size)
    
    @staticmethod
    def morphological_operations(image: np.ndarray, operation: str = 'open',
                                kernel_size: Tuple[int, int] = (3, 3),
                                kernel_shape: str = 'ellipse',
                                iterations: int = 1) -> np.ndarray:
        shape_map = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        kernel = cv2.getStructuringElement(
            shape_map.get(kernel_shape, cv2.MORPH_ELLIPSE),
            kernel_size
        )
        
        op_map = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'erode': cv2.MORPH_ERODE,
            'dilate': cv2.MORPH_DILATE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT
        }
        
        return cv2.morphologyEx(image, op_map.get(operation, cv2.MORPH_OPEN), 
                               kernel, iterations=iterations)
