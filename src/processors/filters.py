"""
Image Filtering Module
Görüntü Filtreleme Modülü
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FilterProcessor:
    """Image filtering operations"""
    
    @staticmethod
    def gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                     sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur
        
        Args:
            image: Input image
            kernel_size: Kernel size (width, height) - must be odd numbers
            sigma: Standard deviation (0 = auto-calculate)
        
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    @staticmethod
    def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter
        
        Args:
            image: Input image
            kernel_size: Kernel size (must be odd, e.g., 3, 5, 7)
        
        Returns:
            Filtered image
        """
        return cv2.medianBlur(image, kernel_size)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9, 
                        sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter (edge-preserving)
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
        
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def box_filter(image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                  normalize: bool = True) -> np.ndarray:
        """
        Apply box filter
        
        Args:
            image: Input image
            kernel_size: Kernel size (width, height)
            normalize: Whether to normalize the kernel
        
        Returns:
            Filtered image
        """
        ddepth = -1 if normalize else cv2.CV_8U
        return cv2.boxFilter(image, ddepth, kernel_size)
    
    @staticmethod
    def morphological_operations(image: np.ndarray, operation: str = 'open',
                                kernel_size: Tuple[int, int] = (3, 3),
                                kernel_shape: str = 'ellipse',
                                iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations
        
        Args:
            image: Input binary image
            operation: 'open', 'close', 'erode', 'dilate', 'gradient', 'tophat', 'blackhat'
            kernel_size: Kernel size (width, height)
            kernel_shape: 'rect', 'ellipse', 'cross'
            iterations: Number of iterations
        
        Returns:
            Processed image
        """
        # Create kernel
        shape_map = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        kernel = cv2.getStructuringElement(
            shape_map.get(kernel_shape, cv2.MORPH_ELLIPSE),
            kernel_size
        )
        
        # Apply operation
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

