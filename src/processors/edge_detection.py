"""
Edge Detection Module
Kenar Tespit Modülü
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class EdgeDetectionProcessor:
    """Edge detection operations"""
    
    @staticmethod
    def canny(image: np.ndarray, threshold1: float = 100,
              threshold2: float = 200, aperture_size: int = 3) -> np.ndarray:
        """
        Apply Canny edge detection
        
        Args:
            image: Input grayscale image
            threshold1: First threshold for hysteresis
            threshold2: Second threshold for hysteresis
            aperture_size: Aperture size for Sobel operator
        
        Returns:
            Edge detected image
        """
        return cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)
    
    @staticmethod
    def sobel(image: np.ndarray, direction: str = 'x',
              ksize: int = 3) -> np.ndarray:
        """
        Apply Sobel edge detection
        
        Args:
            image: Input grayscale image
            direction: 'x', 'y', or 'both'
            ksize: Kernel size (1, 3, 5, or 7)
        
        Returns:
            Edge detected image
        """
        if direction == 'x':
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        elif direction == 'y':
            return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        else:  # both
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            return np.sqrt(sobelx**2 + sobely**2)
    
    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Apply Laplacian edge detection
        
        Args:
            image: Input grayscale image
            ksize: Aperture size (1, 3, 5, or 7)
        
        Returns:
            Edge detected image
        """
        return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    
    @staticmethod
    def scharr(image: np.ndarray, direction: str = 'x') -> np.ndarray:
        """
        Apply Scharr edge detection
        
        Args:
            image: Input grayscale image
            direction: 'x' or 'y'
        
        Returns:
            Edge detected image
        """
        if direction == 'x':
            return cv2.Scharr(image, cv2.CV_64F, 1, 0)
        else:
            return cv2.Scharr(image, cv2.CV_64F, 0, 1)

