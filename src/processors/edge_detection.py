import cv2
import numpy as np
from typing import Tuple

class EdgeDetectionProcessor:
    @staticmethod
    def canny(image: np.ndarray, threshold1: float = 100,
              threshold2: float = 200, aperture_size: int = 3) -> np.ndarray:
        return cv2.Canny(image, threshold1, threshold2, apertureSize=aperture_size)
    
    @staticmethod
    def sobel(image: np.ndarray, direction: str = 'x',
              ksize: int = 3) -> np.ndarray:
        if direction == 'x':
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        elif direction == 'y':
            return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            return np.sqrt(sobelx**2 + sobely**2)
    
    @staticmethod
    def laplacian(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        return cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    
    @staticmethod
    def scharr(image: np.ndarray, direction: str = 'x') -> np.ndarray:
        if direction == 'x':
            return cv2.Scharr(image, cv2.CV_64F, 1, 0)
        else:
            return cv2.Scharr(image, cv2.CV_64F, 0, 1)
