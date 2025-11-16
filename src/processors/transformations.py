"""
Image Transformation Module
Görüntü Dönüşüm Modülü
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class TransformationProcessor:
    """Image transformation operations"""
    
    @staticmethod
    def resize(image: np.ndarray, size: Optional[Tuple[int, int]] = None,
              scale: Optional[float] = None,
              interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image
        
        Args:
            image: Input image
            size: Target size (width, height) or None
            scale: Scale factor (if size is None)
            interpolation: Interpolation method
        
        Returns:
            Resized image
        """
        if size:
            return cv2.resize(image, size, interpolation=interpolation)
        elif scale:
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
        return image
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None,
              scale: float = 1.0) -> np.ndarray:
        """
        Rotate image
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            center: Rotation center (None = image center)
            scale: Scale factor
        
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Translate (shift) image
        
        Args:
            image: Input image
            tx: Translation in x direction
            ty: Translation in y direction
        
        Returns:
            Translated image
        """
        h, w = image.shape[:2]
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (w, h))
    
    @staticmethod
    def flip(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """
        Flip image
        
        Args:
            image: Input image
            direction: 'horizontal', 'vertical', or 'both'
        
        Returns:
            Flipped image
        """
        flip_map = {
            'horizontal': 1,
            'vertical': 0,
            'both': -1
        }
        return cv2.flip(image, flip_map.get(direction, 1))
    
    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Crop image
        
        Args:
            image: Input image
            x: Starting x coordinate
            y: Starting y coordinate
            width: Crop width
            height: Crop height
        
        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        width = min(width, w - x)
        height = min(height, h - y)
        return image[y:y+height, x:x+width]
    
    @staticmethod
    def perspective_transform(image: np.ndarray, src_points: np.ndarray,
                             dst_points: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation
        
        Args:
            image: Input image
            src_points: Source points (4 points)
            dst_points: Destination points (4 points)
        
        Returns:
            Transformed image
        """
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        h, w = image.shape[:2]
        return cv2.warpPerspective(image, M, (w, h))

