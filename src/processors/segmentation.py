"""
Image Segmentation Module
Görüntü Segmentasyon Modülü
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


class SegmentationProcessor:
    """Image segmentation operations"""
    
    @staticmethod
    def threshold(image: np.ndarray, method: str = 'binary',
                 threshold_value: int = 127,
                 max_value: int = 255) -> Tuple[np.ndarray, float]:
        """
        Apply thresholding
        
        Args:
            image: Input grayscale image
            method: 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv', 'otsu', 'adaptive_mean', 'adaptive_gaussian'
            threshold_value: Threshold value (for non-adaptive methods)
            max_value: Maximum value
        
        Returns:
            Binary image and threshold value
        """
        if method == 'otsu':
            threshold_value, binary = cv2.threshold(
                image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary, threshold_value
        elif method == 'adaptive_mean':
            binary = cv2.adaptiveThreshold(
                image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary, threshold_value
        elif method == 'adaptive_gaussian':
            binary = cv2.adaptiveThreshold(
                image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
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
            threshold_value, binary = cv2.threshold(
                image, threshold_value, max_value, thresh_map.get(method, cv2.THRESH_BINARY)
            )
            return binary, threshold_value
    
    @staticmethod
    def find_contours(binary_image: np.ndarray, mode: str = 'external',
                     method: str = 'simple') -> List[np.ndarray]:
        """
        Find contours in binary image
        
        Args:
            binary_image: Binary input image
            mode: 'external', 'list', 'tree', 'ccomp'
            method: 'none', 'simple', 'tc89_l1', 'tc89_kcos'
        
        Returns:
            List of contours
        """
        mode_map = {
            'external': cv2.RETR_EXTERNAL,
            'list': cv2.RETR_LIST,
            'tree': cv2.RETR_TREE,
            'ccomp': cv2.RETR_CCOMP
        }
        
        method_map = {
            'none': cv2.CHAIN_APPROX_NONE,
            'simple': cv2.CHAIN_APPROX_SIMPLE,
            'tc89_l1': cv2.CHAIN_APPROX_TC89_L1,
            'tc89_kcos': cv2.CHAIN_APPROX_TC89_KCOS
        }
        
        contours, _ = cv2.findContours(
            binary_image, mode_map.get(mode, cv2.RETR_EXTERNAL),
            method_map.get(method, cv2.CHAIN_APPROX_SIMPLE)
        )
        return contours
    
    @staticmethod
    def connected_components(binary_image: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find connected components
        
        Args:
            binary_image: Binary input image
            connectivity: 4 or 8 connectivity
        
        Returns:
            (num_labels, labels, stats, centroids)
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=connectivity
        )
        return num_labels, labels, stats, centroids
    
    @staticmethod
    def filter_components(binary_image: np.ndarray, min_area: int = 100,
                         max_area: Optional[int] = None,
                         connectivity: int = 8) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Filter connected components by area
        
        Args:
            binary_image: Binary input image
            min_area: Minimum component area
            max_area: Maximum component area (None = no limit)
            connectivity: 4 or 8 connectivity
        
        Returns:
            (filtered_mask, count, stats)
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_image, connectivity=connectivity
        )
        
        mask = np.zeros_like(binary_image)
        keep_stats = []
        count = 0
        
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            if area >= min_area and (max_area is None or area <= max_area):
                mask[labels == idx] = 255
                keep_stats.append(stats[idx])
                count += 1
        
        return mask, count, np.array(keep_stats) if keep_stats else np.array([])
    
    @staticmethod
    def watershed(image: np.ndarray, markers: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply watershed segmentation
        
        Args:
            image: Input image
            markers: Marker image (None = auto-generate)
        
        Returns:
            Segmented image
        """
        if markers is None:
            # Simple marker generation (can be improved)
            _, markers = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            markers = cv2.connectedComponents(markers)[1]
        
        markers = cv2.watershed(image, markers)
        return markers

