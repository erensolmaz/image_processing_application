"""
Image Visualization Module
Görüntü Görselleştirme Modülü
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class ImageVisualizer:
    """Image visualization operations"""
    
    @staticmethod
    def draw_contours(image: np.ndarray, contours: List[np.ndarray],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2) -> np.ndarray:
        """
        Draw contours on image
        
        Args:
            image: Input image
            contours: List of contours
            color: Contour color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with contours
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return cv2.drawContours(image.copy(), contours, -1, color, thickness)
    
    @staticmethod
    def draw_bounding_boxes(image: np.ndarray, stats: np.ndarray,
                           color: Tuple[int, int, int] = (0, 0, 255),
                           thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            stats: Component statistics from connectedComponentsWithStats
            color: Box color (BGR)
            thickness: Line thickness
        
        Returns:
            Image with bounding boxes
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        result = image.copy()
        for stat in stats:
            x, y, w, h, _ = stat
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        return result
    
    @staticmethod
    def create_comparison_grid(images: List[np.ndarray],
                              titles: Optional[List[str]] = None,
                              grid_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create comparison grid from multiple images
        
        Args:
            images: List of images
            titles: Optional list of titles
            grid_size: Grid size (rows, cols) or None for auto
        
        Returns:
            Combined grid image
        """
        if not images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Determine grid size
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(len(images))))
            rows = int(np.ceil(len(images) / cols))
        else:
            rows, cols = grid_size
        
        # Resize images to same size
        h, w = images[0].shape[:2]
        small_h, small_w = h // 3, w // 3
        
        resized_images = []
        for img in images:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(img, (small_w, small_h))
            resized_images.append(resized)
        
        # Create grid
        rows_list = []
        for i in range(rows):
            row_images = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(resized_images):
                    img = resized_images[idx].copy()
                    # Add title if provided
                    if titles and idx < len(titles):
                        cv2.putText(img, titles[idx][:20], (5, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    row_images.append(img)
                else:
                    # Empty space
                    row_images.append(np.zeros((small_h, small_w, 3), dtype=np.uint8))
            rows_list.append(np.hstack(row_images))
        
        return np.vstack(rows_list)
    
    @staticmethod
    def overlay_mask(image: np.ndarray, mask: np.ndarray,
                    color: Tuple[int, int, int] = (0, 255, 0),
                    alpha: float = 0.5) -> np.ndarray:
        """
        Overlay mask on image
        
        Args:
            image: Input image
            mask: Binary mask
            color: Overlay color (BGR)
            alpha: Transparency (0-1)
        
        Returns:
            Image with overlaid mask
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        overlay = image.copy()
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        return cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)

