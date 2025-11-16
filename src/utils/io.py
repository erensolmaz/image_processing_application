"""
Image I/O Module
Görüntü Giriş/Çıkış Modülü
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ImageIO:
    """Image input/output operations"""
    
    @staticmethod
    def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
        """
        Load image from file
        
        Args:
            path: Image file path
            grayscale: Load as grayscale
        
        Returns:
            Image array
        
        Raises:
            FileNotFoundError: If image cannot be loaded
        """
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        
        return img
    
    @staticmethod
    def save_image(image: np.ndarray, path: Path, 
                  create_dirs: bool = True) -> None:
        """
        Save image to file
        
        Args:
            image: Image array
            path: Output file path
            create_dirs: Create parent directories if needed
        """
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(path), image)
    
    @staticmethod
    def save_with_title(image: np.ndarray, title: str, path: Path,
                       header_height: int = 30) -> None:
        """
        Save image with title header
        
        Args:
            image: Image array
            title: Title text
            path: Output file path
            header_height: Height of header in pixels
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if image.ndim == 2:
            header = np.full((header_height, image.shape[1]), 255, dtype=np.uint8)
            cv2.putText(header, title[:60], (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
            stacked = np.vstack([header, image])
        else:
            header = np.full((header_height, image.shape[1], 3), 255, dtype=np.uint8)
            cv2.putText(header, title[:60], (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            stacked = np.vstack([header, image])
        
        cv2.imwrite(str(path), stacked)
    
    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """
        Ensure directory exists
        
        Args:
            path: Directory path
        
        Returns:
            Path object
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

