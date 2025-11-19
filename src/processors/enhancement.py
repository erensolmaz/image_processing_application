import cv2
import numpy as np
from typing import Tuple, Optional

class EnhancementProcessor:
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                   tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        return cv2.equalizeHist(image)
    
    @staticmethod
    def contrast_stretch(image: np.ndarray, min_val: Optional[int] = None,
                        max_val: Optional[int] = None) -> np.ndarray:
        if min_val is None or max_val is None:
            return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.normalize(image, None, min_val, max_val, cv2.NORM_MINMAX)
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def log_transform(image: np.ndarray, c: Optional[float] = None) -> np.ndarray:
        img_float = image.astype(np.float32)
        if c is None:
            c = 255 / np.log(1 + np.max(img_float))
        log_transformed = c * np.log(1 + img_float)
        return np.clip(log_transformed, 0, 255).astype(np.uint8)
    
    @staticmethod
    def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength
        kernel[1, 1] = 8 * strength + 1
        return cv2.filter2D(image, -1, kernel)
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0,
                                   beta: int = 0) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
