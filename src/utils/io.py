import cv2
import numpy as np
from pathlib import Path

class ImageIO:
    @staticmethod
    def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {path}")
        return img
    
    @staticmethod
    def save_image(image: np.ndarray, path: Path, 
                  create_dirs: bool = True) -> None:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    
    @staticmethod
    def save_with_title(image: np.ndarray, title: str, path: Path,
                       header_height: int = 30) -> None:
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
        path.mkdir(parents=True, exist_ok=True)
        return path
