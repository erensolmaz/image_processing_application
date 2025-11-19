import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .processors import (
    FilterProcessor, TransformationProcessor,
    SegmentationProcessor, EnhancementProcessor
)
from .utils import ImageIO, ImageVisualizer

@dataclass
class PipelineStep:
    name: str
    processor: Callable
    params: Dict[str, Any]
    save_output: bool = True

class ImageProcessingPipeline:
    def __init__(self, output_dir: Optional[Path] = None):
        self.steps: List[PipelineStep] = []
        self.output_dir = output_dir or Path('outputs')
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        
        self.filters = FilterProcessor()
        self.transformations = TransformationProcessor()
        self.segmentation = SegmentationProcessor()
        self.enhancement = EnhancementProcessor()
        self.io = ImageIO()
        self.visualizer = ImageVisualizer()
    
    def load_image(self, image_path: Path, grayscale: bool = False) -> 'ImageProcessingPipeline':
        self.original_image = self.io.load_image(image_path, grayscale=grayscale)
        self.current_image = self.original_image.copy()
        return self
    
    def add_step(self, name: str, processor: Callable, 
                params: Dict[str, Any], save_output: bool = True) -> 'ImageProcessingPipeline':
        step = PipelineStep(name, processor, params, save_output)
        self.steps.append(step)
        return self
    
    def execute(self, save_intermediate: bool = True) -> np.ndarray:
        if self.current_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        self.io.ensure_dir(self.output_dir)
        
        for i, step in enumerate(self.steps):
            self.current_image = step.processor(self.current_image, **step.params)
            
            if save_intermediate and step.save_output:
                output_path = self.output_dir / f"step{i+1:02d}_{step.name}.png"
                self.io.save_with_title(
                    self.current_image, 
                    step.name,
                    output_path
                )
        
        return self.current_image
    
    def reset(self) -> 'ImageProcessingPipeline':
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        self.steps.clear()
        return self
    
    def get_image(self) -> Optional[np.ndarray]:
        return self.current_image
    
    def get_original(self) -> Optional[np.ndarray]:
        return self.original_image
