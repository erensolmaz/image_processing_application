"""
Image Processing Pipeline
Görüntü İşleme Pipeline'ı
"""

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
    """Single pipeline step"""
    name: str
    processor: Callable
    params: Dict[str, Any]
    save_output: bool = True


class ImageProcessingPipeline:
    """Image processing pipeline manager"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize pipeline
        
        Args:
            output_dir: Output directory for saving results
        """
        self.steps: List[PipelineStep] = []
        self.output_dir = output_dir or Path('outputs')
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        
        # Initialize processors
        self.filters = FilterProcessor()
        self.transformations = TransformationProcessor()
        self.segmentation = SegmentationProcessor()
        self.enhancement = EnhancementProcessor()
        self.io = ImageIO()
        self.visualizer = ImageVisualizer()
    
    def load_image(self, image_path: Path, grayscale: bool = False) -> 'ImageProcessingPipeline':
        """
        Load image into pipeline
        
        Args:
            image_path: Path to image file
            grayscale: Load as grayscale
        
        Returns:
            Self for method chaining
        """
        self.original_image = self.io.load_image(image_path, grayscale=grayscale)
        self.current_image = self.original_image.copy()
        return self
    
    def add_step(self, name: str, processor: Callable, 
                params: Dict[str, Any], save_output: bool = True) -> 'ImageProcessingPipeline':
        """
        Add processing step to pipeline
        
        Args:
            name: Step name
            processor: Processing function
            params: Parameters for processor
            save_output: Whether to save output
        
        Returns:
            Self for method chaining
        """
        step = PipelineStep(name, processor, params, save_output)
        self.steps.append(step)
        return self
    
    def execute(self, save_intermediate: bool = True) -> np.ndarray:
        """
        Execute pipeline
        
        Args:
            save_intermediate: Save intermediate results
        
        Returns:
            Final processed image
        """
        if self.current_image is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        self.io.ensure_dir(self.output_dir)
        
        for i, step in enumerate(self.steps):
            # Apply processing step
            self.current_image = step.processor(self.current_image, **step.params)
            
            # Save intermediate result if requested
            if save_intermediate and step.save_output:
                output_path = self.output_dir / f"step{i+1:02d}_{step.name}.png"
                self.io.save_with_title(
                    self.current_image, 
                    step.name,
                    output_path
                )
        
        return self.current_image
    
    def reset(self) -> 'ImageProcessingPipeline':
        """
        Reset pipeline to original image
        
        Returns:
            Self for method chaining
        """
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        self.steps.clear()
        return self
    
    def get_image(self) -> Optional[np.ndarray]:
        """
        Get current processed image
        
        Returns:
            Current image or None
        """
        return self.current_image
    
    def get_original(self) -> Optional[np.ndarray]:
        """
        Get original image
        
        Returns:
            Original image or None
        """
        return self.original_image

