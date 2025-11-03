"""
Abstract interfaces for Vision2Slope components.

This module defines the core interfaces that components must implement,
enabling loose coupling and easy extensibility.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from PIL import Image

from .types import ProcessingResult


class ImageProcessor(ABC):
    """
    Abstract base for image processors.
    
    An ImageProcessor coordinates the entire processing pipeline for a single image.
    """
    
    @abstractmethod
    def process(self, image_path: str) -> ProcessingResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            ProcessingResult with all analysis results
        """
        pass


class SegmentationProvider(ABC):
    """
    Abstract interface for semantic segmentation.
    
    Implementations provide semantic segmentation maps from images.
    """
    
    @abstractmethod
    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Semantic segmentation map as (H, W) numpy array
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get information about the segmentation model."""
        pass


class SkewDetectionProvider(ABC):
    """
    Abstract interface for skew detection.
    
    Implementations detect vertical skew/tilt in panoramic images.
    """
    
    @abstractmethod
    def detect_skew(self, image: Image.Image, semantic_map: np.ndarray) -> Tuple[float, int]:
        """
        Detect skew angle in an image.
        
        Args:
            image: PIL Image object
            semantic_map: Semantic segmentation result
            
        Returns:
            Tuple of (skew_angle, confidence_score)
        """
        pass
    
    @abstractmethod
    def get_vertical_lines(self, image: Image.Image, semantic_map: np.ndarray) -> np.ndarray:
        """
        Get near-vertical lines for visualization.
        
        Args:
            image: PIL Image object
            semantic_map: Semantic segmentation result
            
        Returns:
            Array of detected lines
        """
        pass


class CorrectionProvider(ABC):
    """
    Abstract interface for image correction.
    
    Implementations correct geometric distortions in images.
    """
    
    @abstractmethod
    def correct(self, image: np.ndarray, skew_angle: float) -> np.ndarray:
        """
        Correct image based on skew angle.
        
        Args:
            image: Input image array (BGR format)
            skew_angle: Skew angle in degrees
            
        Returns:
            Corrected image array
        """
        pass


class SlopeAnalysisProvider(ABC):
    """
    Abstract interface for road slope analysis.
    
    Implementations analyze road geometry and estimate slopes.
    """
    
    @abstractmethod
    def analyze(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Analyze road slope from semantic segmentation.
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area)
        """
        pass
    
    @abstractmethod
    def analyze_with_details(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, Optional[np.ndarray]]:
        """
        Analyze road slope with intermediate results.
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area, road_mask, edge_points)
        """
        pass


class VisualizationProvider(ABC):
    """
    Abstract interface for visualization output.
    
    Implementations handle saving various visualization outputs.
    """
    
    @abstractmethod
    def save_segmentation_mask(self, semantic_map: np.ndarray, filename: str):
        """Save semantic segmentation visualization."""
        pass
    
    @abstractmethod
    def save_road_mask(self, road_mask: np.ndarray, filename: str):
        """Save road mask visualization."""
        pass
    
    @abstractmethod
    def save_corrected_image(self, corrected_image: np.ndarray, filename: str) -> str:
        """Save corrected image and return filename."""
        pass
    
    @abstractmethod
    def save_comprehensive_visualization(self, original_image: Image.Image, 
                                        corrected_image: Image.Image,
                                        result: ProcessingResult, 
                                        semantic_map: np.ndarray,
                                        road_mask: np.ndarray):
        """Save comprehensive visualization with all results."""
        pass
