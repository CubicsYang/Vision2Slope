"""
Vision2Slope: Integrated Pipeline for Road Slope Analysis
========================================================

A comprehensive pipeline for road slope analysis from street view images.

Author: Cubics Yang
Date: October 2025

"""

# Configuration system
from .config import (
    PipelineConfig,
    ModelConfig,
    DetectionConfig,
    AnalysisConfig,
    VisualizationConfig,
    ProcessingConfig
)

# Main pipeline
from .pipeline import Vision2SlopePipeline
from .processor import StandardImageProcessor

# Component implementations
from .models import SegmentationModel
from .detectors import SkewDetector
from .correctors import ImageCorrector
from .analyzers import RoadSlopeAnalyzer
from .visualizers import Visualizer
from .utils import Utils
from .pano2perspective import PanoramaTransformer

# Core abstractions
from .core import (
    ImageProcessor,
    SegmentationProvider,
    SkewDetectionProvider,
    CorrectionProvider,
    SlopeAnalysisProvider,
    VisualizationProvider,
    Vision2SlopeException,
    ConfigurationError,
    ProcessingError,
    ProcessingResult,
    ProcessingStatus,
    ProcessingStage
)

__version__ = "2.0.0"

__all__ = [
    # Configuration
    "PipelineConfig",
    "ModelConfig",
    "DetectionConfig",
    "AnalysisConfig",
    "VisualizationConfig",
    "ProcessingConfig",
    
    # Pipeline
    "Vision2SlopePipeline",
    "StandardImageProcessor",
    
    # Components
    "SegmentationModel",
    "SkewDetector",
    "ImageCorrector",
    "RoadSlopeAnalyzer",
    "Visualizer",
    "Utils",
    "PanoramaTransformer",
    
    # Core abstractions (interfaces)
    "ImageProcessor",
    "SegmentationProvider",
    "SkewDetectionProvider",
    "CorrectionProvider",
    "SlopeAnalysisProvider",
    "VisualizationProvider",
    
    # Exceptions
    "Vision2SlopeException",
    "ConfigurationError",
    "ProcessingError",
    
    # Types
    "ProcessingResult",
    "ProcessingStatus",
    "ProcessingStage",
]
