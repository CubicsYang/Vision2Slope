"""
Core abstractions and interfaces for Vision2Slope.
"""

from .interfaces import (
    ImageProcessor,
    SegmentationProvider,
    SkewDetectionProvider,
    CorrectionProvider,
    SlopeAnalysisProvider,
    VisualizationProvider
)
from .exceptions import (
    Vision2SlopeException,
    ConfigurationError,
    ProcessingError,
    ModelLoadError,
    NoLinesDetectedError,
    NoRoadDetectedError
)
from .types import ProcessingResult, ProcessingStatus, ProcessingStage

__all__ = [
    # Interfaces
    'ImageProcessor',
    'SegmentationProvider',
    'SkewDetectionProvider',
    'CorrectionProvider',
    'SlopeAnalysisProvider',
    'VisualizationProvider',
    
    # Exceptions
    'Vision2SlopeException',
    'ConfigurationError',
    'ProcessingError',
    'ModelLoadError',
    'NoLinesDetectedError',
    'NoRoadDetectedError',
    
    # Types
    'ProcessingResult',
    'ProcessingStatus',
    'ProcessingStage',
]
