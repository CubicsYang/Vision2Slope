"""
Custom exceptions for Vision2Slope.

This module defines a hierarchy of exceptions for better error handling.
"""


class Vision2SlopeException(Exception):
    """Base exception for all Vision2Slope errors."""
    pass


class ConfigurationError(Vision2SlopeException):
    """Raised when there is a configuration error."""
    pass


class ProcessingError(Vision2SlopeException):
    """Base exception for processing errors."""
    
    def __init__(self, message: str, filename: str = None, stage: str = None):
        super().__init__(message)
        self.filename = filename
        self.stage = stage


class ModelLoadError(Vision2SlopeException):
    """Raised when model loading fails."""
    pass


class NoLinesDetectedError(ProcessingError):
    """Raised when no lines are detected for skew estimation."""
    pass


class NoRoadDetectedError(ProcessingError):
    """Raised when no road is detected in the image."""
    pass


class SlopeEstimationError(ProcessingError):
    """Raised when slope estimation fails."""
    pass


class ImageLoadError(ProcessingError):
    """Raised when image loading fails."""
    pass
