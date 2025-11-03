"""
Type definitions and data structures for Vision2Slope.

Moved from data_types.py for better organization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProcessingStage(Enum):
    """Processing stage indicators."""
    INITIALIZATION = "initialization"
    SEGMENTATION = "segmentation"
    SKEW_DETECTION = "skew_detection"
    IMAGE_CORRECTION = "image_correction"
    SLOPE_ESTIMATION = "slope_estimation"
    COMPLETE = "complete"


class ProcessingStatus(Enum):
    """Status codes for processing results."""
    SUCCESS = "success"
    SKEW_FAILED = "skew_detection_failed"
    CORRECTION_FAILED = "correction_failed"
    SLOPE_FAILED = "slope_estimation_failed"
    NO_LINES = "no_lines_detected"
    NO_ROAD = "no_road_detected"
    ERROR = "processing_error"


@dataclass
class ProcessingResult:
    """Container for processing results of a single image."""
    filename: str
    pano_id: str
    
    # Skew detection results
    skew_angle: float = -999.0
    num_lines_detected: int = 0
    skew_confidence: float = 0.0
    
    # Image correction results
    correction_applied: bool = False
    corrected_filename: Optional[str] = None
    
    # Slope estimation results
    road_edge_line_slope: float = -999.0
    road_edge_line_intercept: float = -999.0
    road_edge_line_angle: float = -999.0
    road_area: int = 0
    
    # Processing status
    stage_completed: ProcessingStage = ProcessingStage.INITIALIZATION
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    error_message: str = ""
    
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return (self.status == ProcessingStatus.SUCCESS and 
                self.stage_completed == ProcessingStage.COMPLETE)
    
    def to_dict(self) -> dict:
        """Convert result to dictionary for DataFrame."""
        return {
            'filename': self.filename,
            'pano_id': self.pano_id,
            'skew_angle': self.skew_angle,
            'skew_confidence': self.skew_confidence,
            'num_lines_detected': self.num_lines_detected,
            'correction_applied': self.correction_applied,
            'corrected_filename': self.corrected_filename,
            'road_edge_line_slope': self.road_edge_line_slope,
            'road_edge_line_intercept': self.road_edge_line_intercept,
            'road_edge_line_angle': self.road_edge_line_angle,
            'road_area': self.road_area,
            'stage_completed': self.stage_completed.value,
            'status': self.status.value,
            'error_message': self.error_message
        }
