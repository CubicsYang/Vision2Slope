"""
Image correction module for Vision2Slope pipeline.
"""

import logging
import numpy as np
import cv2

from .config import PipelineConfig
from .core.interfaces import CorrectionProvider


class ImageCorrector(CorrectionProvider):
    """Class for correcting image skew."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize image corrector.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def extract_pano_id(filename: str) -> str:
        """
        Extract panorama ID from filename.
        
        Args:
            filename: Image filename
            
        Returns:
            Panorama ID
        """
        return filename.split('_Direction_')[0]
    
    def correct(self, image: np.ndarray, skew_angle: float) -> np.ndarray:
        """
        Correct image skew by rotation.
        
        Args:
            image: Input image array
            skew_angle: Skew angle in degrees
            
        Returns:
            Corrected image array
        """
        try:
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Apply rotation
            corrected_image = cv2.warpAffine(
                image, 
                rotation_matrix, 
                (w, h), 
                flags=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return corrected_image
            
        except Exception as e:
            self.logger.error(f"Image correction failed: {e}")
            raise