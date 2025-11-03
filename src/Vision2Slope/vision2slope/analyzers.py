"""
Road slope analysis module for Vision2Slope pipeline.
"""

import logging
import os
from typing import Optional, Tuple
import numpy as np
import cv2
from skimage import measure
from sklearn.linear_model import RANSACRegressor

from .config import PipelineConfig
from .core.interfaces import SlopeAnalysisProvider


class RoadSlopeAnalyzer(SlopeAnalysisProvider):
    """Class for analyzing road slopes from corrected images."""
    
    # Mapillary Vistas semantic segmentation class IDs
    ROAD_CLASS_ID = 13
    ROAD_MARKING_CLASS_ID = 24
    ROAD_CROSSWALK_CLASS_ID = 23

    def __init__(self, config: PipelineConfig):
        """
        Initialize road slope analyzer.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Analyze road slope from semantic segmentation (interface method).
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area)
        """
        return self.analyze_road_slope(semantic_map)
    
    def analyze_with_details(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, Optional[np.ndarray]]:
        """
        Analyze road slope with intermediate results (interface method).
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area, road_mask, edge_points)
        """
        return self.analyze_road_slope_with_details(semantic_map)
    
    def create_road_mask(self, semantic_map: np.ndarray) -> np.ndarray:
        """
        Create a binary road mask from semantic segmentation.
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Binary road mask
        """
        # Combine road and road marking classes
        road_mask = np.where(semantic_map == self.ROAD_CLASS_ID, 1, 0)
        road_marking_mask = np.where(semantic_map == self.ROAD_MARKING_CLASS_ID, 1, 0)
        road_crosswalk_mask = np.where(semantic_map == self.ROAD_CROSSWALK_CLASS_ID, 1, 0)
        combined_mask = road_mask + road_marking_mask + road_crosswalk_mask

        # Apply morphological opening to remove noise
        # Use configurable kernel size
        kernel_size = getattr(self.config, 'morphology_kernel_size', 15)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned_mask = cv2.morphologyEx(
            combined_mask.astype(np.uint8), 
            cv2.MORPH_OPEN, 
            kernel
        )
        
        return cleaned_mask
    
    def extract_road_edge(self, road_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the upper edge points of the road mask.
        
        Args:
            road_mask: Binary road mask
            
        Returns:
            Array of edge points or None
        """
        contours = measure.find_contours(road_mask.astype(float), level=0.5)
        
        if not contours:
            return None
        
        # Get the longest contour (main road boundary)
        main_contour = max(contours, key=len)
        
        # Sort by column (x-coordinate) and get unique points for upper edge
        sorted_contour = main_contour[np.argsort(main_contour[:, 1])]
        _, unique_indices = np.unique(sorted_contour[:, 1].astype(int), return_index=True)
        upper_edge_points = sorted_contour[unique_indices]
        
        return upper_edge_points
    
    def fit_line_ransac(self, edge_points: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit a line to edge points using RANSAC regression.
        
        Args:
            edge_points: Array of edge points
            
        Returns:
            Tuple of (slope, intercept, angle_degrees)
        """
        x = edge_points[:, 1]  # Column coordinates
        y = edge_points[:, 0]  # Row coordinates
        
        # Use configurable RANSAC parameters
        residual_threshold = getattr(self.config, 'ransac_residual_threshold', 1.0)
        max_trials = getattr(self.config, 'ransac_max_trials', 1000)
        random_state = getattr(self.config, 'ransac_random_state', 42)
        
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state
        )
        
        ransac.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        
        slope = float(ransac.estimator_.coef_[0][0])
        intercept = float(ransac.estimator_.intercept_[0])
        angle_degrees = np.arctan(slope) * 180 / np.pi
        
        return slope, intercept, angle_degrees
    
    def analyze_road_slope(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int]:
        """
        Analyze road slope from semantic segmentation.
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area)
        """
        try:
            # Create road mask
            road_mask = self.create_road_mask(semantic_map)
            road_area = int(np.sum(road_mask))
            
            if road_area == 0:
                return -999.0, -999.0, -999.0, 0
            
            # Extract road edge points
            edge_points = self.extract_road_edge(road_mask)
            
            if edge_points is None or len(edge_points) < self.config.min_edge_points:
                return -999.0, -999.0, -999.0, road_area
            
            # Fit line to edge points
            slope, intercept, angle = self.fit_line_ransac(edge_points)
            
            return slope, intercept, angle, road_area
            
        except Exception as e:
            self.logger.error(f"Road slope analysis failed: {e}")
            return -999.0, -999.0, -999.0, 0
    
    def analyze_road_slope_with_details(self, semantic_map: np.ndarray) -> Tuple[float, float, float, int, np.ndarray, np.ndarray]:
        """
        Analyze road slope from semantic segmentation with detailed intermediate results.
        
        Args:
            semantic_map: Semantic segmentation map
            
        Returns:
            Tuple of (slope, intercept, angle, road_area, road_mask, edge_points)
        """
        try:
            # Create road mask
            road_mask = self.create_road_mask(semantic_map)
            road_area = int(np.sum(road_mask))
            
            if road_area == 0:
                return -999.0, -999.0, -999.0, 0, road_mask, None
            
            # Extract road edge points
            edge_points = self.extract_road_edge(road_mask)
            
            if edge_points is None or len(edge_points) < self.config.min_edge_points:
                return -999.0, -999.0, -999.0, road_area, road_mask, edge_points
            
            # Fit line to edge points
            slope, intercept, angle = self.fit_line_ransac(edge_points)
            
            return slope, intercept, angle, road_area, road_mask, edge_points
            
        except Exception as e:
            self.logger.error(f"Road slope analysis failed: {e}")
            return -999.0, -999.0, -999.0, 0, None, None
    
    def save_road_mask_visualization(self, road_mask: np.ndarray, filename: str, output_dir: str):
        """
        Save road mask visualization.
        
        Args:
            road_mask: Binary road mask
            filename: Original filename
            output_dir: Output directory
        """
        try:
            output_path = os.path.join(output_dir, filename.replace('.png', '_road_mask.png').replace('.jpg', '_road_mask.jpg'))
            cv2.imwrite(output_path, road_mask * 255)
            
        except Exception as e:
            self.logger.error(f"Failed to save road mask visualization: {e}")
