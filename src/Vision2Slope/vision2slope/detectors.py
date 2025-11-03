"""
Skew detection module for Vision2Slope pipeline.
"""

import logging
import os
from typing import List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .config import PipelineConfig
from .core.interfaces import SkewDetectionProvider


class SkewDetector(SkewDetectionProvider):
    """Class for detecting skew angles in images."""
    
    # Mapillary Vistas class IDs
    VEGETATION_CLASS_ID = 30  # exclude mask
    BUILDING_CLASS_ID = 17    # include mask
    FENCE_CLASS_ID = 3        # include mask
    GUARD_RAIL_CLASS_ID = 4   # include mask
    WALL_CLASS_ID = 6         # include mask
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize skew detector.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_lines(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines using Hough Transform.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Detected lines or None if no lines found
        """
        try:
            # Convert to grayscale and apply Gaussian blur
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(
                blurred, 
                self.config.canny_threshold1, 
                self.config.canny_threshold2
            )
            
            # Detect lines using Hough Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1.0,
                theta=np.pi/180,
                threshold=self.config.hough_threshold,
                minLineLength=self.config.min_line_length,
                maxLineGap=self.config.max_line_gap
            )
            
            return lines
            
        except Exception as e:
            self.logger.error(f"Line detection failed: {e}")
            return None
    
    def filter_lines_by_mask(self, lines: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Filter lines based on intersection with a binary mask.
        
        Args:
            lines: Detected lines
            mask: Binary mask
            
        Returns:
            Filtered lines
        """
        if lines is None or lines.size == 0:
            return np.array([])
        
        filtered_lines = []
        mask_points = np.where(mask > 0)
        
        if len(mask_points[0]) == 0:
            return np.array([])
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line intersects with mask
            line_points = self._get_line_points(x1, y1, x2, y2)
            intersection = False
            
            for px, py in line_points:
                if 0 <= py < mask.shape[0] and 0 <= px < mask.shape[1]:
                    if mask[py, px] > 0:
                        intersection = True
                        break
            
            if intersection:
                filtered_lines.append(line)
        
        return np.array(filtered_lines) if filtered_lines else np.array([])
    
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Get points along a line using Bresenham's algorithm."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return points
    
    def calculate_line_angle(self, line: np.ndarray) -> float:
        """
        Calculate the angle of a line segment.
        
        Normalized to [-45, 45] for near-vertical lines.
        
        Args:
            line: Line segment
            
        Returns:
            Angle in degrees
        """
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))

        # Normalize angles close to vertical to [-45, 45], others marked as NaN
        if 45 <= angle <= 135:
            angle -= 90
        elif -135 <= angle <= -45:
            angle += 90
        else:
            angle = np.nan

        return angle

    def estimate_skew_angle(self, lines: np.ndarray) -> Tuple[float, int]:
        """
        Estimate skew angle from detected lines.
        
        Args:
            lines: Detected lines
            
        Returns:
            Tuple of (skew_angle, confidence_score)
        """
        if lines.size == 0:
            return 0.0, 0
        
        # Calculate angles for all lines
        angles = []
        for line in lines:
            angle = self.calculate_line_angle(line)
            if not np.isnan(angle):
                angles.append(angle)
        
        angles = np.array(angles)
        
        # Filter angles close to coordinate axes
        close_to_axes = np.abs(angles) < self.config.angle_tolerance
        
        if not close_to_axes.any():
            return 0.0, 0
        
        filtered_angles = angles[close_to_axes]
        skew_angle = float(np.median(filtered_angles))
        
        # Confidence is the number of lines contributing to the skew angle
        confidence = len(filtered_angles)
        
        return skew_angle, confidence
    
    def get_masked_lines(self, image: Image.Image, semantic_map: np.ndarray) -> np.ndarray:
        """
        Get lines filtered by semantic segmentation masks.
        
        This function detects lines in the image and filters them based on 
        semantic segmentation. It excludes lines intersecting with vegetation 
        and keeps only lines intersecting with building, fence, guard rail, 
        and wall structures.
        
        Args:
            image: PIL Image object
            semantic_map: Semantic segmentation result
            
        Returns:
            Filtered lines as numpy array. Returns empty array if no lines found.
        """
        try:
            # Convert PIL image to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create masks for vegetation
            vegetation_mask = (semantic_map == self.VEGETATION_CLASS_ID).astype(np.uint8)
            
            # Create masks for building, fence, guard rail, and wall
            building_mask = (semantic_map == self.BUILDING_CLASS_ID).astype(np.uint8)
            fence_mask = (semantic_map == self.FENCE_CLASS_ID).astype(np.uint8)
            guard_rail_mask = (semantic_map == self.GUARD_RAIL_CLASS_ID).astype(np.uint8)
            wall_mask = (semantic_map == self.WALL_CLASS_ID).astype(np.uint8)
            
            # Combine masks, including building, fence, guard rail, and wall
            intersection_mask = (
                building_mask | fence_mask | guard_rail_mask | wall_mask
            )
            
            # Detect lines
            lines = self.detect_lines(cv_image)
            
            if lines is None or lines.size == 0:
                return np.array([])
            
            # Filter lines that intersect with vegetation mask
            tree_lines = self.filter_lines_by_mask(lines, vegetation_mask)
            if tree_lines is not None and tree_lines.size > 0:
                # Remove tree lines from lines
                lines = np.array([line for line in lines 
                                 if not any(np.all(line == tree_line) for tree_line in tree_lines)])
            
            # Filter lines using semantic(building) mask
            filtered_lines = self.filter_lines_by_mask(lines, intersection_mask)
            
            return filtered_lines if filtered_lines is not None else np.array([])
            
        except Exception as e:
            self.logger.error(f"Failed to get masked lines: {e}")
            return np.array([])
    
    def get_vertical_lines(self, image: Image.Image, semantic_map: np.ndarray) -> np.ndarray:
        """
        Get near-vertical lines filtered by semantic segmentation masks.
        
        This function first gets masked lines using semantic segmentation,
        then filters them to keep only near-vertical lines by calculating
        their angles using calculate_line_angle.
        
        Args:
            image: PIL Image object
            semantic_map: Semantic segmentation result
            
        Returns:
            Filtered near-vertical lines as numpy array. Returns empty array if no lines found.
        """
        try:
            # Get lines filtered by semantic masks
            masked_lines = self.get_masked_lines(image, semantic_map)
            
            if masked_lines.size == 0:
                return np.array([])
            
            # Filter lines by angle using calculate_line_angle
            vertical_lines = []
            for line in masked_lines:
                angle = self.calculate_line_angle(line)
                # Only keep lines with valid angles (near-vertical)
                if not np.isnan(angle):
                    vertical_lines.append(line)
            
            return np.array(vertical_lines) if vertical_lines else np.array([])
            
        except Exception as e:
            self.logger.error(f"Failed to get vertical lines: {e}")
            return np.array([])
    
    def detect_skew(self, image: Image.Image, semantic_map: np.ndarray) -> Tuple[float, int]:
        """
        Detect skew angle in an image using semantic segmentation.
        
        Args:
            image: PIL Image object
            semantic_map: Semantic segmentation result
            
        Returns:
            Tuple of (skew_angle, confidence_score)
        """
        try:
            # Get near-vertical lines filtered by semantic masks
            vertical_lines = self.get_vertical_lines(image, semantic_map)
            
            if vertical_lines.size == 0:
                return 0.0, 0
            
            # Estimate skew angle
            skew_angle, confidence = self.estimate_skew_angle(vertical_lines)
            
            return skew_angle, confidence
            
        except Exception as e:
            self.logger.error(f"Skew detection failed: {e}")
            return 0.0, 0
    
    def save_edge_detection_visualization(self, image: np.ndarray, filename: str, output_dir: str):
        """
        Save edge detection visualization.
        
        Args:
            image: Input image in BGR format
            filename: Original filename
            output_dir: Output directory
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, self.config.canny_threshold1, self.config.canny_threshold2)
            
            output_path = os.path.join(output_dir, filename.replace('.png', '_edges.png').replace('.jpg', '_edges.jpg'))
            cv2.imwrite(output_path, edges)
            
        except Exception as e:
            self.logger.error(f"Failed to save edge detection visualization: {e}")
    
    def save_line_detection_visualization(self, image: np.ndarray, lines: np.ndarray, 
                                         filename: str, output_dir: str):
        """
        Save line detection visualization.
        
        Args:
            image: Input image in BGR format
            lines: Detected lines
            filename: Original filename
            output_dir: Output directory
        """
        try:
            vis_image = image.copy()
            if lines is not None and lines.size > 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            output_path = os.path.join(output_dir, filename.replace('.png', '_lines.png').replace('.jpg', '_lines.jpg'))
            cv2.imwrite(output_path, vis_image)
            
        except Exception as e:
            self.logger.error(f"Failed to save line detection visualization: {e}")
