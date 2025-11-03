"""
Visualization module for Vision2Slope pipeline.
"""

import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image

from .config import VisualizationConfig
from .core.types import ProcessingResult
from .core.interfaces import VisualizationProvider
from .utils import Utils


class Visualizer(VisualizationProvider):
    """Class for creating visualizations of processing results."""
    
    def __init__(self, viz_config: VisualizationConfig, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            viz_config: Visualization configuration
            output_dir: Base output directory
        """
        self.config = viz_config
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        self._setup_directories()
    
    def _setup_directories(self):
        """Create visualization subdirectories."""
        if self.config.save_visualizations:
            self.viz_dir = self.output_dir / self.config.viz_dir_name
            self.viz_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_corrected_images:
            self.corrected_dir = self.output_dir / self.config.corrected_dir_name
            self.corrected_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_intermediate_results:
            self.intermediate_dir = self.output_dir / self.config.intermediate_dir_name
            self.intermediate_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_segmentation_masks:
            self.masks_dir = self.output_dir / self.config.masks_dir_name
            self.masks_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_edge_detection:
            self.edges_dir = self.output_dir / self.config.edges_dir_name
            self.edges_dir.mkdir(exist_ok=True, parents=True)

        if self.config.save_ppht_results:
            self.ppht_dir = self.output_dir / self.config.ppht_dir_name
            self.ppht_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_line_detection:
            self.lines_dir = self.output_dir / self.config.lines_dir_name
            self.lines_dir.mkdir(exist_ok=True, parents=True)
        
        if self.config.save_road_edge_fitting:
            self.road_edge_fitting_dir = self.output_dir / self.config.road_edge_fitting_dir_name
            self.road_edge_fitting_dir.mkdir(exist_ok=True, parents=True)
    
    def save_segmentation_mask(self, semantic_map: np.ndarray, filename: str):
        """
        Save semantic segmentation mask visualization.
        
        Args:
            semantic_map: Semantic segmentation array
            filename: Original filename
        """
        if not self.config.save_segmentation_masks:
            return
        
        try:
            semantic_rgb = Utils.render_semantic_segmentation(semantic_map)
            output_path = self.masks_dir / filename.replace('.png', '_segmentation.png').replace('.jpg', '_segmentation.jpg')
            cv2.imwrite(str(output_path), cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            self.logger.error(f"Failed to save segmentation mask: {e}")
    
    def save_road_mask(self, road_mask: np.ndarray, filename: str):
        """
        Save road mask visualization.
        
        Args:
            road_mask: Binary road mask
            filename: Original filename
        """
        if not self.config.save_road_masks:
            return
        
        try:
            output_path = self.masks_dir / filename.replace('.png', '_road_mask.png').replace('.jpg', '_road_mask.jpg')
            cv2.imwrite(str(output_path), road_mask * 255)
        except Exception as e:
            self.logger.error(f"Failed to save road mask: {e}")
    
    def save_edge_detection(self, image: np.ndarray, filename: str):
        """
        Save edge detection visualization.
        
        Args:
            image: Input image in BGR format
            filename: Original filename
        """
        if not self.config.save_edge_detection:
            return
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            output_path = self.edges_dir / filename.replace('.png', '_edges.png').replace('.jpg', '_edges.jpg')
            cv2.imwrite(str(output_path), edges)
        except Exception as e:
            self.logger.error(f"Failed to save edge detection: {e}")

    def save_ppht_results(self, ppht_image: np.ndarray, filename: str):
        """
        Save Probabilistic Progressive Hough Transform (PPHT) results visualization.
        
        Args:
            ppht_image: Image with PPHT lines drawn
            filename: Original filename
        """
        if not self.config.save_ppht_results:
            return

        try:
            # Save the ppht visualization image directly
            output_path = self.ppht_dir / filename.replace('.png', '_ppht.png').replace('.jpg', '_ppht.jpg')
            cv2.imwrite(str(output_path), ppht_image)
            self.logger.debug(f"Saved PPHT results to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save PPHT results: {e}")
    
    def save_line_detection(self, image: np.ndarray, lines: np.ndarray, filename: str, has_background: bool = True):
        """
        Save line detection visualization.
        
        Args:
            image: Input image in BGR format
            lines: Detected lines
            filename: Original filename
        """
        if not self.config.save_line_detection:
            return
        
        try:
            if not has_background:
                vis_image = np.zeros_like(image)
            else:
                vis_image = image.copy()
            if lines is not None and lines.size > 0:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(vis_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            output_path = self.lines_dir / filename.replace('.png', '_lines.png').replace('.jpg', '_lines.jpg')
            cv2.imwrite(str(output_path), vis_image)
        except Exception as e:
            self.logger.error(f"Failed to save line detection: {e}")
    
    def save_corrected_image(self, corrected_image: np.ndarray, filename: str) -> str:
        """
        Save corrected image.
        
        Args:
            corrected_image: Corrected image array
            filename: Original filename
            
        Returns:
            Corrected filename
        """
        if not self.config.save_corrected_images:
            return ""
        
        try:
            corrected_filename = filename.replace('.png', '_corrected.png').replace('.jpg', '_corrected.jpg')
            output_path = self.corrected_dir / corrected_filename
            cv2.imwrite(str(output_path), corrected_image)
            return corrected_filename
        except Exception as e:
            self.logger.error(f"Failed to save corrected image: {e}")
            return ""
    
    def save_road_edge_fitting_visualization(self, image: np.ndarray, road_mask: np.ndarray, 
                                            edge_points: np.ndarray, slope: float, 
                                            intercept: float, filename: str):
        """
        Save visualization of road edge points and RANSAC fitted line.
        
        Args:
            image: Input image (BGR format)
            road_mask: Binary road mask
            edge_points: Extracted edge points (y, x format from skimage)
            slope: RANSAC fitted line slope
            intercept: RANSAC fitted line intercept
            filename: Original filename
        """
        if not self.config.save_road_edge_fitting:
            return
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=self.config.figure_dpi)
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Plot 1: Original image with road mask overlay
            axes[0].imshow(image_rgb)
            axes[0].imshow(road_mask, cmap='jet', alpha=0.4)
            axes[0].set_title('Road Mask Overlay')
            axes[0].axis('off')
            
            # Plot 2: Road mask with edge points
            axes[1].imshow(road_mask, cmap='gray')
            if edge_points is not None and len(edge_points) > 0:
                # edge_points format: (row, col) or (y, x)
                axes[1].scatter(edge_points[:, 1], edge_points[:, 0], 
                              c='red', s=3, alpha=0.8)
            axes[1].set_title(f'Extracted Edge Points\n(n={len(edge_points) if edge_points is not None else 0})')
            axes[1].axis('off')
            
            # Plot 3: Image with edge points and RANSAC fitted line
            axes[2].imshow(image_rgb)
            
            if edge_points is not None and len(edge_points) > 0:
                # Plot edge points
                axes[2].scatter(edge_points[:, 1], edge_points[:, 0], 
                              c='red', s=4, alpha=0.8)
                
                # Plot RANSAC fitted line
                if slope != -999.0 and intercept != -999.0:
                    # Get image dimensions
                    height, width = image.shape[:2]
                    
                    # Generate line points across the image width
                    x_line = np.array([0, width])
                    y_line = slope * x_line + intercept
                    
                    # Clip line to image boundaries
                    valid_mask = (y_line >= 0) & (y_line < height)
                    if np.any(valid_mask):
                        axes[2].plot(x_line, y_line, 'b-', linewidth=5, alpha=1.0)
                
            
            axes[2].set_title('Edge Points + RANSAC Fitted Line')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            output_filename = filename.replace('.png', '_road_edge_fitting.png').replace('.jpg', '_road_edge_fitting.jpg')
            output_path = self.road_edge_fitting_dir / output_filename
            plt.savefig(output_path, bbox_inches='tight', dpi=self.config.figure_dpi)
            plt.close(fig)
            
            self.logger.debug(f"Saved road edge fitting visualization to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save road edge fitting visualization: {e}")
    
    def save_comprehensive_visualization(self, original_image: Image.Image, 
                                        corrected_image: Image.Image,
                                        result: ProcessingResult, 
                                        semantic_map: np.ndarray,
                                        road_mask: np.ndarray):
        """
        Save comprehensive visualization with all results.
        
        Args:
            original_image: Original PIL Image
            corrected_image: Corrected PIL Image
            result: Processing result object
            semantic_map: Semantic segmentation map
            road_mask: Binary road mask
        """
        if not self.config.save_visualizations:
            return
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), 
                                    dpi=self.config.figure_dpi)
            
            # Original image
            axes[0].imshow(original_image)
            axes[0].set_title(f"Original Image\n{result.filename}")
            axes[0].axis('off')
            
            # Semantic segmentation overlay
            axes[1].imshow(corrected_image)
            semantic_rgb = Utils.render_semantic_segmentation(semantic_map)
            axes[1].imshow(semantic_rgb, alpha=self.config.overlay_alpha)
            axes[1].set_title("Semantic Segmentation")
            axes[1].axis('off')
            
            # Road mask overlay
            axes[2].imshow(corrected_image)
            axes[2].imshow(road_mask, cmap='gray', alpha=0.5)
            axes[2].set_title(f"Road Mask\nArea: {result.road_area} pixels\nRoad Angle: {result.road_angle:.2f}°")
            axes[2].axis('off')
            
            plt.tight_layout()
            
            viz_filename = result.filename.replace('.png', '_analysis.png').replace('.jpg', '_analysis.jpg')
            viz_path = self.viz_dir / viz_filename
            plt.savefig(viz_path, bbox_inches='tight', dpi=self.config.figure_dpi)
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Failed to save comprehensive visualization: {e}")
