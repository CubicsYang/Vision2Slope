"""
Improved image processor implementation with better error handling and modularity.
"""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from .core.interfaces import (
    ImageProcessor,
    SegmentationProvider,
    SkewDetectionProvider,
    CorrectionProvider,
    SlopeAnalysisProvider,
    VisualizationProvider
)
from .core.types import ProcessingResult, ProcessingStatus, ProcessingStage
from .core.exceptions import (
    ProcessingError,
    NoLinesDetectedError,
    NoRoadDetectedError,
    ImageLoadError
)


class StandardImageProcessor(ImageProcessor):
    """
    Standard implementation of image processor.
    
    This class coordinates all processing steps while remaining loosely coupled
    through dependency injection of provider interfaces.
    """
    
    def __init__(
        self,
        segmentation_provider: SegmentationProvider,
        skew_detector: SkewDetectionProvider,
        corrector: CorrectionProvider,
        slope_analyzer: SlopeAnalysisProvider,
        visualizer: Optional[VisualizationProvider] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize processor with injected dependencies.
        
        Args:
            segmentation_provider: Semantic segmentation provider
            skew_detector: Skew detection provider
            corrector: Image correction provider
            slope_analyzer: Slope analysis provider
            visualizer: Optional visualization provider
            logger: Optional logger instance
        """
        self.segmentation_provider = segmentation_provider
        self.skew_detector = skew_detector
        self.corrector = corrector
        self.slope_analyzer = slope_analyzer
        self.visualizer = visualizer
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, image_path: str) -> ProcessingResult:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            ProcessingResult with all analysis results
        """
        path = Path(image_path)
        filename = path.name
        pano_id = self._extract_pano_id(filename)
        result = ProcessingResult(filename=filename, pano_id=pano_id)
        
        try:
            # Step 1: Load image
            self.logger.debug(f"Processing {filename}: Loading image")
            image = self._load_image(path)
            result.stage_completed = ProcessingStage.INITIALIZATION
            
            # Step 2: Semantic segmentation
            self.logger.debug(f"Processing {filename}: Semantic segmentation")
            semantic_map = self.segmentation_provider.segment(image)
            result.stage_completed = ProcessingStage.SEGMENTATION
            
            if self.visualizer:
                self.visualizer.save_segmentation_mask(semantic_map, filename)
            
            # Step 3: Skew detection
            self.logger.debug(f"Processing {filename}: Skew detection")
            skew_angle, confidence = self.skew_detector.detect_skew(image, semantic_map)
            result.skew_angle = skew_angle
            result.skew_confidence = confidence
            result.stage_completed = ProcessingStage.SKEW_DETECTION
            
            # Get lines for visualization
            lines = self.skew_detector.get_vertical_lines(image, semantic_map)
            result.num_lines_detected = len(lines) if lines is not None else 0
            
            if self.visualizer and lines is not None:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                self.visualizer.save_edge_detection(cv_image, filename)
                self.visualizer.save_line_detection(cv_image, lines, filename, has_background=False)
            
            if confidence == 0:
                raise NoLinesDetectedError(
                    "No valid lines detected for skew estimation",
                    filename=filename,
                    stage="skew_detection"
                )
            
            # Step 4: Image correction
            if abs(skew_angle) > 0.1:
                self.logger.debug(f"Processing {filename}: Image correction (angle={skew_angle:.2f}°)")
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                corrected_image = self.corrector.correct(cv_image, skew_angle)
                
                if self.visualizer:
                    corrected_filename = self.visualizer.save_corrected_image(corrected_image, filename)
                    result.corrected_filename = corrected_filename
                
                # Re-segment corrected image
                corrected_pil = Image.fromarray(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
                corrected_semantic_map = self.segmentation_provider.segment(corrected_pil)
                result.correction_applied = True
            else:
                corrected_pil = image
                corrected_semantic_map = semantic_map
            
            result.stage_completed = ProcessingStage.IMAGE_CORRECTION
            
            # Step 5: Road slope analysis
            self.logger.debug(f"Processing {filename}: Road slope analysis")
            slope, intercept, angle, road_area, road_mask, edge_points = \
                self.slope_analyzer.analyze_with_details(corrected_semantic_map)
            
            result.road_slope = slope
            result.road_intercept = intercept
            result.road_angle = angle
            result.road_area = road_area
            result.stage_completed = ProcessingStage.SLOPE_ESTIMATION
            
            if road_area == 0:
                raise NoRoadDetectedError(
                    "No road detected in the image",
                    filename=filename,
                    stage="slope_estimation"
                )
            
            if slope == -999.0:
                raise ProcessingError(
                    "Slope estimation failed",
                    filename=filename,
                    stage="slope_estimation"
                )
            
            # Save visualizations
            if self.visualizer:
                if road_mask is not None:
                    self.visualizer.save_road_mask(road_mask, filename)
                
                if edge_points is not None:
                    corrected_cv_image = cv2.cvtColor(np.array(corrected_pil), cv2.COLOR_RGB2BGR)
                    self.visualizer.save_road_edge_fitting_visualization(
                        corrected_cv_image, road_mask, edge_points, slope, intercept, filename
                    )
                
                self.visualizer.save_comprehensive_visualization(
                    image, corrected_pil, result, corrected_semantic_map, road_mask
                )
            
            # Mark as complete
            result.stage_completed = ProcessingStage.COMPLETE
            result.status = ProcessingStatus.SUCCESS
            self.logger.info(f"Successfully processed {filename}")
            
        except NoLinesDetectedError as e:
            self.logger.warning(f"No lines detected for {filename}")
            result.status = ProcessingStatus.NO_LINES
            result.error_message = str(e)
            
        except NoRoadDetectedError as e:
            self.logger.warning(f"No road detected in {filename}")
            result.status = ProcessingStatus.NO_ROAD
            result.error_message = str(e)
            
        except ProcessingError as e:
            self.logger.error(f"Processing error for {filename} at stage {e.stage}: {e}")
            if e.stage == "skew_detection":
                result.status = ProcessingStatus.SKEW_FAILED
            elif e.stage == "correction":
                result.status = ProcessingStatus.CORRECTION_FAILED
            elif e.stage == "slope_estimation":
                result.status = ProcessingStatus.SLOPE_FAILED
            else:
                result.status = ProcessingStatus.ERROR
            result.error_message = str(e)
            
        except Exception as e:
            self.logger.error(f"Unexpected error processing {filename}: {e}", exc_info=True)
            result.status = ProcessingStatus.ERROR
            result.error_message = str(e)
        
        return result
    
    def _load_image(self, path: Path) -> Image.Image:
        """
        Load image from path.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image
            
        Raises:
            ImageLoadError: If image cannot be loaded
        """
        try:
            image = Image.open(path)
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ImageLoadError(
                f"Failed to load image: {e}",
                filename=path.name,
                stage="initialization"
            )
    
    @staticmethod
    def _extract_pano_id(filename: str) -> str:
        """
        Extract panorama ID from filename.
        
        Args:
            filename: Image filename
            
        Returns:
            Panorama ID
        """
        return filename.split('_Direction_')[0] if '_Direction_' in filename else filename.rsplit('.', 1)[0]
