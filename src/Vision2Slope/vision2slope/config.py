"""
Configuration system for Vision2Slope.

Provides a hierarchical configuration structure for better organization:
- ModelConfig: Model-related settings
- DetectionConfig: Skew and line detection parameters
- AnalysisConfig: Road slope analysis parameters
- VisualizationConfig: Visualization output settings
- PipelineConfig: Top-level pipeline configuration
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .core.exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for segmentation model."""
    
    # Model selection
    model_name: str = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
    
    # Device configuration
    device: Optional[str] = None  # None = auto-detect (cuda if available)
    
    # Model loading options
    cache_dir: Optional[str] = None
    
    def validate(self):
        """Validate model configuration."""
        if not self.model_name:
            raise ConfigurationError("model_name cannot be empty")


@dataclass
class DetectionConfig:
    """Configuration for skew and line detection."""
    
    # Canny edge detection parameters
    canny_threshold1: float = 50.0
    canny_threshold2: float = 150.0
    
    # Hough line detection parameters
    hough_threshold: int = 50
    min_line_length: int = 50
    max_line_gap: int = 10
    
    # Angle filtering
    angle_tolerance: int = 10  # degrees
    
    def validate(self):
        """Validate detection configuration."""
        if self.canny_threshold1 <= 0 or self.canny_threshold2 <= 0:
            raise ConfigurationError("Canny thresholds must be positive")
        if self.canny_threshold1 >= self.canny_threshold2:
            raise ConfigurationError("canny_threshold1 must be less than canny_threshold2")
        if self.hough_threshold <= 0:
            raise ConfigurationError("hough_threshold must be positive")
        if self.min_line_length <= 0:
            raise ConfigurationError("min_line_length must be positive")
        if self.angle_tolerance < 0 or self.angle_tolerance > 90:
            raise ConfigurationError("angle_tolerance must be between 0 and 90 degrees")


@dataclass
class AnalysisConfig:
    """Configuration for road slope analysis."""
    
    # Road mask parameters
    morphology_kernel_size: int = 15
    
    # Edge extraction parameters
    min_edge_points: int = 10
    
    # RANSAC parameters
    ransac_residual_threshold: float = 1.0
    ransac_max_trials: int = 1000
    ransac_random_state: int = 42
    
    # Bi-directional slope estimation
    use_weighted_average: bool = True
    filter_slope_angle: int = 10  # degrees
    
    def validate(self):
        """Validate analysis configuration."""
        if self.morphology_kernel_size <= 0:
            raise ConfigurationError("morphology_kernel_size must be positive")
        if self.min_edge_points <= 0:
            raise ConfigurationError("min_edge_points must be positive")
        if self.ransac_residual_threshold <= 0:
            raise ConfigurationError("ransac_residual_threshold must be positive")
        if self.ransac_max_trials <= 0:
            raise ConfigurationError("ransac_max_trials must be positive")


@dataclass
class VisualizationConfig:
    """Configuration for visualization outputs."""
    
    # Main visualization toggles
    save_visualizations: bool = True
    save_corrected_images: bool = True
    save_intermediate_results: bool = True
    
    # Step-specific visualizations
    save_segmentation_masks: bool = False
    save_road_masks: bool = False
    save_edge_detection: bool = False
    save_line_detection: bool = False
    save_skew_visualization: bool = False
    save_slope_visualization: bool = True
    save_ppht_results: bool = False
    save_road_edge_fitting: bool = False
    
    # Visualization parameters
    overlay_alpha: float = 0.7
    figure_dpi: int = 150
    figure_size: tuple = (15, 12)
    
    # Output subdirectory names
    viz_dir_name: str = "visualizations"
    corrected_dir_name: str = "corrected_images"
    intermediate_dir_name: str = "intermediate_results"
    masks_dir_name: str = "masks"
    edges_dir_name: str = "edge_detection"
    lines_dir_name: str = "line_detection"
    ppht_dir_name: str = "ppht_results"
    road_edge_fitting_dir_name: str = "road_edge_fitting"
    
    def validate(self):
        """Validate visualization configuration."""
        if not 0 <= self.overlay_alpha <= 1:
            raise ConfigurationError("overlay_alpha must be between 0 and 1")
        if self.figure_dpi <= 0:
            raise ConfigurationError("figure_dpi must be positive")
        if len(self.figure_size) != 2 or any(s <= 0 for s in self.figure_size):
            raise ConfigurationError("figure_size must be a tuple of two positive numbers")


@dataclass
class ProcessingConfig:
    """Configuration for processing options."""
    
    # Logging
    log_level: str = "INFO"
    
    # Parallel processing
    num_workers: int = 1
    use_multiprocessing: bool = False
    
    # Batch processing
    batch_size: int = 1
    
    # Image file extensions
    image_extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Panorama processing
    is_panorama: bool = False  # Whether input images are panoramic
    panorama_output_dir: str = "panorama_perspectives"  # Subdirectory for perspective views
    panorama_fov: float = 90.0  # Field of view for perspective projection
    panorama_phi: float = 0.0  # Vertical angle
    panorama_aspects: tuple = (10, 10)  # Aspect ratio
    panorama_show_size: int = 100  # Scale factor
    
    def validate(self):
        """Validate processing configuration."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")
        if self.num_workers < 0:
            raise ConfigurationError("num_workers must be non-negative")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        if self.panorama_fov <= 0 or self.panorama_fov > 180:
            raise ConfigurationError("panorama_fov must be between 0 and 180 degrees")
        if len(self.panorama_aspects) != 2 or any(a <= 0 for a in self.panorama_aspects):
            raise ConfigurationError("panorama_aspects must be a tuple of two positive numbers")


@dataclass
class PipelineConfig:
    """Top-level configuration for the Vision2Slope pipeline."""
    
    # Input/Output paths
    input_dir: str
    output_dir: str
    
    # Sub-configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    detection_config: DetectionConfig = field(default_factory=DetectionConfig)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    viz_config: VisualizationConfig = field(default_factory=VisualizationConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert to Path objects for easier handling
        self.input_dir = str(Path(self.input_dir).expanduser().resolve())
        self.output_dir = str(Path(self.output_dir).expanduser().resolve())
        
        # Validate paths
        if not os.path.exists(self.input_dir):
            raise ConfigurationError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert dict configs to dataclass instances if needed
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        if isinstance(self.detection_config, dict):
            self.detection_config = DetectionConfig(**self.detection_config)
        if isinstance(self.analysis_config, dict):
            self.analysis_config = AnalysisConfig(**self.analysis_config)
        if isinstance(self.viz_config, dict):
            self.viz_config = VisualizationConfig(**self.viz_config)
        if isinstance(self.processing_config, dict):
            self.processing_config = ProcessingConfig(**self.processing_config)
        
        # Validate all sub-configurations
        self.model_config.validate()
        self.detection_config.validate()
        self.analysis_config.validate()
        self.viz_config.validate()
        self.processing_config.validate()
    
    @classmethod
    def from_args(cls, args):
        """
        Create configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            PipelineConfig instance
        """
        # Build sub-configurations
        model_config = ModelConfig(
            model_name=args.model
        )
        
        detection_config = DetectionConfig(
            canny_threshold1=args.canny_low,
            canny_threshold2=args.canny_high,
            hough_threshold=args.hough_threshold,
            min_line_length=args.min_line_length,
            max_line_gap=args.max_line_gap,
            angle_tolerance=args.angle_tolerance
        )
        
        analysis_config = AnalysisConfig(
            min_edge_points=args.min_edge_points,
            use_weighted_average=args.weighted_average,
            filter_slope_angle=args.angle_tolerance
        )
        
        viz_config = VisualizationConfig(
            save_visualizations=not args.no_visualizations,
            save_corrected_images=not args.no_corrected_images,
            save_intermediate_results=not args.no_intermediate,
            save_segmentation_masks=args.save_segmentation_masks,
            save_road_masks=args.save_road_masks,
            save_edge_detection=args.save_edge_detection,
            save_line_detection=args.save_line_detection,
            save_skew_visualization=args.save_skew_visualization,
            save_slope_visualization=args.save_slope_visualization,
            save_ppht_results=getattr(args, 'save_ppht_results', True),
            save_road_edge_fitting=getattr(args, 'save_road_edge_fitting', True)
        )
        
        processing_config = ProcessingConfig(
            log_level=args.log_level,
            num_workers=args.num_workers,
            use_multiprocessing=args.use_multiprocessing,
            is_panorama=getattr(args, 'is_panorama', False),
            panorama_fov=getattr(args, 'panorama_fov', 90.0),
            panorama_phi=getattr(args, 'panorama_phi', 0.0)
        )
        
        return cls(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_config=model_config,
            detection_config=detection_config,
            analysis_config=analysis_config,
            viz_config=viz_config,
            processing_config=processing_config
        )
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'model_config': self.model_config.__dict__,
            'detection_config': self.detection_config.__dict__,
            'analysis_config': self.analysis_config.__dict__,
            'viz_config': self.viz_config.__dict__,
            'processing_config': self.processing_config.__dict__
        }
