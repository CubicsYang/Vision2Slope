"""
Vision2Slope pipeline with clean architecture and parallel processing support.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig
from .core.types import ProcessingResult
from .core.exceptions import ConfigurationError
from .processor import StandardImageProcessor
from .models import SegmentationModel
from .detectors import SkewDetector  
from .correctors import ImageCorrector
from .analyzers import RoadSlopeAnalyzer
from .visualizers import Visualizer
from .utils import Utils
from .pano2perspective import PanoramaTransformer


class LegacyConfig:
    """Flat config object expected by the detection, correction and analysis components."""


def _build_legacy_config(config: PipelineConfig) -> LegacyConfig:
    """
    Create a legacy config object for compatibility with existing components.

    This is a temporary adapter until all components are refactored.
    Defined at module level so it can be pickled for worker processes.
    """
    cfg = LegacyConfig()

    # Detection parameters
    cfg.canny_threshold1 = config.detection_config.canny_threshold1
    cfg.canny_threshold2 = config.detection_config.canny_threshold2
    cfg.hough_threshold = config.detection_config.hough_threshold
    cfg.min_line_length = config.detection_config.min_line_length
    cfg.max_line_gap = config.detection_config.max_line_gap
    cfg.angle_tolerance = config.detection_config.angle_tolerance

    # Analysis parameters
    cfg.min_edge_points = config.analysis_config.min_edge_points
    cfg.use_weighted_average = config.analysis_config.use_weighted_average
    cfg.morphology_kernel_size = config.analysis_config.morphology_kernel_size
    cfg.ransac_residual_threshold = config.analysis_config.ransac_residual_threshold
    cfg.ransac_max_trials = config.analysis_config.ransac_max_trials
    cfg.ransac_random_state = config.analysis_config.ransac_random_state

    return cfg


# Processor owned by each worker process of process_batch_parallel
_worker_processor: Optional[StandardImageProcessor] = None


def _init_worker(config: PipelineConfig):
    """Pool initializer: build one processor (and load the model once) per worker."""
    global _worker_processor

    legacy_config = _build_legacy_config(config)

    # No visualizer in workers to avoid I/O contention
    _worker_processor = StandardImageProcessor(
        segmentation_provider=SegmentationModel(
            config.model_config.model_name,
            device=config.model_config.device,
            cache_dir=config.model_config.cache_dir
        ),
        skew_detector=SkewDetector(legacy_config),
        corrector=ImageCorrector(legacy_config),
        slope_analyzer=RoadSlopeAnalyzer(legacy_config),
        visualizer=None,
        logger=logging.getLogger('vision2slope.worker')
    )


def _process_image_worker(image_path: str) -> ProcessingResult:
    """Worker function for multiprocessing (must be module-level to be picklable)."""
    return _worker_processor.process(image_path)


class Vision2SlopePipeline:
    """
    Vision2Slope pipeline with clean architecture.
    
    Features:
    - Hierarchical configuration management
    - Dependency injection for testability
    - Optimized parallel processing
    - Comprehensive error handling
    - Modular component design
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize Vision2Slope pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.output_path = Path(config.output_dir)
        self.logger = self._setup_logging()
        
        # Handle panorama transformation if needed
        if self.config.processing_config.is_panorama:
            self._prepare_panorama_images()
        
        self.processor = self._create_processor()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file_path = self.output_path / 'pipeline.log'
        
        # Close and remove handlers left by a previous pipeline (avoids leaking log files)
        logger = logging.getLogger('vision2slope')
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []
        logger.setLevel(getattr(logging, self.config.processing_config.log_level))
        
        # File handler
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.processing_config.log_level))
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _prepare_panorama_images(self):
        """
        Convert panoramic images to perspective views if needed.
        
        This method will:
        1. Create a subdirectory for perspective views
        2. Transform panoramic images to right (90°) and left (270°) views
        3. Update input_dir to point to the perspective views directory
        """
        self.logger.info("=" * 60)
        self.logger.info("PANORAMA PREPROCESSING")
        self.logger.info("=" * 60)
        
        # Create output directory for perspective views
        perspective_dir = self.output_path / self.config.processing_config.panorama_output_dir
        perspective_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Input directory (panoramic images): {self.config.input_dir}")
        self.logger.info(f"Output directory (perspective views): {perspective_dir}")
        
        # Initialize panorama transformer
        panorama_transformer = PanoramaTransformer(self.config.processing_config)
        
        try:
            # Transform panoramic images to perspective views
            generated_files = panorama_transformer.transform_panorama(
                input_dir=self.config.input_dir,
                output_dir=str(perspective_dir),
                generate_left_right=True  # Generate right (90°) and left (270°) views
            )
            
            self.logger.info(f"Generated {len(generated_files)} perspective views")
            self.logger.info("Panorama preprocessing completed successfully")
            self.logger.info("=" * 60)
            
            # Update config to process perspective views instead of originals
            # PanoramaTransformer writes views into nested subdirectories
            # (perspective/, mirroring input subfolders); find where they are
            actual_perspective_dir = self._find_actual_perspective_dir(perspective_dir)
            self.config.input_dir = str(actual_perspective_dir)
            self.logger.info(f"Updated input directory to: {self.config.input_dir}")
            
        except Exception as e:
            self.logger.error(f"Panorama preprocessing failed: {e}")
            raise ConfigurationError(f"Failed to process panoramic images: {e}")
    
    def _find_actual_perspective_dir(self, base_dir: Path) -> Path:
        """
        Find the actual directory containing perspective images.
        
        PanoramaTransformer writes views under a 'perspective/' subdirectory
        (mirroring input subfolders), so search for the actual location.
        
        Args:
            base_dir: Base directory to search from
            
        Returns:
            Path to directory containing perspective images
        """
        # Search for directories containing perspective images
        for root, dirs, files in os.walk(base_dir):
            # Check if this directory contains image files
            image_files = [f for f in files if Path(f).suffix.lower() in self.config.processing_config.image_extensions]
            if image_files:
                found_dir = Path(root)
                self.logger.debug(f"Found {len(image_files)} perspective images in: {found_dir}")
                return found_dir
        
        # If no images found, return the original directory
        self.logger.warning(f"No perspective images found in subdirectories, using: {base_dir}")
        return base_dir
    
    def _create_processor(self) -> StandardImageProcessor:
        """
        Create image processor with all components.
        
        Returns:
            Configured StandardImageProcessor
        """
        self.logger.info("Initializing pipeline components...")
        
        # Create components (adapting old classes to new interfaces)
        segmentation_provider = SegmentationModel(
            self.config.model_config.model_name,
            device=self.config.model_config.device,
            cache_dir=self.config.model_config.cache_dir
        )
        
        # Create legacy-compatible config for old components
        legacy_config = _build_legacy_config(self.config)

        skew_detector = SkewDetector(legacy_config)
        corrector = ImageCorrector(legacy_config)
        slope_analyzer = RoadSlopeAnalyzer(legacy_config)
        visualizer = Visualizer(self.config.viz_config, self.config.output_dir)

        processor = StandardImageProcessor(
            segmentation_provider=segmentation_provider,
            skew_detector=skew_detector,
            corrector=corrector,
            slope_analyzer=slope_analyzer,
            visualizer=visualizer,
            logger=self.logger
        )

        self.logger.info("Pipeline components initialized successfully")
        return processor

    def process_batch(self) -> pd.DataFrame:
        """
        Process all images in the input directory.

        Delegates to process_batch_parallel when
        processing_config.use_multiprocessing is True.

        Returns:
            DataFrame of successful views kept after angle filtering, with
            per-panorama slope estimates (empty if none). Results for all
            images are saved to vision2slope_results_<timestamp>.csv.
        """
        if self.config.processing_config.use_multiprocessing:
            return self.process_batch_parallel()

        image_files = self._find_image_files()
        
        if not image_files:
            self.logger.warning(f"No image files found in {self.config.input_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process all images
        results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            result = self.processor.process(str(image_file))
            results.append(result) 

        # Convert results to DataFrame and save
        df = self._results_to_dataframe(results)

        csv_path = self._save_results(df)
        
        # Post-process for bi-directional slope estimation
        bi_estimate_df = self._bi_slope_estimate(df, csv_path)
        
        # Save intermediate results if requested
        if self.config.viz_config.save_intermediate_results:
            self._save_intermediate_results(df)
        
        # Print summary
        self._print_summary(df, bi_estimate_df)
        
        return bi_estimate_df
    
    def process_batch_parallel(self, num_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Process all images using multiprocessing.
        
        Args:
            num_workers: Number of worker processes (None = use config)
            
        Returns:
            DataFrame of successful views kept after angle filtering, with
            per-panorama slope estimates (empty if none). Results for all
            images are saved to vision2slope_results_<timestamp>.csv.
        """
        if num_workers is None:
            num_workers = self.config.processing_config.num_workers
        if num_workers <= 0:
            num_workers = cpu_count()

        image_files = self._find_image_files()
        
        if not image_files:
            self.logger.warning(f"No image files found in {self.config.input_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(image_files)} images to process")

        # Each worker loads its own model once, so don't start more than needed
        num_workers = min(num_workers, len(image_files))
        self.logger.info(f"Using {num_workers} worker processes")

        # Process in parallel
        with Pool(processes=num_workers, initializer=_init_worker, initargs=(self.config,)) as pool:
            results = list(tqdm(
                pool.imap(_process_image_worker, [str(f) for f in image_files]),
                total=len(image_files),
                desc="Processing images (parallel)"
            ))
        
        # Convert results to DataFrame and save
        df = self._results_to_dataframe(results)
        csv_path = self._save_results(df)
        
        # Post-process
        bi_estimate_df = self._bi_slope_estimate(df, csv_path)
        
        if self.config.viz_config.save_intermediate_results:
            self._save_intermediate_results(df)
        
        self._print_summary(df, bi_estimate_df)

        return bi_estimate_df

    def _find_image_files(self) -> List[Path]:
        """Find all image files in input directory."""
        input_path = Path(self.config.input_dir)
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in self.config.processing_config.image_extensions
        ]
        return sorted(image_files)
    
    def _results_to_dataframe(self, results: List[ProcessingResult]) -> pd.DataFrame:
        """Convert list of ProcessingResult to DataFrame."""
        results_data = [result.to_dict() for result in results]
        return pd.DataFrame(results_data)
    
    def _save_results(self, df: pd.DataFrame) -> Path:
        """Save results DataFrame to CSV."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_filename = f"vision2slope_results_{timestamp}.csv"
        csv_path = self.output_path / csv_filename
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Results saved to: {csv_path}")
        return csv_path
    
    def _bi_slope_estimate(self, df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        """
        Estimate slope using bi-directional approach.

        If metadata_csv is provided (with heading and edge_bearing columns),
        computes signed slope along the OSM edge (u→v) direction.
        Otherwise falls back to absolute-value estimation.

        Args:
            df: DataFrame with results
            csv_path: Path where original CSV was saved

        Returns:
            DataFrame with adjusted slopes
        """
        filtered_df = df[df['status'] == 'success'].copy()

        if filtered_df.empty:
            self.logger.warning("No successful results for bi-directional slope estimation")
            return filtered_df

        # Extract pano_id and perspective_angle from filename
        filtered_df['pano_id'] = filtered_df['filename'].apply(Utils.get_pano_id_from_path)
        filtered_df['perspective_angle'] = filtered_df['filename'].apply(
            lambda x: Utils.get_perspective_angle_from_path(x)
            if '_Direction_' in x else 0.0
        )

        # Filter by angle threshold
        angle_threshold = self.config.analysis_config.filter_slope_angle
        filtered_df = filtered_df[filtered_df['road_edge_line_angle'].abs() <= angle_threshold]

        if filtered_df.empty:
            self.logger.warning("No valid segments after angle filtering")
            return filtered_df

        self.logger.info(f"Found {len(filtered_df)} valid segments for bi-directional estimation")

        # Check if metadata is available for signed slope estimation
        metadata_csv_path = self.config.processing_config.metadata_csv
        use_signed = False

        if metadata_csv_path is not None:
            metadata_df = pd.read_csv(metadata_csv_path)
            metadata_df['pano_id'] = metadata_df['pano_id'].astype(str)
            metadata_df = metadata_df.drop_duplicates(subset='pano_id', keep='first')

            required_cols = {'pano_id', 'heading', 'edge_bearing'}
            if required_cols.issubset(metadata_df.columns):
                # Older GSVDownloader versions wrote streetlevel's heading in radians
                headings = metadata_df['heading'].dropna()
                if len(headings) > 1 and headings.abs().max() <= 2 * np.pi:
                    self.logger.warning(
                        "All headings in metadata_csv are within [0, 2π] and look like radians; "
                        "heading must be in degrees (north=0, clockwise), otherwise slope signs are wrong"
                    )
                filtered_df = filtered_df.merge(
                    metadata_df[['pano_id', 'heading', 'edge_bearing']],
                    on='pano_id',
                    how='left'
                )
                has_metadata = filtered_df['heading'].notna() & filtered_df['edge_bearing'].notna()

                if has_metadata.any():
                    use_signed = True
                    self.logger.info("Using signed slope estimation with heading and edge_bearing metadata")
                else:
                    self.logger.warning("No matching metadata found for any pano_id, falling back to absolute slope")
            else:
                missing = required_cols - set(metadata_df.columns)
                self.logger.warning(f"metadata_csv missing columns {missing}, falling back to absolute slope")

        if use_signed:
            # Compute signed slope for each row
            filtered_df['signed_slope'] = filtered_df.apply(
                lambda row: Utils.compute_signed_slope(
                    row['road_edge_line_angle'],
                    row['perspective_angle'],
                    row['heading'],
                    row['edge_bearing']
                ) if pd.notna(row.get('heading')) and pd.notna(row.get('edge_bearing'))
                else np.nan,
                axis=1
            )

            # Aggregate signed slope per pano_id
            estimate_slope_list = {}
            grouped = filtered_df.groupby('pano_id')

            for pano_id, group in tqdm(grouped, desc="Computing signed road slopes"):
                valid = group['signed_slope'].dropna()
                if valid.empty:
                    continue
                if self.config.analysis_config.use_weighted_average:
                    valid_mask = group['signed_slope'].notna()
                    slope_per_pano = np.average(
                        group.loc[valid_mask, 'signed_slope'],
                        weights=group.loc[valid_mask, 'road_area']
                    )
                else:
                    slope_per_pano = valid.mean()
                estimate_slope_list[pano_id] = slope_per_pano

            filtered_df['road_estimated_slope'] = filtered_df['pano_id'].map(estimate_slope_list)
            filtered_df['road_estimated_slope_abs'] = filtered_df['road_estimated_slope'].abs()
        else:
            # Original absolute-value behavior
            estimate_slope_list = {}
            grouped = filtered_df.groupby('pano_id')

            for pano_id, group in tqdm(grouped, desc="Computing road slopes"):
                if self.config.analysis_config.use_weighted_average:
                    slope_per_pano = np.average(
                        np.abs(group['road_edge_line_angle']),
                        weights=group['road_area']
                    )
                else:
                    slope_per_pano = np.abs(group['road_edge_line_angle']).mean()
                estimate_slope_list[pano_id] = slope_per_pano

            filtered_df['road_estimated_slope'] = np.abs(filtered_df['pano_id'].map(estimate_slope_list))

        # Save estimated results
        estimate_csv_path = csv_path.parent / csv_path.name.replace('.csv', '_estimate.csv')
        filtered_df.to_csv(estimate_csv_path, index=False)
        self.logger.info(f"Estimated results saved to: {estimate_csv_path}")

        return filtered_df
    
    def _save_intermediate_results(self, df: pd.DataFrame):
        """Save intermediate processing results."""
        intermediate_dir = self.output_path / "intermediate_results"
        intermediate_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Skew detection results
        skew_df = df[['filename', 'pano_id', 'skew_angle', 'skew_confidence', 'num_lines_detected']].copy()
        skew_path = intermediate_dir / f"skew_detection_{timestamp}.csv"
        skew_df.to_csv(skew_path, index=False)
        
        # Slope estimation results
        slope_df = df[['filename', 'pano_id', 'road_edge_line_slope', 'road_edge_line_intercept', 
                      'road_edge_line_angle', 'road_area']].copy()
        slope_path = intermediate_dir / f"slope_estimation_{timestamp}.csv"
        slope_df.to_csv(slope_path, index=False)
        
        self.logger.info(f"Intermediate results saved to: {intermediate_dir}")
    
    def _print_summary(self, df: pd.DataFrame, estimate_df: pd.DataFrame):
        """
        Print processing summary statistics.

        Args:
            df: Per-image results for all processed images
            estimate_df: Filtered and aggregated rows from _bi_slope_estimate
        """
        if df is None or df.empty:
            self.logger.warning("No data available for summary")
            return
        
        total = len(df)
        status_counts = df['status'].value_counts()
        
        self.logger.info("=" * 60)
        self.logger.info("VISION2SLOPE PROCESSING SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total images: {total}")
        
        for status, count in status_counts.items():
            percentage = (count / total * 100) if total > 0 else 0
            self.logger.info(f"  {status}: {count} ({percentage:.1f}%)")
        
        # Statistics for successful results
        successful_df = df[df['status'] == 'success']
        if len(successful_df) > 0:
            self.logger.info("\nSuccessful Results Statistics:")
            self.logger.info(f"  Avg skew angle: {successful_df['skew_angle'].mean():.2f}°")
            self.logger.info(f"  Avg road edge line slope: {successful_df['road_edge_line_slope'].mean():.4f}")
            self.logger.info(f"  Avg road edge line angle: {successful_df['road_edge_line_angle'].mean():.2f}°")
            self.logger.info(f"  Avg road area: {successful_df['road_area'].mean():.0f} pixels")

        # Statistics for the filtered, per-panorama estimates
        if 'road_estimated_slope' in estimate_df.columns:
            per_pano = estimate_df.drop_duplicates(subset='pano_id')['road_estimated_slope'].dropna()
            self.logger.info("\nSlope Estimates:")
            self.logger.info(f"  Views kept after angle filtering: {len(estimate_df)}")
            self.logger.info(f"  Panoramas with an estimate: {len(per_pano)}")
            if len(per_pano) > 0:
                self.logger.info(f"  Avg |road estimated slope| per panorama: {per_pano.abs().mean():.2f}°")
        else:
            self.logger.info("\nSlope Estimates: none (no views passed processing and angle filtering)")

        self.logger.info("=" * 60)
