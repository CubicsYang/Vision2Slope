"""
Command-line interface for Vision2Slope pipeline.
"""

import argparse
import sys

from .config import (
    PipelineConfig,
    ModelConfig,
    DetectionConfig,
    AnalysisConfig,
    VisualizationConfig,
    ProcessingConfig
)
from .pipeline import Vision2SlopePipeline


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Vision2Slope: Integrated pipeline for road slope analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing images")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for results")
    
    # Model configuration
    parser.add_argument("--model", default="facebook/mask2former-swin-large-mapillary-vistas-semantic",
                       help="Segmentation model name")
    
    # Skew detection parameters
    parser.add_argument("--canny-low", type=float, default=50.0,
                       help="Lower threshold for Canny edge detection")
    parser.add_argument("--canny-high", type=float, default=150.0,
                       help="Upper threshold for Canny edge detection")
    parser.add_argument("--hough-threshold", type=int, default=50,
                       help="Threshold for Hough line detection")
    parser.add_argument("--min-line-length", type=int, default=50,
                       help="Minimum length of line segments")
    parser.add_argument("--max-line-gap", type=int, default=10,
                       help="Maximum gap between line segments")
    parser.add_argument("--angle-tolerance", type=int, default=10,
                       help="Tolerance for angle estimation in degrees")
    
    # Image correction parameters
    parser.add_argument("--weighted-average", action="store_true",
                       help="Use weighted average for angle calculation")
    
    # Road analysis parameters
    parser.add_argument("--min-edge-points", type=int, default=10,
                       help="Minimum number of edge points for line fitting")
    
    # Output options - main toggles
    parser.add_argument("--no-visualizations", action="store_true",
                       help="Skip saving comprehensive visualization images")
    parser.add_argument("--no-corrected-images", action="store_true",
                       help="Skip saving corrected images")
    parser.add_argument("--no-intermediate", action="store_true",
                       help="Skip saving intermediate results")
    
    # Output options - detailed visualizations
    parser.add_argument("--save-segmentation-masks", action="store_true",
                       help="Save semantic segmentation masks")
    parser.add_argument("--save-road-masks", action="store_true",
                       help="Save road masks")
    parser.add_argument("--save-edge-detection", action="store_true",
                       help="Save edge detection visualizations")
    parser.add_argument("--save-line-detection", action="store_true",
                       help="Save line detection visualizations")
    parser.add_argument("--save-skew-visualization", action="store_true",
                       help="Save skew detection visualizations")
    parser.add_argument("--save-slope-visualization", action="store_true",
                       help="Save slope estimation visualizations")
    
    # Processing options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of worker processes for multiprocessing (0 for auto)")
    parser.add_argument("--use-multiprocessing", action="store_true",
                       help="Enable multiprocessing for batch processing")
    
    # Panorama processing options
    parser.add_argument("--is-panorama", action="store_true",
                       help="Input images are panoramic (will generate left/right perspective views)")
    parser.add_argument("--panorama-fov", type=float, default=90.0,
                       help="Field of view for panorama transformation")
    parser.add_argument("--panorama-phi", type=float, default=0.0,
                       help="Vertical angle for panorama transformation")
    
    return parser


def main():
    """Main function for command-line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration from arguments
        config = PipelineConfig.from_args(args)
        
        # Run pipeline
        pipeline = Vision2SlopePipeline(config)
        
        if args.use_multiprocessing:
            results_df = pipeline.process_batch_parallel(args.num_workers)
        else:
            results_df = pipeline.process_batch()
        
        if len(results_df) > 0:
            print(f"\nProcessing completed! {len(results_df)} images processed.")
        else:
            print("No images were processed successfully.")
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
