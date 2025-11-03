"""
Example usage of Vision2Slope pipeline (V2.0).
"""

from vision2slope import (
    PipelineConfig,
    ModelConfig,
    DetectionConfig,
    AnalysisConfig,
    VisualizationConfig,
    ProcessingConfig,
    Vision2SlopePipeline
)


def example_basic_usage():
    """Basic usage example with default settings."""
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output"
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()
    
    print(f"Processed {len(results_df)} images")
    print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")


def example_custom_visualization():
    """Example with custom visualization settings."""
    
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output",
        viz_config=VisualizationConfig(
            save_visualizations=True,
            save_corrected_images=True,
            save_intermediate_results=True,
            save_segmentation_masks=True,
            save_road_masks=True,
            save_edge_detection=True,
            save_line_detection=True,
            overlay_alpha=0.7,
            figure_dpi=150
        ),
        processing_config=ProcessingConfig(
            log_level="DEBUG"
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()


def example_custom_parameters():
    """Example with custom detection parameters."""
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output",
        detection_config=DetectionConfig(
            canny_threshold1=50.0,
            canny_threshold2=150.0,
            hough_threshold=50,
            min_line_length=50,
            max_line_gap=10,
            angle_tolerance=10
        ),
        analysis_config=AnalysisConfig(
            min_edge_points=10,
            use_weighted_average=True
        ),
        processing_config=ProcessingConfig(
            log_level="INFO"
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()


def example_multiprocessing():
    """Example using multiprocessing for faster batch processing."""
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output",
        processing_config=ProcessingConfig(
            num_workers=8,  # Use 8 worker processes
            use_multiprocessing=True
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch_parallel()
    
    print(f"Processed {len(results_df)} images using multiprocessing")


def example_minimal_visualization():
    """Example with minimal visualization output."""
    
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output",
        viz_config=VisualizationConfig(
            save_visualizations=False,  # No comprehensive visualizations
            save_corrected_images=True,  # Only save corrected images
            save_intermediate_results=False,
            save_segmentation_masks=False,
            save_road_masks=False,
            save_edge_detection=False,
            save_line_detection=False,
            save_ppht_results=True,  # Save PPHT results
            save_road_edge_fitting=True  # Save road edge fitting visualization
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()


def example_road_edge_visualization():
    """Example focusing on road edge and RANSAC fitting visualization."""
    
    config = PipelineConfig(
        input_dir="examples/input",
        output_dir="examples/output",
        viz_config=VisualizationConfig(
            save_visualizations=False,  # Disable comprehensive visualization
            save_corrected_images=False,
            save_intermediate_results=False,
            save_segmentation_masks=False,
            save_road_masks=True,  # Keep road masks
            save_edge_detection=False,
            save_line_detection=False,
            save_ppht_results=False,
            save_road_edge_fitting=True,  # Enable road edge fitting visualization
            figure_dpi=200  # Higher quality for detailed analysis
        ),
        processing_config=ProcessingConfig(
            log_level="DEBUG"  # Detailed logging
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()
    
    print(f"\nRoad edge fitting visualizations saved to:")
    print(f"  {config.output_dir}/road_edge_fitting/")



if __name__ == "__main__":
    # Run the basic example
    print("Running basic usage example...")
    example_basic_usage()
    
    # Uncomment to run other examples:
    # example_custom_visualization()
    # example_custom_parameters()
    # example_multiprocessing()
    # example_minimal_visualization()
    # example_road_edge_visualization()
