"""
Example usage of Vision2Slope pipeline with panoramic images.
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


def example_panorama_basic():
    """Basic usage with panoramic images."""
    config = PipelineConfig(
        input_dir="examples/input_panorama",
        output_dir="examples/output_panorama",
        processing_config=ProcessingConfig(
            is_panorama=True,  # Enable panorama preprocessing
            log_level="INFO"
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()
    
    print(f"Processed {len(results_df)} perspective views from panoramic images")
    print(f"Successful: {len(results_df[results_df['status'] == 'success'])}")


def example_panorama_custom_parameters():
    """Custom panorama transformation parameters."""
    config = PipelineConfig(
        input_dir="examples/input_panorama",
        output_dir="examples/output_panorama",
        processing_config=ProcessingConfig(
            is_panorama=True,
            panorama_fov=90.0,  # Field of view
            panorama_phi=0.0,   # Vertical angle
            panorama_aspects=(10, 10),  # Aspect ratio
            panorama_show_size=100,  # Scale factor
            log_level="DEBUG"
        ),
        viz_config=VisualizationConfig(
            save_visualizations=True,
            save_corrected_images=True,
            save_intermediate_results=True,
            save_segmentation_masks=True,
            save_road_edge_fitting=True
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()


def example_panorama_multiprocessing():
    """Process panoramic images with multiprocessing."""
    config = PipelineConfig(
        input_dir="examples/input_panorama",
        output_dir="examples/output_panorama",
        processing_config=ProcessingConfig(
            is_panorama=True,
            num_workers=4,
            use_multiprocessing=True,
            log_level="INFO"
        ),
        viz_config=VisualizationConfig(
            save_visualizations=False,  # Minimal visualization for speed
            save_corrected_images=True
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch_parallel()
    
    print(f"Processed {len(results_df)} images using multiprocessing")


def example_mixed_input():
    """
    Process mixed input (both panoramic and regular perspective images).
    
    Note: Use is_panorama=False for regular perspective images.
    If you have mixed inputs, process them separately.
    """
    # Process panoramic images
    pano_config = PipelineConfig(
        input_dir="examples/input_panorama",
        output_dir="examples/output_mixed/panorama",
        processing_config=ProcessingConfig(
            is_panorama=True,
            log_level="INFO"
        )
    )
    
    pano_pipeline = Vision2SlopePipeline(pano_config)
    pano_results = pano_pipeline.process_batch()
    
    # Process regular perspective images
    regular_config = PipelineConfig(
        input_dir="examples/input_regular",
        output_dir="examples/output_mixed/regular",
        processing_config=ProcessingConfig(
            is_panorama=False,  # Regular images
            log_level="INFO"
        )
    )
    
    regular_pipeline = Vision2SlopePipeline(regular_config)
    regular_results = regular_pipeline.process_batch()
    
    print(f"Panoramic: {len(pano_results)} results")
    print(f"Regular: {len(regular_results)} results")
    print(f"Total: {len(pano_results) + len(regular_results)} results")


if __name__ == "__main__":
    # Run the basic panorama example
    print("Running basic panorama example...")
    example_panorama_basic()
    
    # Uncomment to run other examples:
    # example_panorama_custom_parameters()
    # example_panorama_multiprocessing()
    # example_mixed_input()
