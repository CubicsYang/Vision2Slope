"""
Example usage of Vision2Slope pipeline with panoramic images.
"""

from vision2slope import (
    PipelineConfig,
    VisualizationConfig,
    ProcessingConfig,
    Vision2SlopePipeline
)


def example_panorama_basic():
    """Basic usage with panoramic images."""
    config = PipelineConfig(
        input_dir="examples/input_pano",
        output_dir="examples/output_pano",
        processing_config=ProcessingConfig(
            is_panorama=True,  # Enable panorama preprocessing
            panorama_fov=90,  # Field of view
            panorama_phi=0.0,   # Vertical angle
            panorama_aspects=(10, 10),  # Aspect ratio
            panorama_show_size=100  # Scale factor
        )
    )
    
    pipeline = Vision2SlopePipeline(config)
    results_df = pipeline.process_batch()
    
    print(f"Perspective views with slope estimates: {len(results_df)}")


if __name__ == "__main__":
    # Run the basic panorama example
    print("Running basic panorama example...")
    example_panorama_basic()
