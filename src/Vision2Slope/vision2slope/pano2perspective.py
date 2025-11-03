"""
Panorama to perspective transformation module for Vision2Slope pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional
from zensvi.transform import ImageTransformer


class PanoramaTransformer:
    """Class for converting panoramic images to perspective views."""
    
    def __init__(self, config=None):
        """
        Initialize panorama transformer.
        
        Args:
            config: Optional configuration object with transformation parameters
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Default transformation parameters
        self.fov = getattr(config, 'panorama_fov', 90)
        self.phi = getattr(config, 'panorama_phi', 0)
        self.aspects = getattr(config, 'panorama_aspects', (10, 10))
        self.show_size = getattr(config, 'panorama_show_size', 100)
    
    def transform_panorama(
        self, 
        input_dir: str, 
        output_dir: str,
        generate_left_right: bool = True
    ) -> List[Path]:
        """
        Transform panoramic images to perspective views.
        
        Args:
            input_dir: Input directory containing panoramic images
            output_dir: Output directory for perspective images
            generate_left_right: If True, generate left (90°) and right (270°) views
            
        Returns:
            List of paths to generated perspective images
        """
        self.logger.info(f"Transforming panoramic images from {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        if generate_left_right:
            # Generate left view (90°)
            self.logger.info("Generating left view (Direction 90°)...")
            left_transformer = ImageTransformer(
                dir_input=input_dir,
                dir_output=output_dir
            )
            left_transformer.transform_images(
                style_list="perspective",
                FOV=self.fov,
                theta=90,  # Left view
                phi=self.phi,
                aspects=self.aspects,
                show_size=self.show_size
            )
            
            # Generate right view (270°)
            self.logger.info("Generating right view (Direction 270°)...")
            right_transformer = ImageTransformer(
                dir_input=input_dir,
                dir_output=output_dir
            )
            right_transformer.transform_images(
                style_list="perspective",
                FOV=self.fov,
                theta=270,  # Right view
                phi=self.phi,
                aspects=self.aspects,
                show_size=self.show_size
            )
            
            # Collect generated files
            for file in output_path.iterdir():
                if file.is_file() and ('Direction_90' in file.name or 'Direction_270' in file.name):
                    generated_files.append(file)
            
            self.logger.info(f"Generated {len(generated_files)} perspective views")
        else:
            # Generate all standard directions if not specifically left/right
            self.logger.warning("generate_left_right=False: transforming all images as-is")
            transformer = ImageTransformer(
                dir_input=input_dir,
                dir_output=output_dir
            )
            transformer.transform_images(
                style_list="perspective",
                FOV=self.fov,
                theta=90,  # Default view
                phi=self.phi,
                aspects=self.aspects,
                show_size=self.show_size
            )
            
            for file in output_path.iterdir():
                if file.is_file():
                    generated_files.append(file)
        
        return generated_files
    
    def is_panoramic_image(self, image_path: str) -> bool:
        """
        Check if an image is panoramic based on aspect ratio.
        
        A typical panoramic image has aspect ratio close to 2:1.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image appears to be panoramic
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            aspect_ratio = width / height
            
            # Panoramic images typically have aspect ratio between 1.8 and 2.2
            is_panoramic = 1.8 <= aspect_ratio <= 2.2
            
            if is_panoramic:
                self.logger.debug(f"{image_path}: Detected as panoramic (aspect ratio: {aspect_ratio:.2f})")
            
            return is_panoramic
            
        except Exception as e:
            self.logger.error(f"Failed to check if image is panoramic: {e}")
            return False