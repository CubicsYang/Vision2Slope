"""
Panorama to perspective transformation module for Vision2Slope pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional
from zensvi.transform import ImageTransformer
import glob


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
            generate_left_right: If True, generate right (90°) and left (270°) views
            
        Returns:
            List of paths to generated perspective images
        """
        self.logger.info(f"Transforming panoramic images from {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        self.logger.warning("generate_left_right=False: transforming all images as-is")
        transformer = ImageTransformer(
                dir_input=input_dir,
                dir_output=output_dir
            )
        print(output_dir)
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
        
        # Delete redundant perspective images with 'Direction_0' and 'Direction_180'
        perspective_dir = Path(output_dir) / "perspective"
        deleted_count = 0

        # Search for Direction_0 and Direction_180 images
        for direction in ['Direction_0', 'Direction_180']:
            pattern = str(perspective_dir / "**" / f"*{direction}*.png")
            matching_files = glob.glob(pattern, recursive=True)
            
            for file_path in matching_files:
                try:
                    Path(file_path).unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {file_path}: {e}")

        if deleted_count > 0:
            print(f"\n✓ Cleaned up {deleted_count} redundant perspective images")
        else:
            print("\nℹ No redundant images found to delete")
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