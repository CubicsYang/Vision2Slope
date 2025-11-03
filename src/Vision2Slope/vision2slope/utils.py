"""
Utility functions for Vision2Slope pipeline.
"""

import os
import re
import numpy as np


class Utils:
    """Utility functions for the Vision2Slope pipeline."""
    
    @staticmethod
    def get_pano_id_from_path(image_path: str) -> str:
        """
        Extract the panorama ID from the image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Panorama ID extracted from the image path
        """
        return str(os.path.basename(image_path).split('_Direction_')[0])
    
    @staticmethod
    def get_perspective_angle_from_path(image_path: str) -> float:
        """
        Extract the perspective angle from the image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Perspective angle extracted from the image path
        """
        match = re.search(r'_Direction_(\d+)_FOV_', os.path.basename(image_path))
        if match:
            return float(match.group(1))
        else:
            raise ValueError(f"Perspective angle not found in image path: {image_path}")

    @staticmethod
    def get_semantic_colormap():
        """
        Returns a dictionary mapping semantic class IDs to RGB colors.
        """
        return {
            0: (165, 42, 42),     # Bird
            1: (0, 192, 0),       # Ground Animal
            2: (196, 196, 196),   # Curb
            3: (190, 153, 153),   # Fence
            4: (180, 165, 180),   # Guard Rail
            5: (102, 102, 156),   # Barrier
            6: (102, 102, 156),   # Wall
            7: (128, 64, 255),    # Bike Lane
            8: (140, 140, 200),   # Crosswalk - Plain
            9: (170, 170, 170),   # Curb Cut
            10: (250, 170, 160),  # Parking
            11: (96, 96, 96),     # Pedestrian Area
            12: (230, 150, 140),  # Rail Track
            13: (128, 64, 128),   # Road
            14: (110, 110, 110),  # Service Lane
            15: (244, 35, 232),   # Sidewalk
            16: (150, 100, 100),  # Bridge
            17: (70, 70, 70),     # Building
            18: (150, 120, 90),   # Tunnel
            19: (220, 20, 60),    # Person
            20: (255, 0, 0),      # Bicyclist
            21: (255, 0, 0),      # Motorcyclist
            22: (255, 0, 0),      # Other Rider
            23: (200, 128, 128),  # Lane Marking - Crosswalk
            24: (255, 255, 255),  # Lane Marking - General
            25: (64, 170, 64),    # Mountain
            26: (230, 160, 50),   # Sand
            27: (70, 130, 180),   # Sky
            28: (190, 255, 255),  # Snow
            29: (152, 251, 152),  # Terrain
            30: (107, 142, 35),   # Vegetation
            31: (0, 170, 30),     # Water
            32: (255, 220, 0),    # Banner
            33: (255, 0, 0),      # Bench
            34: (255, 0, 0),      # Bike Rack
            35: (255, 0, 0),      # Billboard
            36: (255, 0, 0),      # Catch Basin
            37: (255, 0, 0),      # CCTV Camera
            38: (255, 0, 0),      # Fire Hydrant
            39: (255, 0, 0),      # Junction Box
            40: (255, 0, 0),      # Mailbox
            41: (255, 0, 0),      # Manhole
            42: (255, 0, 0),      # Phone Booth
            43: (255, 0, 0),      # Pothole
            44: (255, 0, 0),      # Street Light
            45: (255, 0, 0),      # Pole
            46: (255, 0, 0),      # Traffic Sign Frame
            47: (255, 0, 0),      # Utility Pole
            48: (255, 0, 0),      # Traffic Light
            49: (255, 0, 0),      # Traffic Sign (Back)
            50: (255, 0, 0),      # Traffic Sign (Front)
            51: (255, 0, 0),      # Trash Can
            52: (119, 11, 32),    # Bicycle
            53: (0, 0, 142),      # Boat
            54: (0, 60, 100),     # Bus
            55: (0, 0, 142),      # Car
            56: (0, 0, 90),       # Caravan
            57: (0, 0, 230),      # Motorcycle
            58: (0, 80, 100),     # On Rails
            59: (128, 64, 64),    # Other Vehicle
            60: (0, 0, 110),      # Trailer
            61: (0, 0, 70),       # Truck
            62: (0, 0, 192),      # Wheeled Slow
            63: (32, 32, 32),     # Car Mount
            64: (120, 10, 10),    # Ego Vehicle
        }

    @staticmethod
    def render_semantic_segmentation(semantic_array: np.ndarray) -> np.ndarray:
        """
        Generate RGB rendering from semantic segmentation labels.

        Args:
            semantic_array: (H, W) semantic segmentation array with class IDs
            
        Returns:
            rgb_array: (H, W, 3) RGB image, dtype=uint8
        """
        label_color_map = Utils.get_semantic_colormap()
        h, w = semantic_array.shape
        rgb_array = np.zeros((h, w, 3), dtype=np.uint8)

        for label, color in label_color_map.items():
            mask = semantic_array == label
            rgb_array[mask] = color

        return rgb_array
