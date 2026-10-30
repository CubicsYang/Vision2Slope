"""
Panorama to perspective transformation module for Vision2Slope pipeline.

The equirectangular-to-perspective projection (_xyz2lonlat, _lonlat2XY and
equirectangular_to_perspective) is adapted from zensvi's
zensvi.transform.ImageTransformer (https://github.com/koito19960406/ZenSVI),
MIT License, Copyright (c) 2024. It reproduces zensvi's output under numpy 1
pixel for pixel (also under numpy 2, see the float32 casts), and the output
file names match zensvi's, so results equal those of earlier zensvi-based runs
without depending on zensvi.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

# View directions in degrees clockwise from the panorama center (camera heading)
SIDE_VIEW_DIRECTIONS = (90, 270)      # right and left side views
ALL_VIEW_DIRECTIONS = (0, 90, 180, 270)

PANORAMA_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _xyz2lonlat(xyz: np.ndarray) -> np.ndarray:
    """Convert 3D ray directions (..., 3) to longitude/latitude in radians (..., 2)."""
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)
    return np.concatenate([lon, lat], axis=-1)


def _lonlat2XY(lonlat: np.ndarray, shape: tuple) -> np.ndarray:
    """Convert longitude/latitude (..., 2) to pixel coordinates in an image of `shape`."""
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    return np.concatenate([X, Y], axis=-1)


def equirectangular_to_perspective(img: np.ndarray, fov: float, theta: float, phi: float,
                                   height: int, width: int) -> np.ndarray:
    """
    Render a perspective view from an equirectangular panorama.

    Args:
        img: Equirectangular panorama (H, W[, C]); its center column is the camera heading
        fov: Horizontal field of view in degrees
        theta: View direction in degrees, clockwise from the panorama center
        phi: Tilt angle in degrees
        height: Output height in pixels
        width: Output width in pixels

    Returns:
        Perspective image (height, width[, C])
    """
    f = 0.5 * width * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array(
        [
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1],
        ],
        np.float32,
    )
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    # Explicit float32 angles: numpy 1 kept these products float32, numpy 2 (NEP 50)
    # would promote them to float64 and shift some output pixels
    R1, _ = cv2.Rodrigues(y_axis * np.float32(np.radians(theta)))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.float32(np.radians(phi)))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = _xyz2lonlat(xyz)
    XY = _lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
    return cv2.remap(img, XY[..., 0], XY[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


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

        Panoramas are searched recursively. Each view is written to
        <output_dir>/perspective/<relative path of the panorama>_Direction_<θ>_FOV_<fov>_aspect_<h>--<w>_raw.png;
        views that already exist are kept.

        Args:
            input_dir: Input directory containing panoramic images
            output_dir: Output directory for perspective images
            generate_left_right: If True, generate only the right (90°) and left (270°) views;
                otherwise also the front (0°) and back (180°) views

        Returns:
            List of paths to the perspective images
        """
        self.logger.info(f"Transforming panoramic images from {input_dir}")
        self.logger.info(f"Output directory: {output_dir}")

        input_path = Path(input_dir)
        perspective_root = Path(output_dir) / "perspective"
        directions = SIDE_VIEW_DIRECTIONS if generate_left_right else ALL_VIEW_DIRECTIONS

        height = int(self.aspects[0] * self.show_size)
        width = int(self.aspects[1] * self.show_size)
        aspect_name = f"{self.aspects[0]}--{self.aspects[1]}"

        panoramas = sorted(
            p for p in input_path.rglob('*')
            if p.is_file() and p.suffix.lower() in PANORAMA_EXTENSIONS
        )

        generated_files = []
        for pano_path in tqdm(panoramas, desc="Converting panoramas to perspective views"):
            view_base = perspective_root / pano_path.relative_to(input_path).with_suffix('')
            view_base.parent.mkdir(parents=True, exist_ok=True)

            img = None
            for theta in directions:
                view_path = view_base.with_name(
                    f"{view_base.name}_Direction_{theta}_FOV_{self.fov}_aspect_{aspect_name}_raw.png"
                )
                if not view_path.exists():
                    if img is None:
                        img = self._read_panorama(pano_path)
                        if img is None:
                            break
                    view = equirectangular_to_perspective(img, self.fov, theta, self.phi, height, width)
                    cv2.imwrite(str(view_path), view)
                generated_files.append(view_path)

        return generated_files

    def _read_panorama(self, path: Path):
        """Read a panorama as BGR (alpha kept); return None if it cannot be read."""
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.logger.warning(f"Could not read panorama, skipping: {path}")
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

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
