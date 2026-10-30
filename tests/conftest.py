"""
Shared fixtures for the Vision2Slope tests.

The tests never load Mask2Former: FakeSegmentation returns a hand-made
semantic map, so they run offline in a few seconds.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Test the package in this checkout even if another copy is installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "Vision2Slope"))

from vision2slope import SegmentationProvider  # noqa: E402

HEIGHT, WIDTH = 400, 600

# Mapillary Vistas class IDs used by the pipeline
BUILDING, ROAD, VEGETATION = 17, 13, 30


class FakeSegmentation(SegmentationProvider):
    """Segmentation provider with a building band on top and road at the bottom."""

    # Slope (rows per column) of the road's upper edge, about 2.9°
    ROAD_EDGE_SLOPE = 0.05

    def __init__(self, model_name: str = "fake", device=None, cache_dir=None,
                 building: bool = True, road: bool = True):
        self.building = building
        self.road = road

    def segment(self, image):
        width, height = image.size
        semantic_map = np.zeros((height, width), dtype=np.int64)
        if self.building:
            semantic_map[: height // 2, :] = BUILDING
        if self.road:
            rows, cols = np.indices(semantic_map.shape)
            semantic_map[rows >= height * 0.625 + self.ROAD_EDGE_SLOPE * cols] = ROAD
        return semantic_map

    def get_model_info(self) -> dict:
        return {"model_name": "fake"}


def write_facade_image(path: Path, tilt: float = 0.0) -> Path:
    """Write a gray image with near-vertical facade edges; tilt is dx/dy of each edge."""
    image = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    for x in range(100, 500, 60):
        cv2.line(image, (x, 20), (int(x + tilt * 160), 180), (255, 255, 255), 3)
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def facade_image(tmp_path):
    return write_facade_image(tmp_path / "pano_Direction_90_FOV_90_aspect_10--10_raw.png")
