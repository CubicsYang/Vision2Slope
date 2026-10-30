import math

import pytest

from vision2slope import Utils
from vision2slope.downloader import _heading_degrees


def test_pano_id_and_perspective_angle_from_filename():
    name = "a_b-c_Direction_270_FOV_90_aspect_10--10_raw.png"
    assert Utils.get_pano_id_from_path(f"/some/dir/{name}") == "a_b-c"
    assert Utils.get_perspective_angle_from_path(name) == 270.0
    with pytest.raises(ValueError):
        Utils.get_perspective_angle_from_path("plain.png")


@pytest.mark.parametrize("a, b, expected", [(10, 350, 20), (0, 180, 180), (90, 160, 70), (359, 1, 2)])
def test_angle_difference(a, b, expected):
    assert Utils.angle_difference(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("angle, view, heading, bearing, expected", [
    (3.0, 90, 90, 90, 3.0),     # right view keeps the sign
    (3.0, 270, 90, 90, -3.0),   # left view inverts it
    (3.0, 90, 90, 270, -3.0),   # heading opposite to the OSM edge flips it
    (-2.0, 270, 0, 350, 2.0),
])
def test_compute_signed_slope(angle, view, heading, bearing, expected):
    assert Utils.compute_signed_slope(angle, view, heading, bearing) == pytest.approx(expected)


class _Pano:
    def __init__(self, heading):
        self.heading = heading


@pytest.mark.parametrize("radians, degrees", [(0.0, 0.0), (math.pi / 2, 90.0), (-math.pi / 2, 270.0)])
def test_downloader_converts_heading_to_degrees(radians, degrees):
    assert _heading_degrees(_Pano(radians)) == pytest.approx(degrees)


def test_downloader_missing_heading_is_nan():
    assert math.isnan(_heading_degrees(_Pano(None)))
