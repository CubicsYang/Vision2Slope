import hashlib
import types

import cv2
import numpy as np
import pytest

from vision2slope.pano2perspective import PanoramaTransformer, equirectangular_to_perspective


def _config(fov=90.0, phi=0.0):
    return types.SimpleNamespace(panorama_fov=fov, panorama_phi=phi, panorama_aspects=(2, 2), panorama_show_size=50)


def _longitude_panorama(height=200, width=400):
    """Panorama whose blue channel encodes the column (longitude), so views can be located."""
    pano = np.zeros((height, width, 3), np.uint8)
    pano[..., 0] = (np.arange(width) * 255 // (width - 1)).astype(np.uint8)
    pano[..., 1] = 128
    return pano


@pytest.mark.parametrize("theta, expected_column_fraction", [(0, 0.5), (90, 0.75), (270, 0.25)])
def test_view_center_looks_clockwise_from_panorama_center(theta, expected_column_fraction):
    pano = _longitude_panorama()
    view = equirectangular_to_perspective(pano, 90, theta, 0, 101, 101)
    center_value = int(view[50, 50, 0])
    expected = int(round(expected_column_fraction * 255))
    assert abs(center_value - expected) <= 2


def test_views_match_reference_checksum():
    # Reference computed with numpy 1.26 / OpenCV 4.11 (the zensvi-based setup behind the
    # paper); numpy 2 promotion or an OpenCV change that alters pixels breaks this.
    # Sizes are large enough that float32 vs float64 rotation angles change pixels.
    pano = np.random.default_rng(0).integers(0, 256, (512, 1024, 3), dtype=np.uint8)
    digest = hashlib.sha256()
    for theta, phi in [(90, 0.0), (270, 0.0), (90, 5.0)]:
        digest.update(equirectangular_to_perspective(pano, 90.0, theta, phi, 256, 256).tobytes())
    assert digest.hexdigest() == "12a427a2c8a2e974e3b9e293ad1be4f997776ea09eaff4def4b10f54864623fe"


def test_transform_panorama_writes_side_views_with_zensvi_names(tmp_path):
    (tmp_path / "in" / "sub").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "in" / "abc.jpg"), _longitude_panorama())
    cv2.imwrite(str(tmp_path / "in" / "sub" / "def.png"), _longitude_panorama())

    views = PanoramaTransformer(_config()).transform_panorama(str(tmp_path / "in"), str(tmp_path / "out"))

    names = sorted(p.relative_to(tmp_path / "out").as_posix() for p in views)
    assert names == [
        "perspective/abc_Direction_270_FOV_90.0_aspect_2--2_raw.png",
        "perspective/abc_Direction_90_FOV_90.0_aspect_2--2_raw.png",
        "perspective/sub/def_Direction_270_FOV_90.0_aspect_2--2_raw.png",
        "perspective/sub/def_Direction_90_FOV_90.0_aspect_2--2_raw.png",
    ]
    assert all(cv2.imread(str(p)).shape == (100, 100, 3) for p in views)


def test_transform_panorama_keeps_existing_views_and_can_render_all_directions(tmp_path):
    (tmp_path / "in").mkdir()
    cv2.imwrite(str(tmp_path / "in" / "abc.png"), _longitude_panorama())
    transformer = PanoramaTransformer(_config())

    first = transformer.transform_panorama(str(tmp_path / "in"), str(tmp_path / "out"))
    stamp = first[0].stat().st_mtime_ns
    all_views = transformer.transform_panorama(str(tmp_path / "in"), str(tmp_path / "out"), generate_left_right=False)

    assert first[0].stat().st_mtime_ns == stamp
    assert len(all_views) == 4


def test_matches_zensvi_pixel_for_pixel(tmp_path):
    zensvi_transform = pytest.importorskip("zensvi.transform")
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        pytest.skip("reference is zensvi under numpy 1; under numpy 2 zensvi itself promotes to float64")
    (tmp_path / "in").mkdir()
    rng = np.random.default_rng(0)
    cv2.imwrite(str(tmp_path / "in" / "abc.png"), rng.integers(0, 256, (200, 400, 3), dtype=np.uint8))

    for fov, phi in [(90.0, 0.0), (75, 5)]:
        out_z, out_o = tmp_path / f"z_{fov}", tmp_path / f"o_{fov}"
        zensvi_transform.ImageTransformer(dir_input=str(tmp_path / "in"), dir_output=str(out_z), verbosity=0).transform_images(
            style_list="perspective", FOV=fov, theta=90, phi=phi, aspects=(2, 2), show_size=50)
        for view in PanoramaTransformer(_config(fov, phi)).transform_panorama(str(tmp_path / "in"), str(out_o)):
            reference = out_z / view.relative_to(out_o)
            assert np.array_equal(cv2.imread(str(view)), cv2.imread(str(reference)))
