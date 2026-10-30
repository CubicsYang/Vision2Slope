import numpy as np
import pytest
from PIL import Image

from vision2slope import (
    ImageCorrector,
    PipelineConfig,
    ProcessingStatus,
    RoadSlopeAnalyzer,
    SkewDetector,
    StandardImageProcessor,
)
from vision2slope.pipeline import _build_legacy_config

from conftest import ROAD, VEGETATION, FakeSegmentation, write_facade_image


@pytest.fixture
def legacy_config(tmp_path):
    (tmp_path / "in").mkdir()
    return _build_legacy_config(PipelineConfig(input_dir=str(tmp_path / "in"), output_dir=str(tmp_path / "out")))


def make_processor(legacy_config, segmentation=None):
    return StandardImageProcessor(
        segmentation_provider=segmentation or FakeSegmentation(),
        skew_detector=SkewDetector(legacy_config),
        corrector=ImageCorrector(legacy_config),
        slope_analyzer=RoadSlopeAnalyzer(legacy_config),
    )


def test_skew_detector_finds_vertical_facade_lines(legacy_config, facade_image):
    image = Image.open(facade_image).convert("RGB")
    skew, confidence = SkewDetector(legacy_config).detect_skew(image, FakeSegmentation().segment(image))
    assert confidence > 0
    assert abs(skew) < 1.0


def test_skew_detector_ignores_lines_on_vegetation(legacy_config, facade_image):
    image = Image.open(facade_image).convert("RGB")
    semantic_map = np.full((image.height, image.width), VEGETATION)
    assert SkewDetector(legacy_config).detect_skew(image, semantic_map) == (0.0, 0)


def test_road_analyzer_recovers_edge_angle(legacy_config):
    semantic_map = np.zeros((400, 600), dtype=np.int64)
    rows, cols = np.indices(semantic_map.shape)
    semantic_map[rows >= 200 + 0.1 * cols] = ROAD  # upper edge with slope 0.1
    slope, _, angle, area, _, _ = RoadSlopeAnalyzer(legacy_config).analyze_road_slope_with_details(semantic_map)
    assert area > 0
    assert slope == pytest.approx(0.1, abs=0.01)
    assert angle == pytest.approx(np.degrees(np.arctan(0.1)), abs=0.5)


def test_road_analyzer_without_road_returns_sentinel(legacy_config):
    result = RoadSlopeAnalyzer(legacy_config).analyze(np.zeros((50, 50), dtype=np.int64))
    assert result == (-999.0, -999.0, -999.0, 0)


def test_processor_success(legacy_config, facade_image):
    result = make_processor(legacy_config).process(str(facade_image))
    assert result.status == ProcessingStatus.SUCCESS
    assert result.pano_id == "pano"


@pytest.mark.parametrize("segmentation, status", [
    (FakeSegmentation(building=False), ProcessingStatus.NO_LINES),
    (FakeSegmentation(road=False), ProcessingStatus.NO_ROAD),
])
def test_processor_reports_missing_inputs(legacy_config, facade_image, segmentation, status):
    assert make_processor(legacy_config, segmentation).process(str(facade_image)).status == status


def _raise(*args, **kwargs):
    raise RuntimeError("injected failure")


@pytest.mark.parametrize("component, method, tilt, status", [
    ("skew_detector", "detect_lines", 0.0, ProcessingStatus.SKEW_FAILED),
    ("corrector", "correct", 0.05, ProcessingStatus.CORRECTION_FAILED),
    ("slope_analyzer", "fit_line_ransac", 0.0, ProcessingStatus.SLOPE_FAILED),
])
def test_processor_maps_component_errors_to_stage_status(legacy_config, tmp_path, component, method, tilt, status):
    image_path = write_facade_image(tmp_path / "x.png", tilt=tilt)
    processor = make_processor(legacy_config)
    setattr(getattr(processor, component), method, _raise)
    result = processor.process(str(image_path))
    assert result.status == status
    assert "injected failure" in result.error_message
