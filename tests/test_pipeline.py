import logging

import numpy as np
import pandas as pd
import pytest

import vision2slope.pipeline as pipeline_module
from vision2slope import (
    ConfigurationError,
    PipelineConfig,
    ProcessingConfig,
    Vision2SlopePipeline,
    Visualizer,
    VisualizationConfig,
)

from conftest import FakeSegmentation, write_facade_image


@pytest.fixture
def input_dir(tmp_path):
    directory = tmp_path / "in"
    directory.mkdir()
    for pano in ("p1", "p2"):
        for view in (90, 270):
            write_facade_image(directory / f"{pano}_Direction_{view}_FOV_90_aspect_10--10_raw.png")
    return directory


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(pipeline_module, "SegmentationModel", FakeSegmentation)


def test_config_normalizes_log_level_and_converts_dicts(tmp_path):
    config = PipelineConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                            processing_config={"log_level": "info"})
    assert config.processing_config.log_level == "INFO"


def test_config_rejects_invalid_values(tmp_path):
    with pytest.raises(ConfigurationError):
        PipelineConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                       detection_config={"canny_threshold1": 200, "canny_threshold2": 100})
    with pytest.raises(ConfigurationError):
        PipelineConfig(input_dir=str(tmp_path / "missing"), output_dir=str(tmp_path / "out"))


def test_process_batch_end_to_end(input_dir, tmp_path):
    output_dir = tmp_path / "out"
    config = PipelineConfig(input_dir=str(input_dir), output_dir=str(output_dir),
                            processing_config=ProcessingConfig(log_level="warning"))
    estimate = Vision2SlopePipeline(config).process_batch()

    assert len(estimate) == 4
    assert estimate["pano_id"].nunique() == 2
    expected = np.degrees(np.arctan(FakeSegmentation.ROAD_EDGE_SLOPE))
    assert estimate["road_estimated_slope"].to_numpy() == pytest.approx(expected, abs=0.3)
    all_results = [p for p in output_dir.glob("vision2slope_results_*.csv") if "_estimate" not in p.name]
    assert len(pd.read_csv(all_results[0])) == 4


def test_process_batch_returns_empty_table_when_nothing_passes(input_dir, tmp_path):
    config = PipelineConfig(input_dir=str(input_dir), output_dir=str(tmp_path / "out"),
                            analysis_config={"filter_slope_angle": -1})
    estimate = Vision2SlopePipeline(config).process_batch()
    assert estimate.empty
    assert "status" in estimate.columns


def test_signed_estimate_uses_metadata(input_dir, tmp_path):
    metadata = tmp_path / "meta.csv"
    pd.DataFrame({"pano_id": ["p1", "p2"], "heading": [90.0, 90.0], "edge_bearing": [90.0, 270.0]}).to_csv(metadata, index=False)
    config = PipelineConfig(input_dir=str(input_dir), output_dir=str(tmp_path / "out"),
                            processing_config=ProcessingConfig(metadata_csv=str(metadata)))
    estimate = Vision2SlopePipeline(config).process_batch()
    per_view = estimate.set_index("filename")["signed_slope"]
    # Both views see the same positive edge angle; the 270° view inverts its sign
    assert per_view["p1_Direction_90_FOV_90_aspect_10--10_raw.png"] > 1.0
    assert per_view["p1_Direction_270_FOV_90_aspect_10--10_raw.png"] < -1.0

    per_pano = estimate.drop_duplicates("pano_id").set_index("pano_id")
    # Same imagery, opposite edge direction: equal magnitude, opposite sign
    assert per_pano.loc["p1", "road_estimated_slope"] == pytest.approx(-per_pano.loc["p2", "road_estimated_slope"])
    assert {"road_estimated_slope_abs"} <= set(estimate.columns)


def test_new_pipeline_closes_previous_log_handlers(input_dir, tmp_path):
    config = PipelineConfig(input_dir=str(input_dir), output_dir=str(tmp_path / "out"))
    first = Vision2SlopePipeline(config)
    file_handler = next(h for h in first.logger.handlers if hasattr(h, "baseFilename"))
    Vision2SlopePipeline(config)
    assert file_handler.stream is None


def test_pipeline_logs_are_not_duplicated_by_root_handlers(input_dir, tmp_path):
    # Colab/Jupyter attach a root handler; records must not reach it as well
    root_records = []
    root_handler = logging.Handler()
    root_handler.emit = root_records.append
    logging.getLogger().addHandler(root_handler)
    try:
        config = PipelineConfig(input_dir=str(input_dir), output_dir=str(tmp_path / "out"))
        Vision2SlopePipeline(config)
        logging.getLogger("vision2slope.detectors").warning("from a module logger")
    finally:
        logging.getLogger().removeHandler(root_handler)
    assert root_records == []
    assert "from a module logger" in (tmp_path / "out" / "pipeline.log").read_text()


def test_visualizer_output_names_keep_extension(tmp_path):
    visualizer = Visualizer(VisualizationConfig(save_road_masks=True), str(tmp_path))
    assert (tmp_path / "masks").is_dir()
    assert visualizer._output_name("a.jpeg", "road_mask") == "a_road_mask.jpeg"
    assert visualizer._output_name("a.PNG", "edges") == "a_edges.png"
    assert visualizer._output_name("a.bmp", "analysis", figure=True) == "a_analysis.png"
    assert visualizer._output_name("a.bmp", "corrected") == "a_corrected.bmp"


def test_cli_defaults_to_weighted_average():
    from vision2slope.cli import create_argument_parser

    parser = create_argument_parser()
    required = ["--input_dir", "x", "--output_dir", "y"]
    assert parser.parse_args(required).weighted_average is True
    assert parser.parse_args(required + ["--no-weighted-average"]).weighted_average is False
