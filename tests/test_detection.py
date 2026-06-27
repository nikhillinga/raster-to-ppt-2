"""Tests for phase 1 detection (YOLO detector)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.models import DetectedElement


# ---------------------------------------------------------------------------
# Helpers to build mock YOLO results
# ---------------------------------------------------------------------------

def _make_mock_result(detections):
    """Build a mock ultralytics Result object.

    Args:
        detections: list of dicts with keys
            x1, y1, x2, y2, conf, cls_id, class_name.
    """
    if not detections:
        result = MagicMock()
        result.boxes = MagicMock()
        result.boxes.__len__ = lambda self: 0
        result.boxes.xyxy = []
        result.boxes.conf = []
        result.boxes.cls = []
        result.names = {}
        return result

    import torch

    xyxy = torch.tensor([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections])
    conf = torch.tensor([d["conf"] for d in detections])
    cls = torch.tensor([d["cls_id"] for d in detections], dtype=torch.float32)
    names = {d["cls_id"]: d["class_name"] for d in detections}

    boxes = MagicMock()
    boxes.xyxy = xyxy
    boxes.conf = conf
    boxes.cls = cls
    boxes.__len__ = lambda self: len(detections)

    result = MagicMock()
    result.boxes = boxes
    result.names = names
    return result


def _fake_image():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_yolo_missing_weights():
    """Patching YOLO_WEIGHTS to a nonexistent path raises FileNotFoundError."""
    fake_path = Path("nonexistent/weights/fake.pt")
    with patch("pipeline.phase1_detection.yolo_detector.config") as mock_config:
        mock_config.YOLO_WEIGHTS = fake_path
        mock_config.YOLO_IMAGE_SIZE = 1280
        mock_config.YOLO_CONF_THRESHOLD = 0.25
        from pipeline.phase1_detection.yolo_detector import run_yolo
        with pytest.raises(FileNotFoundError):
            run_yolo(_fake_image())


def test_yolo_returns_list():
    """Mock YOLO returning fake detections → output is List[DetectedElement]."""
    detections = [
        {"x1": 10, "y1": 20, "x2": 110, "y2": 120, "conf": 0.9, "cls_id": 0, "class_name": "header"},
        {"x1": 200, "y1": 50, "x2": 400, "y2": 250, "conf": 0.8, "cls_id": 1, "class_name": "content_block"},
    ]
    mock_result = _make_mock_result(detections)

    with patch("pipeline.phase1_detection.yolo_detector.config") as mock_config, \
         patch("pipeline.phase1_detection.yolo_detector.YOLO") as MockYOLO:
        mock_config.YOLO_WEIGHTS = MagicMock()
        mock_config.YOLO_WEIGHTS.exists.return_value = True
        mock_config.YOLO_IMAGE_SIZE = 1280
        mock_config.YOLO_CONF_THRESHOLD = 0.25
        MockYOLO.return_value = MagicMock(return_value=[mock_result])

        from pipeline.phase1_detection.yolo_detector import run_yolo
        result = run_yolo(_fake_image())

    assert isinstance(result, list)
    assert len(result) == 2
    for elem in result:
        assert isinstance(elem, DetectedElement)


def test_yolo_semantic_mapping():
    """Mock model returning class_name='header' → semantic_type='Header'."""
    detections = [
        {"x1": 0, "y1": 0, "x2": 100, "y2": 50, "conf": 0.95, "cls_id": 0, "class_name": "header"},
    ]
    mock_result = _make_mock_result(detections)

    with patch("pipeline.phase1_detection.yolo_detector.config") as mock_config, \
         patch("pipeline.phase1_detection.yolo_detector.YOLO") as MockYOLO:
        mock_config.YOLO_WEIGHTS = MagicMock()
        mock_config.YOLO_WEIGHTS.exists.return_value = True
        mock_config.YOLO_IMAGE_SIZE = 1280
        mock_config.YOLO_CONF_THRESHOLD = 0.25
        MockYOLO.return_value = MagicMock(return_value=[mock_result])

        from pipeline.phase1_detection.yolo_detector import run_yolo
        result = run_yolo(_fake_image())

    assert len(result) == 1
    assert result[0].semantic_type == "Header"


def test_yolo_empty_returns_list():
    """Mock model returning zero detections → returns [] not an exception."""
    mock_result = _make_mock_result([])

    with patch("pipeline.phase1_detection.yolo_detector.config") as mock_config, \
         patch("pipeline.phase1_detection.yolo_detector.YOLO") as MockYOLO:
        mock_config.YOLO_WEIGHTS = MagicMock()
        mock_config.YOLO_WEIGHTS.exists.return_value = True
        mock_config.YOLO_IMAGE_SIZE = 1280
        mock_config.YOLO_CONF_THRESHOLD = 0.25
        MockYOLO.return_value = MagicMock(return_value=[mock_result])

        from pipeline.phase1_detection.yolo_detector import run_yolo
        result = run_yolo(_fake_image())

    assert result == []
    assert isinstance(result, list)
