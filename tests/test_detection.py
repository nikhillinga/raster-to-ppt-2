"""Tests for phase 1 detection (YOLO, Gemini, merger)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.models import BBox, DetectedElement


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
# Tests for YOLO
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


# ---------------------------------------------------------------------------
# Tests for Merger
# ---------------------------------------------------------------------------

from pipeline.phase1_detection.merger import calculate_iou, calculate_containment, merge_detections

def test_calculate_iou():
    box1 = BBox(x=0, y=0, w=10, h=10)
    box2 = BBox(x=5, y=5, w=10, h=10)
    # Area box1 = 100, Area box2 = 100. Intersection = 5x5 = 25.
    # Union = 100 + 100 - 25 = 175. IoU = 25 / 175 = 1/7 ~= 0.1428
    iou = calculate_iou(box1, box2)
    assert abs(iou - (25 / 175)) < 1e-5

def test_calculate_containment():
    outer = BBox(x=0, y=0, w=100, h=100)
    inner = BBox(x=10, y=10, w=50, h=50)
    # inner area = 2500, all inside outer.
    containment = calculate_containment(inner, outer)
    assert containment == 1.0
    
    partial = BBox(x=50, y=50, w=100, h=100)
    # partial area = 10000. Intersection with outer = 50x50 = 2500.
    # containment of partial in outer = 2500 / 10000 = 0.25
    containment = calculate_containment(partial, outer)
    assert abs(containment - 0.25) < 1e-5

def test_merge_detections_basic():
    yolo_elem = DetectedElement(
        id="yolo1", bbox=BBox(x=0, y=0, w=100, h=100), detected_by="yolo", confidence=0.8
    )
    # New gemini element, no correction, disjoint
    gemini_elem = DetectedElement(
        id="gem1", bbox=BBox(x=200, y=200, w=50, h=50), detected_by="gemini", confidence=0.9
    )
    
    tree = merge_detections([yolo_elem], [gemini_elem])
    assert len(tree.roots) == 2
    ids = {r.id for r in tree.roots}
    assert "yolo1" in ids
    assert "gem1" in ids

def test_merge_detections_containment():
    outer = DetectedElement(
        id="outer", bbox=BBox(x=0, y=0, w=100, h=100), detected_by="yolo", confidence=0.8
    )
    inner = DetectedElement(
        id="inner", bbox=BBox(x=10, y=10, w=20, h=20), detected_by="gemini", confidence=0.9
    )
    
    tree = merge_detections([outer], [inner])
    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert root.id == "outer"
    assert len(root.children) == 1
    assert root.children[0].id == "inner"

def test_merge_detections_correction():
    from pipeline.phase1_detection.gemini_detector import GeminiDetectedElement
    yolo_elem = DetectedElement(
        id="yolo1", bbox=BBox(x=10, y=10, w=50, h=50), detected_by="yolo", confidence=0.8
    )
    # Gemini correction for a box near yolo1, and it's larger
    gemini_elem = GeminiDetectedElement(
        id="gem1", 
        bbox=BBox(x=5, y=5, w=100, h=100), 
        detected_by="gemini", 
        confidence=0.9,
        correction_for=BBox(x=10, y=10, w=50, h=50)
    )
    
    tree = merge_detections([yolo_elem], [gemini_elem])
    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert root.id == "gem1"
    assert root.bbox.w == 100
