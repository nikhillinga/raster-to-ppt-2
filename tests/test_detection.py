"""Tests for phase 1 detection (YOLO, Gemini, merger)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.models import BBox, DetectedElement
from pipeline.phase1_detection.gemini_detector import GeminiDetectedElement


# ---------------------------------------------------------------------------
# Helpers to build mock YOLO results
# ---------------------------------------------------------------------------

def _make_mock_result(detections):
    """Build a mock ultralytics Result object."""
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
    fake_path = Path("nonexistent/weights/fake.pt")
    with patch("pipeline.phase1_detection.yolo_detector.config") as mock_config:
        mock_config.YOLO_WEIGHTS = fake_path
        mock_config.YOLO_IMAGE_SIZE = 1280
        mock_config.YOLO_CONF_THRESHOLD = 0.25
        from pipeline.phase1_detection.yolo_detector import run_yolo
        with pytest.raises(FileNotFoundError):
            run_yolo(_fake_image())


def test_yolo_returns_list():
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


def test_yolo_semantic_mapping():
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

    assert result[0].semantic_type == "Header"


def test_yolo_empty_returns_list():
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


# ---------------------------------------------------------------------------
# Tests for Merger
# ---------------------------------------------------------------------------

from pipeline.phase1_detection.merger import iou, containment_ratio, merge_detections


def test_iou_no_overlap():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=20, y=20, w=10, h=10)
    assert iou(a, b) == 0.0


def test_iou_identical():
    a = BBox(x=5, y=5, w=20, h=20)
    b = BBox(x=5, y=5, w=20, h=20)
    assert iou(a, b) == 1.0


def test_iou_partial():
    a = BBox(x=0, y=0, w=10, h=10)  # Area 100
    b = BBox(x=5, y=0, w=10, h=10)  # Area 100
    # Intersection is x from 5 to 10 (width 5), y from 0 to 10 (height 10) = 50.
    # Union = 100 + 100 - 50 = 150.
    # IoU = 50 / 150 = 0.3333...
    assert abs(iou(a, b) - 0.33333333) < 1e-5


def test_containment_fully_inside():
    larger = BBox(x=0, y=0, w=100, h=100)
    smaller = BBox(x=10, y=10, w=20, h=20)
    assert containment_ratio(smaller, larger) == 1.0


def test_containment_no_overlap():
    larger = BBox(x=0, y=0, w=100, h=100)
    smaller = BBox(x=150, y=150, w=20, h=20)
    assert containment_ratio(smaller, larger) == 0.0


def test_merge_dedup():
    # Two elements with high IoU (approx 0.8)
    # Box A: 100x100
    # Box B: 90x90 inside Box A (intersection 8100. Union = 10000 + 8100 - 8100 = 10000. IoU = 0.81)
    a = DetectedElement(id="a", bbox=BBox(x=0, y=0, w=100, h=100), confidence=0.9, detected_by="yolo")
    b = DetectedElement(id="b", bbox=BBox(x=0, y=0, w=90, h=90), confidence=0.8, detected_by="gemini")
    
    with patch("pipeline.phase1_detection.merger.config") as mock_config:
        mock_config.MERGE_IOU_DUPLICATE_THRESHOLD = 0.8
        mock_config.MERGE_CONTAINMENT_THRESHOLD = 0.85
        tree = merge_detections([a], [b])
        
    assert len(tree.roots) == 1
    assert tree.roots[0].id == "a"


def test_merge_nesting():
    # Element 90% inside another (we make the smaller box clearly fit 100% inside to hit nesting)
    a = DetectedElement(id="a", bbox=BBox(x=0, y=0, w=100, h=100), confidence=0.9, detected_by="yolo")
    b = DetectedElement(id="b", bbox=BBox(x=10, y=10, w=30, h=30), confidence=0.8, detected_by="gemini")
    
    with patch("pipeline.phase1_detection.merger.config") as mock_config:
        mock_config.MERGE_IOU_DUPLICATE_THRESHOLD = 0.8
        mock_config.MERGE_CONTAINMENT_THRESHOLD = 0.85
        tree = merge_detections([a], [b])
        
    assert len(tree.roots) == 1
    assert tree.roots[0].id == "a"
    assert len(tree.roots[0].children) == 1
    assert tree.roots[0].children[0].id == "b"


def test_merge_tree_all_elements():
    # 1 root with 2 children -> all_elements() returns 3 items
    root = DetectedElement(id="root", bbox=BBox(x=0, y=0, w=200, h=200), detected_by="yolo")
    child1 = DetectedElement(id="c1", bbox=BBox(x=10, y=10, w=20, h=20), detected_by="yolo")
    child2 = DetectedElement(id="c2", bbox=BBox(x=40, y=40, w=20, h=20), detected_by="gemini")
    
    with patch("pipeline.phase1_detection.merger.config") as mock_config:
        mock_config.MERGE_IOU_DUPLICATE_THRESHOLD = 0.8
        mock_config.MERGE_CONTAINMENT_THRESHOLD = 0.85
        tree = merge_detections([root, child1], [child2])
        
    assert len(tree.all_elements()) == 3


def test_merge_roots_only_top_level():
    # Nested element not in roots list
    root = DetectedElement(id="root", bbox=BBox(x=0, y=0, w=200, h=200), detected_by="yolo")
    child = DetectedElement(id="child", bbox=BBox(x=10, y=10, w=20, h=20), detected_by="gemini")
    
    with patch("pipeline.phase1_detection.merger.config") as mock_config:
        mock_config.MERGE_IOU_DUPLICATE_THRESHOLD = 0.8
        mock_config.MERGE_CONTAINMENT_THRESHOLD = 0.85
        tree = merge_detections([root], [child])
        
    ids = {r.id for r in tree.roots}
    assert "root" in ids
    assert "child" not in ids
