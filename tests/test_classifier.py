"""Tests for phase 2 decision classifier."""

import numpy as np
import pytest

from pipeline.models import BBox, DetectedElement, ElementTree
from pipeline.phase2_decision.classifier import classify_element, classify_tree


def _fake_image():
    # 200x200 BGR image
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_always_crop_types():
    # semantic_type="BackgroundArt" → processing_path must be "crop" regardless of what image is passed
    element = DetectedElement(
        bbox=BBox(x=0, y=0, w=100, h=100),
        semantic_type="BackgroundArt"
    )
    result = classify_element(element, _fake_image())
    assert result == "crop"
    assert element.processing_path == "crop"


def test_always_reconstruct_types():
    # semantic_type="Header" → must be "reconstruct"
    element = DetectedElement(
        bbox=BBox(x=0, y=0, w=100, h=100),
        semantic_type="Header"
    )
    result = classify_element(element, _fake_image())
    assert result == "reconstruct"
    assert element.processing_path == "reconstruct"


def test_solid_color_reconstruct():
    # pass a solid blue 200x200 image region → entropy should be low → path should be "reconstruct"
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255  # Solid blue
    
    element = DetectedElement(
        bbox=BBox(x=0, y=0, w=200, h=200),
        semantic_type="Section" # not an override type
    )
    result = classify_element(element, image)
    assert result == "reconstruct"
    assert element.processing_path == "reconstruct"


def test_photo_crop():
    # pass a photograph-like numpy array with high entropy → path should be "crop"
    # Create random noise to simulate high entropy and edges
    np.random.seed(42)
    image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    element = DetectedElement(
        bbox=BBox(x=0, y=0, w=200, h=200),
        semantic_type="Section" # not an override type
    )
    result = classify_element(element, image)
    assert result == "crop"
    assert element.processing_path == "crop"


def test_classify_tree_sets_all():
    # create tree with 3 elements (1 root + 2 children) →
    # after classify_tree all 3 must have processing_path != "undecided"
    child1 = DetectedElement(bbox=BBox(x=10, y=10, w=20, h=20), semantic_type="Section")
    child2 = DetectedElement(bbox=BBox(x=40, y=40, w=20, h=20), semantic_type="Section")
    root = DetectedElement(bbox=BBox(x=0, y=0, w=100, h=100), semantic_type="Section", children=[child1, child2])
    
    tree = ElementTree(roots=[root])
    
    # Ensure they start as undecided
    assert root.processing_path == "undecided"
    assert child1.processing_path == "undecided"
    assert child2.processing_path == "undecided"
    
    classify_tree(tree, _fake_image())
    
    assert root.processing_path != "undecided"
    assert child1.processing_path != "undecided"
    assert child2.processing_path != "undecided"


def test_zero_area_element():
    # element with w=0, h=0 → no exception, path="crop"
    element = DetectedElement(
        bbox=BBox(x=10, y=10, w=0, h=0),
        semantic_type="Section"
    )
    # Area will be 0
    assert element.bbox.area == 0
    
    result = classify_element(element, _fake_image())
    assert result == "crop"
    assert element.processing_path == "crop"
