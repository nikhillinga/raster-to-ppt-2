"""Tests for pipeline data models (BBox, DetectedElement, ElementTree)."""

import pytest
from pipeline.models import BBox, DetectedElement, ElementTree


class TestDetectedElementDescendants:
    """Test 1: DetectedElement with children — all_descendants and child_ids."""

    def test_all_descendants_returns_parent_and_children(self):
        parent = DetectedElement(
            id="parent",
            bbox=BBox(x=0, y=0, w=100, h=100),
            children=[
                DetectedElement(id="child1", bbox=BBox(x=10, y=10, w=30, h=30)),
                DetectedElement(id="child2", bbox=BBox(x=50, y=50, w=30, h=30)),
            ],
        )
        descendants = parent.all_descendants()
        assert len(descendants) == 3
        ids = {d.id for d in descendants}
        assert ids == {"parent", "child1", "child2"}

    def test_child_ids_excludes_parent(self):
        parent = DetectedElement(
            id="parent",
            bbox=BBox(x=0, y=0, w=100, h=100),
            children=[
                DetectedElement(id="child1", bbox=BBox(x=10, y=10, w=30, h=30)),
                DetectedElement(id="child2", bbox=BBox(x=50, y=50, w=30, h=30)),
            ],
        )
        cids = parent.child_ids()
        assert cids == {"child1", "child2"}
        assert "parent" not in cids


class TestModelRebuild:
    """Test 2: model_rebuild does not raise."""

    def test_model_rebuild_no_exception(self):
        DetectedElement.model_rebuild()


class TestElementTree:
    """Test 3: ElementTree.all_elements with nested roots."""

    def test_all_elements_returns_roots_and_children(self):
        tree = ElementTree(
            roots=[
                DetectedElement(
                    id="root1",
                    bbox=BBox(x=0, y=0, w=200, h=200),
                    children=[
                        DetectedElement(id="r1c1", bbox=BBox(x=10, y=10, w=50, h=50)),
                    ],
                ),
                DetectedElement(
                    id="root2",
                    bbox=BBox(x=300, y=0, w=200, h=200),
                    children=[
                        DetectedElement(id="r2c1", bbox=BBox(x=310, y=10, w=50, h=50)),
                    ],
                ),
            ],
        )
        elements = tree.all_elements()
        assert len(elements) == 4
        ids = {e.id for e in elements}
        assert ids == {"root1", "r1c1", "root2", "r2c1"}


class TestBBoxProperties:
    """Test 4: BBox computed properties."""

    def test_x2(self):
        assert BBox(x=10, y=20, w=100, h=50).x2 == 110

    def test_y2(self):
        assert BBox(x=10, y=20, w=100, h=50).y2 == 70

    def test_area(self):
        assert BBox(x=10, y=20, w=100, h=50).area == 5000
