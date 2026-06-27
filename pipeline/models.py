from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import uuid

SemanticType = Literal[
    "Section", "NestedSection", "TextBlock", "Header",
    "Icon", "DiagramShape", "Arrow", "Table", "Chart", "BackgroundArt"
]

ProcessingPath = Literal["reconstruct", "crop", "undecided"]

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int: return self.x + self.w
    @property
    def y2(self) -> int: return self.y + self.h
    @property
    def area(self) -> int: return self.w * self.h
    @property
    def cx(self) -> float: return self.x + self.w / 2
    @property
    def cy(self) -> float: return self.y + self.h / 2

class OCRLine(BaseModel):
    text: str
    bbox: BBox
    font_size_pt: float
    color_rgb: tuple[int, int, int]
    bold: bool
    is_art: bool = False

class DetectedElement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    bbox: BBox
    confidence: float = 1.0
    semantic_type: SemanticType = "TextBlock"
    shape_type: Optional[str] = None
    fill_color: Optional[tuple[int, int, int]] = None
    border_color: Optional[tuple[int, int, int]] = None
    detected_by: str = "yolo"
    processing_path: ProcessingPath = "undecided"
    ocr_lines: List[OCRLine] = []
    tile_path: Optional[str] = None
    arrow_start: Optional[tuple[int, int]] = None
    arrow_end: Optional[tuple[int, int]] = None
    arrow_direction: Optional[tuple[float, float]] = None
    children: List["DetectedElement"] = []

    def all_descendants(self) -> List["DetectedElement"]:
        result = [self]
        for child in self.children:
            result.extend(child.all_descendants())
        return result

    def child_ids(self) -> set[str]:
        ids = set()
        for child in self.children:
            ids.add(child.id)
            ids.update(child.child_ids())
        return ids

DetectedElement.model_rebuild()

class ElementTree(BaseModel):
    roots: List[DetectedElement] = []
    source_image_path: str = ""
    source_image_w: int = 0
    source_image_h: int = 0

    def all_elements(self) -> List[DetectedElement]:
        result = []
        for root in self.roots:
            result.extend(root.all_descendants())
        return result
