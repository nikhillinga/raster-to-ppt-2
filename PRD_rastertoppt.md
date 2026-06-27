# Raster-to-PPTX Pipeline v2 — Product Requirements Document

**Author:** Nikhil Arya Linga  
**Client:** DealVerse AI LLC (Kal Kilpi)  
**Version:** 1.0  
**Date:** June 2026  
**Status:** Pre-build — approved for implementation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Repository Structure](#4-repository-structure)
5. [Phase 0 — Prerequisites](#5-phase-0--prerequisites)
6. [Phase 1 — Detection](#6-phase-1--detection)
7. [Phase 2 — Reconstruct vs. Crop Decision](#7-phase-2--reconstruct-vs-crop-decision)
8. [Phase 3A — Reconstruct Path](#8-phase-3a--reconstruct-path)
9. [Phase 3B — Crop Path](#9-phase-3b--crop-path)
10. [Phase 4 — PPTX Assembly](#10-phase-4--pptx-assembly)
11. [Phase 5 — QA / SSIM Validation](#11-phase-5--qa--ssim-validation)
12. [Data Model](#12-data-model)
13. [Agent Prompts](#13-agent-prompts)
14. [Testing Strategy](#14-testing-strategy)
15. [Known Limitations and Future Work](#15-known-limitations-and-future-work)

---

## 1. Overview

This document specifies a ground-up rebuild of the raster-to-PPTX conversion pipeline. The goal is to convert complex raster slide images (dark backgrounds, nested layouts, illustrated icons, custom shapes, organic connectors) into fully editable PowerPoint files that:

- **Look visually identical** to the source raster on human review
- **Contain genuinely editable objects** — shapes, text, and containers that can be repositioned, resized, and rearranged independently in PowerPoint
- **Preserve logical hierarchy** — nested sections group as a unit while remaining individually editable within the group
- **Handle complexity gracefully** — complex illustrated regions are preserved as high-quality image crops rather than mangled reconstruction attempts

### Why a rebuild?

The previous pipeline accumulated significant technical debt across multiple bug-fix rounds: 8 overlapping suppression mechanisms, conflicting OCR paths, VLM integration bolted onto an architecture not designed for it, and a fundamental assumption (reconstruct everything as native shapes) that does not hold for the real test data. The clean rebuild eliminates these by making the right architectural decisions from the start.

### Key architectural decisions (locked)

1. **VLM-first detection** — Gemini is the primary detector; YOLO seeds the first prompt and reduces Gemini's work, but does not determine final detections
2. **Hierarchy built from Phase 1** — elements are a tree from the moment detection completes; no retrofitting
3. **Crop-and-preserve as the DEFAULT for complex regions** — reconstruct native shapes only where the result will be visually equivalent
4. **Shared text overlay logic** — both the reconstruct and crop paths use the same per-line OCR and text box placement code
5. **One decision point** — the reconstruct-vs-crop classifier is a single explicit gate, not scattered heuristics across the pipeline

---

## 2. Goals and Non-Goals

### Goals

- Convert raster slide images to editable `.pptx` files
- Support slides with 3+ levels of nesting (Section → NestedSection → child elements)
- Maintain visual fidelity sufficient to pass human review ("looks as good as the raster")
- Text must be editable as coherent, selectable text blocks — not per-glyph or per-word objects
- Objects must be independently movable and resizable in PowerPoint
- Handle dark backgrounds, illustrated icons, curved connectors, and dense icon grids
- Produce QA scores (per-element and per-type SSIM) for every output
- Be runnable as a CLI on a single image or batch of images

### Non-Goals

- Exact pixel-perfect recreation of custom illustrated artwork (postman, Viking ship, etc.) — these are preserved as image crops, not reconstructed
- Curved/organic connectors as native PowerPoint bezier paths — these are preserved as background crops
- Font family detection (Arial is used as a safe universal substitute throughout)
- Animation or slide transition support
- Multi-slide presentation input (each slide is processed independently)

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: raster image (.png / .jpg)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1 — DETECTION                                         │
│                                                              │
│  1a. YOLO pre-pass (fast, seeds Gemini context)              │
│  1b. Gemini iterative refinement (find missing, correct      │
│      clipped boxes, assign semantic types)                   │
│  1c. Merge + deduplicate (IoU dedup, containment → nesting)  │
│  1d. Output: ElementTree (parent-child hierarchy)            │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2 — RECONSTRUCT vs. CROP DECISION                     │
│                                                              │
│  Per element: score complexity (entropy + edge density +     │
│  sub-element count) → route to Phase 3A or 3B               │
│  Semantic type overrides: always-crop / always-reconstruct   │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│  PHASE 3A            │    │  PHASE 3B                        │
│  RECONSTRUCT PATH    │    │  CROP PATH                       │
│                      │    │                                  │
│  - Shape classify    │    │  - Crop tile from source image   │
│  - Render native     │    │  - Detect background color       │
│    PowerPoint shape  │    │  - Strip text via local fill     │
│  - Per-line OCR      │    │  - Per-line OCR on original      │
│  - Text overlay      │    │  - Text overlay on clean tile    │
└──────────────┬───────┘    └──────────────┬──────────────────┘
               │                           │
               └────────────┬──────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4 — PPTX ASSEMBLY                                     │
│                                                              │
│  Walk element tree top-down:                                 │
│  - Section/NestedSection → GroupShape                        │
│  - DiagramShape → native MSO_SHAPE                           │
│  - Image tile → add_picture                                  │
│  - TextBlock → per-line text boxes                           │
│  - Arrow → MSO_CONNECTOR.STRAIGHT + XML arrowhead            │
│  - Z-order: background → shapes → text (always top)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5 — QA / SSIM VALIDATION                              │
│                                                              │
│  - Render output .pptx back to image                         │
│  - SSIM: overall, per-element (tree walk), per semantic type │
│  - Flag elements below threshold                             │
│  - Save visual diff image                                    │
│  - Export JSON QA report                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
raster_to_pptx_v2/
├── config.py                    # All tunable parameters (single source of truth)
├── requirements.txt             # Pinned dependencies
├── run.py                       # CLI entry point (single image)
├── run_batch.py                 # CLI entry point (directory of images)
├── .env.example                 # API key template
│
├── pipeline/
│   ├── __init__.py
│   ├── models.py                # ElementTree data model (Pydantic)
│   ├── pipeline.py              # Top-level orchestrator: runs phases 1-5 in sequence
│   │
│   ├── phase1_detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py     # YOLO pre-pass: fast bounding box detection
│   │   ├── gemini_detector.py   # Gemini iterative refinement + semantic typing
│   │   ├── merger.py            # IoU dedup + containment-to-nesting merge
│   │   └── prompts.py           # All Gemini prompt strings (single file)
│   │
│   ├── phase2_decision/
│   │   ├── __init__.py
│   │   └── classifier.py        # Reconstruct vs. crop complexity classifier
│   │
│   ├── phase3a_reconstruct/
│   │   ├── __init__.py
│   │   ├── shape_classifier.py  # Contour → shape_type string
│   │   ├── shape_renderer.py    # shape_type → MSO_SHAPE + fill/border
│   │   └── ocr.py               # Per-line OCR (shared with Phase 3B)
│   │
│   ├── phase3b_crop/
│   │   ├── __init__.py
│   │   ├── tiler.py             # Crop element region from source image
│   │   ├── background.py        # Detect dark vs. light background
│   │   ├── text_remover.py      # Local-fill text removal → clean tile
│   │   └── ocr.py               # Imports from phase3a_reconstruct/ocr.py (shared)
│   │
│   ├── phase4_assembly/
│   │   ├── __init__.py
│   │   ├── assembler.py         # Walk element tree → write .pptx
│   │   ├── group_builder.py     # Build PowerPoint GroupShapes for nested elements
│   │   ├── text_placer.py       # Place per-line text boxes at correct positions
│   │   └── arrow_writer.py      # Write connectors + XML arrowheads
│   │
│   └── phase5_qa/
│       ├── __init__.py
│       ├── renderer.py          # Render .pptx to image via LibreOffice headless
│       ├── ssim.py              # SSIM computation (overall + per-element + per-type)
│       └── reporter.py          # JSON QA report + visual diff image
│
├── output/                      # Pipeline outputs (gitignored)
│   ├── tiles/                   # Cropped image tiles
│   └── qa/                      # SSIM diff images + JSON reports
│
├── weights/                     # YOLO model weights (gitignored)
├── samples/                     # Test images
├── tests/                       # Pytest test suite
│   ├── test_models.py
│   ├── test_detection.py
│   ├── test_classifier.py
│   ├── test_ocr.py
│   ├── test_assembler.py
│   └── test_qa.py
└── logs/                        # Log files (gitignored)
```

---

## 5. Phase 0 — Prerequisites

Before writing any pipeline code, complete these prerequisites. They are mandatory — every later phase depends on them.

### 5.1 Environment setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file from `.env.example`:
```
GEMINI_API_KEY=your_key_here
```

Install Tesseract binary separately (not via pip):
- Ubuntu: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: installer at https://github.com/UB-Mannheim/tesseract/wiki

Install LibreOffice (for PPTX-to-image rendering in Phase 5):
- Ubuntu: `sudo apt-get install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: download from libreoffice.org

### 5.2 `models.py` — Element data model

Build this first. Every other module imports from here.

```python
# pipeline/models.py

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
    x: int       # left edge, pixels from image origin
    y: int       # top edge, pixels from image origin
    w: int       # width in pixels
    h: int       # height in pixels

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
    is_art: bool = False   # True if line has no alphanumeric chars → skip overlay

class DetectedElement(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    bbox: BBox
    confidence: float = 1.0
    semantic_type: SemanticType = "TextBlock"
    shape_type: Optional[str] = None    # "rectangle", "circle", "triangle", "star", etc.
    fill_color: Optional[tuple[int, int, int]] = None
    border_color: Optional[tuple[int, int, int]] = None
    detected_by: str = "yolo"           # "yolo" | "gemini" | "shape_cv"
    processing_path: ProcessingPath = "undecided"
    ocr_lines: List[OCRLine] = []
    tile_path: Optional[str] = None     # Path to cropped image tile (crop path only)
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
    roots: List[DetectedElement] = []    # Top-level elements only
    source_image_path: str = ""
    source_image_w: int = 0
    source_image_h: int = 0

    def all_elements(self) -> List[DetectedElement]:
        result = []
        for root in self.roots:
            result.extend(root.all_descendants())
        return result
```

### 5.3 `run.py` — CLI entry point

```python
# run.py
import argparse
from pathlib import Path
from pipeline.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Convert raster slide to editable PPTX")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output .pptx path")
    parser.add_argument("--no-vlm", action="store_true", help="Skip Gemini refinement")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    output_path = args.output or Path(args.image).stem + "_output.pptx"
    run_pipeline(
        image_path=args.image,
        output_path=output_path,
        vlm_enabled=not args.no_vlm,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
```

---

## 6. Phase 1 — Detection

### 6.1 YOLO pre-pass (`phase1_detection/yolo_detector.py`)

**Purpose:** Fast first-pass detection to seed Gemini's context. YOLO gives us approximate bounding boxes at high speed. We do NOT rely on YOLO boxes as final detections — they are inputs to Gemini, not outputs of Phase 1.

**Inputs:** Source image (numpy array, BGR)  
**Outputs:** List of `DetectedElement` with `detected_by="yolo"`, `processing_path="undecided"`

**Implementation spec:**

```python
def run_yolo(image: np.ndarray) -> List[DetectedElement]:
    """
    Run YOLO inference on the input image.
    Returns a list of DetectedElement — one per detected box.
    Filters by YOLO_CONF_THRESHOLD from config.
    Maps YOLO class indices to semantic types via YOLO_CLASSES list in config.
    """
```

- Load weights from `config.YOLO_WEIGHTS`; raise clear error if file not found
- Run inference at `config.YOLO_IMAGE_SIZE`
- Apply `config.YOLO_CONF_THRESHOLD` filter
- Map each YOLO class index to the nearest matching `SemanticType` (define the mapping explicitly in a dict in this file — do not hardcode inline)
- Return `List[DetectedElement]`; return empty list (not exception) if YOLO detects nothing

### 6.2 Gemini iterative detection (`phase1_detection/gemini_detector.py`)

**Purpose:** Primary detection. Gemini sees the source image (and in iterations 2+, an annotated version showing existing boxes) and returns structured JSON describing all detected elements including their semantic types. Runs iteratively to recover elements missed in one shot.

**Critical constraint:** The annotated image sent to Gemini (with boxes drawn on it) must NEVER be used anywhere else in the pipeline. Always maintain a clean copy of the source image for all CV operations.

**Inputs:** Source image, initial YOLO detections  
**Outputs:** Final list of `DetectedElement` (merged YOLO + Gemini)

**Iteration logic:**

```
Pass 1: Send source image. Ask Gemini to detect ALL elements.
Pass 2: Draw Pass 1 boxes on a COPY of the image (magenta, 1px). 
        Send annotated copy. Ask: "find missing elements AND correct clipped boxes."
Pass 3: Draw all current boxes. Same prompt.
Stop when: Gemini returns 0 new/corrected boxes OR config.GEMINI_ITER_MAX reached.
```

**Gemini response schema** (enforce via prompt, parse with json.loads):

```json
{
  "elements": [
    {
      "bbox": {"x": 100, "y": 50, "w": 300, "h": 80},
      "semantic_type": "Header",
      "confidence": 0.95,
      "correction_for": null,
      "label": "Slide title"
    }
  ]
}
```

`correction_for`: if non-null, contains the approximate bbox of an existing box this is meant to correct/replace. Used in merger to trigger the replace logic.

**Retry logic:** Use `tenacity` library:
```python
from tenacity import retry, stop_after_attempt, wait_fixed
@retry(stop=stop_after_attempt(config.GEMINI_RETRY_MAX), 
       wait=wait_fixed(config.GEMINI_RETRY_BACKOFF[0]))
def call_gemini(...): ...
```

On all retries exhausted: log warning, return whatever partial results exist.

### 6.3 Merger (`phase1_detection/merger.py`)

**Purpose:** Merge YOLO and Gemini detections into a clean, deduplicated element tree with proper parent-child hierarchy.

**Steps (run in this exact order):**

**Step 1 — Apply Gemini corrections:**
For each Gemini element with `correction_for` populated:
- Find the closest existing YOLO element by spatial proximity to `correction_for` bbox
- Compare Gemini's bbox area vs YOLO's: if Gemini's box is larger (more complete), replace the YOLO box
- If YOLO's box is already larger, keep YOLO and discard the correction (don't let Gemini shrink a working box)

**Step 2 — Add genuinely new Gemini detections:**
For each Gemini element without `correction_for`:
- Check IoU against all existing elements
- If IoU < `config.MERGE_IOU_DUPLICATE_THRESHOLD` with all existing → add as new element
- If IoU ≥ threshold → discard (duplicate)

**Step 3 — Deduplication pass:**
For all elements (YOLO + added Gemini), run IoU dedup:
- For each pair with IoU ≥ `config.MERGE_IOU_DUPLICATE_THRESHOLD` → keep higher-confidence one

**Step 4 — Containment → nesting:**
Sort elements by area descending (largest first). For each pair where smaller is ≥ `config.MERGE_CONTAINMENT_THRESHOLD` inside larger:
- Do NOT suppress. Append smaller to `larger.children`.
- Remove smaller from the flat top-level list (it lives in the tree, not at root level).
- Exception: if BOTH elements are the same semantic type and nearly identical size (within 5%), treat as duplicate instead of parent-child.

**Step 5 — Build ElementTree:**
Remaining flat list (after nesting) becomes `ElementTree.roots`.

### 6.4 Prompts (`phase1_detection/prompts.py`)

All Gemini prompt strings live in this single file. No prompt strings anywhere else in the codebase.

```python
DETECTION_PASS_1 = """
You are analyzing a slide image to identify all distinct visual elements.

Detect every element visible in this slide. For each element, return:
- bbox: {x, y, w, h} in pixels from the top-left corner of the image
- semantic_type: one of [Section, NestedSection, TextBlock, Header, Icon, DiagramShape, Arrow, Table, Chart, BackgroundArt]
- confidence: 0.0–1.0
- label: a short descriptive label (e.g. "Tier 1 container", "Year 1943 milestone card")
- correction_for: null (always null in Pass 1)

Semantic type definitions:
- Section: large container that holds other elements (e.g. a tier panel, a major card group)
- NestedSection: smaller container inside a Section (e.g. an individual milestone card, a sub-panel)
- TextBlock: body text, bullet lists, descriptions
- Header: slide title or major section heading
- Icon: small graphic symbol, logo, illustration (not reconstructible as a simple shape)
- DiagramShape: geometric shape with fill color (rectangle, circle, triangle, star, hexagon, pill/capsule)
- Arrow: directional connector between elements
- Table: tabular data grid
- Chart: data visualization (bar, pie, line chart)
- BackgroundArt: decorative element — texture, border art, illustrated corner decoration

Rules:
- Include ALL elements, including decorative ones
- Detect nested elements as separate entries (a Section AND its children are separate entries)
- For text inside a shape, the shape and its text are SEPARATE elements
- Do NOT merge overlapping elements — return them all
- Return ONLY valid JSON. No explanation text.

Return format:
{"elements": [...]}
"""

DETECTION_PASS_N = """
The image shows a slide with boxes already drawn (in magenta) around previously detected elements.

Your task:
1. Find any elements that do NOT yet have a magenta box around them — return these as new detections (correction_for: null)
2. Find any existing magenta boxes that look CLIPPED or TOO SMALL relative to the actual element they cover — return a corrected, larger box for these (set correction_for to the approximate bbox of the clipped box)

Use the same schema as before. Return ONLY valid JSON.
{"elements": [...]}
"""

COMPLEXITY_ASSESSMENT = """
Look at this cropped region of a slide. 

Assess whether this region is:
- SIMPLE: contains only plain colored shapes and/or clean text that could be accurately recreated as native PowerPoint shapes (rectangles, circles, triangles) with text overlaid
- COMPLEX: contains illustrations, photographs, detailed icons, textures, gradients, or content that cannot be faithfully recreated as simple geometric shapes

Return ONLY one word: SIMPLE or COMPLEX
"""
```

---

## 7. Phase 2 — Reconstruct vs. Crop Decision

**File:** `phase2_decision/classifier.py`

**Purpose:** For each element in the tree, decide whether to process it via the reconstruct path (Phase 3A — native PowerPoint shapes) or the crop path (Phase 3B — image tile + text overlay).

### Decision logic

```python
def classify_element(element: DetectedElement, source_image: np.ndarray) -> ProcessingPath:
    """
    Assign element.processing_path = "reconstruct" or "crop".
    Sets the path on the element in-place. Returns the path for logging.
    """
```

**Step 1 — Semantic type overrides (check first, skip scoring if matched):**

```python
ALWAYS_CROP = {"BackgroundArt", "Chart", "Icon"}
ALWAYS_RECONSTRUCT = {"Header", "TextBlock", "Arrow"}

if element.semantic_type in ALWAYS_CROP: return "crop"
if element.semantic_type in ALWAYS_RECONSTRUCT: return "reconstruct"
```

**Step 2 — Complexity scoring (for ambiguous types: Section, NestedSection, DiagramShape, Table):**

Crop the element's region from the source image, then compute three scores:

```python
# 1. Pixel entropy (0–8 bits; high = complex texture/illustration)
from skimage.measure import shannon_entropy
entropy_score = shannon_entropy(crop_gray) / 8.0   # normalize to 0–1

# 2. Edge density (Canny edges / total pixels; high = complex fine detail)
edges = cv2.Canny(crop_gray, 50, 150)
edge_score = np.sum(edges > 0) / (element.bbox.w * element.bbox.h)
edge_score = min(edge_score / 0.3, 1.0)            # normalize (0.3 = empirical max for simple shapes)

# 3. Sub-element density (children per 100k pixels; high = many nested things)
child_count = len(element.children)
area_100k = element.bbox.area / 100_000
density_score = min(child_count / max(area_100k, 0.1) / 5.0, 1.0)  # normalize

complexity = (
    entropy_score * config.COMPLEXITY_ENTROPY_WEIGHT +
    edge_score * config.COMPLEXITY_EDGE_DENSITY_WEIGHT +
    density_score * config.COMPLEXITY_DETECTION_DENSITY_WEIGHT
)

return "crop" if complexity >= config.CROP_THRESHOLD else "reconstruct"
```

**Step 3 — Run recursively on all descendants:**

```python
def classify_tree(tree: ElementTree, source_image: np.ndarray) -> None:
    for element in tree.all_elements():
        classify_element(element, source_image)
```

---

## 8. Phase 3A — Reconstruct Path

### 8.1 Shape classifier (`phase3a_reconstruct/shape_classifier.py`)

**Purpose:** For elements on the reconstruct path with a visual shape (DiagramShape, Section, NestedSection), classify the shape type from the source image.

```python
def classify_shape(element: DetectedElement, source_image: np.ndarray) -> str:
    """
    Returns shape_type string: "rectangle", "circle", "triangle", "star",
    "pentagon", "hexagon", "pill", "polygon", "arrow_triangle"
    Sets element.shape_type and element.fill_color in-place.
    """
```

**Implementation:**

1. Crop the element region. Convert to grayscale. Threshold to binary.
2. Find the largest contour (the shape boundary).
3. Approximate contour with `cv2.approxPolyDP` using epsilon = `config.SHAPE_VERTEX_TOLERANCE * perimeter`.
4. Count vertices → classify:

```python
n = len(approx)
if n == 3: shape_type = "triangle"
elif n == 4:
    # check aspect ratio: if very elongated and thin → probably arrow head
    ar = w / h if w > h else h / w
    shape_type = "arrow_triangle" if ar > 3.0 else "rectangle"
elif n >= 5:
    hull = cv2.convexHull(contour, returnPoints=False)
    hull_ratio = len(hull) / n
    if hull_ratio < config.SHAPE_STAR_CONCAVITY_RATIO:
        shape_type = "star"
    elif n == 5: shape_type = "pentagon"
    elif n == 6: shape_type = "hexagon"
    else: shape_type = "polygon"
else:
    # Check circularity: 4π*area / perimeter²
    circularity = 4 * math.pi * area / (perimeter ** 2)
    if circularity > 0.85: shape_type = "circle"
    # Check pill/capsule: rectangle with high corner roundness
    elif aspect_ratio < 3.0 and circularity > 0.6: shape_type = "pill"
    else: shape_type = "rectangle"
```

5. Sample fill color: median RGB of pixels inside the contour (excluding border pixels).
6. Sample border color: median RGB of contour border pixels.

### 8.2 Shape renderer (`phase3a_reconstruct/shape_renderer.py`)

**Purpose:** Map shape_type → `MSO_SHAPE` constant → create PowerPoint shape with correct fill and border.

**Shape type → MSO_SHAPE mapping:**

```python
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt
from pptx.dml.color import RGBColor

SHAPE_MAP = {
    "rectangle":      PP_PLACEHOLDER.BODY,       # use add_shape with MSO_SHAPE.RECTANGLE
    "circle":         "OVAL",
    "triangle":       "ISOSCELES_TRIANGLE",
    "star":           "STAR_5_POINT",
    "pentagon":       "PENTAGON",
    "hexagon":        "HEXAGON",
    "pill":           "ROUNDED_RECTANGLE",        # set corner_radius to max
    "polygon":        "RECTANGLE",               # fallback for unknown polygon
    "arrow_triangle": "RIGHT_TRIANGLE",
}
```

For `"pill"` shape: after creating the `ROUNDED_RECTANGLE`, set the corner rounding to maximum:
```python
shape.adjustments[0] = 50000  # 50000 = maximum corner radius in pptx units
```

Apply fill and border:
```python
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(*element.fill_color)
shape.line.color.rgb = RGBColor(*element.border_color)
shape.line.width = Pt(1.5)
```

### 8.3 Per-line OCR (`phase3a_reconstruct/ocr.py`)

**Purpose:** Extract text from an element region at per-line granularity with font metadata. This module is SHARED between the reconstruct and crop paths — import it in both.

```python
def extract_lines(element: DetectedElement, source_image: np.ndarray) -> List[OCRLine]:
    """
    Run Tesseract on the element's region (or its safe_text_area for non-rectangular shapes).
    Returns one OCRLine per visual line of text.
    """
```

**Implementation steps:**

1. **Determine OCR region:** For rectangular elements, use `element.bbox`. For non-rectangular shapes (triangle, star, hexagon, pill), compute a "safe text area" — an inset rectangle that fits within the actual shape silhouette:
   - Triangle: lower-center 60% of bbox (triangles are widest at bottom)
   - Star: center 40% of bbox (star inner region)
   - Pill/Circle: center 70% of bbox (avoid curved ends)
   - Default: full bbox

2. **Run Tesseract:** Use `pytesseract.image_to_data()` with `output_type=Output.DICT`. Request `--psm 6` (assume uniform block of text).

3. **Group into visual lines:** Group word-level Tesseract output by `line_num`. For each group:
   - `text`: join words with spaces, strip
   - `bbox`: union of all word bboxes in the line, converted back to absolute image coordinates
   - `font_size_pt`: `line_height_px * config.FONT_SIZE_HEIGHT_RATIO`
   - `bold`: `median_stroke_thickness / font_size_px > config.FONT_BOLD_STROKE_RATIO AND font_size_px >= config.FONT_BOLD_MIN_SIZE_PX`
   - `color_rgb`: sample from ink pixels in the line bbox (pixels darker than surroundings)
   - `is_art`: `True` if text contains no `[a-zA-Z0-9]` characters

4. **Filter:** Skip lines where `text.strip() == ""`. Skip lines where `is_art == True` (preserve position, but will not be overlaid).

5. **Stroke thickness for bold detection:**
```python
line_crop_gray = crop to line_bbox, convert to grayscale
binary = cv2.threshold(line_crop_gray, 128, 255, cv2.THRESH_BINARY_INV)[1]
# For each row, count consecutive dark pixel runs
horizontal_runs = [len of each run of 1s in each row]
median_stroke = np.median(horizontal_runs) if horizontal_runs else 0
```

---

## 9. Phase 3B — Crop Path

### 9.1 Background detection (`phase3b_crop/background.py`)

**Purpose:** Determine if the slide (or a specific element region) has a dark or light background. This affects how local-fill text removal works.

```python
def is_dark_background(image: np.ndarray, bbox: Optional[BBox] = None) -> bool:
    """
    Returns True if the region (or full image) has a dark background.
    Samples from corners and edges of the region to avoid sampling foreground content.
    """
    region = crop(image, bbox) if bbox else image
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Sample 20px border around the region (background is usually at edges)
    border_pixels = get_border_pixels(gray, border_width=20)
    mean_brightness = np.mean(border_pixels)
    return mean_brightness < config.DARK_BACKGROUND_THRESHOLD
```

### 9.2 Tiler (`phase3b_crop/tiler.py`)

**Purpose:** Crop the element's region from the source image, save as a tile PNG.

```python
def crop_tile(element: DetectedElement, source_image: np.ndarray) -> str:
    """
    Crops element.bbox from source_image.
    Saves to config.TILE_OUTPUT_DIR / f"{element.id}.png".
    Sets element.tile_path in-place.
    Returns the saved path.
    """
```

### 9.3 Text remover (`phase3b_crop/text_remover.py`)

**Purpose:** Given a cropped tile and its OCR lines, fill each text region with the local background color, producing a clean text-free tile.

```python
def remove_text(tile: np.ndarray, ocr_lines: List[OCRLine], 
                tile_bbox: BBox, dark_bg: bool) -> np.ndarray:
    """
    For each OCR line (that is not is_art):
    1. Sample background color in a ring around the line bbox
    2. Fill the line bbox with that sampled color
    Returns the cleaned tile (text-free).
    """
```

**Ring sampling (critical detail):**

```python
def sample_background_color(tile: np.ndarray, line_bbox: BBox, 
                             element_bbox: BBox, dark_bg: bool) -> tuple[int, int, int]:
    ring_w = config.FILL_RING_WIDTH_PX
    
    # Clamp ring to element bounds to prevent sampling outside the element
    rx1 = max(line_bbox.x - ring_w, element_bbox.x)
    ry1 = max(line_bbox.y - ring_w, element_bbox.y)
    rx2 = min(line_bbox.x2 + ring_w, element_bbox.x2)
    ry2 = min(line_bbox.y2 + ring_w, element_bbox.y2)
    
    # Build mask: ring area MINUS the text bbox itself
    ring_mask = np.zeros(tile.shape[:2], dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    ring_mask[line_bbox.y:line_bbox.y2, line_bbox.x:line_bbox.x2] = False
    
    # Exclude ink pixels from ring sample
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    ink_mask = gray < config.FILL_INK_DARKNESS_THRESHOLD
    sample_mask = ring_mask & ~ink_mask
    
    if np.sum(sample_mask) < config.FILL_RING_MIN_SAMPLE_PX:
        # Fallback: sample center region of the element (away from text and edges)
        cx, cy = element_bbox.w // 2, element_bbox.h // 2
        margin = int(min(element_bbox.w, element_bbox.h) * config.FILL_FALLBACK_SAMPLE_REGION)
        sample_region = tile[cy-margin:cy+margin, cx-margin:cx+margin]
        return tuple(np.median(sample_region.reshape(-1, 3), axis=0).astype(int))
    
    sampled_pixels = tile[sample_mask]
    return tuple(np.median(sampled_pixels, axis=0).astype(int))
```

---

## 10. Phase 4 — PPTX Assembly

### 10.1 Assembler (`phase4_assembly/assembler.py`)

**Purpose:** Walk the ElementTree and write a `.pptx` file. The top-level orchestrator for Phase 4.

```python
def assemble(tree: ElementTree, output_path: str) -> str:
    """
    Creates a new PowerPoint presentation from the ElementTree.
    Returns the path to the saved .pptx file.
    """
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation()
    prs.slide_width = Emu(config.SLIDE_WIDTH_EMU)
    prs.slide_height = Emu(config.SLIDE_HEIGHT_EMU)

    slide_layout = prs.slide_layouts[6]  # blank layout
    slide = prs.slides.add_slide(slide_layout)

    rendered_ids = set()

    # Render roots in z-order (largest area first = furthest back)
    sorted_roots = sorted(tree.roots, key=lambda e: e.bbox.area, reverse=True)
    for element in sorted_roots:
        render_element(element, slide, rendered_ids, tree)

    prs.save(output_path)
    return output_path
```

### 10.2 Element rendering (`phase4_assembly/assembler.py`)

```python
def render_element(element: DetectedElement, target, rendered_ids: set, tree: ElementTree):
    """
    target: either a slide object or a GroupShape's .shapes collection.
    Renders element and recursively renders its children into a GroupShape.
    """
    if element.id in rendered_ids:
        return
    rendered_ids.add(element.id)
    # Pre-register all children so containment guard never re-processes them
    rendered_ids.update(element.child_ids())

    if element.children:
        render_group(element, target, rendered_ids, tree)
    else:
        render_leaf(element, target)
```

**Group rendering:**

```python
def render_group(element: DetectedElement, target, rendered_ids: set, tree: ElementTree):
    """Create an empty GroupShape, render parent + all children INTO it."""
    grp = target.shapes.add_group_shape()

    # Render parent element into group (its own shape/tile if it has one)
    if element.processing_path == "reconstruct" and element.shape_type:
        render_shape_into(element, grp)
    elif element.processing_path == "crop" and element.tile_path:
        render_tile_into(element, grp)

    # Render children into group (sorted largest-first for z-order)
    for child in sorted(element.children, key=lambda e: e.bbox.area, reverse=True):
        render_element(child, grp, rendered_ids, tree)

    # Render text boxes always last (always on top within the group)
    render_text_lines(element, grp)
```

**Leaf rendering (elements with no children):**

```python
def render_leaf(element: DetectedElement, target):
    route = {
        "Header": render_text_block,
        "TextBlock": render_text_block,
        "Arrow": render_arrow,
        "BackgroundArt": render_tile,
    }
    if element.processing_path == "reconstruct":
        if element.shape_type: render_shape(element, target)
        render_text_lines(element, target)
    elif element.processing_path == "crop":
        render_tile(element, target)
        render_text_lines(element, target)
    elif element.semantic_type == "Arrow":
        render_arrow(element, target)
    else:
        render_text_block(element, target)
```

### 10.3 Coordinate conversion

All coordinates throughout the pipeline are in pixels relative to the source image. Convert to EMU for PPTX:

```python
def px_to_emu(px: int, axis: str = "x") -> int:
    """axis: 'x' uses SLIDE_WIDTH, 'y' uses SLIDE_HEIGHT"""
    if axis == "x":
        return int(px * config.SLIDE_WIDTH_EMU / config.SLIDE_WIDTH_PX)
    else:
        return int(px * config.SLIDE_HEIGHT_EMU / config.SLIDE_HEIGHT_PX)
```

### 10.4 Arrow writer (`phase4_assembly/arrow_writer.py`)

**Purpose:** Write a PowerPoint connector with a real arrowhead. Must write XML directly — python-pptx's high-level arrowhead API is a no-op.

```python
from pptx.oxml.ns import qn
from pptx.util import Pt, Emu
from pptx.enum.shapes import MSO_CONNECTOR

def render_arrow(element: DetectedElement, target):
    """
    Creates a STRAIGHT connector from element.arrow_start to element.arrow_end.
    Writes arrowhead directly to XML.
    """
    if not element.arrow_start or not element.arrow_end:
        return   # No geometry → skip (do not draw fallback rectangle)

    x1, y1 = element.arrow_start
    x2, y2 = element.arrow_end

    connector = target.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Emu(px_to_emu(x1, "x")), Emu(px_to_emu(y1, "y")),
        Emu(px_to_emu(x2, "x")), Emu(px_to_emu(y2, "y"))
    )
    connector.line.color.rgb = RGBColor(0, 0, 0)
    connector.line.width = Pt(config.ARROW_LINE_WIDTH_PT)

    # Write arrowhead directly to XML (python-pptx high-level API is a no-op for this)
    _set_arrowhead(connector)

def _set_arrowhead(connector, end="tail", arrow_type="triangle", width="med", length="med"):
    ln = connector.line._get_or_add_ln()
    tag = qn("a:tailEnd") if end == "tail" else qn("a:headEnd")
    existing = ln.find(tag)
    if existing is not None:
        ln.remove(existing)
    el = ln.makeelement(tag, {"type": arrow_type, "w": width, "len": length})
    ln.append(el)
```

### 10.5 Text placer (`phase4_assembly/text_placer.py`)

```python
def render_text_lines(element: DetectedElement, target):
    """
    Place each OCRLine as its own no-wrap text box at its true position.
    Skip lines where is_art=True.
    """
    for line in element.ocr_lines:
        if line.is_art or not line.text.strip():
            continue

        txBox = target.shapes.add_textbox(
            Emu(px_to_emu(line.bbox.x, "x")),
            Emu(px_to_emu(line.bbox.y, "y")),
            Emu(px_to_emu(line.bbox.w, "x") + px_to_emu(config.TEXT_BOX_PADDING_PX, "x")),
            Emu(px_to_emu(line.bbox.h, "y")),
        )
        tf = txBox.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = line.text
        run.font.name = config.FONT_FAMILY
        run.font.size = Pt(
            max(config.FONT_SIZE_MIN_PT, 
                min(line.font_size_pt, config.FONT_SIZE_MAX_PT))
        )
        run.font.bold = line.bold
        run.font.color.rgb = RGBColor(*line.color_rgb)
```

---

## 11. Phase 5 — QA / SSIM Validation

### 11.1 PPTX renderer (`phase5_qa/renderer.py`)

**Purpose:** Render the output `.pptx` back to a PNG image for SSIM comparison. Uses LibreOffice headless.

```python
import subprocess

def render_pptx_to_image(pptx_path: str, output_dir: str) -> str:
    """
    Uses LibreOffice headless to render the first slide to a PNG.
    Returns path to rendered image.
    """
    cmd = [
        "libreoffice", "--headless", "--convert-to", "png",
        "--outdir", output_dir, pptx_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice render failed: {result.stderr.decode()}")
    stem = Path(pptx_path).stem
    return str(Path(output_dir) / f"{stem}.png")
```

### 11.2 SSIM computation (`phase5_qa/ssim.py`)

```python
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

def compute_ssim(source_path: str, rendered_path: str, tree: ElementTree) -> dict:
    """
    Returns a dict with:
    - overall_ssim: float
    - passed: bool (overall_ssim >= config.SSIM_PASS_THRESHOLD)
    - element_scores: list of {id, semantic_type, ssim, flagged}
    - type_summary: {SemanticType: mean_ssim}
    """
    source = cv2.imread(source_path)
    rendered = cv2.imread(rendered_path)

    # Resize rendered to match source dimensions
    if source.shape != rendered.shape:
        rendered = cv2.resize(rendered, (source.shape[1], source.shape[0]))

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)

    # Overall SSIM
    overall, diff = ssim(source_gray, rendered_gray, 
                         win_size=config.SSIM_WINDOW_SIZE, full=True)

    # Per-element SSIM (walk full tree)
    element_scores = []
    type_scores = {}

    for elem in tree.all_elements():
        b = elem.bbox
        src_crop = source_gray[b.y:b.y2, b.x:b.x2]
        ren_crop = rendered_gray[b.y:b.y2, b.x:b.x2]

        if src_crop.size == 0 or ren_crop.size == 0:
            continue

        if src_crop.shape != ren_crop.shape:
            ren_crop = cv2.resize(ren_crop, (src_crop.shape[1], src_crop.shape[0]))

        # Need minimum size for SSIM window
        if min(src_crop.shape) < config.SSIM_WINDOW_SIZE:
            continue

        elem_ssim = ssim(src_crop, ren_crop, win_size=config.SSIM_WINDOW_SIZE)
        flagged = elem_ssim < config.SSIM_ELEMENT_FLAG_THRESHOLD

        element_scores.append({
            "id": elem.id,
            "semantic_type": elem.semantic_type,
            "ssim": round(elem_ssim, 4),
            "flagged": flagged
        })

        if elem.semantic_type not in type_scores:
            type_scores[elem.semantic_type] = []
        type_scores[elem.semantic_type].append(elem_ssim)

    type_summary = {t: round(float(np.mean(v)), 4) for t, v in type_scores.items()}

    return {
        "overall_ssim": round(float(overall), 4),
        "passed": overall >= config.SSIM_PASS_THRESHOLD,
        "element_scores": element_scores,
        "type_summary": type_summary
    }
```

### 11.3 Reporter (`phase5_qa/reporter.py`)

```python
import json
from pathlib import Path

def save_report(qa_result: dict, output_path: str, diff_image=None):
    """
    Saves JSON QA report and optional visual diff image.
    """
    report_path = Path(output_path).with_suffix(".qa.json")
    with open(report_path, "w") as f:
        json.dump(qa_result, f, indent=2)

    if diff_image is not None and config.QA_DIFF_SAVE:
        diff_path = Path(output_path).with_suffix(".diff.png")
        cv2.imwrite(str(diff_path), diff_image)
```

---

## 12. Data Model

See Section 5.2 for the full `models.py` specification. Key design decisions:

- **`ElementTree.roots`** contains only top-level elements; nested elements live exclusively in `children` lists
- **`ElementTree.all_elements()`** returns every element in the tree (flat, depth-first) — used by QA, classifier, and any phase that needs to iterate all elements
- **`DetectedElement.child_ids()`** returns a recursive set of all descendant IDs — used to pre-register children in `rendered_ids` before rendering begins
- **`BBox`** stores coordinates in source image pixels throughout — conversion to EMU happens only in Phase 4 at render time
- **`OCRLine.is_art`** is the universal flag for non-text content that looks like text — set during OCR, checked in text removal and text overlay

---

## 13. Agent Prompts

Use these prompts to instruct AI coding agents (Antigravity, Codex, Claude, etc.) to implement each phase. Always include `config.py`, `requirements.txt`, and `models.py` in the agent's context before using these prompts.

---

### PROMPT: Phase 0 — Project setup

```
You are setting up a new Python project for a computer vision pipeline that converts
raster slide images into editable PowerPoint files. This is a clean build — no existing
code to modify or build on top of.

Create the following:
1. The directory structure exactly as shown in Section 4 of the PRD (all __init__.py
   files, all module files as empty stubs with correct docstrings)
2. Implement models.py exactly as specified in Section 5.2 — this is the data model
   used by every other module
3. Implement run.py exactly as specified in Section 5.3
4. Write a basic test in tests/test_models.py that:
   - Creates a DetectedElement with two children
   - Calls all_descendants() and verifies it returns 3 elements (parent + 2 children)
   - Calls child_ids() and verifies it returns the 2 children's IDs
   - Verifies that model_rebuild() resolves the self-reference without error
5. Run the test and confirm it passes before stopping

Do not implement any pipeline logic yet. Only models.py, run.py, directory structure,
and the model test. Commit with message: "feat: project structure + data model"
```

---

### PROMPT: Phase 1A — YOLO detection

```
Context: This is phase1_detection/yolo_detector.py in a raster-to-PPTX pipeline.
Read config.py and models.py before writing any code.

Implement run_yolo(image: np.ndarray) -> List[DetectedElement] as specified in
Section 6.1 of the PRD. Specifically:

1. Load YOLO model from config.YOLO_WEIGHTS using ultralytics.YOLO
2. Handle the case where weights file does not exist — raise FileNotFoundError
   with a clear message telling the user where to place the weights file
3. Run inference at config.YOLO_IMAGE_SIZE with config.YOLO_CONF_THRESHOLD
4. Map YOLO class indices to SemanticType using config.YOLO_CLASSES
   Define a YOLO_TO_SEMANTIC dict at the top of the file:
   {"header": "Header", "content_block": "Section", "text_zone": "TextBlock",
    "chart_area": "Chart", "table": "Table", "image": "BackgroundArt",
    "icon": "Icon", "arrow": "Arrow"}
5. Return List[DetectedElement] — one per detection above confidence threshold
6. Return empty list (not exception) if no detections

Write a test in tests/test_detection.py:
- Mock the YOLO model to return fake detections
- Verify the output is a list of DetectedElement
- Verify mapping from class index to semantic_type is correct
- Verify empty list is returned when confidence is below threshold

Commit: "feat: phase 1A — YOLO pre-pass detector"
```

---

### PROMPT: Phase 1B — Gemini detection

```
Context: This is phase1_detection/gemini_detector.py. Read config.py, models.py,
and phase1_detection/prompts.py before writing any code.

Implement the Gemini iterative detection as specified in Section 6.2 of the PRD.

Key requirements:
1. Use the tenacity library for retry logic — wrap the Gemini API call with
   @retry(stop=stop_after_attempt(config.GEMINI_RETRY_MAX),
          wait=wait_fixed(config.GEMINI_RETRY_BACKOFF[0]))
2. Load GEMINI_API_KEY from environment variable via python-dotenv
3. Pass 1: send source image + DETECTION_PASS_1 prompt from prompts.py
4. Pass 2+: draw current boxes on a DEEP COPY of the source image (never modify
   the original), send + DETECTION_PASS_N prompt
5. Annotate with magenta (255, 0, 255), 1px thickness, cv2.rectangle
6. CRITICAL: the annotated copy must NEVER be returned or stored — it is only
   for Gemini's input; the original source image remains clean
7. Parse Gemini's JSON response safely — wrap in try/except, log failures, skip
   unparseable responses
8. Stop iteration when response has 0 elements OR config.GEMINI_ITER_MAX reached
9. Cache responses during development: if config.USE_CACHE=True, save/load
   Gemini responses to config.CACHE_DIR keyed by hash of (image_hash, pass_number)

Test requirements:
- Mock the Gemini API
- Verify annotated image is never returned (source image unchanged after call)
- Verify retry fires on API exception
- Verify iteration stops at GEMINI_ITER_MAX
- Verify JSON parse failures are handled gracefully

Commit: "feat: phase 1B — Gemini iterative detection"
```

---

### PROMPT: Phase 1C — Merger

```
Context: This is phase1_detection/merger.py. Read config.py and models.py.

Implement merge_detections(yolo_elements, gemini_elements) -> ElementTree as
specified in Section 6.3 of the PRD. The 5 steps must run in EXACTLY the order
listed in the PRD:

1. Apply Gemini corrections (correction_for is not null)
2. Add genuinely new Gemini detections (IoU < MERGE_IOU_DUPLICATE_THRESHOLD)
3. IoU dedup pass on the full merged set
4. Containment → nesting (threshold: MERGE_CONTAINMENT_THRESHOLD)
5. Build ElementTree from remaining flat roots

For containment check, use this formula:
   intersection_area = area of overlap between box_a and box_b
   containment_ratio = intersection_area / smaller_box.area
   if containment_ratio >= threshold → nest smaller inside larger

Add a helper function iou(bbox_a, bbox_b) -> float at the top of the file.
Add a helper function containment_ratio(smaller, larger) -> float.

Tests:
- iou() returns 0.0 for non-overlapping boxes, 1.0 for identical boxes
- Two elements with IoU 0.8 → only one survives dedup
- One element 90% inside another → becomes a child, not suppressed
- ElementTree.roots contains only top-level elements
- ElementTree.all_elements() returns all elements including nested children

Commit: "feat: phase 1C — detection merger with containment nesting"
```

---

### PROMPT: Phase 2 — Reconstruct vs. Crop Classifier

```
Context: This is phase2_decision/classifier.py. Read config.py and models.py.

Implement classify_element(element, source_image) -> ProcessingPath and
classify_tree(tree, source_image) -> None as specified in Section 7 of the PRD.

Requirements:
1. Semantic type overrides FIRST (check before scoring):
   ALWAYS_CROP = config.ALWAYS_CROP_TYPES
   ALWAYS_RECONSTRUCT = config.ALWAYS_RECONSTRUCT_TYPES
2. For unoveridden types, compute three scores as in Section 7
3. Use skimage.measure.shannon_entropy for entropy score
4. Use cv2.Canny for edge detection
5. All thresholds from config — no hardcoded values
6. classify_tree() walks tree.all_elements() and sets processing_path in-place
7. Log the decision (semantic_type, score, path) for every element at DEBUG level

Tests:
- BackgroundArt always → "crop" regardless of complexity score
- Header always → "reconstruct" regardless of complexity score  
- A solid blue rectangle → should score low complexity → "reconstruct"
- A photograph → should score high complexity → "crop"
- classify_tree() sets processing_path on all elements including nested children

Commit: "feat: phase 2 — reconstruct vs crop classifier"
```

---

### PROMPT: Phase 3A — Reconstruct path (shape + OCR)

```
Context: This implements phase3a_reconstruct/. Read config.py and models.py.
There are two files to implement: shape_classifier.py and ocr.py.

SHAPE CLASSIFIER (shape_classifier.py):
Implement classify_shape(element, source_image) -> str as in Section 8.1.
- Use cv2.approxPolyDP with epsilon = config.SHAPE_VERTEX_TOLERANCE * perimeter
- Implement the full vertex-count → shape_type decision tree from Section 8.1
- Implement concavity check for star detection (hull_ratio < config.SHAPE_STAR_CONCAVITY_RATIO)
- Sample fill_color and border_color and set them on the element in-place
- Return "rectangle" as fallback for any unclassified shape

OCR (ocr.py):
Implement extract_lines(element, source_image) -> List[OCRLine] as in Section 8.3.
- Implement safe_text_area() that returns an inset BBox per shape_type
- Run pytesseract.image_to_data() with --psm 6
- Group by line_num, compute per-line font_size, color, bold, is_art
- For bold: implement median horizontal dark-run measurement, gate to FONT_BOLD_MIN_SIZE_PX
- Filter empty lines and is_art lines

Tests:
- An equilateral triangle contour → "triangle"
- A 5-pointed star contour → "star"
- A circle contour → "circle"
- A very wide thin rectangle → "rectangle" (not "arrow_triangle" unless thin)
- OCR on image with known text → returns correct OCRLine with correct text
- is_art=True for line containing only "→" or "●"
- is_art=False for line containing "1943"

Commit: "feat: phase 3A — shape classifier and per-line OCR"
```

---

### PROMPT: Phase 3B — Crop path

```
Context: This implements phase3b_crop/. Read config.py and models.py.
Implement background.py, tiler.py, and text_remover.py as in Section 9.

BACKGROUND (background.py):
- Implement is_dark_background(image, bbox=None) as in Section 9.1
- Sample from 20px border of the region (not center — center may be foreground)
- Threshold at config.DARK_BACKGROUND_THRESHOLD

TILER (tiler.py):
- Implement crop_tile(element, source_image) -> str
- Create config.TILE_OUTPUT_DIR if it doesn't exist
- Save as PNG, set element.tile_path in-place
- Return the saved path

TEXT REMOVER (text_remover.py):
- Implement remove_text(tile, ocr_lines, tile_bbox, dark_bg) -> np.ndarray
- Implement sample_background_color() EXACTLY as in Section 9.3
  including the ring-clamping and ink-exclusion logic
- Fill text bboxes using cv2.rectangle with the sampled color
- Skip lines where is_art=True
- Return the cleaned tile as a numpy array

Tests:
- A white slide → is_dark_background returns False
- A dark navy slide → is_dark_background returns True
- crop_tile saves a file and sets element.tile_path
- remove_text on an image with "TEST" in white on blue background:
  the returned image should have blue pixels where "TEST" was

Commit: "feat: phase 3B — crop path: tiler, background detection, text removal"
```

---

### PROMPT: Phase 4 — PPTX Assembly

```
Context: This implements phase4_assembly/. Read config.py and models.py.
Implement assembler.py, group_builder.py, text_placer.py, and arrow_writer.py
as specified in Section 10.

Key constraints that are easy to get wrong — read these carefully:
1. GroupShape API: add_group_shape() creates EMPTY group. Shapes are added INTO
   the group via grp.shapes.add_shape(), grp.shapes.add_textbox(), etc.
   Do NOT try to pass existing shapes into add_group_shape().
2. Arrow arrowhead: connector.line.end_arrowhead = N is a NO-OP in python-pptx.
   Use _set_arrowhead() with lxml XML manipulation as in Section 10.4.
   Always verify the XML contains <a:tailEnd> after calling _set_arrowhead().
3. Coordinate conversion: ALL pixel→EMU conversion uses px_to_emu() from Section 10.3.
   Never hardcode EMU values.
4. Text boxes: set tf.word_wrap = False. Place each OCRLine as its own text box.
   Do NOT try to place multiple lines in one text frame.
5. Z-order: children sort largest-area-first (background shapes render behind smaller shapes).
   Text overlay always renders last (on top of everything else within the group).

Implement px_to_emu() first and unit-test it before anything else.

Tests:
- px_to_emu(0, "x") == 0
- px_to_emu(config.SLIDE_WIDTH_PX, "x") == config.SLIDE_WIDTH_EMU  
- An element with 2 children → creates a GroupShape in output
- An Arrow element with arrow_start and arrow_end → creates a connector
- An Arrow element with NO arrow geometry → nothing is rendered (no fallback rectangle)
- Verify via XML that <a:tailEnd type="triangle"> is present in connector output

Commit: "feat: phase 4 — PPTX assembly with group shapes and arrow XML"
```

---

### PROMPT: Phase 5 — QA and SSIM

```
Context: This implements phase5_qa/. Read config.py and models.py.
Implement renderer.py, ssim.py, and reporter.py as in Section 11.

RENDERER:
- Implement render_pptx_to_image() using LibreOffice headless as in Section 11.1
- Handle timeout (60s) with subprocess.TimeoutExpired
- Raise RuntimeError with clear message if LibreOffice is not installed
  (check with shutil.which("libreoffice"))
- Return path to rendered PNG

SSIM:
- Implement compute_ssim() as in Section 11.2
- Walk tree.all_elements() — score ALL descendants, not just roots
- Use seen-ID set to avoid double-counting any element
- Skip elements where crop is too small for SSIM window (min dimension < SSIM_WINDOW_SIZE)
- Return the full dict structure from Section 11.2

REPORTER:
- Save JSON report with json.dump (indent=2)
- Save visual diff image only if config.QA_DIFF_SAVE=True

Then implement pipeline/pipeline.py — the top-level orchestrator that calls
phases 1 through 5 in sequence and passes outputs between them:

def run_pipeline(image_path, output_path, vlm_enabled=True, debug=False):
    source_image = cv2.imread(image_path)
    # Phase 1
    yolo_elements = run_yolo(source_image)
    gemini_elements = run_gemini(source_image, yolo_elements) if vlm_enabled else []
    tree = merge_detections(yolo_elements, gemini_elements)
    # Phase 2
    classify_tree(tree, source_image)
    # Phase 3 (both paths, per element)
    for element in tree.all_elements():
        if element.processing_path == "reconstruct":
            classify_shape(element, source_image)
            element.ocr_lines = extract_lines(element, source_image)
        elif element.processing_path == "crop":
            crop_tile(element, source_image)
            dark_bg = is_dark_background(source_image, element.bbox)
            element.ocr_lines = extract_lines(element, source_image)
            tile = cv2.imread(element.tile_path)
            clean_tile = remove_text(tile, element.ocr_lines, element.bbox, dark_bg)
            cv2.imwrite(element.tile_path, clean_tile)
    # Phase 4
    output_pptx = assemble(tree, output_path)
    # Phase 5
    rendered_img = render_pptx_to_image(output_pptx, str(config.QA_OUTPUT_DIR))
    qa_result = compute_ssim(image_path, rendered_img, tree)
    save_report(qa_result, output_pptx)
    return qa_result

Tests:
- Full end-to-end test on sample_slide.png (simple slide)
- Verify QA JSON is saved
- Verify overall_ssim key exists in output
- Verify element_scores includes at least one entry

Commit: "feat: phase 5 — QA SSIM loop + pipeline orchestrator"
```

---

## 14. Testing Strategy

### Automated tests (pytest)

Run with: `python -m pytest tests/ -v --cov=pipeline`

Each phase has its own test file. Tests must pass at 100% before the phase is considered complete. No exceptions.

### Manual verification protocol (after all phases are implemented)

Run on each of the following test cases and record SSIM scores:

| Test case | Expected behavior | Pass criteria |
|---|---|---|
| Simple slide (header + blue box + arrow) | All 3 elements detected, arrow has arrowhead, text correct | Overall SSIM ≥ 0.85 |
| Complex slide (triangle + star + double arrow) | Triangle and star rendered as native shapes, no garbage OCR | Overall SSIM ≥ 0.80 |
| Dark background slide (IKEA timeline) | Dark bg detected, tiles clean, text legible | No white artifacts in tiles |
| Nested sections slide (Kal's example) | 3 levels of nesting, GroupShapes present in output | Elements individually selectable in PowerPoint |

### Human visual review protocol

Open each output `.pptx` in PowerPoint (or LibreOffice Impress) and verify:
1. Every grouped container: drag → children move with it
2. Double-click into group: individual children are selectable
3. Click any text block: cursor appears, text is editable
4. Text reads correctly (no garbled characters, no is_art garbage)
5. Colors match source image visually

---

## 15. Known Limitations and Future Work

### Out of scope for v2 (document for future)

| Limitation | Reason | Future path |
|---|---|---|
| Curved/organic connectors (e.g. IKEA snake path) | No reliable CV reconstruction path for bezier curves | Preserve as background crop; explore SVG trace in v3 |
| Font family detection | Requires font fingerprinting dataset | Default to Arial; add font matching in v3 |
| Exact pixel-perfect fidelity on 4+ level nesting | Complexity compounds per nesting level | Iterative fine-tuning on more data |
| Animated elements | Outside PPTX scope for this project | N/A |
| Multi-slide batch with shared style | Each slide treated independently | Shared style inference in v3 |

### YOLO license disclosure

The YOLOv8 model (Ultralytics) is licensed under AGPL-3.0, which has copyleft implications for commercial use. Disclose this dependency to DealVerse AI. Commercial licensing available from Ultralytics if required.

### Shapely (future addition)

`shapely` is listed in `requirements.txt` for future use in polygon containment math (non-rectangular shape intersection calculations). It is not used in v2 but is pre-included to avoid a dependency change during that work.
