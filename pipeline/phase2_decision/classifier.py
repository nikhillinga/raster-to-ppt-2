"""Phase 2: Decision Classifier — Reconstruct vs. Crop."""

import cv2
import numpy as np
from loguru import logger
from skimage.measure import shannon_entropy

import config
from pipeline.models import DetectedElement, ElementTree


def classify_element(element: DetectedElement, source_image: np.ndarray) -> str:
    """Assign element.processing_path in-place based on complexity scoring. Return the path string."""
    
    # Step 1 — Semantic type overrides (check FIRST, return immediately if matched):
    if element.semantic_type in config.ALWAYS_CROP_TYPES:
        element.processing_path = "crop"
        return "crop"
    if element.semantic_type in config.ALWAYS_RECONSTRUCT_TYPES:
        element.processing_path = "reconstruct"
        return "reconstruct"
        
    # Edge cases to handle:
    # - If element.bbox.area == 0: set processing_path="crop" (can't score empty region)
    if element.bbox.area == 0:
        element.processing_path = "crop"
        return "crop"
        
    # Crop the element region from source_image using element.bbox
    y1, y2 = element.bbox.y, element.bbox.y2
    x1, x2 = element.bbox.x, element.bbox.x2
    
    # Bound the crop to image dimensions
    h_img, w_img = source_image.shape[:2]
    y1 = max(0, min(y1, h_img))
    y2 = max(0, min(y2, h_img))
    x1 = max(0, min(x1, w_img))
    x2 = max(0, min(x2, w_img))
    
    crop = source_image[y1:y2, x1:x2]
    
    # - If crop from source_image is empty (bbox outside image bounds): set processing_path="crop"
    if crop.size == 0:
        element.processing_path = "crop"
        return "crop"

    # Step 2 — Complexity scoring:
    # Convert to grayscale.
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Score 1 — Pixel entropy (0.0 to 1.0):
    entropy_score = shannon_entropy(crop_gray) / 8.0
    
    # Score 2 — Edge density (0.0 to 1.0):
    edges = cv2.Canny(crop_gray, 50, 150)
    edge_score = min(np.sum(edges > 0) / max(element.bbox.area, 1) / 0.3, 1.0)
    
    # Score 3 — Sub-element density (0.0 to 1.0):
    child_count = len(element.children)
    area_100k = element.bbox.area / 100_000
    density_score = min(child_count / max(area_100k, 0.1) / 5.0, 1.0)
    
    complexity = (
        entropy_score * config.COMPLEXITY_ENTROPY_WEIGHT +
        edge_score * config.COMPLEXITY_EDGE_DENSITY_WEIGHT +
        density_score * config.COMPLEXITY_DETECTION_DENSITY_WEIGHT
    )
    
    path = "crop" if complexity >= config.CROP_THRESHOLD else "reconstruct"
    element.processing_path = path
    
    # Step 3 — Log the decision:
    logger.debug(f"Element {element.id} ({element.semantic_type}): "
                 f"complexity={complexity:.3f} → {path}")
                 
    return path


def classify_tree(tree: ElementTree, source_image: np.ndarray) -> None:
    """Call classify_element on every element in the tree (all descendants)."""
    for element in tree.all_elements():
        classify_element(element, source_image)
