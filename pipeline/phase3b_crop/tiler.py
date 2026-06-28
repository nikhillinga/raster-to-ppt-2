"""Phase 3B: Tiler (saving crops of detected elements)."""

import cv2
import numpy as np
from loguru import logger

import config
from pipeline.models import DetectedElement


def crop_tile(element: DetectedElement, source_image: np.ndarray) -> str:
    """Crop an element from the source image and save it as a tile."""
    config.TILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    y1 = max(0, element.bbox.y)
    y2 = min(source_image.shape[0], element.bbox.y2)
    x1 = max(0, element.bbox.x)
    x2 = min(source_image.shape[1], element.bbox.x2)
    
    crop = source_image[y1:y2, x1:x2]
    
    if crop.size == 0:
        logger.warning(f"Empty crop for element {element.id}")
        return ""
        
    tile_path = config.TILE_OUTPUT_DIR / f"{element.id}.png"
    cv2.imwrite(str(tile_path), crop)
    
    element.tile_path = str(tile_path)
    return str(tile_path)
