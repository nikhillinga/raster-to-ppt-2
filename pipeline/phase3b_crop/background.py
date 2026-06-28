"""Phase 3B: Background detection."""

from typing import Optional

import cv2
import numpy as np

import config
from pipeline.models import BBox


def _get_border_pixels(gray: np.ndarray, border_width: int = 20) -> np.ndarray:
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    
    # Ensure border_width doesn't exceed image dimensions
    border_width = min(border_width, h // 2, w // 2)
    if border_width <= 0:
        return gray.flatten()
        
    mask[:border_width, :] = True    # top
    mask[-border_width:, :] = True   # bottom
    mask[:, :border_width] = True    # left
    mask[:, -border_width:] = True   # right
    return gray[mask]


def is_dark_background(image: np.ndarray, bbox: Optional[BBox] = None) -> bool:
    """Detect if an image or region has a dark background by sampling its outer border."""
    if bbox is not None:
        y1 = max(0, bbox.y)
        y2 = min(image.shape[0], bbox.y2)
        x1 = max(0, bbox.x)
        x2 = min(image.shape[1], bbox.x2)
        crop = image[y1:y2, x1:x2]
    else:
        crop = image
        
    if crop.size == 0:
        return False
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    border_pixels = _get_border_pixels(gray)
    
    if border_pixels.size == 0:
        return False
        
    mean_brightness = np.mean(border_pixels)
    return mean_brightness < config.DARK_BACKGROUND_THRESHOLD
