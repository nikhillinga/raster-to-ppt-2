"""Phase 3B: Text removal from image tiles."""

from typing import List

import cv2
import numpy as np

import config
from pipeline.models import BBox, OCRLine


def _sample_ring(tile: np.ndarray, line_tile_bbox: BBox, element_bbox: BBox, dark_bg: bool) -> tuple[int, int, int]:
    """Sample the background color from a ring around the text line."""
    ring_w = config.FILL_RING_WIDTH_PX

    # Sampling ring around the text line, CLAMPED to element bounds
    # (element bounds in tile coords = full tile dimensions)
    rx1 = max(line_tile_bbox.x - ring_w, 0)
    ry1 = max(line_tile_bbox.y - ring_w, 0)
    rx2 = min(line_tile_bbox.x2 + ring_w, tile.shape[1])
    ry2 = min(line_tile_bbox.y2 + ring_w, tile.shape[0])

    # Build ring mask: outer rect MINUS inner (text) rect
    ring_mask = np.zeros(tile.shape[:2], dtype=bool)
    ring_mask[ry1:ry2, rx1:rx2] = True
    
    ix1 = max(line_tile_bbox.x, 0)
    iy1 = max(line_tile_bbox.y, 0)
    ix2 = min(line_tile_bbox.x2, tile.shape[1])
    iy2 = min(line_tile_bbox.y2, tile.shape[0])
    
    ring_mask[iy1:iy2, ix1:ix2] = False

    # Exclude ink pixels
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    if dark_bg:
        ink_mask = gray > (255 - config.FILL_INK_DARKNESS_THRESHOLD)
    else:
        ink_mask = gray < config.FILL_INK_DARKNESS_THRESHOLD
        
    sample_mask = ring_mask & ~ink_mask

    if np.sum(sample_mask) < config.FILL_RING_MIN_SAMPLE_PX:
        # Fallback: sample center region of tile
        ch, cw = tile.shape[0] // 2, tile.shape[1] // 2
        m = max(int(min(tile.shape[0], tile.shape[1]) * config.FILL_FALLBACK_SAMPLE_REGION), 5)
        
        c_y1 = max(0, ch - m)
        c_y2 = min(tile.shape[0], ch + m)
        c_x1 = max(0, cw - m)
        c_x2 = min(tile.shape[1], cw + m)
        
        center = tile[c_y1:c_y2, c_x1:c_x2]
        if center.size > 0:
            med = np.median(center.reshape(-1, 3), axis=0)
            return (int(med[0]), int(med[1]), int(med[2]))
        return (255, 255, 255)  # last resort

    sampled = tile[sample_mask]
    med = np.median(sampled, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def remove_text(tile: np.ndarray, ocr_lines: List[OCRLine], element_bbox: BBox, dark_bg: bool) -> np.ndarray:
    """Remove text from the tile by filling the text regions with sampled background colors."""
    result = tile.copy()
    
    for line in ocr_lines:
        if line.is_art:
            continue
            
        # Convert line bbox from source image coords to tile coords
        tile_line_x = line.bbox.x - element_bbox.x
        tile_line_y = line.bbox.y - element_bbox.y
        
        # Clamp to tile dimensions
        tile_line_x = max(0, min(tile_line_x, tile.shape[1]))
        tile_line_y = max(0, min(tile_line_y, tile.shape[0]))
        tile_line_w = max(0, min(line.bbox.w, tile.shape[1] - tile_line_x))
        tile_line_h = max(0, min(line.bbox.h, tile.shape[0] - tile_line_y))
        
        if tile_line_w <= 0 or tile_line_h <= 0:
            continue
            
        tile_line_bbox = BBox(x=tile_line_x, y=tile_line_y, w=tile_line_w, h=tile_line_h)
        
        # Sample background color using _sample_ring (BGR to be used in cv2)
        fill_color_bgr = _sample_ring(tile, tile_line_bbox, element_bbox, dark_bg)
        
        # Fill the text region with sampled color
        cv2.rectangle(
            result,
            (tile_line_bbox.x, tile_line_bbox.y),
            (tile_line_bbox.x2, tile_line_bbox.y2),
            color=fill_color_bgr,
            thickness=-1
        )
        
    return result
