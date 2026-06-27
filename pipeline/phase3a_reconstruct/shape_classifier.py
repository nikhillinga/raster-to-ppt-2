"""Phase 3A: Shape classification."""

import math
import cv2
import numpy as np

import config
from pipeline.models import DetectedElement


def classify_shape(element: DetectedElement, source_image: np.ndarray) -> str:
    """Classify the geometric shape of the element, set properties in-place, and return shape_type."""
    y1, y2 = element.bbox.y, element.bbox.y2
    x1, x2 = element.bbox.x, element.bbox.x2
    
    h_img, w_img = source_image.shape[:2]
    y1 = max(0, min(y1, h_img))
    y2 = max(0, min(y2, h_img))
    x1 = max(0, min(x1, w_img))
    x2 = max(0, min(x2, w_img))
    
    crop = source_image[y1:y2, x1:x2]
    
    fallback_type = "rectangle"
    
    if crop.size == 0:
        element.shape_type = fallback_type
        return fallback_type
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Heuristic to ensure the shape itself is white (255) and background is black (0)
    # Check the corners of the image. If they are mostly white, it's likely a dark shape on light bg,
    # so we need to invert it.
    corners = [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]]
    if sum(corners) > 255 * 2: 
        binary = cv2.bitwise_not(binary)
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        element.shape_type = fallback_type
        return fallback_type
        
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    
    if area < config.SHAPE_MIN_AREA_PX:
        element.shape_type = fallback_type
        return fallback_type
        
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, config.SHAPE_VERTEX_TOLERANCE * perimeter, True)
    n = len(approx)
    
    if n == 3:
        shape_type = "triangle"
    elif n == 4:
        x_b, y_b, w_b, h_b = cv2.boundingRect(approx)
        ar = max(w_b, h_b) / max(min(w_b, h_b), 1)
        shape_type = "arrow_triangle" if ar > config.SHAPE_ARROW_ASPECT_RATIO else "rectangle"
    elif n >= 5:
        hull = cv2.convexHull(contour, returnPoints=False)
        hull_ratio = len(hull) / max(n, 1)
        if hull_ratio < config.SHAPE_STAR_CONCAVITY_RATIO:
            shape_type = "star"
        elif n == 5:
            shape_type = "pentagon"
        elif n == 6:
            shape_type = "hexagon"
        else:
            circularity = 4 * math.pi * area / max(perimeter ** 2, 1)
            if circularity > 0.85:
                shape_type = "circle"
            elif circularity > 0.6:
                shape_type = "pill"
            else:
                shape_type = "polygon"
    else:
        shape_type = "rectangle"

    # Sample fill color
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Exclude 3px border using a 7x7 kernel (which erodes ~3 pixels on each edge)
    kernel = np.ones((7, 7), np.uint8)
    mask_shrunk = cv2.erode(mask, kernel, iterations=1)
    
    if cv2.countNonZero(mask_shrunk) > 0:
        pixels = crop[mask_shrunk == 255]
        median_bgr = np.median(pixels, axis=0)
        fill_color = (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))
    else:
        fill_color = None
        
    # Sample border color
    mask_border = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask_border, [contour], -1, 255, 1)
    
    if cv2.countNonZero(mask_border) > 0:
        pixels_border = crop[mask_border == 255]
        median_border_bgr = np.median(pixels_border, axis=0)
        border_color = (int(median_border_bgr[2]), int(median_border_bgr[1]), int(median_border_bgr[0]))
    else:
        border_color = None

    element.shape_type = shape_type
    element.fill_color = fill_color
    element.border_color = border_color
    
    return shape_type
