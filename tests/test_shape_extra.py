import cv2
import numpy as np
from pipeline.models import BBox, DetectedElement
from pipeline.phase3a_reconstruct.shape_classifier import classify_shape

def _white_image(w=200, h=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img.fill(255)
    return img

def test_star_shape_type():
    img = _white_image()
    # Create points for a 5-pointed star
    pts = []
    import math
    for i in range(10):
        angle = i * math.pi / 5
        r = 80 if i % 2 == 0 else 30
        pts.append([int(100 + r * math.sin(angle)), int(100 - r * math.cos(angle))])
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (0, 0, 0))
    
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    shape = classify_shape(element, img)
    assert shape == "star"

def test_wide_thin_rectangle():
    img = _white_image()
    cv2.rectangle(img, (10, 100), (190, 110), (0, 0, 0), -1)
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    shape = classify_shape(element, img)
    assert shape == "rectangle"
