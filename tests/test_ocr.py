"""Tests for phase 3A shape classification and OCR."""

import cv2
import numpy as np

from pipeline.models import BBox, DetectedElement, OCRLine
from pipeline.phase3a_reconstruct.shape_classifier import classify_shape
from pipeline.phase3a_reconstruct.ocr import extract_lines, safe_text_area
from pipeline.phase3b_crop.background import is_dark_background
from pipeline.phase3b_crop.tiler import crop_tile
from pipeline.phase3b_crop.text_remover import remove_text


def _white_image(w=200, h=200):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img.fill(255)
    return img


def test_triangle_shape_type():
    img = _white_image()
    # Draw a filled black triangle
    pts = np.array([[100, 20], [180, 180], [20, 180]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (0, 0, 0))
    
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    shape = classify_shape(element, img)
    assert shape == "triangle"


def test_circle_shape_type():
    img = _white_image()
    # Draw a filled black circle
    cv2.circle(img, (100, 100), 80, (0, 0, 0), -1)
    
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    shape = classify_shape(element, img)
    # Could be circle or pill depending on exact circularity, but perfectly drawn circle should hit > 0.85
    assert shape == "circle"


def test_rectangle_shape_type():
    img = _white_image()
    # Draw a filled black rectangle
    cv2.rectangle(img, (20, 20), (180, 180), (0, 0, 0), -1)
    
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    shape = classify_shape(element, img)
    assert shape == "rectangle"


def test_fill_color_sampled():
    img = _white_image()
    # Draw a solid red circle (BGR: 0, 0, 255)
    cv2.circle(img, (100, 100), 80, (0, 0, 255), -1)
    
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    classify_shape(element, img)
    
    assert element.fill_color is not None
    # Tuple should be RGB (255, 0, 0)
    assert abs(element.fill_color[0] - 255) <= 5
    assert element.fill_color[1] <= 5
    assert element.fill_color[2] <= 5


def test_extract_lines_empty_image():
    from unittest.mock import patch
    img = _white_image()
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    with patch("pytesseract.image_to_data", return_value={"text": [], "line_num": [], "left": [], "top": [], "width": [], "height": []}):
        lines = extract_lines(element, img)
    assert lines == []


def test_is_art_flag():
    # We can test the regex logic directly by creating an OCRLine object
    line = OCRLine(
        text="→", 
        bbox=BBox(x=0,y=0,w=10,h=10), 
        font_size_pt=12.0, 
        color_rgb=(0,0,0), 
        bold=False, 
        is_art=True
    )
    assert line.is_art is True
    
    import re
    is_art = not bool(re.search(r'[a-zA-Z0-9]', "→"))
    assert is_art is True


def test_is_art_alphanumeric():
    import re
    is_art = not bool(re.search(r'[a-zA-Z0-9]', "1943"))
    assert is_art is False


def test_dark_background_detection():
    dark_img = np.full((200, 200, 3), 30, dtype=np.uint8)
    assert is_dark_background(dark_img) == True
    
    light_img = np.full((200, 200, 3), 220, dtype=np.uint8)
    assert is_dark_background(light_img) == False


def test_crop_tile_saves_file():
    import os
    # Create a fake 400x300 blue image
    img = np.full((300, 400, 3), (255, 0, 0), dtype=np.uint8)
    element = DetectedElement(bbox=BBox(x=50, y=50, w=100, h=100))
    
    tile_path = crop_tile(element, img)
    assert element.tile_path is not None
    assert element.tile_path == tile_path
    assert os.path.exists(tile_path)


def test_remove_text_fills_region():
    # Create a 200x200 solid light gray image (BGR 200,200,200).
    img = np.full((200, 200, 3), (200, 200, 200), dtype=np.uint8)
    
    # Add black text area in the center by setting pixels to 0.
    img[80:120, 80:120] = 0
    
    element_bbox = BBox(x=0, y=0, w=200, h=200)
    
    # Create an OCRLine with bbox covering that black region, is_art=False.
    line = OCRLine(
        text="text", 
        bbox=BBox(x=80, y=80, w=40, h=40), 
        font_size_pt=12.0, 
        color_rgb=(0,0,0), 
        bold=False, 
        is_art=False
    )
    
    result = remove_text(img, [line], element_bbox, False)
    
    # Assert: the text region in result is no longer black (filled with light gray).
    sampled_color = result[100, 100]
    
    # It should be light gray (200, 200, 200)
    assert sampled_color[0] == 200
    assert sampled_color[1] == 200
    assert sampled_color[2] == 200


def test_remove_text_blue_background():
    # Create a 200x200 solid blue image (BGR 255, 0, 0)
    img = np.full((200, 200, 3), (255, 0, 0), dtype=np.uint8)
    
    # Add white text area in the center (BGR 255, 255, 255)
    img[80:120, 80:120] = 255
    
    element_bbox = BBox(x=0, y=0, w=200, h=200)
    
    # Create an OCRLine with bbox covering that white region, is_art=False
    line = OCRLine(
        text="TEST", 
        bbox=BBox(x=80, y=80, w=40, h=40), 
        font_size_pt=12.0, 
        color_rgb=(255, 255, 255), 
        bold=False, 
        is_art=False
    )
    
    result = remove_text(img, [line], element_bbox, dark_bg=True)
    
    # Assert: the text region in result is now blue
    sampled_color = result[100, 100]
    
    # It should be blue (255, 0, 0)
    assert sampled_color[0] == 255
    assert sampled_color[1] == 0
    assert sampled_color[2] == 0


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


def test_extract_lines_known_text():
    from unittest.mock import patch
    img = _white_image()
    element = DetectedElement(bbox=BBox(x=0, y=0, w=200, h=200))
    
    fake_data = {
        "text": ["", "Hello", "World"],
        "line_num": [0, 1, 1],
        "left": [0, 10, 60],
        "top": [0, 10, 10],
        "width": [0, 40, 40],
        "height": [0, 15, 15]
    }
    with patch("pytesseract.image_to_data", return_value=fake_data):
        lines = extract_lines(element, img)
        
    assert len(lines) == 1
    assert lines[0].text == "Hello World"
    assert lines[0].bbox.x == 10
    assert lines[0].bbox.y == 10
    assert lines[0].bbox.w == 90
    assert lines[0].bbox.h == 15
