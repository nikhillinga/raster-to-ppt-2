"""Phase 3A: OCR extraction for reconstruct path."""

import re
from typing import List
from collections import defaultdict

import cv2
import numpy as np
import pytesseract

import config
from pipeline.models import BBox, DetectedElement, OCRLine


def safe_text_area(element: DetectedElement, source_image: np.ndarray) -> BBox:
    """Compute an inset bounding box to fit text within the shape silhouette."""
    bbox = element.bbox
    shape_type = element.shape_type or "rectangle"
    
    if shape_type == "triangle":
        # use lower 60% of height, center 70% of width
        y_inset = int(bbox.h * 0.4)
        h_new = bbox.h - y_inset
        x_inset = int(bbox.w * 0.15)
        w_new = int(bbox.w * 0.7)
        new_bbox = BBox(x=bbox.x + x_inset, y=bbox.y + y_inset, w=w_new, h=h_new)
    elif shape_type == "star":
        # center 60% of both dimensions
        x_inset = int(bbox.w * 0.2)
        y_inset = int(bbox.h * 0.2)
        w_new = int(bbox.w * 0.6)
        h_new = int(bbox.h * 0.6)
        new_bbox = BBox(x=bbox.x + x_inset, y=bbox.y + y_inset, w=w_new, h=h_new)
    elif shape_type in ["pill", "circle"]:
        # center 70% of both dimensions
        x_inset = int(bbox.w * 0.15)
        y_inset = int(bbox.h * 0.15)
        w_new = int(bbox.w * 0.7)
        h_new = int(bbox.h * 0.7)
        new_bbox = BBox(x=bbox.x + x_inset, y=bbox.y + y_inset, w=w_new, h=h_new)
    else:
        # full bbox (no inset)
        new_bbox = BBox(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h)
        
    # Clamp result to be within source_image bounds
    h_img, w_img = source_image.shape[:2]
    new_bbox.x = max(0, min(new_bbox.x, w_img))
    new_bbox.y = max(0, min(new_bbox.y, h_img))
    new_bbox.w = max(0, min(new_bbox.w, w_img - new_bbox.x))
    new_bbox.h = max(0, min(new_bbox.h, h_img - new_bbox.y))
    
    return new_bbox


def extract_lines(element: DetectedElement, source_image: np.ndarray) -> List[OCRLine]:
    """Run Tesseract OCR on the safe area and extract lines."""
    
    safe_bbox = safe_text_area(element, source_image)
    if safe_bbox.w <= 0 or safe_bbox.h <= 0:
        return []
        
    crop = source_image[safe_bbox.y:safe_bbox.y2, safe_bbox.x:safe_bbox.x2]
    if crop.size == 0:
        return []
        
    data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, config="--psm 6")
    
    lines = defaultdict(list)
    for i in range(len(data["text"])):
        text = data["text"][i]
        if not text.strip():
            continue
        conf = int(data["conf"][i])
        
        line_num = data["line_num"][i]
        lines[line_num].append({
            "text": text,
            "conf": conf,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
        })
        
    ocr_lines = []
    h_img, w_img = source_image.shape[:2]
    
    for line_num, words in lines.items():
        text_str = " ".join(w["text"] for w in words if w["text"].strip())
        stripped = text_str.strip()
        if not stripped:
            continue
            
        valid_confs = [w["conf"] for w in words if w["conf"] >= 0]
        if not valid_confs:
            continue
        avg_conf = sum(valid_confs) / len(valid_confs)
        if avg_conf < config.OCR_MIN_CONFIDENCE:
            continue
            
        # Bug 4 extended filters: drop phantom/artifact lines
        if len(stripped) <= 3 and not stripped.isalnum():
            continue
        if "_" in stripped or stripped.startswith("'"):
            continue
            
        lefts = [w["left"] for w in words]
        tops = [w["top"] for w in words]
        right_edges = [w["left"] + w["width"] for w in words]
        bottom_edges = [w["top"] + w["height"] for w in words]
        
        # convert back to source image coords
        line_x = min(lefts) + safe_bbox.x
        line_y = min(tops) + safe_bbox.y
        line_w = max(right_edges) - min(lefts)
        line_h = max(bottom_edges) - min(tops)
        
        # Clamp to bounds
        line_x = max(0, min(line_x, w_img))
        line_y = max(0, min(line_y, h_img))
        line_w = max(0, min(line_w, w_img - line_x))
        line_h = max(0, min(line_h, h_img - line_y))
        
        if line_w <= 0 or line_h <= 0:
            continue
            
        line_height_px = line_h
        font_size_pt = max(
            config.FONT_SIZE_MIN_PT,
            min(line_height_px * config.FONT_SIZE_HEIGHT_RATIO, config.FONT_SIZE_MAX_PT)
        )
        
        # Color & Bold sampling
        line_crop = source_image[line_y:line_y+line_h, line_x:line_x+line_w]
        if line_crop.size == 0:
            color_rgb = (0, 0, 0)
            bold = False
        else:
            gray_line = cv2.cvtColor(line_crop, cv2.COLOR_BGR2GRAY)
            # Find darkest pixels inside the bbox
            p5 = np.percentile(gray_line, 5)
            ink_mask = gray_line <= p5
            
            if np.any(ink_mask):
                pixels = line_crop[ink_mask]
                med = np.median(pixels, axis=0)
                color_rgb = (int(med[2]), int(med[1]), int(med[0]))
            else:
                color_rgb = (0, 0, 0)
                
            # Bold detection: consecutive ink runs
            run_lengths = []
            for row in ink_mask:
                current_run = 0
                for val in row:
                    if val:
                        current_run += 1
                    else:
                        if current_run > 0:
                            run_lengths.append(current_run)
                            current_run = 0
                if current_run > 0:
                    run_lengths.append(current_run)
                    
            median_stroke = np.median(run_lengths) if run_lengths else 0
            bold = (median_stroke / max(line_height_px, 1) > config.FONT_BOLD_STROKE_RATIO 
                    and line_height_px >= config.FONT_BOLD_MIN_SIZE_PX)
                    
        is_art = not any(c.isalnum() for c in stripped)
        
        ocr_lines.append(OCRLine(
            text=text_str,
            bbox=BBox(x=line_x, y=line_y, w=line_w, h=line_h),
            font_size_pt=font_size_pt,
            color_rgb=color_rgb,
            bold=bold,
            is_art=is_art
        ))
        
    return ocr_lines
