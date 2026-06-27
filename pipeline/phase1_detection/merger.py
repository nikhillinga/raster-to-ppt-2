"""Merging and deduplication of YOLO and Gemini detections."""

from typing import List
import math

import config
from pipeline.models import BBox, DetectedElement, ElementTree


def iou(a: BBox, b: BBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def containment_ratio(smaller: BBox, larger: BBox) -> float:
    """What fraction of 'smaller' is inside 'larger'?"""
    ix1 = max(smaller.x, larger.x)
    iy1 = max(smaller.y, larger.y)
    ix2 = min(smaller.x2, larger.x2)
    iy2 = min(smaller.y2, larger.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    return inter / smaller.area if smaller.area > 0 else 0.0


def center_distance(a: BBox, b: BBox) -> float:
    """Compute the Euclidean distance between the centers of two bounding boxes."""
    return math.hypot(a.cx - b.cx, a.cy - b.cy)


def merge_detections(
    yolo_elements: List[DetectedElement],
    gemini_elements: List[DetectedElement],
    source_image_path: str = "",
    source_w: int = 0,
    source_h: int = 0,
) -> ElementTree:
    """Merge YOLO and Gemini detections following the 5 steps."""
    
    # We will work with a shallow copy of YOLO elements to start with.
    working_list = list(yolo_elements)

    # ---------------------------------------------------------
    # Step 1 — Apply Gemini corrections
    # ---------------------------------------------------------
    gemini_corrections = [e for e in gemini_elements if getattr(e, "correction_for", None) is not None]
    
    for g_elem in gemini_corrections:
        corr_bbox = g_elem.correction_for
        closest_yolo = None
        min_dist = float('inf')
        
        for y_elem in working_list:
            if y_elem.detected_by == "yolo":
                dist = center_distance(corr_bbox, y_elem.bbox)
                if dist < min_dist:
                    min_dist = dist
                    closest_yolo = y_elem
                    
        # If closest YOLO element is within 50px AND gemini bbox area > yolo bbox area * 1.05
        if closest_yolo is not None and min_dist <= 50:
            if g_elem.bbox.area > closest_yolo.bbox.area * 1.05:
                # replace the YOLO element's bbox with gemini element's bbox
                closest_yolo.bbox = g_elem.bbox
                # Note: The prompt says "replace the YOLO element's bbox", it didn't explicitly say 
                # replace the whole element, just the bbox. We will update the bbox.
        # Otherwise: discard the correction

    # ---------------------------------------------------------
    # Step 2 — Add genuinely new Gemini detections
    # ---------------------------------------------------------
    gemini_new = [e for e in gemini_elements if getattr(e, "correction_for", None) is None]
    
    for g_elem in gemini_new:
        max_iou = 0.0
        for w_elem in working_list:
            current_iou = iou(g_elem.bbox, w_elem.bbox)
            if current_iou > max_iou:
                max_iou = current_iou
                
        if max_iou < config.MERGE_IOU_DUPLICATE_THRESHOLD:
            working_list.append(g_elem)
        # Else: discard (duplicate)

    # ---------------------------------------------------------
    # Step 3 — IoU dedup pass
    # ---------------------------------------------------------
    # Greedy approach: sort by confidence desc
    working_list.sort(key=lambda x: x.confidence, reverse=True)
    kept_list = []
    
    for elem in working_list:
        should_keep = True
        for kept in kept_list:
            if iou(elem.bbox, kept.bbox) >= config.MERGE_IOU_DUPLICATE_THRESHOLD:
                should_keep = False
                break
        if should_keep:
            kept_list.append(elem)
            
    working_list = kept_list

    # ---------------------------------------------------------
    # Step 4 — Containment → nesting
    # ---------------------------------------------------------
    working_list.sort(key=lambda x: x.bbox.area, reverse=True)
    nested_ids = set()
    
    for i in range(len(working_list)):
        larger = working_list[i]
        
        for j in range(i + 1, len(working_list)):
            smaller = working_list[j]
            
            # Skip if smaller is already nested, as we only want one parent? 
            # The prompt doesn't explicitly mention skipping, but it says "remove all nested (marked) elements from the flat list."
            if smaller.id in nested_ids:
                continue
                
            if containment_ratio(smaller.bbox, larger.bbox) >= config.MERGE_CONTAINMENT_THRESHOLD:
                # If the two elements have the same semantic_type AND smaller.area > larger.area * 0.8
                if (smaller.semantic_type == larger.semantic_type) and (smaller.bbox.area > larger.bbox.area * 0.8):
                    # treat as duplicate (keep larger, discard smaller)
                    nested_ids.add(smaller.id)
                else:
                    # append smaller to larger.children; mark smaller as nested
                    larger.children.append(smaller)
                    nested_ids.add(smaller.id)
                    
    # After all pairs processed: remove all nested (marked) elements from the flat list.
    remaining_flat_list = [e for e in working_list if e.id not in nested_ids]

    # ---------------------------------------------------------
    # Step 5 — Build ElementTree
    # ---------------------------------------------------------
    tree = ElementTree(
        roots=remaining_flat_list,
        source_image_path=source_image_path,
        source_image_w=source_w,
        source_image_h=source_h,
    )
    
    return tree
