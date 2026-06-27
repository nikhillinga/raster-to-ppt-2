"""Merging and deduplication of YOLO and Gemini detections."""

from typing import List

from loguru import logger

import config
from pipeline.models import BBox, DetectedElement, ElementTree


def calculate_iou(box1: BBox, box2: BBox) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x_left = max(box1.x, box2.x)
    y_top = max(box1.y, box2.y)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = box1.area
    box2_area = box2.area

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def calculate_containment(inner: BBox, outer: BBox) -> float:
    """Calculate the fraction of 'inner' box that is contained within 'outer' box."""
    x_left = max(inner.x, outer.x)
    y_top = max(inner.y, outer.y)
    x_right = min(inner.x2, outer.x2)
    y_bottom = min(inner.y2, outer.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area / float(inner.area)


def calculate_center_distance(box1: BBox, box2: BBox) -> float:
    """Calculate the Euclidean distance between the centers of two boxes."""
    return ((box1.cx - box2.cx) ** 2 + (box1.cy - box2.cy) ** 2) ** 0.5


def merge_detections(
    yolo_elements: List[DetectedElement],
    gemini_elements: List[DetectedElement],
    source_image_path: str = "",
    source_image_w: int = 0,
    source_image_h: int = 0,
) -> ElementTree:
    """Merge YOLO and Gemini detections into a clean, deduplicated ElementTree.

    Follows the 5-step process defined in the PRD:
    1. Apply Gemini corrections
    2. Add genuinely new Gemini detections
    3. Deduplication pass
    4. Containment -> nesting
    5. Build ElementTree
    """
    final_elements = list(yolo_elements)
    
    # Step 1: Apply Gemini corrections
    gemini_corrections = [e for e in gemini_elements if getattr(e, "correction_for", None) is not None]
    
    for gem_elem in gemini_corrections:
        corr_bbox = gem_elem.correction_for
        # Find closest existing YOLO element by spatial proximity
        best_match = None
        min_dist = float('inf')
        best_idx = -1
        
        for i, yolo_elem in enumerate(final_elements):
            if yolo_elem.detected_by != "yolo":
                continue
            dist = calculate_center_distance(corr_bbox, yolo_elem.bbox)
            # Need some reasonable proximity threshold? The PRD just says "closest"
            if dist < min_dist:
                min_dist = dist
                best_match = yolo_elem
                best_idx = i
                
        if best_match is not None:
            # Compare areas: if Gemini's is larger, replace YOLO box
            if gem_elem.bbox.area > best_match.bbox.area:
                final_elements[best_idx] = gem_elem
                logger.debug(f"Replaced YOLO box {best_match.id} with Gemini correction {gem_elem.id}")
            else:
                logger.debug(f"Discarded Gemini correction {gem_elem.id}: YOLO box is larger")

    # Step 2: Add genuinely new Gemini detections
    gemini_new = [e for e in gemini_elements if getattr(e, "correction_for", None) is None]
    
    for gem_elem in gemini_new:
        is_duplicate = False
        for exist_elem in final_elements:
            iou = calculate_iou(gem_elem.bbox, exist_elem.bbox)
            if iou >= config.MERGE_IOU_DUPLICATE_THRESHOLD:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_elements.append(gem_elem)
            
    # Step 3: Deduplication pass
    # We will build a new list without duplicates
    deduped_elements = []
    
    for elem in sorted(final_elements, key=lambda e: e.confidence, reverse=True):
        is_duplicate = False
        for kept_elem in deduped_elements:
            iou = calculate_iou(elem.bbox, kept_elem.bbox)
            if iou >= config.MERGE_IOU_DUPLICATE_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped_elements.append(elem)
            
    # Step 4: Containment -> nesting
    # Sort elements by area descending (largest first)
    sorted_elements = sorted(deduped_elements, key=lambda e: e.bbox.area, reverse=True)
    
    # We need to track which elements are nested so we don't include them in the root list
    # Because elements are already objects, we can just append to their children list
    # but we must ensure we don't modify elements that are shared. 
    # Actually, we can modify them in place since they are specific to this run.
    roots = []
    nested_ids = set()
    
    for i in range(len(sorted_elements)):
        larger = sorted_elements[i]
        
        # If this element is already nested inside something else, it can still be a parent
        # to even smaller things. But we will process it anyway.
        
        for j in range(i + 1, len(sorted_elements)):
            smaller = sorted_elements[j]
            if smaller.id in nested_ids:
                # Already nested inside something else? Wait, an element should only have one direct parent
                continue
                
            containment = calculate_containment(smaller.bbox, larger.bbox)
            if containment >= config.MERGE_CONTAINMENT_THRESHOLD:
                # Check exception: same semantic type and nearly identical size (within 5%)
                area_ratio = smaller.bbox.area / float(larger.bbox.area)
                if (smaller.semantic_type == larger.semantic_type) and (area_ratio >= 0.95):
                    # Treat as duplicate -> essentially suppress it. 
                    nested_ids.add(smaller.id) 
                    continue
                
                # Nest it!
                larger.children.append(smaller)
                nested_ids.add(smaller.id)

    # Roots are the elements that are not nested inside anything
    for elem in sorted_elements:
        if elem.id not in nested_ids:
            roots.append(elem)

    # Step 5: Build ElementTree
    tree = ElementTree(
        roots=roots,
        source_image_path=source_image_path,
        source_image_w=source_image_w,
        source_image_h=source_image_h,
    )
    
    logger.info(f"Merge complete: {len(roots)} roots, {len(deduped_elements)} total elements")
    return tree
