"""YOLO-based object detection for slide elements."""

from typing import List

import numpy as np
from loguru import logger
from ultralytics import YOLO

import config
from pipeline.models import BBox, DetectedElement

YOLO_TO_SEMANTIC = {
    "header": "Header",
    "content_block": "Section",
    "text_zone": "TextBlock",
    "chart_area": "Chart",
    "table": "Table",
    "image": "BackgroundArt",
    "icon": "Icon",
    "arrow": "Arrow",
}


def run_yolo(image: np.ndarray) -> List[DetectedElement]:
    """Run YOLO object detection on a slide image.

    Args:
        image: Input image as a numpy array (H, W, C).

    Returns:
        List of DetectedElement instances for each detection above the
        confidence threshold. Returns an empty list if no detections.

    Raises:
        FileNotFoundError: If the YOLO weights file does not exist.
    """
    if not config.YOLO_WEIGHTS.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {config.YOLO_WEIGHTS}. Place your .pt file there."
        )

    model = YOLO(str(config.YOLO_WEIGHTS))
    results = model(image, imgsz=config.YOLO_IMAGE_SIZE, conf=config.YOLO_CONF_THRESHOLD)

    elements: List[DetectedElement] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            x1, y1, x2, y2 = xyxy
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            class_name = result.names.get(cls_id, "unknown")
            semantic_type = YOLO_TO_SEMANTIC.get(class_name, "TextBlock")

            bbox = BBox(x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1))
            element = DetectedElement(
                bbox=bbox,
                confidence=conf,
                semantic_type=semantic_type,
                detected_by="yolo",
                processing_path="undecided",
            )
            elements.append(element)

    logger.info(f"YOLO detected {len(elements)} elements")
    return elements
