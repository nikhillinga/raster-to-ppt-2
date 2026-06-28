"""Gemini VLM-based element detection and iterative refinement."""

import json
import os
from typing import List, Optional, Tuple

import cv2
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed

import config
from pipeline.models import BBox, DetectedElement
from pipeline.phase1_detection.prompts import DETECTION_PASS_1, DETECTION_PASS_N

# Load .env and configure Gemini once when the module loads
load_dotenv()
_gemini_key = os.getenv("GEMINI_API_KEY", "")
print(f"Gemini key loaded: {_gemini_key[:8]}..." if _gemini_key else "Gemini key loaded: MISSING")
genai.configure(api_key=_gemini_key)


@retry(
    stop=stop_after_attempt(config.GEMINI_RETRY_MAX),
    wait=wait_fixed(config.GEMINI_RETRY_BACKOFF[0]),
)
def _call_gemini(image: Image.Image, prompt: str) -> dict:
    """Make a call to the Gemini API and return the parsed JSON response.

    Uses tenacity for retries in case of API errors or malformed JSON.
    """
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=config.GEMINI_TEMPERATURE,
            max_output_tokens=config.GEMINI_MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content([image, prompt])
    
    logger.debug(f"Gemini raw response: {response.text[:500]}")
    
    if not response.text:
        raise ValueError("Empty response from Gemini")
        
    return json.loads(response.text)


def _draw_boxes(image: np.ndarray, elements: List[DetectedElement]) -> np.ndarray:
    """Draw bounding boxes on a copy of the image."""
    annotated = image.copy()
    for el in elements:
        x1, y1 = el.bbox.x, el.bbox.y
        x2, y2 = el.bbox.x2, el.bbox.y2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 1)
    return annotated


class GeminiDetectedElement(DetectedElement):
    correction_for: Optional[BBox] = None


_LABEL_TO_SEMANTIC: List[Tuple[List[str], str]] = [
    (["header", "title"], "Header"),
    (["text", "paragraph", "block"], "TextBlock"),
    (["arrow", "connector"], "Arrow"),
    (["chart", "graph", "plot"], "Chart"),
    (["icon", "logo", "symbol"], "Icon"),
    (["table"], "Table"),
    (["triangle", "star", "circle", "hexagon", "rectangle", "shape", "diagram"], "DiagramShape"),
    (["background", "bg", "art"], "BackgroundArt"),
]


def _map_label_to_semantic_type(label: str) -> str:
    """Map a free-text label from Gemini to a known semantic_type."""
    label_lower = label.lower()
    for keywords, sem_type in _LABEL_TO_SEMANTIC:
        if any(w in label_lower for w in keywords):
            return sem_type
    return "TextBlock"


def _parse_gemini_response(
    response_data: dict, image_width: int = 1, image_height: int = 1
) -> List[GeminiDetectedElement]:
    """Parse the JSON response into a list of GeminiDetectedElement instances.

    Handles both the expected bbox dict format and Gemini 2.5 Flash's
    box_2d [y1, x1, y2, x2] normalized (0-1000) format.
    """
    elements = []
    items = response_data.get("elements", [])
    
    for item in items:
        # --- Extract bbox: support both formats ---
        box_2d = item.get("box_2d")
        b = item.get("bbox")

        if box_2d and isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
            # Gemini 2.5 Flash format: [y1, x1, y2, x2] normalised 0-1000
            y1_n, x1_n, y2_n, x2_n = box_2d
            x = int(x1_n / 1000 * image_width)
            y = int(y1_n / 1000 * image_height)
            w = int((x2_n - x1_n) / 1000 * image_width)
            h = int((y2_n - y1_n) / 1000 * image_height)
            logger.debug(
                f"box_2d converted: [{y1_n},{x1_n},{y2_n},{x2_n}] → "
                f"x={x} y={y} w={w} h={h} (img {image_width}x{image_height})"
            )
        elif b and isinstance(b, dict):
            # Expected dict format: {x, y, w, h}
            x, y = b.get("x", 0), b.get("y", 0)
            w, h = b.get("w", 0), b.get("h", 0)
        else:
            logger.warning(f"Element missing both bbox and box_2d, skipping: {item}")
            continue
        
        # We need a valid width and height
        if w <= 0 or h <= 0:
            continue
            
        bbox = BBox(x=int(x), y=int(y), w=int(w), h=int(h))
        
        # --- Resolve semantic_type: prefer explicit, fall back to label mapping ---
        semantic_type = item.get("semantic_type")
        if not semantic_type:
            label = item.get("label", "")
            semantic_type = _map_label_to_semantic_type(label)
            logger.debug(f"Mapped label '{label}' → semantic_type '{semantic_type}'")
        
        # Check correction_for
        correction_for = None
        c = item.get("correction_for")
        if c and isinstance(c, dict):
            cw, ch = c.get("w", 0), c.get("h", 0)
            if cw > 0 and ch > 0:
                correction_for = BBox(
                    x=int(c.get("x", 0)), 
                    y=int(c.get("y", 0)), 
                    w=int(c.get("w", 0)), 
                    h=int(c.get("h", 0))
                )
                
        element = GeminiDetectedElement(
            bbox=bbox,
            confidence=float(item.get("confidence", 1.0)),
            semantic_type=semantic_type,
            detected_by="gemini",
            processing_path="undecided",
            correction_for=correction_for
        )
            
        elements.append(element)
        
    return elements


def run_gemini_detection(
    image: np.ndarray, initial_elements: List[DetectedElement] = None
) -> List[DetectedElement]:
    """Run iterative Gemini detection to find elements and correct clipped boxes.

    Args:
        image: Source image as a numpy array (H, W, C) in BGR.
        initial_elements: Optional list of initial YOLO detections to seed the process.

    Returns:
        List of all DetectedElement instances found by Gemini.
    """
    if initial_elements is None:
        initial_elements = []
        
    all_gemini_elements = []
    current_boxes = list(initial_elements)
    
    for i in range(config.GEMINI_ITER_MAX):
        logger.info(f"Starting Gemini detection iteration {i+1}/{config.GEMINI_ITER_MAX}")
        
        if i == 0 and not current_boxes:
            # Pass 1 without any seeding
            prompt = DETECTION_PASS_1
            input_img = image
        else:
            # Pass N (or Pass 1 with YOLO seeding)
            prompt = DETECTION_PASS_N
            input_img = _draw_boxes(image, current_boxes)
            
            if config.DEBUG_SAVE_ANNOTATED_IMAGE:
                out_path = config.OUTPUT_DIR / f"debug_gemini_pass_{i+1}.png"
                cv2.imwrite(str(out_path), input_img)
                logger.debug(f"Saved annotated image to {out_path}")
                
        # Convert BGR numpy array to RGB PIL Image
        rgb_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        img_h, img_w = image.shape[:2]
        try:
            response_data = _call_gemini(pil_img, prompt)
            new_elements = _parse_gemini_response(response_data, img_w, img_h)
        except Exception as e:
            logger.warning(f"Gemini API call failed after retries on iteration {i+1}: {e}")
            break
            
        if len(new_elements) <= config.GEMINI_ITER_STOP_THRESHOLD:
            logger.info(f"Gemini returned {len(new_elements)} elements. Stopping iterations.")
            break
            
        all_gemini_elements.extend(new_elements)
        current_boxes.extend(new_elements)
        
        logger.info(f"Gemini iteration {i+1} found {len(new_elements)} elements.")
        
    return all_gemini_elements
