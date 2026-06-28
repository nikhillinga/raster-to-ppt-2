"""Top-level orchestrator: runs phases 1-5 in sequence."""

import cv2
import config

from pipeline.phase1_detection.yolo_detector import run_yolo
from pipeline.phase1_detection.gemini_detector import run_gemini_detection as run_gemini
from pipeline.phase1_detection.merger import merge_detections
from pipeline.phase2_decision.classifier import classify_tree
from pipeline.phase3a_reconstruct.shape_classifier import classify_shape
from pipeline.phase3a_reconstruct.ocr import extract_lines
from pipeline.phase3b_crop.tiler import crop_tile
from pipeline.phase3b_crop.background import is_dark_background
from pipeline.phase3b_crop.text_remover import remove_text
from pipeline.phase4_assembly.assembler import assemble
from pipeline.phase5_qa.renderer import render_pptx_to_image
from pipeline.phase5_qa.ssim import compute_ssim
from pipeline.phase5_qa.reporter import save_report


def run_pipeline(
    image_path: str,
    output_path: str,
    vlm_enabled: bool = True,
    debug: bool = False,
) -> dict:
    source_image = cv2.imread(image_path)
    
    # Phase 1
    yolo_elements = run_yolo(source_image)
    gemini_elements = run_gemini(source_image, yolo_elements) if vlm_enabled else []
    tree = merge_detections(yolo_elements, gemini_elements)
    
    # Phase 2
    classify_tree(tree, source_image)
    
    # Phase 3 (both paths, per element)
    for element in tree.all_elements():
        if element.processing_path == "reconstruct":
            classify_shape(element, source_image)
            element.ocr_lines = extract_lines(element, source_image)
        elif element.processing_path == "crop":
            crop_tile(element, source_image)
            dark_bg = is_dark_background(source_image, element.bbox)
            element.ocr_lines = extract_lines(element, source_image)
            tile = cv2.imread(element.tile_path)
            clean_tile = remove_text(tile, element.ocr_lines, element.bbox, dark_bg)
            cv2.imwrite(element.tile_path, clean_tile)
            
    # Phase 4
    output_pptx = assemble(tree, output_path)
    
    # Phase 5
    config.QA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered_img = render_pptx_to_image(output_pptx, str(config.QA_OUTPUT_DIR))
    qa_result = compute_ssim(image_path, rendered_img, tree)
    save_report(qa_result, output_pptx)
    
    return qa_result
