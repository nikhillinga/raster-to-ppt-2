"""Tests for QA / SSIM validation and Pipeline orchestrator."""

import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np

import config
from pipeline.models import BBox, DetectedElement, ElementTree
from pipeline.pipeline import run_pipeline


@patch("pipeline.pipeline.run_yolo")
@patch("pipeline.pipeline.run_gemini")
@patch("pipeline.pipeline.merge_detections")
@patch("pipeline.pipeline.classify_tree")
@patch("pipeline.pipeline.classify_shape")
@patch("pipeline.pipeline.extract_lines")
@patch("pipeline.pipeline.crop_tile")
@patch("pipeline.pipeline.is_dark_background")
@patch("pipeline.pipeline.remove_text")
@patch("pipeline.pipeline.assemble")
@patch("pipeline.pipeline.render_pptx_to_image")
def test_pipeline_orchestrator(
    mock_render_pptx, mock_assemble, mock_remove_text, mock_is_dark,
    mock_crop_tile, mock_extract, mock_classify_shape, mock_classify_tree,
    mock_merge, mock_gemini, mock_yolo, tmp_path
):
    # Setup mock data
    image_path = str(tmp_path / "sample_slide.png")
    # Create a dummy image large enough for SSIM window (e.g. 100x100)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(image_path, dummy_img)
    
    # Mock render_pptx_to_image to return another dummy image
    rendered_path = str(tmp_path / "rendered.png")
    cv2.imwrite(rendered_path, dummy_img)
    mock_render_pptx.return_value = rendered_path
    
    # Mock tree with one element that is large enough (w=50, h=50) for SSIM window (usually 7x7)
    elem = DetectedElement(
        bbox=BBox(x=0, y=0, w=50, h=50),
        semantic_type="TextBlock",
        processing_path="reconstruct"
    )
    tree = ElementTree(roots=[elem])
    mock_merge.return_value = tree
    
    # Mock assemble output
    output_pptx = str(tmp_path / "output.pptx")
    mock_assemble.return_value = output_pptx
    
    # Force QA diff saving to test logic
    config.QA_DIFF_SAVE = True
    config.QA_OUTPUT_DIR = tmp_path / "qa"
    config.QA_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run pipeline
    qa_result = run_pipeline(image_path, output_pptx, vlm_enabled=False)
    
    # Verify QA JSON is saved
    json_path = Path(output_pptx).with_suffix(".qa.json")
    assert json_path.exists()
    
    # Verify contents of JSON
    with open(json_path, "r") as f:
        saved_data = json.load(f)
        
    assert "overall_ssim" in saved_data
    assert "element_scores" in saved_data
    assert len(saved_data["element_scores"]) >= 1
