"""Top-level orchestrator: runs phases 1-5 in sequence."""


def run_pipeline(
    image_path: str,
    output_path: str,
    vlm_enabled: bool = True,
    debug: bool = False,
) -> str:
    """
    Execute the full raster-to-PPTX conversion pipeline.

    Phases:
        1. Detection (YOLO + Gemini iterative refinement + merge)
        2. Reconstruct vs. Crop decision
        3A. Reconstruct path (native shapes + OCR text overlay)
        3B. Crop path (image tile + text removal + OCR text overlay)
        4. PPTX assembly
        5. QA / SSIM validation

    Args:
        image_path: Path to the input raster slide image.
        output_path: Path for the output .pptx file.
        vlm_enabled: If False, skip Gemini VLM refinement (YOLO-only).
        debug: If True, save intermediate debug images.

    Returns:
        Path to the saved .pptx file.
    """
    raise NotImplementedError("Pipeline logic not yet implemented.")
