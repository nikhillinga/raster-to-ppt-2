"""Phase 5: SSIM validation."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity

import config
from pipeline.models import ElementTree


def compute_ssim(source_path: str, rendered_path: str, tree: ElementTree) -> dict:
    """
    Returns a dict with overall SSIM and per-element SSIM.
    """
    source = cv2.imread(source_path)
    rendered = cv2.imread(rendered_path)

    # Resize rendered to match source dimensions
    if source.shape != rendered.shape:
        rendered = cv2.resize(rendered, (source.shape[1], source.shape[0]))

    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    rendered_gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)

    # Overall SSIM
    overall, diff = structural_similarity(
        source_gray, rendered_gray, 
        win_size=config.SSIM_WINDOW_SIZE, full=True,
        data_range=255  # For 8-bit images
    )

    # Per-element SSIM (walk full tree)
    element_scores = []
    type_scores = {}
    seen_ids = set()

    for elem in tree.all_elements():
        if elem.id in seen_ids:
            continue
        seen_ids.add(elem.id)

        b = elem.bbox
        src_crop = source_gray[b.y:b.y2, b.x:b.x2]
        ren_crop = rendered_gray[b.y:b.y2, b.x:b.x2]

        if src_crop.size == 0 or ren_crop.size == 0:
            continue

        if src_crop.shape != ren_crop.shape:
            ren_crop = cv2.resize(ren_crop, (src_crop.shape[1], src_crop.shape[0]))

        # Need minimum size for SSIM window
        if min(src_crop.shape) < config.SSIM_WINDOW_SIZE:
            continue

        elem_ssim = structural_similarity(
            src_crop, ren_crop, 
            win_size=config.SSIM_WINDOW_SIZE,
            data_range=255
        )
        flagged = bool(elem_ssim < config.SSIM_ELEMENT_FLAG_THRESHOLD)

        element_scores.append({
            "id": elem.id,
            "semantic_type": elem.semantic_type,
            "ssim": round(float(elem_ssim), 4),
            "flagged": flagged
        })

        if elem.semantic_type not in type_scores:
            type_scores[elem.semantic_type] = []
        type_scores[elem.semantic_type].append(float(elem_ssim))

    type_summary = {t: round(float(np.mean(v)), 4) for t, v in type_scores.items()}

    return {
        "overall_ssim": round(float(overall), 4),
        "passed": bool(overall >= config.SSIM_PASS_THRESHOLD),
        "element_scores": element_scores,
        "type_summary": type_summary
    }
