"""Phase 5: JSON QA report."""

import json
from pathlib import Path
import cv2

import config


def save_report(qa_result: dict, output_path: str, diff_image=None):
    """
    Saves JSON QA report and optional visual diff image.
    """
    report_path = Path(output_path).with_suffix(".qa.json")
    with open(report_path, "w") as f:
        json.dump(qa_result, f, indent=2)

    if diff_image is not None and config.QA_DIFF_SAVE:
        diff_path = Path(output_path).with_suffix(".diff.png")
        cv2.imwrite(str(diff_path), diff_image)
