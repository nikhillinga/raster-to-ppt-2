"""Phase 5: PPTX Renderer via LibreOffice."""

import subprocess
import shutil
from pathlib import Path


def render_pptx_to_image(pptx_path: str, output_dir: str) -> str:
    """
    Uses LibreOffice headless to render the first slide to a PNG.
    Returns path to rendered image.
    """
    if not shutil.which("libreoffice"):
        raise RuntimeError("LibreOffice is not installed or not in PATH.")

    cmd = [
        "libreoffice", "--headless", "--convert-to", "png",
        "--outdir", output_dir, pptx_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice render failed: {result.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("LibreOffice render failed: TimeoutExpired after 60s")
    
    stem = Path(pptx_path).stem
    return str(Path(output_dir) / f"{stem}.png")
