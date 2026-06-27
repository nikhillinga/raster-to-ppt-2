import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert raster slide to editable PPTX")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Output .pptx path")
    parser.add_argument("--no-vlm", action="store_true", help="Skip Gemini refinement")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()
    output_path = args.output or Path(args.image).stem + "_output.pptx"
    from pipeline.pipeline import run_pipeline
    run_pipeline(
        image_path=args.image,
        output_path=output_path,
        vlm_enabled=not args.no_vlm,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
