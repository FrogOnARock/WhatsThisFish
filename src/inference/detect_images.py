"""
Batch image detection for fish detection model.
"""

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def detect_images(model_path: str, source: str, output_dir: str,
                  conf: float = 0.25, imgsz: int = 640):
    """Run detection on a directory of images or a single image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=conf,
        iou=0.45,
        max_det=300,
        imgsz=imgsz,
        save=True,
        save_txt=True,
        project=str(output_dir),
        name="detections",
        exist_ok=True,
    )

    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    logger.info("Detected %d fish across %d images", total_dets, len(results))
    return results


def main():
    parser = argparse.ArgumentParser(description="Detect fish in images")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--source", type=str, required=True,
                        help="Image file or directory")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    detect_images(args.model, args.source, args.output,
                  conf=args.conf, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
