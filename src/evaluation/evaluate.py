"""
Evaluation metrics for fish detection model.

Metrics and targets:
    mAP@0.5                  >= 0.75   Primary detection quality
    mAP@0.5:0.95             >= 0.50   Localization precision
    Recall@0.5 (conf=0.15)   >= 0.90   Ecological survey completeness
    Cross-domain mAP         Within 10%  Generalization
"""

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGETS = {
    "mAP50": 0.75,
    "mAP50-95": 0.50,
    "recall": 0.90,
}


def evaluate_model(model_path: str, data_yaml: str, split: str = "val",
                   conf: float = 0.25, imgsz: int = 640) -> dict:
    """Run evaluation on the specified split and return metrics."""
    logger.info("Loading model from %s", model_path)
    model = YOLO(model_path)

    logger.info("Evaluating on split='%s' with conf=%.2f", split, conf)
    results = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=0.45,
        imgsz=imgsz,
        max_det=300,
    )

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    logger.info("=== Evaluation Results ===")
    for metric, value in metrics.items():
        target = TARGETS.get(metric)
        status = ""
        if target:
            status = " ✓" if value >= target else f" ✗ (target: {target:.2f})"
        logger.info("  %-12s %.4f%s", metric, value, status)

    return metrics


def evaluate_high_recall(model_path: str, data_yaml: str, imgsz: int = 640) -> dict:
    """Evaluate at low confidence for ecological survey (high-recall mode)."""
    logger.info("Running high-recall evaluation (conf=0.15)")
    return evaluate_model(model_path, data_yaml, conf=0.15, imgsz=imgsz)


def evaluate_cross_domain(model_path: str, data_yaml: str,
                          holdout_sources: list[str] | None = None,
                          imgsz: int = 640) -> dict:
    """Evaluate cross-domain generalization on held-out source datasets.

    Recommended holdout: Coralscapes, DeepFish, and one river dataset.
    """
    logger.info("Cross-domain evaluation on holdout sources: %s", holdout_sources)
    # For cross-domain eval, we need a separate etl YAML that only contains
    # the held-out sources. This requires a pre-prepared split.
    # For now, run standard val and log for manual comparison.
    return evaluate_model(model_path, data_yaml, imgsz=imgsz)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fish detection model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--etl", type=str, required=True,
                        help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="val",
                        help="Evaluation split (train/val/test)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")
    parser.add_argument("--high_recall", action="store_true",
                        help="Run high-recall evaluation (conf=0.15)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save metrics JSON to this path")
    args = parser.parse_args()

    if args.high_recall:
        metrics = evaluate_high_recall(args.model, args.data, imgsz=args.imgsz)
    else:
        metrics = evaluate_model(
            args.model, args.data, split=args.split,
            conf=args.conf, imgsz=args.imgsz,
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
