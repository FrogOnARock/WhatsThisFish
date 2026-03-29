"""
Main training entrypoint for fish detection.

Three-stage transfer learning pipeline:
  Stage 1: COCO pretrained YOLO11l (automatic from base weights)
  Stage 2: Fish domain fine-tuning on LILA
  Stage 3: Target domain adaptation (optional, user-provided data)
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_stage2(config: dict, data_yaml: str, resume: str | None = None):
    """Stage 2: Fish domain fine-tuning on LILA dataset.

    - Loads COCO pretrained YOLO11l
    - Freezes backbone for initial epochs
    - Trains detection head on LILA with domain-balanced sampling
    - Unfreezes backbone with lower LR after warmup
    """
    model_weights = config.get("model", "yolo11l.pt")

    if resume:
        logger.info("Resuming training from %s", resume)
        model = YOLO(resume)
    else:
        logger.info("Loading pretrained weights: %s", model_weights)
        model = YOLO(model_weights)

    # Build training args from config, excluding non-ultralytics keys
    train_args = {
        "data": data_yaml,
        "epochs": config.get("epochs", 100),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", -1),
        "workers": config.get("workers", 8),
        "patience": config.get("patience", 20),
        "optimizer": config.get("optimizer", "SGD"),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.01),
        "momentum": config.get("momentum", 0.937),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 3),
        "warmup_momentum": config.get("warmup_momentum", 0.8),
        "warmup_bias_lr": config.get("warmup_bias_lr", 0.1),
        "box": config.get("box", 7.5),
        "cls": config.get("cls", 0.5),
        "dfl": config.get("dfl", 1.5),
        "hsv_h": config.get("hsv_h", 0.04),
        "hsv_s": config.get("hsv_s", 0.40),
        "hsv_v": config.get("hsv_v", 0.30),
        "flipud": config.get("flipud", 0.0),
        "fliplr": config.get("fliplr", 0.5),
        "mosaic": config.get("mosaic", 1.0),
        "close_mosaic": config.get("close_mosaic", 10),
        "mixup": config.get("mixup", 0.15),
        "scale": config.get("scale", 0.5),
        "translate": config.get("translate", 0.1),
        "freeze": config.get("freeze", 10),
        "project": "runs/detect",
        "name": "stage2",
        "exist_ok": True,
    }

    logger.info("Starting Stage 2 training with %d epochs", train_args["epochs"])
    results = model.train(**train_args)
    logger.info("Stage 2 training complete. Best weights: runs/detect/stage2/weights/best.pt")
    return results


def train_stage3(config: dict, data_yaml: str, base_model: str):
    """Stage 3: Target domain adaptation (optional).

    Fine-tunes the Stage 2 model on user-provided annotated samples
    from their specific dive footage with very low LR.
    """
    logger.info("Loading Stage 2 model from %s", base_model)
    model = YOLO(base_model)

    train_args = {
        "data": data_yaml,
        "epochs": config.get("finetune_epochs", 20),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", -1),
        "workers": config.get("workers", 8),
        "patience": 10,
        "lr0": config.get("lr0", 0.01) * 0.1,
        "lrf": config.get("lrf", 0.01),
        "freeze": 0,
        "project": "runs/detect",
        "name": "stage3",
        "exist_ok": True,
    }

    logger.info("Starting Stage 3 fine-tuning with LR=%.5f", train_args["lr0"])
    results = model.train(**train_args)
    logger.info("Stage 3 training complete. Best weights: runs/detect/stage3/weights/best.pt")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train fish detection model")
    parser.add_argument("--config", type=str, default="config/train_config.yaml",
                        help="Path to training config YAML")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML (Ultralytics format)")
    parser.add_argument("--stage", type=int, default=2, choices=[2, 3],
                        help="Training stage (2=LILA fine-tune, 3=target domain)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Path to base model for Stage 3")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage == 2:
        train_stage2(config, args.data, resume=args.resume)
    elif args.stage == 3:
        if not args.base_model:
            args.base_model = "runs/detect/stage2/weights/best.pt"
        train_stage3(config, args.data, args.base_model)


if __name__ == "__main__":
    main()
