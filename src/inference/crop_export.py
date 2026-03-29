"""
Export tracked fish crops for the species classification stage.

For each track, selects the N best frames (highest confidence, least blur)
and exports cropped fish regions with padding.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_laplacian_variance(img: np.ndarray) -> float:
    """Compute Laplacian variance as a sharpness measure."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def crop_with_padding(frame: np.ndarray, bbox: list[float],
                      padding: float = 0.15) -> np.ndarray:
    """Crop a region from the frame with padding around the bbox.

    Args:
        frame: Full frame (H, W, 3)
        bbox: Normalized [x1, y1, x2, y2] in [0, 1]
        padding: Padding ratio around the bbox

    Returns:
        Cropped region
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Convert to pixel coords
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)

    # Add padding
    bw, bh = px2 - px1, py2 - py1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    px1 = max(0, px1 - pad_x)
    py1 = max(0, py1 - pad_y)
    px2 = min(w, px2 + pad_x)
    py2 = min(h, py2 + pad_y)

    return frame[py1:py2, px1:px2]


def export_crops(detections_json: str, video_path: str, output_dir: str,
                 top_k: int = 5, padding: float = 0.15, crop_size: int = 224):
    """Export tracked fish crops for classification.

    Args:
        detections_json: Path to detection results JSON
        video_path: Path to source video
        output_dir: Output directory for crops
        top_k: Number of best frames per track
        padding: Padding ratio around bounding boxes
        crop_size: Resize crops to this size
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(detections_json) as f:
        data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    manifest = []

    for track in data["tracks"]:
        track_id = track["track_id"]
        dets = track["detections"]

        # Sort by confidence, take top_k
        dets_sorted = sorted(dets, key=lambda d: d["confidence"], reverse=True)
        dets_selected = dets_sorted[:top_k]

        track_dir = output_dir / f"track_{track_id:04d}"
        track_dir.mkdir(exist_ok=True)

        track_crops = []
        for det in dets_selected:
            frame_idx = det["frame_idx"]
            if frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            crop = crop_with_padding(frame, det["bbox"], padding=padding)
            if crop.size == 0:
                continue

            # Measure sharpness
            sharpness = compute_laplacian_variance(crop)

            # Resize
            crop_resized = cv2.resize(crop, (crop_size, crop_size))

            filename = f"frame_{frame_idx:06d}_conf{det['confidence']:.2f}.jpg"
            crop_path = track_dir / filename
            cv2.imwrite(str(crop_path), crop_resized)

            track_crops.append({
                "track_id": track_id,
                "frame_idx": frame_idx,
                "confidence": det["confidence"],
                "sharpness": sharpness,
                "crop_path": str(crop_path),
            })

        manifest.extend(track_crops)

    cap.release()

    manifest_path = output_dir / "crop_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Exported %d crops for %d tracks to %s",
                len(manifest), len(data["tracks"]), output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export fish crops for classification")
    parser.add_argument("--detections", type=str, required=True,
                        help="Path to detection results JSON")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source video")
    parser.add_argument("--output", type=str, default="crops/")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--crop_size", type=int, default=224)
    args = parser.parse_args()

    export_crops(args.detections, args.source, args.output,
                 top_k=args.top_k, padding=args.padding, crop_size=args.crop_size)


if __name__ == "__main__":
    main()
