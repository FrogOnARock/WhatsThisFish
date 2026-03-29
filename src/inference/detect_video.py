"""
Video detection pipeline: raw video → detected + tracked fish.

Phase 1 (baseline): Simple frame extraction + 3-channel YOLO detection.
Phase 5 will add SSIM frame sampling, 5-channel preprocessing, and BoT-SORT tracking.
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(video_path: str, target_fps: float = 5.0) -> list[dict]:
    """Extract frames from video at the target FPS.

    Returns list of dicts with 'frame_idx', 'timestamp_sec', 'frame' (numpy array).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(src_fps / target_fps)))

    logger.info("Video: %s (%.1f fps, %d frames)", video_path, src_fps, total_frames)
    logger.info("Extracting every %d frames (target %.1f fps)", frame_interval, target_fps)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append({
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / src_fps,
                "frame": frame,
            })
        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames", len(frames))
    return frames


def detect_frames(model: YOLO, frames: list[dict], conf: float = 0.25,
                  iou: float = 0.45) -> list[dict]:
    """Run detection on extracted frames. Returns detections per frame."""
    all_detections = []

    for frame_data in tqdm(frames, desc="Detecting"):
        results = model.predict(
            source=frame_data["frame"],
            conf=conf,
            iou=iou,
            max_det=300,
            verbose=False,
        )

        frame_dets = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                # Get normalized xyxy boxes
                h, w = frame_data["frame"].shape[:2]
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    score = float(boxes.conf[i].cpu())
                    frame_dets.append({
                        "bbox": [
                            float(xyxy[0] / w),
                            float(xyxy[1] / h),
                            float(xyxy[2] / w),
                            float(xyxy[3] / h),
                        ],
                        "confidence": score,
                    })

        all_detections.append({
            "frame_idx": frame_data["frame_idx"],
            "timestamp_sec": frame_data["timestamp_sec"],
            "detections": frame_dets,
        })

    total_dets = sum(len(d["detections"]) for d in all_detections)
    logger.info("Total detections: %d across %d frames", total_dets, len(all_detections))
    return all_detections


def build_output(video_path: str, detections: list[dict], fps_processed: float) -> dict:
    """Build output JSON structure.

    Phase 1: No tracking, each detection is its own 'track'.
    Phase 5 will add proper BoT-SORT tracking with persistent IDs.
    """
    tracks = []
    track_id = 0

    for frame_data in detections:
        for det in frame_data["detections"]:
            tracks.append({
                "track_id": track_id,
                "species": None,
                "detections": [{
                    "frame_idx": frame_data["frame_idx"],
                    "timestamp_sec": frame_data["timestamp_sec"],
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                }],
            })
            track_id += 1

    return {
        "video_path": str(video_path),
        "fps_processed": fps_processed,
        "tracks": tracks,
        "summary": {
            "total_unique_fish": len(tracks),
            "frames_processed": len(detections),
        },
    }


def detect_video(model_path: str, video_path: str, output_dir: str,
                 fps: float = 5.0, conf: float = 0.25):
    """End-to-end video detection pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    frames = extract_frames(video_path, target_fps=fps)
    detections = detect_frames(model, frames, conf=conf)
    output = build_output(video_path, detections, fps)

    video_name = Path(video_path).stem
    output_path = output_dir / f"{video_name}_detections.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Summary: %d fish detected across %d frames",
                output["summary"]["total_unique_fish"],
                output["summary"]["frames_processed"])
    return output


def main():
    parser = argparse.ArgumentParser(description="Detect fish in video")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model weights")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to input video")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Target FPS for frame extraction")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold")
    args = parser.parse_args()

    detect_video(args.model, args.source, args.output,
                 fps=args.fps, conf=args.conf)


if __name__ == "__main__":
    main()
