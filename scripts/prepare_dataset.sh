#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Downloading LILA dataset ==="
python "$PROJECT_DIR/src/data/download_lila.py" \
    --output_dir "$PROJECT_DIR/data/raw/lila" \
    --sources DeepFish Coralscapes \
    --max_images 5000

echo "=== Converting COCO to YOLO format ==="
python "$PROJECT_DIR/src/data/download_lila.py" \
    --output_dir "$PROJECT_DIR/data/raw/lila" \
    --convert_only

echo "=== Dataset preparation complete ==="
