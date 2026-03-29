#!/bin/bash
set -euo pipefail

echo "Setting up fish detection environment..."

conda create -n fishdet python=3.11 -y
conda activate fishdet

pip install ultralytics>=8.3.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless>=4.9
pip install albumentations>=1.4
pip install ensemble-boxes
pip install pycocotools
pip install gcsfs
pip install scikit-image
pip install supervision

echo "Environment setup complete."
