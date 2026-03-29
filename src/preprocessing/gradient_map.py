"""
Gradient magnitude map using Scharr operator.

Produces a continuous edge-magnitude channel that highlights
object boundaries — useful for detecting camouflaged fish whose
texture differs subtly from the background.
"""

import cv2
import numpy as np


def compute_gradient_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using the Scharr operator.

    Args:
        img_bgr: BGR image (H, W, 3), uint8

    Returns:
        gradient_mag: (H, W), float32, normalized to [0, 1]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val

    return magnitude.astype(np.float32)
