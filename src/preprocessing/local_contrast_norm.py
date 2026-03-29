"""
Local contrast normalization.

Divides each pixel by local mean and standard deviation,
making local anomalies (e.g., a camouflaged fish slightly
different from its surroundings) visible even when global
contrast is low.
"""

import cv2
import numpy as np


def local_contrast_normalize(img_bgr: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Compute local contrast normalization.

    Args:
        img_bgr: BGR image (H, W, 3), uint8
        kernel_size: Gaussian kernel size for local statistics (must be odd)

    Returns:
        lcn: (H, W), float32, rescaled to [0, 1]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    diff = gray - local_mean
    local_var = cv2.GaussianBlur(diff ** 2, (kernel_size, kernel_size), 0)
    local_std = np.sqrt(local_var + 1e-6)

    lcn = diff / local_std

    # Clip to [-3, 3] and rescale to [0, 1]
    lcn = np.clip(lcn, -3.0, 3.0)
    lcn = (lcn + 3.0) / 6.0

    return lcn.astype(np.float32)
