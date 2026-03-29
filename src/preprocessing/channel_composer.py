"""
Compose 5-channel tensor from BGR image.

Channels: RGB(3) + gradient_magnitude(1) + local_contrast_norm(1)
"""

import cv2
import numpy as np

from whatsthatfish.src.preprocessing.gradient_map import compute_gradient_magnitude
from whatsthatfish.src.preprocessing.local_contrast_norm import local_contrast_normalize


def compose_channels(img_bgr: np.ndarray, lcn_kernel_size: int = 31) -> np.ndarray:
    """Compose 5-channel tensor from BGR image.

    Args:
        img_bgr: BGR image (H, W, 3), uint8
        lcn_kernel_size: Kernel size for local contrast normalization

    Returns:
        tensor: (H, W, 5), float32, all channels in [0, 1]
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    grad = compute_gradient_magnitude(img_bgr)
    lcn = local_contrast_normalize(img_bgr, kernel_size=lcn_kernel_size)

    return np.dstack([rgb, grad, lcn])
