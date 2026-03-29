"""
Image quality scoring for underwater images.

UIQM (Underwater Image Quality Measure) + histogram entropy scoring.
Used by the selective enhancement pipeline to identify images that
would benefit from enhancement vs. genuinely hard examples.
"""

import cv2
import numpy as np
from scipy import ndimage


def compute_uiqm(img_bgr: np.ndarray) -> float:
    """Underwater Image Quality Measure.

    Combines colorfulness, sharpness, and contrast into a single score.
    Higher is better. Typical range: [0, 5].

    Components:
        UICM: Underwater Image Colorfulness Measure
        UISM: Underwater Image Sharpness Measure
        UIConM: Underwater Image Contrast Measure

    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    """
    c1, c2, c3 = 0.0282, 0.2953, 3.5753

    uicm = _compute_uicm(img_bgr)
    uism = _compute_uism(img_bgr)
    uiconm = _compute_uiconm(img_bgr)

    return c1 * uicm + c2 * uism + c3 * uiconm


def _compute_uicm(img_bgr: np.ndarray) -> float:
    """Underwater Image Colorfulness Measure.

    Based on the difference between RG and YB color opponent channels.
    """
    b, g, r = img_bgr[:, :, 0].astype(np.float64), img_bgr[:, :, 1].astype(np.float64), img_bgr[:, :, 2].astype(np.float64)

    rg = r - g
    yb = (r + g) / 2.0 - b

    rg_mean, rg_std = rg.mean(), rg.std()
    yb_mean, yb_std = yb.mean(), yb.std()

    return -0.0268 * np.sqrt(rg_mean ** 2 + yb_mean ** 2) + 0.1586 * np.sqrt(rg_std ** 2 + yb_std ** 2)


def _compute_uism(img_bgr: np.ndarray) -> float:
    """Underwater Image Sharpness Measure.

    Uses Sobel edge detection on each channel.
    """
    sharpness = 0.0
    for ch in range(3):
        channel = img_bgr[:, :, ch].astype(np.float64)
        sx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sx ** 2 + sy ** 2)
        # EME (Enhancement Measure Estimation)
        block_size = 8
        h, w = edge.shape
        eme = 0.0
        count = 0
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = edge[i:i + block_size, j:j + block_size]
                bmax = block.max()
                bmin = block.min()
                if bmin > 0 and bmax > 0:
                    eme += 20.0 * np.log(bmax / bmin)
                    count += 1
        if count > 0:
            sharpness += eme / count

    return sharpness / 3.0


def _compute_uiconm(img_bgr: np.ndarray) -> float:
    """Underwater Image Contrast Measure.

    LogAMEE (Logarithmic Average of Maximum-to-Minimum Entropy Error).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    block_size = 8
    h, w = gray.shape
    amee = 0.0
    count = 0

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i + block_size, j:j + block_size]
            bmax = block.max()
            bmin = block.min()
            if bmin > 0 and bmax > 0:
                ratio = bmax / bmin
                amee += np.log(ratio + 1e-8)
                count += 1

    return amee / max(count, 1)


def compute_entropy(img_bgr: np.ndarray) -> float:
    """Color histogram entropy as a proxy for image information content.

    Low entropy → washed out / low contrast → likely degraded.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Compute histogram over H, S, V channels
    total_entropy = 0.0
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-8)
        # Shannon entropy
        nonzero = hist[hist > 0]
        total_entropy -= np.sum(nonzero * np.log2(nonzero))

    return total_entropy / 3.0


def compute_blur_score(img_bgr: np.ndarray) -> float:
    """Laplacian variance as a blur detection metric.

    Higher values = sharper image. Typical threshold: < 100 is blurry.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def classify_image_quality(img_bgr: np.ndarray,
                           uiqm_threshold: float = 2.5,
                           entropy_threshold: float = 6.0) -> str:
    """Classify image into quality categories.

    Returns:
        'bad_quality': Low UIQM + low entropy → enhancement will help
        'hard_example': Decent UIQM but may be genuinely difficult
        'good': Adequate quality
    """
    uiqm = compute_uiqm(img_bgr)
    entropy = compute_entropy(img_bgr)

    if uiqm < uiqm_threshold and entropy < entropy_threshold:
        return "bad_quality"
    elif uiqm < uiqm_threshold:
        return "hard_example"
    else:
        return "good"
