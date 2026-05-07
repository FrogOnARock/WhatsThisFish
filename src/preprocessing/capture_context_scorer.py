"""Capture-context scoring (underwater vs above-water).

iNaturalist fish observations are a mix of underwater dive shots, fishing-deck
photos, aquarium shots, market photos, and lab specimens. For a dive-companion
classifier we want to bias training toward underwater captures only.

The v1 heuristic exploits a physical fact: water absorbs red light within the
first few meters of depth. Underwater photos have systematically suppressed red
and elevated blue/green; above-water photos do not. This produces a strong
univariate signal in the per-channel means.

Pure scoring functions: input is an OpenCV BGR image (np.uint8, HxWx3), output
is a tuple of raw channel means plus a Boolean underwater verdict. The raw
means are stored alongside the verdict so thresholds and rules can be retuned
later via SQL UPDATE rather than re-scoring 1M images.

If this heuristic underperforms on a manually-labeled spot-check sample,
swap in a CLIP-based scorer with the same return shape — the table schema
already accommodates it.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any

# Please review image_rgb.ipynb for experimentation to understand the threshold set for standard deviation
# of the red, green, blue chromaticity
# There was not a clear linear boundary between under water and above water images that could be attained
# through just red chromaticity nor a combination of red/blue chromaticity. The standard deviation
# of channel chromaticity was determined to be a good determinant in what WAS an above water image
# Choosing to proceed with filtering what we know is an above water image helps us retain some of the images from
# shallower depths while removing the large portion of images we know is not valuable.
# Anything under 0.015 has been denoted above water, while between 0.15 and 0.25 considered ambiguous, while anything
# above 0.25 is considered underwater



class ContextScorer:
    def __init__(self):
        self._std_chrom: float = 0.015

    @staticmethod
    def compute_channel_means(img_bgr: np.ndarray) -> tuple[float, float, float]:
        """Per-channel pixel-mean of a BGR image.
    
        Returns (mean_r, mean_g, mean_b) in conventional R-G-B order — note the
        index swap, since OpenCV's native layout is BGR. Means are computed in
        float64 to avoid uint8 overflow on the sum.
        """
        img = img_bgr.astype(np.float64)
        return (
            float(img[..., 2].mean()),  # R — BGR index 2
            float(img[..., 1].mean()),  # G — BGR index 1
            float(img[..., 0].mean()),  # B — BGR index 0
        )
    
    
    def classify_underwater(
        self,
        mean_r: float,
        mean_g: float,
        mean_b: float,
    ) -> tuple[int, Any]:
        """
        Combined channel standard deviation metric to determine what images
        are confidently above water
        """
    
        total = mean_r + mean_g + mean_b
        if total == 0:
            return 0, 0.0  # or surface as unscoreable to the caller
    
        red_chrom = mean_r / total
        blue_chrom = mean_b / total
        green_chrom = mean_g / total
        std_dev = np.std([red_chrom, blue_chrom, green_chrom])
    
        if std_dev < self._std_chrom:
            return 0, std_dev
        elif std_dev < 0.25:
            return 1, std_dev
        else:
            return 2, std_dev
    
    
    def score_capture_context(
        self,
        image_bytes: bytes,
    ) -> ValueError | tuple[float, float, float, Any, int]:
        """Composite scorer: per-channel means plus underwater verdict.
    
        Returns (mean_r, mean_g, mean_b, is_underwater). The means are stored
        in the DB alongside the verdict so the threshold/rule can be retuned
        later without re-scoring images.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return ValueError("Failed to decode image")

        mean_r, mean_g, mean_b = self.compute_channel_means(img_bgr)
        classification, stddev = self.classify_underwater(mean_r, mean_g, mean_b)
        return mean_r, mean_g, mean_b, stddev, classification
