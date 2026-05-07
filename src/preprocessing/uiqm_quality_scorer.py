"""UIQM (Underwater Image Quality Measure) scoring.

Pure scoring functions: input is an OpenCV BGR image (np.uint8, HxWx3),
output is a float (or tuple of floats for the composite).

UIQM = c1 * UICM + c2 * UISM + c3 * UIConM   (Panetta et al. 2015)

Sub-scores are returned alongside the composite so downstream consumers
can re-weight without re-scoring.
"""
from __future__ import annotations

import cv2
import numpy as np


class QualityScorer:
    def __init__(self,
                UIQM_C1_UICM: float = 0.0282,
                UIQM_C2_UISM: float = 0.2953,
                UIQM_C3_UICONM: float = 3.5753,
                BLOCK_SIZE: int = 8,
                LUM_B: float = 0.114,
                LUM_R: float = 0.299,
                LUM_G: float = 0.587):


        # Composite weights from the original UIQM paper.
        # Tunable — kept as module constants so calibration experiments are explicit.
        self.UIQM_C1_UICM = UIQM_C1_UICM
        self.UIQM_C2_UISM = UIQM_C2_UISM
        self.UIQM_C3_UICONM = UIQM_C3_UICONM

        # Block size for UISM and UIConM block-based reductions.
        self.BLOCK_SIZE = BLOCK_SIZE

        # NTSC luminance weights for UISM channel aggregation.
        self.LUM_B = LUM_B
        self.LUM_R = LUM_R
        self.LUM_G = LUM_G

    def _alpha_trimmed_stats(
        self,
        values: np.ndarray,
        alpha_l: float = 0.1,
        alpha_r: float = 0.1,
    ) -> tuple[float, float]:
        """Alpha-trimmed mean and variance of a 1-D float array.

        Drops the bottom alpha_l fraction and top alpha_r fraction of values
        before computing mean and variance — robust against outliers like
        glints or dead pixels.
        """
        sorted_vals = np.sort(values, axis=None)
        n = sorted_vals.size
        lo = int(np.floor(alpha_l * n))
        hi = n - int(np.floor(alpha_r * n))
        trimmed = sorted_vals[lo:hi]
        if trimmed.size == 0:
            return 0.0, 0.0
        return float(trimmed.mean()), float(trimmed.var())


    def compute_uicm(self, img_bgr: np.ndarray) -> float:
        """Underwater Image Colorfulness Measure (UICM).

        Computed on two chrominance channels:
            RG = R - G            (red-green opponency)
            YB = (R + G) / 2 - B  (yellow-blue opponency)

        UICM = -0.0268 * sqrt(mean_RG^2 + mean_YB^2)
               + 0.1586 * sqrt(var_RG + var_YB)

        The negative-mean term penalizes color casts (e.g. uniform blue tint
        from depth attenuation); the positive-variance term rewards
        chromatic spread.
    """
        img = img_bgr.astype(np.float64)
        b, g, r = img[..., 0], img[..., 1], img[..., 2]

        rg = r - g
        yb = 0.5 * (r + g) - b

        mean_rg, var_rg = self._alpha_trimmed_stats(rg)
        mean_yb, var_yb = self._alpha_trimmed_stats(yb)

        mean_term = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        var_term = np.sqrt(var_rg + var_yb)

        return float(-0.0268 * mean_term + 0.1586 * var_term)


    def _block_view(self, arr: np.ndarray) -> np.ndarray:
        """Reshape a 2-D array into non-overlapping (nH, nW, B, B) blocks.

        Crops the right/bottom edges so dimensions are exact multiples of
        `block`. The returned view shares memory with `arr` — no copy.
        """
        block = self.BLOCK_SIZE
        h, w = arr.shape
        h_crop = (h // block) * block
        w_crop = (w // block) * block
        cropped = arr[:h_crop, :w_crop]
        return (
            cropped
            .reshape(h_crop // block, block, w_crop // block, block)
            .transpose(0, 2, 1, 3)
        )


    def _channel_eme(self, channel: np.ndarray) -> float:
        """EME (Enhancement Measure of Enhancement) over non-overlapping blocks.

        EME = (2 / k1*k2) * sum_blocks( log(bmax / bmin) )

        Used by UISM. Skips blocks where bmin == 0 to avoid divide-by-zero;
        those blocks contribute nothing to the score (which is the standard
        handling — a fully-black region has no measurable enhancement).
        """
        blocks = self._block_view(channel)
        bmax = blocks.max(axis=(-2, -1)).astype(np.float64)
        bmin = blocks.min(axis=(-2, -1)).astype(np.float64)

        valid = bmin > 0
        if not valid.any():
            return 0.0

        ratios = np.log(bmax[valid] / bmin[valid])
        n_blocks = blocks.shape[0] * blocks.shape[1]
        return float(2.0 / n_blocks * ratios.sum())


    def compute_uism(self, img_bgr: np.ndarray) -> float:
        """Underwater Image Sharpness Measure (UISM).

        Apply Sobel edge magnitude per channel, then aggregate via per-block
        EME and combine with NTSC luminance weights:

            UISM = lambda_R * EME(|sobel(R)|)
                 + lambda_G * EME(|sobel(G)|)
                 + lambda_B * EME(|sobel(B)|)

        Sobel is multiplied elementwise by the original channel before EME,
        per the original paper — sharper edges in brighter regions
        contribute more.
        """
        img = img_bgr.astype(np.float64)
        b, g, r = img[..., 0], img[..., 1], img[..., 2]

        def edge_weighted(channel: np.ndarray) -> np.ndarray:
            sx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sx * sx + sy * sy)
            return channel * magnitude

        eme_r = self._channel_eme(edge_weighted(r))
        eme_g = self._channel_eme(edge_weighted(g))
        eme_b = self._channel_eme(edge_weighted(b))

        return float(self.LUM_R * eme_r + self.LUM_G * eme_g + self.LUM_B * eme_b)


    def compute_uiconm(self, img_bgr: np.ndarray) -> float:
        """Underwater Image Contrast Measure (UIConM).

        Per-block log-ratio of intensity extremes on the grayscale image:

            UIConM = (1 / nBlocks) * sum_blocks( log(bmax / bmin) )

        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        blocks = self._block_view(gray)

        bmax = blocks.max(axis=(-2, -1)).astype(np.float64)
        bmin = blocks.min(axis=(-2, -1)).astype(np.float64)
        n_blocks = blocks.shape[0] * blocks.shape[1]

        if not (blocks > 0).any():
            return 0.0

        valid = bmin > 0
        if not valid.any():
            return 0.0


        ratios = np.log( bmax[valid] / bmin[valid] )

        return ( ( 1 / n_blocks ) * ratios.sum() ).astype(np.float64)


    def compute_uiqm(
            self,
            image_bytes: bytes,
        ) -> ValueError | tuple[float, float, float, float]:
        """Composite UIQM and its three sub-scores.

        Returns (uicm, uism, uiconm, uiqm). Sub-scores are returned
        alongside the composite so downstream consumers can re-weight
        without re-scoring.
        """
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return ValueError("Failed to decode image")

        uicm = self.compute_uicm(img_bgr)
        uism = self.compute_uism(img_bgr)
        uiconm = self.compute_uiconm(img_bgr)
        uiqm = self.UIQM_C1_UICM * uicm + self.UIQM_C2_UISM * uism + self.UIQM_C3_UICONM * uiconm
        return uicm, uism, uiconm, uiqm
