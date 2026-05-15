"""
Tests for lcn_gradient_map: gradient_map and local_contrast_normalization.

Both functions accept (H, W) float64 grayscale arrays and return (H, W) uint8.
No infrastructure required — all tests use in-memory numpy fixtures.
"""

import numpy as np
import pytest

from whatsthatfish.src.preprocessing.lcn_gradient_map import (
    gradient_map,
    local_contrast_normalization,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def uniform_gray() -> np.ndarray:
    """Solid grey — no edges, no contrast variation."""
    return np.full((64, 64), 128.0, dtype=np.float64)


@pytest.fixture
def step_edge() -> np.ndarray:
    """Left half black, right half white — sharp vertical edge at centre column."""
    arr = np.zeros((64, 64), dtype=np.float64)
    arr[:, 32:] = 255.0
    return arr


# ════════════════════════════════════════════════════════════════════════════════
# gradient_map
# ════════════════════════════════════════════════════════════════════════════════

class TestGradientMap:

    def test_output_is_2d_uint8(self, step_edge):
        out = gradient_map(step_edge)
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self, step_edge):
        out = gradient_map(step_edge)
        assert out.shape == (64, 64)

    def test_values_in_valid_range(self, step_edge):
        out = gradient_map(step_edge)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_edge_pixels_are_nonzero(self, step_edge):
        """The column at the black/white boundary should have high gradient magnitude."""
        out = gradient_map(step_edge)
        assert out[:, 32].mean() > 100

    def test_uniform_image_collapses_to_zeros(self, uniform_gray):
        """Flat image has no gradient — min-max normalisation collapses to all zeros."""
        out = gradient_map(uniform_gray)
        assert out.max() == 0


# ════════════════════════════════════════════════════════════════════════════════
# local_contrast_normalization
# ════════════════════════════════════════════════════════════════════════════════

class TestLocalContrastNormalization:

    def test_output_is_2d_uint8(self, step_edge):
        out = local_contrast_normalization(step_edge)
        assert out.ndim == 2
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self, step_edge):
        out = local_contrast_normalization(step_edge)
        assert out.shape == (64, 64)

    def test_values_in_valid_range(self, step_edge):
        out = local_contrast_normalization(step_edge)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_uniform_image_produces_flat_output(self, uniform_gray):
        """A uniform image has no local contrast — output should be flat."""
        out = local_contrast_normalization(uniform_gray)
        assert out.std() == pytest.approx(0.0, abs=1.0)

    def test_high_contrast_image_spans_full_range(self, step_edge):
        """A sharp black/white step should produce output spanning most of [0, 255]."""
        out = local_contrast_normalization(step_edge)
        assert out.max() - out.min() > 200
