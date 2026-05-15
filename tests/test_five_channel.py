"""
Tests for AddMultiChannel: PIL Image -> (5, H, W) float32 tensor.

Channel layout: 0-2 RGB [0,1], 3 Scharr gradient [0,1], 4 LCN [0,1].
No infrastructure required — all tests use in-memory PIL fixtures.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from whatsthatfish.src.preprocessing.five_channel_conversion import AddMultiChannel


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def transform():
    return AddMultiChannel()


@pytest.fixture
def uniform_pil() -> Image.Image:
    """Solid grey image — no edges, no contrast variation."""
    return Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB")


@pytest.fixture
def step_edge_pil() -> Image.Image:
    """Left half black, right half white — sharp vertical edge at centre."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, 32:, :] = 255
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def random_pil() -> Image.Image:
    rng = np.random.default_rng(42)
    return Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8), mode="RGB")


# ════════════════════════════════════════════════════════════════════════════════
# Output shape and dtype
# ════════════════════════════════════════════════════════════════════════════════

class TestOutputFormat:

    def test_output_shape(self, transform, step_edge_pil):
        out = transform(step_edge_pil)
        assert out.shape == (5, 64, 64)

    def test_output_dtype(self, transform, step_edge_pil):
        out = transform(step_edge_pil)
        assert out.dtype == torch.float32

    def test_values_in_unit_range(self, transform, random_pil):
        out = transform(random_pil)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
# RGB channels (0-2)
# ════════════════════════════════════════════════════════════════════════════════

class TestRGBChannels:

    def test_rgb_normalized(self, transform, random_pil):
        """RGB channels should equal the original pixel values divided by 255."""
        arr = np.array(random_pil.convert("RGB"), dtype=np.float32) / 255.0
        out = transform(random_pil)
        expected = torch.from_numpy(arr.transpose(2, 0, 1))
        assert torch.allclose(out[:3], expected, atol=1e-5)

    def test_uniform_rgb_is_constant(self, transform, uniform_pil):
        out = transform(uniform_pil)
        assert out[:3].std() == pytest.approx(0.0, abs=1e-5)


# ════════════════════════════════════════════════════════════════════════════════
# Gradient channel (3)
# ════════════════════════════════════════════════════════════════════════════════

class TestGradientChannel:

    def test_uniform_image_has_zero_gradient(self, transform, uniform_pil):
        """No edges means gradient channel should collapse to zero."""
        out = transform(uniform_pil)
        assert out[3].max() == pytest.approx(0.0, abs=1e-5)

    def test_edge_image_has_nonzero_gradient(self, transform, step_edge_pil):
        out = transform(step_edge_pil)
        assert out[3].max() > 0.0


# ════════════════════════════════════════════════════════════════════════════════
# LCN channel (4)
# ════════════════════════════════════════════════════════════════════════════════

class TestLCNChannel:

    def test_uniform_image_has_flat_lcn(self, transform, uniform_pil):
        """No local contrast means LCN channel should be flat."""
        out = transform(uniform_pil)
        assert out[4].std() == pytest.approx(0.0, abs=1e-3)

    def test_edge_image_has_nonzero_lcn(self, transform, step_edge_pil):
        out = transform(step_edge_pil)
        assert out[4].max() > 0.0
