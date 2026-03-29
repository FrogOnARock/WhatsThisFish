"""
Basic pipeline tests to verify components work end-to-end.
"""

import numpy as np
import pytest


class TestGradientMap:
    def test_output_shape(self):
        from whatsthatfish.src import compute_gradient_magnitude
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = compute_gradient_magnitude(img)
        assert result.shape == (100, 150)
        assert result.dtype == np.float32

    def test_output_range(self):
        from whatsthatfish.src import compute_gradient_magnitude
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = compute_gradient_magnitude(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_image_low_gradient(self):
        from whatsthatfish.src import compute_gradient_magnitude
        img = np.full((100, 150, 3), 128, dtype=np.uint8)
        result = compute_gradient_magnitude(img)
        assert result.max() < 0.01


class TestLocalContrastNorm:
    def test_output_shape(self):
        from whatsthatfish.src.preprocessing.local_contrast_norm import local_contrast_normalize
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = local_contrast_normalize(img)
        assert result.shape == (100, 150)
        assert result.dtype == np.float32

    def test_output_range(self):
        from whatsthatfish.src.preprocessing.local_contrast_norm import local_contrast_normalize
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = local_contrast_normalize(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestChannelComposer:
    def test_output_shape(self):
        from whatsthatfish.src.preprocessing.channel_composer import compose_channels
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = compose_channels(img)
        assert result.shape == (100, 150, 5)
        assert result.dtype == np.float32

    def test_output_range(self):
        from whatsthatfish.src.preprocessing.channel_composer import compose_channels
        img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = compose_channels(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rgb_channels_preserved(self):
        from whatsthatfish.src.preprocessing.channel_composer import compose_channels
        import cv2
        img_bgr = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = compose_channels(img_bgr)
        expected_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        np.testing.assert_allclose(result[:, :, :3], expected_rgb, atol=1e-6)
