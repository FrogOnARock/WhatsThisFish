"""
Tests for object_detection_collate and ObjectDetectionDataset.

collate tests: pure tensor logic, no mocking needed.
dataset tests: DB and GCS dependencies mocked.
"""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from whatsthatfish.src.models.od_dataloader import object_detection_collate
from whatsthatfish.src.models.od_dataset import ObjectDetectionDataset


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_image_bytes(h: int = 64, w: int = 64) -> bytes:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _image_tensor(h: int = 64, w: int = 64) -> torch.Tensor:
    return torch.rand(3, h, w)


def _label_tensor(*rows) -> torch.Tensor:
    """Build an (N, 5) label tensor from rows of [class_id, x, y, w, h]."""
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(list(rows), dtype=torch.float32)


# ── Fake DB row ────────────────────────────────────────────────────────────────

def _fake_row(file_name: str, uiqm: float, annotations: list[dict]):
    row = MagicMock()
    row.file_name = file_name
    row.uiqm = uiqm
    row.annotation = annotations
    return row


# ════════════════════════════════════════════════════════════════════════════════
# object_detection_collate
# ════════════════════════════════════════════════════════════════════════════════

class TestCollate:

    def test_image_stack_shape(self):
        batch = [(_image_tensor(), _label_tensor()) for _ in range(4)]
        imgs, _ = object_detection_collate(batch)
        assert imgs.shape == (4, 3, 64, 64)

    def test_batch_index_prepended(self):
        batch = [
            (_image_tensor(), _label_tensor([0, 0.5, 0.5, 0.2, 0.2])),
            (_image_tensor(), _label_tensor([0, 0.3, 0.3, 0.1, 0.1])),
        ]
        _, labels = object_detection_collate(batch)
        assert labels[0, 0] == 0.0
        assert labels[1, 0] == 1.0

    def test_label_shape_is_n_by_6(self):
        batch = [
            (_image_tensor(), _label_tensor([0, 0.5, 0.5, 0.2, 0.2], [0, 0.1, 0.1, 0.05, 0.05])),
            (_image_tensor(), _label_tensor([0, 0.3, 0.3, 0.1, 0.1])),
        ]
        _, labels = object_detection_collate(batch)
        assert labels.shape == (3, 6)

    def test_all_negative_batch_returns_empty(self):
        batch = [(_image_tensor(), _label_tensor()) for _ in range(4)]
        _, labels = object_detection_collate(batch)
        assert labels.shape == (0, 6)

    def test_mixed_positive_negative_batch(self):
        batch = [
            (_image_tensor(), _label_tensor([0, 0.5, 0.5, 0.2, 0.2])),
            (_image_tensor(), _label_tensor()),
            (_image_tensor(), _label_tensor([0, 0.3, 0.3, 0.1, 0.1])),
        ]
        _, labels = object_detection_collate(batch)
        assert labels.shape == (2, 6)
        assert set(labels[:, 0].tolist()) == {0.0, 2.0}

    def test_label_values_preserved(self):
        row = [0.0, 0.5, 0.5, 0.2, 0.2]
        batch = [(_image_tensor(), _label_tensor(row))]
        _, labels = object_detection_collate(batch)
        assert torch.allclose(labels[0, 1:], torch.tensor(row, dtype=torch.float32))


# ════════════════════════════════════════════════════════════════════════════════
# ObjectDetectionDataset
# ════════════════════════════════════════════════════════════════════════════════

FISH_ANN = [{"class_id": 0, "is_train": True, "norm_center_x": 0.5, "norm_center_y": 0.5, "norm_width": 0.2, "norm_height": 0.2}]
NEG_ANN  = [{"class_id": 1, "is_train": True, "norm_center_x": 0.0, "norm_center_y": 0.0, "norm_width": 0.0, "norm_height": 0.0}]

FAKE_ROWS = [
    _fake_row("fish_001.jpg", 0.8, FISH_ANN),
    _fake_row("neg_001.jpg",  0.4, NEG_ANN),
    _fake_row("fish_002.jpg", 0.6, FISH_ANN),
]

_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])


@pytest.fixture
def dataset():
    mock_session_instance = MagicMock()
    mock_session_instance.execute.return_value.all.return_value = FAKE_ROWS

    mock_session_factory = MagicMock(return_value=mock_session_instance)
    mock_get_session = MagicMock(return_value=mock_session_factory)

    mock_gcs_config = MagicMock()
    mock_gcs_config.bucket = "whats-that-fish"
    mock_gcs_config.prefixes.get.return_value = "object_detection/"
    mock_config = MagicMock()
    mock_config.gcs = mock_gcs_config

    with patch("whatsthatfish.src.models.od_dataset.get_session_factory", mock_get_session), \
         patch("whatsthatfish.src.models.od_dataset.get_config", return_value=mock_config):
        ds = ObjectDetectionDataset(split="train", transforms=_transform)

    return ds


class TestObjectDetectionDataset:

    def test_len(self, dataset):
        assert len(dataset) == 3

    def test_getitem_image_shape(self, dataset):
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = _make_image_bytes()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with patch("whatsthatfish.src.models.od_dataset.storage") as mock_storage:
            mock_storage.Client.from_service_account_json.return_value.bucket.return_value = mock_bucket
            with patch.dict("os.environ", {"GCS_SECRET": "/fake/path.json"}):
                img, _ = dataset[0]

        assert img.shape == (3, 64, 64)
        assert img.dtype == torch.float32

    def test_getitem_positive_label_shape(self, dataset):
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = _make_image_bytes()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with patch("whatsthatfish.src.models.od_dataset.storage") as mock_storage:
            mock_storage.Client.from_service_account_json.return_value.bucket.return_value = mock_bucket
            with patch.dict("os.environ", {"GCS_SECRET": "/fake/path.json"}):
                _, labels = dataset[0]

        assert labels.shape == (1, 5)
        assert labels.dtype == torch.float32

    def test_getitem_negative_label_is_empty(self, dataset):
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = _make_image_bytes()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        with patch("whatsthatfish.src.models.od_dataset.storage") as mock_storage:
            mock_storage.Client.from_service_account_json.return_value.bucket.return_value = mock_bucket
            with patch.dict("os.environ", {"GCS_SECRET": "/fake/path.json"}):
                _, labels = dataset[1]

        assert labels.shape == (0, 5)

    def test_uiqm_weights_length_matches_dataset(self, dataset):
        weights = [row.uiqm for row in dataset.data]
        assert len(weights) == len(dataset)
