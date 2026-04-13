"""
Unit tests for DataFactory orchestration.

Verifies that the correct pipelines run for each Dataset enum value.
All pipeline classes are mocked — we're testing routing logic, not pipelines.
"""

from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from ..src.etl.factory import DataFactory, Dataset
from ..src.etl.inaturalist_dataset import INaturalistDataset
from ..src.etl.download_lila import LilaDataset
from ..src.etl.photo_transfer import PhotoTransferPipeline


@pytest.fixture
def factory():
    """Create a DataFactory with all external dependencies mocked."""
    with patch("whatsthatfish.src.etl.factory.get_config") as mock_config, \
         patch("whatsthatfish.src.etl.factory.get_session_factory") as mock_sf, \
         patch("whatsthatfish.src.etl.factory.GCSClient"):

        mock_config.return_value = MagicMock()
        mock_sf.return_value = MagicMock()
        yield DataFactory()


# ─── Pipeline routing ─────────────────────────────────────────────────


class TestPipelineRouting:
    """Each Dataset value should trigger exactly the right pipelines."""

    @patch.object(DataFactory, "_build_photo_transfer")
    @patch.object(DataFactory, "_build_lila")
    @patch.object(DataFactory, "_build_inat")
    def test_classification_runs_inat_and_transfer(
        self, mock_inat, mock_lila, mock_transfer, factory
    ):
        mock_inat.return_value = MagicMock(spec=INaturalistDataset)
        mock_lila.return_value = MagicMock(spec=LilaDataset)
        mock_transfer.return_value = MagicMock(spec=PhotoTransferPipeline, run=AsyncMock())

        factory.run(dataset=Dataset.CLASSIFICATION)

        mock_inat.return_value.run.assert_called_once_with(taxa=None)
        mock_lila.assert_not_called()
        mock_transfer.return_value.run.assert_awaited_once()

    @patch.object(DataFactory, "_build_photo_transfer")
    @patch.object(DataFactory, "_build_lila")
    @patch.object(DataFactory, "_build_inat")
    def test_detection_runs_lila_only(
        self, mock_inat, mock_lila, mock_transfer, factory
    ):
        mock_inat.return_value = MagicMock(spec=INaturalistDataset)
        mock_lila.return_value = MagicMock(spec=LilaDataset)
        mock_transfer.return_value = MagicMock(spec=PhotoTransferPipeline, run=AsyncMock())

        factory.run(dataset=Dataset.DETECTION)

        mock_inat.assert_not_called()
        mock_lila.return_value.extract_lila_images.assert_called_once()
        mock_transfer.assert_not_called()

    @patch.object(DataFactory, "_build_photo_transfer")
    @patch.object(DataFactory, "_build_lila")
    @patch.object(DataFactory, "_build_inat")
    def test_all_runs_every_pipeline(
        self, mock_inat, mock_lila, mock_transfer, factory
    ):
        mock_inat.return_value = MagicMock(spec=INaturalistDataset)
        mock_lila.return_value = MagicMock(spec=LilaDataset)
        mock_transfer.return_value = MagicMock(spec=PhotoTransferPipeline, run=AsyncMock())

        factory.run(dataset=Dataset.ALL)

        mock_inat.return_value.run.assert_called_once_with(taxa=None)
        mock_lila.return_value.extract_lila_images.assert_called_once()
        mock_transfer.return_value.run.assert_awaited_once()


# ─── Taxa passthrough ─────────────────────────────────────────────────


class TestTaxaPassthrough:
    """Custom taxa should be forwarded to the iNat pipeline."""

    @patch.object(DataFactory, "_build_photo_transfer")
    @patch.object(DataFactory, "_build_inat")
    def test_custom_taxa_forwarded_to_inat(
        self, mock_inat, mock_transfer, factory
    ):
        mock_inat.return_value = MagicMock(spec=INaturalistDataset)
        mock_transfer.return_value = MagicMock(spec=PhotoTransferPipeline, run=AsyncMock())

        custom_taxa = [12345, 67890]
        factory.run(dataset=Dataset.CLASSIFICATION, taxa=custom_taxa)

        mock_inat.return_value.run.assert_called_once_with(taxa=custom_taxa)

    @patch.object(DataFactory, "_build_photo_transfer")
    @patch.object(DataFactory, "_build_lila")
    @patch.object(DataFactory, "_build_inat")
    def test_taxa_not_passed_to_lila(
        self, mock_inat, mock_lila, mock_transfer, factory
    ):
        """LILA pipeline doesn't accept taxa — ensure no leakage."""
        mock_inat.return_value = MagicMock(spec=INaturalistDataset)
        mock_lila.return_value = MagicMock(spec=LilaDataset)
        mock_transfer.return_value = MagicMock(spec=PhotoTransferPipeline, run=AsyncMock())

        factory.run(dataset=Dataset.ALL, taxa=[47178])

        # extract_lila_images takes no arguments
        mock_lila.return_value.extract_lila_images.assert_called_once_with()


