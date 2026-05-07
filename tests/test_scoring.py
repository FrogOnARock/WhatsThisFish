"""
Tests for the scoring pipeline: scorers, tracker WAL behaviour, and runner orchestration.

Unit / WAL tests (TestQualityScorer, TestContextScorer, TestScoringProgressTrackerWAL,
TestContextRunnerTracking, TestScoreRunnerTracking, TestPreProcessingFactoryRouting)
require no infrastructure — they use in-memory SQLite or mocks.

DB compaction tests (TestScoringProgressTrackerDB) require Postgres:
    docker compose -f docker-compose.test.yml up -d
"""

import asyncio
import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from whatsthatfish.src.config import GCSConfig
from whatsthatfish.src.database.base import Base
from whatsthatfish.src.database.models import (
    InatCaptureContext,
    InatFilteredObservations,
    InatImageQuality,
    InatTaxa,
    LilaCollectedImages,
    LilaImageQuality,
)
from whatsthatfish.src.preprocessing.capture_context_scorer import ContextScorer
from whatsthatfish.src.preprocessing.factory import Dataset, PreProcessingFactory
from whatsthatfish.src.preprocessing.score_runner import (
    ContextRunner,
    ScoreRunner,
    ScoringProgressTracker,
)
from whatsthatfish.src.preprocessing.uiqm_quality_scorer import QualityScorer


# ── Image helpers ──────────────────────────────────────────────────────────────

def _make_image_bytes(r: int, g: int, b: int, size: tuple[int, int] = (64, 64)) -> bytes:
    """Solid-colour JPEG encoded as bytes (OpenCV BGR channel order)."""
    img = np.full((size[0], size[1], 3), (b, g, r), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _make_gradient_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    """Horizontal grayscale gradient — has visible edges so UISM > 0."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for col in range(size[1]):
        img[:, col, :] = int(255 * col / size[1])
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sqlite_session_factory():
    """In-memory SQLite for WAL-only tracker tests.

    Only creates the tables used by the WAL tests. LilaYolo is excluded because
    its JSONB column is PostgreSQL-specific and SQLite cannot render it.
    """
    engine = create_engine("sqlite:///:memory:")
    _SQLITE_TABLES = [
        InatTaxa.__table__,
        InatFilteredObservations.__table__,
        InatCaptureContext.__table__,
        InatImageQuality.__table__,
        LilaCollectedImages.__table__,
        LilaImageQuality.__table__,
    ]
    Base.metadata.create_all(engine, tables=_SQLITE_TABLES)
    return sessionmaker(bind=engine)


@pytest.fixture
def gcs_config():
    return GCSConfig(
        bucket="test-bucket",
        prefixes={"gcs_train": "training/", "gcs_object_detection": "object_detection/"},
    )


@pytest.fixture
def blue_bytes() -> bytes:
    return _make_image_bytes(r=0, g=20, b=200)


@pytest.fixture
def gray_bytes() -> bytes:
    return _make_image_bytes(r=128, g=128, b=128)


@pytest.fixture
def gradient_bytes() -> bytes:
    return _make_gradient_bytes()


# ── Postgres-only fixtures (skipped if no docker) ──────────────────────────────

@pytest.fixture
def seeded_inat_uuid(session_factory):
    """Insert the FK parent chain and return a valid photo_uuid for scoring rows."""
    uuid = "test-score-uuid-001"
    with session_factory() as session:
        session.add(InatTaxa(taxon_id=88888, name="Test Fish", active=True))
        session.add(InatFilteredObservations(
            photo_uuid=uuid,
            photo_id=88888001,
            observation_uuid="test-obs-uuid-001",
            observer_id=1,
            taxon_id=88888,
            extension="jpg",
            license="cc-by",
        ))
        session.commit()
    return uuid


@pytest.fixture
def seeded_lila_filename(session_factory):
    """Insert a LilaCollectedImages parent row and return its file_name."""
    fname = "test-dataset/test-image.jpg"
    with session_factory() as session:
        session.add(LilaCollectedImages(
            id=fname,
            file_name=fname,
            dataset="test-dataset",
            is_train=True,
            width=640,
            height=480,
        ))
        session.commit()
    return fname


# ── Factory fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def factory_mocks():
    """Patch external deps so PreProcessingFactory can be constructed in isolation."""
    with patch("whatsthatfish.src.preprocessing.factory.get_config") as mock_cfg, \
         patch("whatsthatfish.src.preprocessing.factory.get_session_factory") as mock_sf, \
         patch("whatsthatfish.src.preprocessing.factory.GCSClient"):
        mock_cfg.return_value = MagicMock()
        mock_cfg.return_value.gcs = GCSConfig(bucket="b", prefixes={})
        mock_sf.return_value = MagicMock()
        yield


# ════════════════════════════════════════════════════════════════════════════════
# QualityScorer
# ════════════════════════════════════════════════════════════════════════════════

class TestQualityScorer:

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_compute_uiqm_returns_four_floats(self, blue_bytes):
        result = self.scorer.compute_uiqm(blue_bytes)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_compute_uiqm_corrupt_bytes_returns_value_error(self):
        result = self.scorer.compute_uiqm(b"not-an-image")
        assert isinstance(result, ValueError)

    def test_uicm_negative_for_blue_cast(self, blue_bytes):
        """Strong colour cast should produce a negative UICM (mean-term dominates)."""
        arr = np.frombuffer(blue_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert self.scorer.compute_uicm(img) < 0

    def test_uism_positive_for_image_with_edges(self, gradient_bytes):
        arr = np.frombuffer(gradient_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert self.scorer.compute_uism(img) > 0

    def test_uiconm_zero_for_all_black(self):
        """All-black image has no contrast — guard must return 0.0, not crash."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        assert self.scorer.compute_uiconm(black) == 0.0

    def test_composite_equals_weighted_sum_of_subscores(self, gradient_bytes):
        arr = np.frombuffer(gradient_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        uicm = self.scorer.compute_uicm(img)
        uism = self.scorer.compute_uism(img)
        uiconm = self.scorer.compute_uiconm(img)
        expected = (
            self.scorer.UIQM_C1_UICM * uicm
            + self.scorer.UIQM_C2_UISM * uism
            + self.scorer.UIQM_C3_UICONM * uiconm
        )
        _, _, _, uiqm = self.scorer.compute_uiqm(gradient_bytes)
        assert abs(uiqm - expected) < 1e-9


# ════════════════════════════════════════════════════════════════════════════════
# ContextScorer
# ════════════════════════════════════════════════════════════════════════════════

class TestContextScorer:

    def setup_method(self):
        self.scorer = ContextScorer()

    def test_balanced_channels_is_above_water(self):
        """Equal channel means → near-zero chromaticity std → above water (0)."""
        classification, stddev = self.scorer.classify_underwater(128.0, 128.0, 128.0)
        assert classification == 0
        assert stddev < 0.015

    def test_deep_blue_dominant_is_confidently_underwater(self):
        classification, stddev = self.scorer.classify_underwater(0.0, 10.0, 200.0)
        assert classification == 2
        assert stddev >= 0.25

    def test_mildly_tinted_image_is_ambiguous(self):
        """Moderate colour skew → ambiguous (1), between the two thresholds."""
        classification, stddev = self.scorer.classify_underwater(200.0, 100.0, 50.0)
        assert classification == 1
        assert 0.015 <= stddev < 0.25

    def test_all_zero_pixels_returns_above_water_zero_stddev(self):
        """total == 0 guard path must return (0, 0.0), not crash on unpack."""
        classification, stddev = self.scorer.classify_underwater(0.0, 0.0, 0.0)
        assert classification == 0
        assert stddev == 0.0

    def test_score_capture_context_returns_five_values(self, blue_bytes):
        result = self.scorer.score_capture_context(blue_bytes)
        assert len(result) == 5
        mean_r, mean_g, mean_b, stddev, classification = result
        assert isinstance(mean_r, float)
        assert isinstance(stddev, float)

    def test_score_capture_context_blue_image_not_above_water(self, blue_bytes):
        _, _, _, _, classification = self.scorer.score_capture_context(blue_bytes)
        assert classification > 0

    def test_score_capture_context_corrupt_bytes_returns_value_error(self):
        assert isinstance(self.scorer.score_capture_context(b"garbage"), ValueError)


# ════════════════════════════════════════════════════════════════════════════════
# ScoringProgressTracker — WAL behaviour (SQLite, compact never fires)
# ════════════════════════════════════════════════════════════════════════════════

def _context_tracker(tmp_path, session_factory, compact_every=9999):
    return ScoringProgressTracker(
        data_path=str(tmp_path),
        source="inat_context",
        session_factory=session_factory,
        dest_table=InatCaptureContext,
        pk="photo_uuid",
        compact_every=compact_every,
    )


_CONTEXT_ROW = {
    "photo_uuid": "uuid-001",
    "mean_r": 100.0,
    "mean_g": 150.0,
    "mean_b": 200.0,
    "stddev": 0.35,
    "is_underwater": 2,
}


class TestScoringProgressTrackerWAL:

    def test_fresh_load_returns_empty_set(self, tmp_path, sqlite_session_factory):
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        assert tracker.load() == set()
        tracker._wal_file.close()

    def test_record_adds_to_in_memory_completed(self, tmp_path, sqlite_session_factory):
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        tracker.load()
        tracker.record(_CONTEXT_ROW)
        assert tracker.is_completed("uuid-001")
        assert tracker.completed_count == 1
        tracker._wal_file.close()

    def test_wal_written_with_header_on_first_record(self, tmp_path, sqlite_session_factory):
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        tracker.load()
        tracker.record(_CONTEXT_ROW)
        tracker._wal_file.flush()

        wal_path = tmp_path / "inat_context_transfer_progress_wal.csv"
        with open(wal_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["photo_uuid"] == "uuid-001"
        assert "mean_r" in reader.fieldnames
        tracker._wal_file.close()

    def test_multiple_records_all_appear_in_wal(self, tmp_path, sqlite_session_factory):
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        tracker.load()
        for i in range(3):
            tracker.record({**_CONTEXT_ROW, "photo_uuid": f"uuid-{i:03d}"})
        tracker._wal_file.flush()

        wal_path = tmp_path / "inat_context_transfer_progress_wal.csv"
        rows = list(csv.DictReader(open(wal_path)))
        assert len(rows) == 3
        tracker._wal_file.close()

    def test_wal_replay_recovers_uncompacted_records(self, tmp_path, sqlite_session_factory):
        """Simulated crash: a new tracker instance replays the existing WAL."""
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        tracker.load()
        for i in range(3):
            tracker.record({**_CONTEXT_ROW, "photo_uuid": f"uuid-{i:03d}"})
        tracker._wal_file.close()  # don't compact — simulate crash

        tracker2 = _context_tracker(tmp_path, sqlite_session_factory)
        tracker2.load()
        assert tracker2.completed_count == 3
        assert tracker2.is_completed("uuid-000")
        assert tracker2.is_completed("uuid-002")
        tracker2._wal_file.close()

    def test_existing_wal_header_not_duplicated_on_reopen(self, tmp_path, sqlite_session_factory):
        """Reopening an existing WAL should not prepend a second header line."""
        tracker = _context_tracker(tmp_path, sqlite_session_factory)
        tracker.load()
        tracker.record(_CONTEXT_ROW)
        tracker._wal_file.close()

        tracker2 = _context_tracker(tmp_path, sqlite_session_factory)
        tracker2.load()
        tracker2._wal_file.close()

        wal_path = tmp_path / "inat_context_transfer_progress_wal.csv"
        non_empty_lines = [l for l in wal_path.read_text().splitlines() if l.strip()]
        assert len(non_empty_lines) == 2  # header + 1 data row, no duplicate header

    def test_auto_compact_triggered_at_threshold(self, tmp_path, sqlite_session_factory):
        """compact() should fire after compact_every records."""
        tracker = _context_tracker(tmp_path, sqlite_session_factory, compact_every=2)
        tracker.load()

        with patch.object(tracker, "compact") as mock_compact:
            tracker.record({**_CONTEXT_ROW, "photo_uuid": "uuid-001"})
            assert mock_compact.call_count == 0
            tracker.record({**_CONTEXT_ROW, "photo_uuid": "uuid-002"})
            assert mock_compact.call_count == 1

        tracker._wal_file.close()


# ════════════════════════════════════════════════════════════════════════════════
# ScoringProgressTracker — DB compaction (Postgres, requires docker)
# ════════════════════════════════════════════════════════════════════════════════

class TestScoringProgressTrackerDB:

    def test_compact_inat_context_upserts_to_db(self, tmp_path, session_factory, seeded_inat_uuid):
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="inat_context",
            session_factory=session_factory,
            dest_table=InatCaptureContext,
            pk="photo_uuid",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "mean_r": 50.0, "mean_g": 80.0, "mean_b": 180.0,
            "stddev": 0.35, "is_underwater": 2,
        })
        tracker.compact()

        with session_factory() as session:
            row = session.execute(
                select(InatCaptureContext).where(
                    InatCaptureContext.photo_uuid == seeded_inat_uuid
                )
            ).scalar_one()
        assert row.is_underwater == 2
        assert abs(row.mean_b - 180.0) < 0.01

    def test_compact_inat_scoring_upserts_to_db(self, tmp_path, session_factory, seeded_inat_uuid):
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="inat_scoring",
            session_factory=session_factory,
            dest_table=InatImageQuality,
            pk="photo_uuid",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "uicm": 1.5, "uism": 2.3, "uiconm": 0.8, "uiqm": 5.1,
        })
        tracker.compact()

        with session_factory() as session:
            row = session.execute(
                select(InatImageQuality).where(
                    InatImageQuality.photo_uuid == seeded_inat_uuid
                )
            ).scalar_one()
        assert abs(row.uiqm - 5.1) < 0.01

    def test_compact_lila_upserts_to_db(self, tmp_path, session_factory, seeded_lila_filename):
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="lila_scoring",
            session_factory=session_factory,
            dest_table=LilaImageQuality,
            pk="file_name",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "file_name": seeded_lila_filename,
            "uicm": 1.1, "uism": 1.2, "uiconm": 0.5, "uiqm": 3.3,
        })
        tracker.compact()

        with session_factory() as session:
            row = session.execute(
                select(LilaImageQuality).where(
                    LilaImageQuality.file_name == seeded_lila_filename
                )
            ).scalar_one()
        assert abs(row.uiqm - 3.3) < 0.01

    def test_compact_clears_buffer_and_resets_counter(self, tmp_path, session_factory, seeded_inat_uuid):
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="inat_scoring",
            session_factory=session_factory,
            dest_table=InatImageQuality,
            pk="photo_uuid",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "uicm": 1.0, "uism": 1.0, "uiconm": 1.0, "uiqm": 1.0,
        })
        assert len(tracker._wal_buffer) == 1 and tracker._since_last_compact == 1

        tracker.compact()

        assert len(tracker._wal_buffer) == 0
        assert tracker._since_last_compact == 0

    def test_compact_rewrites_wal_with_header_only(self, tmp_path, session_factory, seeded_inat_uuid):
        """After compaction the WAL should be truncated to just the header line."""
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="inat_scoring",
            session_factory=session_factory,
            dest_table=InatImageQuality,
            pk="photo_uuid",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "uicm": 1.0, "uism": 1.0, "uiconm": 1.0, "uiqm": 1.0,
        })
        tracker.compact()
        tracker._wal_file.flush()

        wal_path = tmp_path / "inat_scoring_transfer_progress_wal.csv"
        content = wal_path.read_text().strip()
        expected_header = ",".join(InatImageQuality.__table__.columns.keys())
        assert content == expected_header

    def test_upsert_on_conflict_updates_not_inserts_duplicate(
        self, tmp_path, session_factory, seeded_inat_uuid
    ):
        """Compacting the same photo_uuid twice should update the row, not create a duplicate."""
        tracker = ScoringProgressTracker(
            data_path=str(tmp_path),
            source="inat_scoring",
            session_factory=session_factory,
            dest_table=InatImageQuality,
            pk="photo_uuid",
            compact_every=9999,
        )
        tracker.load()
        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "uicm": 1.0, "uism": 1.0, "uiconm": 1.0, "uiqm": 1.0,
        })
        tracker.compact()

        tracker.record({
            "photo_uuid": seeded_inat_uuid,
            "uicm": 9.9, "uism": 9.9, "uiconm": 9.9, "uiqm": 9.9,
        })
        tracker.compact()

        with session_factory() as session:
            results = session.execute(
                select(InatImageQuality).where(
                    InatImageQuality.photo_uuid == seeded_inat_uuid
                )
            ).scalars().all()
        assert len(results) == 1
        assert abs(results[0].uiqm - 9.9) < 0.01


# ════════════════════════════════════════════════════════════════════════════════
# ContextRunner — tracking behaviour
# ════════════════════════════════════════════════════════════════════════════════

class TestContextRunnerTracking:

    def _make_runner(self, gcs_config, tracker=None):
        return ContextRunner(
            gcs_config=gcs_config,
            session=MagicMock(),
            progress_tracker=tracker or MagicMock(spec=ScoringProgressTracker),
            dataset="inat",
            concurrency=1,
        )

    @pytest.mark.asyncio
    async def test_context_dict_uses_photo_uuid_not_filename(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        runner = self._make_runner(gcs_config, mock_tracker)
        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        await runner._context_with_tracking(
            {"photo_uuid": "real-uuid-abc", "filename": "12345.jpg"},
            mock_storage,
        )

        recorded = mock_tracker.record.call_args[0][0]
        assert recorded["photo_uuid"] == "real-uuid-abc"
        assert "filename" not in recorded

    @pytest.mark.asyncio
    async def test_context_dict_has_all_expected_keys(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        runner = self._make_runner(gcs_config, mock_tracker)
        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        await runner._context_with_tracking(
            {"photo_uuid": "test-uuid", "filename": "99.jpg"},
            mock_storage,
        )

        recorded = mock_tracker.record.call_args[0][0]
        assert set(recorded.keys()) == {
            "photo_uuid", "mean_r", "mean_g", "mean_b", "stddev", "is_underwater"
        }

    @pytest.mark.asyncio
    async def test_run_skips_already_completed_uuids(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        mock_tracker.load.return_value = {"uuid-already-done"}

        runner = self._make_runner(gcs_config, mock_tracker)
        runner._select_all_uploads = MagicMock(
            return_value={"uuid-already-done", "uuid-new"}
        )
        runner._select_files = MagicMock(return_value=[
            {"photo_uuid": "uuid-new", "filename": "new.jpg"},
        ])

        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        with patch("whatsthatfish.src.preprocessing.score_runner.GCSAsyncStorage") as MockGCS:
            MockGCS.return_value.__aenter__ = AsyncMock(return_value=mock_storage)
            MockGCS.return_value.__aexit__ = AsyncMock(return_value=False)
            await runner.run()

        assert mock_tracker.record.call_count == 1

    @pytest.mark.asyncio
    async def test_run_calls_close_in_finally(self, gcs_config):
        """close() must be called even when GCS raises."""
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        mock_tracker.load.return_value = set()
        runner = self._make_runner(gcs_config, mock_tracker)
        runner._select_all_uploads = MagicMock(return_value={"uuid-1"})
        runner._select_files = MagicMock(return_value=[
            {"photo_uuid": "uuid-1", "filename": "1.jpg"}
        ])

        with patch("whatsthatfish.src.preprocessing.score_runner.GCSAsyncStorage") as MockGCS:
            MockGCS.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("gcs down"))
            MockGCS.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError):
                await runner.run()

        mock_tracker.close.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════════
# ScoreRunner — tracking behaviour
# ════════════════════════════════════════════════════════════════════════════════

class TestScoreRunnerTracking:

    def _make_runner(self, gcs_config, dataset, tracker=None):
        return ScoreRunner(
            gcs_config=gcs_config,
            session=MagicMock(),
            progress_tracker=tracker or MagicMock(spec=ScoringProgressTracker),
            dataset=dataset,
            concurrency=1,
        )

    @pytest.mark.asyncio
    async def test_inat_score_dict_uses_photo_uuid(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        runner = self._make_runner(gcs_config, "inat", mock_tracker)
        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        await runner._scoring_with_tracking(
            {"photo_uuid": "real-uuid-xyz", "filename": "55555.jpg", "photo_id": 55555},
            mock_storage,
        )

        recorded = mock_tracker.record.call_args[0][0]
        assert "photo_uuid" in recorded
        assert recorded["photo_uuid"] == "real-uuid-xyz"
        assert "file_name" not in recorded

    @pytest.mark.asyncio
    async def test_lila_score_dict_uses_file_name(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        runner = self._make_runner(gcs_config, "lila", mock_tracker)
        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        await runner._scoring_with_tracking(
            {"filename": "salmon_cv/frame_001.jpg"},
            mock_storage,
        )

        recorded = mock_tracker.record.call_args[0][0]
        assert "file_name" in recorded
        assert recorded["file_name"] == "salmon_cv/frame_001.jpg"
        assert "photo_uuid" not in recorded

    @pytest.mark.asyncio
    async def test_inat_run_deduplicates_by_photo_uuid(self, gcs_config, blue_bytes):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        mock_tracker.load.return_value = {"uuid-done"}

        runner = self._make_runner(gcs_config, "inat", mock_tracker)
        runner._select_all_uploads = MagicMock(return_value={"111", "222"})
        runner._select_files = MagicMock(return_value=[
            {"photo_uuid": "uuid-done", "filename": "111.jpg", "photo_id": 111},
            {"photo_uuid": "uuid-new", "filename": "222.jpg", "photo_id": 222},
        ])

        mock_storage = AsyncMock()
        mock_storage.download.return_value = blue_bytes

        with patch("whatsthatfish.src.preprocessing.score_runner.GCSAsyncStorage") as MockGCS:
            MockGCS.return_value.__aenter__ = AsyncMock(return_value=mock_storage)
            MockGCS.return_value.__aexit__ = AsyncMock(return_value=False)
            await runner.run()

        assert mock_tracker.record.call_count == 1
        assert mock_tracker.record.call_args[0][0]["photo_uuid"] == "uuid-new"

    @pytest.mark.asyncio
    async def test_run_calls_close_in_finally(self, gcs_config):
        mock_tracker = MagicMock(spec=ScoringProgressTracker)
        mock_tracker.load.return_value = set()
        runner = self._make_runner(gcs_config, "inat", mock_tracker)
        runner._select_all_uploads = MagicMock(return_value={"111"})
        runner._select_files = MagicMock(return_value=[
            {"photo_uuid": "uuid-1", "filename": "111.jpg", "photo_id": 111}
        ])

        with patch("whatsthatfish.src.preprocessing.score_runner.GCSAsyncStorage") as MockGCS:
            MockGCS.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("gcs down"))
            MockGCS.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError):
                await runner.run()

        mock_tracker.close.assert_called_once()


# ════════════════════════════════════════════════════════════════════════════════
# PreProcessingFactory — dest_table resolution and pipeline routing
# ════════════════════════════════════════════════════════════════════════════════

class TestPreProcessingFactoryRouting:

    def test_dest_table_scoring_inat_returns_image_quality(self, factory_mocks):
        factory = PreProcessingFactory(type=Dataset.SCORING)
        assert factory._dest_table("inat", runner="scoring") is InatImageQuality

    def test_dest_table_context_inat_returns_capture_context(self, factory_mocks):
        factory = PreProcessingFactory(type=Dataset.ALL)
        assert factory._dest_table("inat", runner="context") is InatCaptureContext

    def test_dest_table_lila_always_returns_lila_quality(self, factory_mocks):
        factory = PreProcessingFactory(type=Dataset.ALL)
        assert factory._dest_table("lila", runner="scoring") is LilaImageQuality
        assert factory._dest_table("lila", runner="context") is LilaImageQuality

    def test_dest_table_unknown_dataset_raises(self, factory_mocks):
        factory = PreProcessingFactory(type=Dataset.ALL)
        with pytest.raises(ValueError):
            factory._dest_table("unknown", runner="scoring")

    @pytest.mark.asyncio
    @patch("whatsthatfish.src.preprocessing.factory.ContextRunner")
    @patch("whatsthatfish.src.preprocessing.factory.ScoringProgressTracker")
    async def test_context_type_builds_context_runner(
        self, mock_tracker, mock_runner, factory_mocks
    ):
        mock_runner.return_value.run = AsyncMock()
        factory = PreProcessingFactory(type=Dataset.CONTEXT)
        await factory.run()
        assert mock_runner.call_count == 1

    @pytest.mark.asyncio
    @patch("whatsthatfish.src.preprocessing.factory.ScoreRunner")
    @patch("whatsthatfish.src.preprocessing.factory.ScoringProgressTracker")
    async def test_scoring_type_builds_two_score_runners(
        self, mock_tracker, mock_runner, factory_mocks
    ):
        mock_runner.return_value.run = AsyncMock()
        factory = PreProcessingFactory(type=Dataset.SCORING)
        await factory.run()
        assert mock_runner.call_count == 2  # inat + lila

    @pytest.mark.asyncio
    @patch("whatsthatfish.src.preprocessing.factory.AnnotationConverter")
    @patch("whatsthatfish.src.preprocessing.factory.ScoreRunner")
    @patch("whatsthatfish.src.preprocessing.factory.ContextRunner")
    @patch("whatsthatfish.src.preprocessing.factory.ScoringProgressTracker")
    async def test_all_type_runs_every_pipeline(
        self, mock_tracker, mock_ctx, mock_score, mock_ann, factory_mocks
    ):
        mock_ctx.return_value.run = AsyncMock()
        mock_score.return_value.run = AsyncMock()
        mock_ann.return_value.run = MagicMock()
        factory = PreProcessingFactory(type=Dataset.ALL)
        await factory.run()
        assert mock_ctx.call_count == 1
        assert mock_score.call_count == 2
        mock_ann.return_value.run.assert_called_once()
