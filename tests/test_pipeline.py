"""
Unit tests for TransferProgressTracker.

Tests the WAL + Postgres crash recovery pattern with both
integer-style (iNat photo_id) and string-style (LILA file_name) identifiers.
Uses an in-memory SQLite database to keep tests fast and isolated.
"""

import polars as pl
import pytest
from pathlib import Path
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from whatsthatfish.src.etl.photo_transfer import TransferProgressTracker
from whatsthatfish.src.database.base import Base
from whatsthatfish.src.database.models import SuccessfulUploads


@pytest.fixture
def tracker_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary directory for each test."""
    return tmp_path


@pytest.fixture
def session_factory():
    """Create an in-memory SQLite engine with the successful_uploads table."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ─── Fresh start ────────────────────────────────────────────────────────


class TestFreshStart:
    def test_load_returns_empty_set(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(str(tracker_dir), source="inat", session_factory=session_factory)
        completed = tracker.load()
        assert completed == set()
        tracker.close()

    def test_completed_count_is_zero(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(str(tracker_dir), source="inat", session_factory=session_factory)
        tracker.load()
        assert tracker.completed_count == 0
        tracker.close()


# ─── Recording with source="inat" ──────────────────────────────────────


class TestRecordingInat:
    def test_record_adds_to_completed_set(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()

        tracker.record("100")
        tracker.record("200")

        assert tracker.is_completed("100")
        assert tracker.is_completed("200")
        assert not tracker.is_completed("999")
        assert tracker.completed_count == 2
        tracker.close()

    def test_record_appends_to_wal_file(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.record("3")

        wal_path = tracker_dir / "inat_transfer_progress_wal.csv"
        assert wal_path.exists()
        df = pl.read_csv(wal_path, has_header=False)
        assert len(df) == 3
        tracker.close()

    def test_compact_writes_to_db(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.record("3")
        tracker.compact()

        with session_factory() as session:
            rows = session.execute(
                select(SuccessfulUploads.identifier)
                .where(SuccessfulUploads.source == "inat")
            ).scalars().all()
            assert set(rows) == {"1", "2", "3"}
        tracker.close()

    def test_record_does_not_write_db_without_compact(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        with session_factory() as session:
            count = session.query(SuccessfulUploads).count()
            assert count == 0


# ─── Recording with source="lila" (string file_name identifiers) ───────


class TestRecordingLila:
    def test_record_string_identifiers(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory, compact_every=9999
        )
        tracker.load()

        tracker.record("salmon_cv/frame_00123.jpg")
        tracker.record("deep_reef/IMG_4501.jpg")

        assert tracker.is_completed("salmon_cv/frame_00123.jpg")
        assert tracker.is_completed("deep_reef/IMG_4501.jpg")
        assert not tracker.is_completed("nonexistent.jpg")
        assert tracker.completed_count == 2
        tracker.close()

    def test_compact_writes_to_db_with_lila_source(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("salmon_cv/frame_00123.jpg")
        tracker.record("deep_reef/IMG_4501.jpg")
        tracker.compact()

        with session_factory() as session:
            rows = session.execute(
                select(SuccessfulUploads.identifier)
                .where(SuccessfulUploads.source == "lila")
            ).scalars().all()
            assert set(rows) == {
                "salmon_cv/frame_00123.jpg",
                "deep_reef/IMG_4501.jpg",
            }
        tracker.close()

    def test_sources_are_isolated(self, tracker_dir: Path, session_factory):
        """iNat and LILA trackers sharing the same DB don't see each other's entries."""
        inat_tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        lila_tracker = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory, compact_every=9999
        )

        inat_tracker.load()
        lila_tracker.load()

        inat_tracker.record("12345")
        lila_tracker.record("salmon_cv/frame_001.jpg")
        inat_tracker.compact()
        lila_tracker.compact()

        # Reload fresh trackers — each should only see its own source
        inat_tracker2 = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory
        )
        lila_tracker2 = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory
        )

        inat_completed = inat_tracker2.load()
        lila_completed = lila_tracker2.load()

        assert "12345" in inat_completed
        assert "salmon_cv/frame_001.jpg" not in inat_completed
        assert "salmon_cv/frame_001.jpg" in lila_completed
        assert "12345" not in lila_completed


# ─── Compaction details ─────────────────────────────────────────────────


class TestCompactionDetails:
    def test_auto_compact_and_truncates_wal(self, tracker_dir: Path, session_factory):
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=2
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        wal_path = tracker_dir / "inat_transfer_progress_wal.csv"
        assert wal_path.exists()
        assert wal_path.read_text() == ""

        with session_factory() as session:
            count = session.query(SuccessfulUploads).count()
            assert count == 2


# ─── Crash recovery ────────────────────────────────────────────────────


class TestCrashRecovery:
    def test_wal_replay_after_crash(self, tracker_dir: Path, session_factory):
        """Records in WAL but never compacted should survive a restart."""
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        # Simulate crash — new tracker, same directory and DB
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        assert tracker.completed_count == 2
        assert tracker.is_completed("1")

    def test_db_plus_wal_combined_recovery(self, tracker_dir: Path, session_factory):
        """Some records in DB, some only in WAL — all should load."""
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.compact()

        tracker.record("3")
        tracker.record("4")

        # Simulate crash
        tracker = TransferProgressTracker(
            str(tracker_dir), source="inat", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        assert tracker.completed_count == 4

    def test_crash_recovery_with_lila_file_names(self, tracker_dir: Path, session_factory):
        """Crash recovery works with string file_name identifiers (LILA)."""
        tracker = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        tracker.record("salmon_cv/frame_001.jpg")
        tracker.record("deep_reef/IMG_100.jpg")
        tracker.compact()
        tracker.record("kelp_forest/shot_42.jpg")

        # Simulate crash
        tracker = TransferProgressTracker(
            str(tracker_dir), source="lila", session_factory=session_factory, compact_every=9999
        )
        tracker.load()
        assert tracker.completed_count == 3
        assert tracker.is_completed("salmon_cv/frame_001.jpg")
        assert tracker.is_completed("kelp_forest/shot_42.jpg")
