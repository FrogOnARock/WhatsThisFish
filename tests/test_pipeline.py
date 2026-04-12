"""
Unit tests for TransferProgressTracker.

Tests the WAL + parquet crash recovery pattern with both
integer-style (iNat photo_id) and string-style (LILA file_name) identifiers.
"""

import polars as pl
import pytest
from pathlib import Path

from whatsthatfish.src.data.photo_transfer import TransferProgressTracker


@pytest.fixture
def tracker_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary directory for each test.

    pytest's tmp_path fixture creates a unique temp directory per test,
    so each test gets isolated state — no cross-contamination.
    """
    return tmp_path


# ─── Fresh start ────────────────────────────────────────────────────────


class TestFreshStart:
    def test_load_returns_empty_set(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir))
        completed = tracker.load()
        assert completed == set()
        tracker.close()

    def test_completed_count_is_zero(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir))
        tracker.load()
        assert tracker.completed_count == 0
        tracker.close()


# ─── Recording with default id_column (photo_id) ───────────────────────


class TestRecordingPhotoId:
    def test_record_adds_to_completed_set(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()

        tracker.record("100")
        tracker.record("200")

        assert tracker.is_completed("100")
        assert tracker.is_completed("200")
        assert not tracker.is_completed("999")
        assert tracker.completed_count == 2
        tracker.close()

    def test_record_appends_to_wal_file(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.record("3")

        wal_path = tracker_dir / "transfer_progress.wal.csv"
        assert wal_path.exists()
        df = pl.read_csv(wal_path, has_header=False)
        assert len(df) == 3
        tracker.close()

    def test_compact_writes_parquet_with_photo_id_column(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.record("3")
        tracker.compact()

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert "photo_id" in df.columns
        assert set(df["photo_id"].to_list()) == {"1", "2", "3"}
        tracker.close()

    def test_record_does_not_create_parquet_without_compact(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert not parquet_path.exists()


# ─── Recording with custom id_column (file_name for LILA) ──────────────


class TestRecordingFileName:
    def test_record_string_identifiers(self, tracker_dir: Path):
        tracker = TransferProgressTracker(
            str(tracker_dir), compact_every=9999, id_column="file_name"
        )
        tracker.load()

        tracker.record("salmon_cv/frame_00123.jpg")
        tracker.record("deep_reef/IMG_4501.jpg")

        assert tracker.is_completed("salmon_cv/frame_00123.jpg")
        assert tracker.is_completed("deep_reef/IMG_4501.jpg")
        assert not tracker.is_completed("nonexistent.jpg")
        assert tracker.completed_count == 2
        tracker.close()

    def test_compact_writes_parquet_with_file_name_column(self, tracker_dir: Path):
        tracker = TransferProgressTracker(
            str(tracker_dir), compact_every=9999, id_column="file_name"
        )
        tracker.load()
        tracker.record("salmon_cv/frame_00123.jpg")
        tracker.record("deep_reef/IMG_4501.jpg")
        tracker.compact()

        parquet_path = tracker_dir / "transfer_progress.parquet"
        df = pl.read_parquet(parquet_path)
        assert "file_name" in df.columns
        assert set(df["file_name"].to_list()) == {
            "salmon_cv/frame_00123.jpg",
            "deep_reef/IMG_4501.jpg",
        }
        tracker.close()

    def test_custom_wal_and_parquet_paths(self, tracker_dir: Path):
        tracker = TransferProgressTracker(
            str(tracker_dir),
            compact_every=9999,
            id_column="file_name",
            parquet_path="lila_progress.parquet",
            wal_path="lila_progress.wal.csv",
        )
        tracker.load()
        tracker.record("some_source/image.jpg")
        tracker.compact()

        assert (tracker_dir / "lila_progress.parquet").exists()
        assert (tracker_dir / "lila_progress.wal.csv").exists()
        # Default paths should NOT exist
        assert not (tracker_dir / "transfer_progress.parquet").exists()
        tracker.close()


# ─── Compaction details ─────────────────────────────────────────────────


class TestCompactionDetails:
    def test_auto_compact_and_truncates_wal(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=2)
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        wal_path = tracker_dir / "transfer_progress.wal.csv"
        assert wal_path.exists()
        assert wal_path.read_text() == ""

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert len(df) == 2


# ─── Crash recovery ────────────────────────────────────────────────────


class TestCrashRecovery:
    def test_wal_replay_after_crash(self, tracker_dir: Path):
        """Records in WAL but never compacted should survive a restart."""
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record("1")
        tracker.record("2")

        # Simulate crash — new tracker, same directory
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        assert tracker.completed_count == 2
        assert tracker.is_completed("1")

    def test_parquet_plus_wal_combined_recovery(self, tracker_dir: Path):
        """Some records in parquet, some only in WAL — all should load."""
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record("1")
        tracker.record("2")
        tracker.compact()

        tracker.record("3")
        tracker.record("4")

        # Simulate crash
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        assert tracker.completed_count == 4

    def test_crash_recovery_with_file_name_ids(self, tracker_dir: Path):
        """Crash recovery works with string file_name identifiers (LILA)."""
        tracker = TransferProgressTracker(
            str(tracker_dir), compact_every=9999, id_column="file_name"
        )
        tracker.load()
        tracker.record("salmon_cv/frame_001.jpg")
        tracker.record("deep_reef/IMG_100.jpg")
        tracker.compact()
        tracker.record("kelp_forest/shot_42.jpg")

        # Simulate crash
        tracker = TransferProgressTracker(
            str(tracker_dir), compact_every=9999, id_column="file_name"
        )
        tracker.load()
        assert tracker.completed_count == 3
        assert tracker.is_completed("salmon_cv/frame_001.jpg")
        assert tracker.is_completed("kelp_forest/shot_42.jpg")
