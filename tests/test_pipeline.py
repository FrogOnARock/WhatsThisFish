"""
Unit tests for TransferProgressTracker.

Tests the WAL + parquet crash recovery pattern.
"""

import polars as pl
import pytest
from pathlib import Path

from typing_extensions import assert_type

from whatsthatfish.src.data.photo_transfer import TransferProgressTracker


@pytest.fixture
def tracker_dir(tmp_path: Path) -> Path:
    """Provides a clean temporary directory for each test.

    pytest's tmp_path fixture creates a unique temp directory per test,
    so each test gets isolated state — no cross-contamination.
    """
    return tmp_path


# ─── Completed tests (examples for you) ───────────────────────────────────


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


class TestRecording:
    def test_record_adds_to_completed_set(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()

        tracker.record(100)
        tracker.record(200)

        assert tracker.is_completed(100)
        assert tracker.is_completed(200)
        assert not tracker.is_completed(999)
        assert tracker.completed_count == 2
        tracker.close()


class TestCompaction:
    def test_compact_writes_parquet(self, tracker_dir: Path):
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()

        tracker.record(1)
        tracker.record(2)
        tracker.record(3)
        tracker.compact()

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert set(df["photo_id"].to_list()) == {1, 2, 3}
        tracker.close()


class TestRecordingWAL:
    def test_record_appends_to_wal_file(self, tracker_dir: Path):
        """Verify that record() actually writes to the WAL CSV on disk."""


        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record(1)
        tracker.record(2)
        tracker.record(3)

        wal_path = tracker_dir / "transfer_progress.wal.csv"
        assert wal_path.exists()
        df = pl.read_csv(wal_path, has_header=False)
        assert len(df) == 3
        tracker.close()


    def test_record_does_not_create_parquet_without_compact(self, tracker_dir: Path):
        """Verify that recording alone doesn't write parquet — only WAL."""

        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record(1)
        tracker.record(2)

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert not parquet_path.exists()

class TestCompactionDetails:
    def test_auto_compact_and_truncates_wal(self, tracker_dir: Path):
        """After compaction, the WAL should be empty."""

        # Auto-compact test and test if compaction truncates the WAL
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=2)
        tracker.load()
        tracker.record(1)
        tracker.record(2)

        wal_path = tracker_dir / "transfer_progress.wal.csv"
        assert wal_path.exists()
        assert wal_path.read_text() == ""

        parquet_path = tracker_dir / "transfer_progress.parquet"
        assert parquet_path.exists()
        df = pl.read_parquet(parquet_path)
        assert len(df) == 2


class TestCrashRecovery:
    def test_wal_replay_after_crash(self, tracker_dir: Path):
        """Records in WAL but never compacted should survive a restart."""

        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        tracker.record(1)
        tracker.record(2)

        # Simulate a crash by reloading the tracker and leveraging .load()
        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        assert tracker.completed_count == 2


    def test_parquet_plus_wal_combined_recovery(self, tracker_dir: Path):
        """Some records in parquet, some only in WAL — all should load."""

        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()
        # Record initial set of values
        tracker.record(1)
        tracker.record(2)

        # Compact the WAL, write to parquet
        tracker.compact()

        # Write additional records and then simulate a crash
        tracker.record(3)
        tracker.record(4)

        tracker = TransferProgressTracker(str(tracker_dir), compact_every=9999)
        tracker.load()

        assert tracker.completed_count == 4
