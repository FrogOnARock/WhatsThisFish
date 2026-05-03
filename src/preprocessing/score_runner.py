import asyncio
import csv
import io
from pathlib import Path
from typing import Type

import aiohttp
import polars as pl
from gcloud.aio.storage import Storage as GCSAsyncStorage
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from ..database.config import get_session_factory

from ..config import _get_logger, GCSConfig
from ..database.models import InatImageQuality, InatCaptureContext, InatFilteredObservations
from whatsthatfish.src.database.base import Base
from ..retry import transfer_retry, db_retry
from .capture_context_scorer import ContextScorer
from .uiqm_quality_scorer import QualityScorer

logger = _get_logger(__name__)

class ScoringProgressTracker:
    """Tracks completed transfers using a WAL + Postgres pattern.

    The WAL (write-ahead log) is a CSV file that receives an append after
    every successful upload. Periodically the WAL is compacted into the
    successful_uploads table in Postgres and the WAL is truncated.

    On startup:
        completed = successful_uploads(source) ∪ WAL replay

    This gives us both crash safety (WAL) and durable storage (Postgres).

    Generic over identifier type — works with photo_id (iNat) or
    file_name (LILA) by configuring source at construction time.
    All identifiers are stored as strings internally.
    """

    def __init__(
        self,
        data_path: str,
        source: str,
        session_factory: sessionmaker,
        dest_table: Type[Base],
        pk: str = "photo_uuid",
        compact_every: int = 1000,
        wal_path: str | None = None,
    ):
        self._data_dir = Path(data_path)
        self._source = source
        self._dest_table = dest_table
        self._session_factory = session_factory
        self._pk = pk
        self._wal_path = self._data_dir / (wal_path or f"{source}_transfer_progress_wal.csv")
        self._compact_every = compact_every

        self._completed: set[str] = set()
        self._wal_buffer: list[dict[str, str]] = []
        self._wal_file: io.TextIOWrapper | None = None
        self._wal_writer: csv.DictWriter | None = None
        self._since_last_compact: int = 0

    def load(self) -> set[str]:
        """Load completed transfers from Postgres + replay any WAL entries."""
        with self._session_factory() as session:
            rows = session.execute(
                select(self._dest_table.__table__.c[self._pk])
            ).scalars().all()
            self._completed = set(rows)
            if self._completed:
                logger.info(f"Loaded {len(self._completed):,} completed transfers from DB")

        wal_count = 0
        if self._wal_path.exists():
            with open(self._wal_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        self._completed.add(row.get("photo_uuid"))
                        self._wal_buffer.append(dict(row))
                        wal_count += 1
            if wal_count:
                logger.info(f"Replayed {wal_count:,} WAL entries")

        self._wal_file = open(self._wal_path, "a", newline="")
        self._wal_writer = csv.DictWriter(self._wal_file, self._dest_table.__table__.columns.keys())

        logger.info(f"Total completed transfers: {len(self._completed):,}")
        return self._completed

    def record(self, row: dict[str, str]) -> None:
        """Record a successful transfer — appends to WAL immediately."""

        photo_uuid = row.get("photo_uuid")
        self._completed.add(photo_uuid)
        self._wal_writer.writerow(row)
        self._wal_file.flush()
        self._wal_buffer.append(row)
        self._since_last_compact += 1

        if self._since_last_compact >= self._compact_every:
            self.compact()

    def compact(self) -> None:
        """Bulk insert WAL entries to Postgres and truncate WAL."""

        if not self._wal_buffer:
            return

        with self._session_factory() as session:
            session.add_all([
                self._dest_table(**row)
                for row in self._wal_buffer
            ])
            session.commit()

        self._wal_buffer.clear()

        # Truncate the WAL
        self._wal_file.close()
        self._wal_file = open(self._wal_path, "w", newline="")
        self._wal_writer = csv.DictWriter(self._wal_file, self._dest_table.__table__.columns.keys())
        self._since_last_compact = 0

        logger.info(f"Compacted {len(self._completed):,} entries to DB")

    def close(self) -> None:
        """Final compaction and cleanup."""
        self.compact()
        if self._wal_file:
            self._wal_file.close()

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def is_completed(self, identifier: str) -> bool:
        return str(identifier) in self._completed



class ScoreRunner:
    def __init__(self, gcs_config: GCSConfig,
                 session: sessionmaker,
                 progress_tracker: ScoringProgressTracker,
                 dataset: str,
                 concurrency: int = 50):

        self._dataset = dataset
        self._gcs_config = gcs_config
        self._session_factory = session
        self._progress_tracker = progress_tracker
        self._semaphore = asyncio.Semaphore(concurrency)
        self.context_scorer = ContextScorer()
        self.quality_scorer = QualityScorer()



    async def _load_image_quality(self, data):

        with self._session_factory() as session:
            stmt = insert(InatImageQuality).values(data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[InatImageQuality.photo_uuid],
                set_= {
                    "uicm": stmt.excluded.uicm,
                    "uism": stmt.excluded.uism,
                    "uiconm": stmt.excluded.uiconm,
                    "uiqm": stmt.excluded.uiqm
                }
            )
            session.execute(stmt)
            session.commit()



    async def _scoring_with_tracking(self, row: dict[str, str],  gcs_storage: GCSAsyncStorage):
        async with self._semaphore:

            bucket_name = self._gcs_config.bucket
            file_name = row.get("filename")
            photo_uuid = row.get("photo_uuid")
            train_path = self._gcs_config.prefixes.get("gcs_train")
            obj_path = self._gcs_config.prefixes.get("gcs_object_detection")


            if self._dataset == "inat":
                try:
                    blob = await gcs_storage.download(bucket_name, train_path + file_name)
                    mean_r, mean_g, mean_b, stddev, classification = self.context_scorer.score_capture_context(blob)
                    uicm, uism, uiconm, uiqm = self.quality_scorer.compute_uiqm(blob)

                    score_dict = {
                        "photo_uuid": photo_uuid,
                        "uicm": uicm,
                        "uism": uism,
                        "uiconm": uiconm,
                        "uiqm": uiqm
                    }

                    context_dict = {
                        "photo_uuid": photo_uuid,
                        "mean_r": mean_r,
                        "mean_g": mean_g,
                        "mean_b": mean_b,
                        "stddev": stddev,
                        "is_underwater": classification
                    }

                    # self.load
                    #
                    # self._progress_tracker.record()

            #
            #
            # elif self._dataset == "lila":
            #     blob = gcs_storage.download()


            return NotImplementedError


    def _select_files(self, files):

        if self._dataset == "inat":

            with self._session_factory() as session:
                rows = session.execute(
                    select(InatFilteredObservations.photo_uuid, func.concat(
                        InatFilteredObservations.photo_id, ".", InatFilteredObservations.extension
                    ).label("filename")
                   ).where(InatFilteredObservations.photo_id.in_(files))
                ).all()

                return [{"photo_uuid": r.photo_uuid, "filename": r.filename} for r in rows]

        elif self._dataset == "lila":
            pass

        return NotImplementedError


    async def runner(self):

        files = []

        """
        Need to ensure that files 
        """

        ids = set(self._progress_tracker.load())
        files = set(files) - ids

        rows = self._select_files(files)

        try:
            async with (
                GCSAsyncStorage() as gcs_storage
            ):
                tasks = [
                    self._scoring_with_tracking(row, gcs_storage)
                    for row in rows
                ]

                await asyncio.gather(*tasks)

        finally:
            self._progress_tracker.close()










