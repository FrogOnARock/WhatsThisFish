"""
Async photo transfer pipeline: iNaturalist S3 → GCS.

Streams individual photos from iNaturalist's public S3 bucket through memory
to GCS, with WAL-based crash recovery and configurable concurrency.

Architecture:
    1. TransferProgressTracker — WAL + Postgres for crash-safe resume tracking
    2. PhotoTransferPipeline  — async orchestrator with semaphore-controlled concurrency
"""

import asyncio
import csv
import io
from pathlib import Path

import aiohttp
import polars as pl
from gcloud.aio.storage import Storage as GCSAsyncStorage
from sqlalchemy import select, func
from sqlalchemy.orm import sessionmaker

from ..config import _get_logger, GCSConfig, S3Config
from ..database.models import SuccessfulUploads, InatFilteredObservations
from ..retry import transfer_retry, db_retry

logger = _get_logger("PhotoTransfer")

class TransferProgressTracker:
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
        compact_every: int = 1000,
        wal_path: str | None = None,
    ):
        self._data_dir = Path(data_path)
        self._source = source
        self._session_factory = session_factory
        self._wal_path = self._data_dir / (wal_path or f"{source}_transfer_progress_wal.csv")
        self._compact_every = compact_every

        self._completed: set[str] = set()
        self._wal_buffer: list[str] = []
        self._wal_file: io.TextIOWrapper | None = None
        self._wal_writer: csv.writer | None = None
        self._since_last_compact: int = 0

    def load(self) -> set[str]:
        """Load completed transfers from Postgres + replay any WAL entries."""
        with self._session_factory() as session:
            rows = session.execute(
                select(SuccessfulUploads.identifier)
                .where(SuccessfulUploads.source == self._source)
            ).scalars().all()
            self._completed = set(rows)
            if self._completed:
                logger.info(f"Loaded {len(self._completed):,} completed transfers from DB")

        wal_count = 0
        if self._wal_path.exists():
            with open(self._wal_path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        self._completed.add(row[0])
                        self._wal_buffer.append(row[0])
                        wal_count += 1
            if wal_count:
                logger.info(f"Replayed {wal_count:,} WAL entries")

        self._wal_file = open(self._wal_path, "a", newline="")
        self._wal_writer = csv.writer(self._wal_file)

        logger.info(f"Total completed transfers: {len(self._completed):,}")
        return self._completed

    def record(self, identifier: str) -> None:
        """Record a successful transfer — appends to WAL immediately."""
        self._completed.add(str(identifier))
        self._wal_writer.writerow([identifier])
        self._wal_file.flush()
        self._wal_buffer.append(str(identifier))
        self._since_last_compact += 1

        if self._since_last_compact >= self._compact_every:
            self.compact()

    def compact(self) -> None:
        """Bulk insert WAL entries to Postgres and truncate WAL."""
        if not self._wal_buffer:
            return

        with self._session_factory() as session:
            session.add_all([
                SuccessfulUploads(identifier=ident, source=self._source)
                for ident in self._wal_buffer
            ])
            session.commit()

        self._wal_buffer.clear()

        # Truncate the WAL
        self._wal_file.close()
        self._wal_file = open(self._wal_path, "w", newline="")
        self._wal_writer = csv.writer(self._wal_file)
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


class PhotoTransferPipeline:
    """Async pipeline for transferring iNaturalist photos from S3 to GCS.

    Downloads photos via HTTPS from iNaturalist's public S3 bucket,
    streams through memory, and uploads to GCS. Uses a semaphore to
    control concurrency and a TransferProgressTracker for crash-safe
    resume capability.
    """

    def __init__(
        self,
        gcs_config,
        s3_config,
        data_path: str,
        session_factory: sessionmaker,
        concurrency: int = 50,
        compact_every: int = 1000,
    ):
        self._gcs_config = gcs_config
        self._s3_config = s3_config
        self._data_path = data_path
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._session_factory = session_factory
        self._tracker = TransferProgressTracker(
            data_path, source="inat", session_factory=session_factory,
            compact_every=compact_every,
        )

        # Counters for progress reporting
        self._transferred: int = 0
        self._failed: int = 0
        self._total: int = 0



    def _load_pending_photos(self) -> pl.DataFrame:
        """Load dataset.parquet and filter out already-completed transfers.

        Returns a DataFrame with columns: photo_id, extension
        """

        with self._session_factory() as session:
            rows = session.execute(
                select(InatFilteredObservations.photo_id, InatFilteredObservations.extension)
            ).all()
            df = pl.DataFrame(rows, schema=["photo_id", "extension"])

        completed = self._tracker.load()
        if completed:
            df = df.filter(~pl.col("photo_id").cast(pl.Utf8).is_in(list(completed)))

        self._total = len(df) + len(completed)
        logger.info(
            f"Dataset: {self._total:,} total photos, "
            f"{len(completed):,} already transferred, "
            f"{len(df):,} remaining"
        )
        return df

    @transfer_retry
    async def _transfer_single(
        self,
        photo_id: int,
        extension: str,
        http_session: aiohttp.ClientSession,
        gcs_client: GCSAsyncStorage,
    ) -> bool | None:
        """Transfer a single photo from S3 to GCS.

        Args:
            photo_id: iNaturalist photo ID
            extension: File extension (e.g., "jpeg", "jpg", "png")
            http_session: aiohttp session for S3 downloads
            gcs_client: async GCS client for uploads

        Returns:
            True if transfer succeeded, False otherwise
        """

        try:
            # Async get and post requests must be awaited
            # Set S3 image url
            s3_image_url = self._s3_config.base_url + f"/{photo_id}/medium.{extension}"

            # Download bytes from S3
            response = await http_session.get(s3_image_url)
            response.raise_for_status()

            image_bytes = await response.read()

            # Assign bucket name and prefix from config -> in this case it is for the object detection coming from S3
            bucket_name = self._gcs_config.bucket
            prefix = self._gcs_config.prefixes["gcs_train"]

            # Upload to GCS
            await gcs_client.upload(bucket_name, f"{prefix}/{photo_id}.{extension}", image_bytes)
            return True

        except aiohttp.ClientResponseError as e:
            logger.warning(f"Connection error: {e}")
            return False

        except Exception as e:
            logger.warning(f"Error transferring photo {photo_id}: {e}")
            return False


    async def _transfer_with_tracking(
        self,
        photo_id: int,
        extension: str,
        http_session: aiohttp.ClientSession,
        gcs_client: GCSAsyncStorage,
    ) -> None:
        """Wraps _transfer_single with semaphore, progress tracking, and logging."""
        async with self._semaphore:
            success = await self._transfer_single(
                photo_id, extension, http_session, gcs_client
            )

        if success:
            self._tracker.record(str(photo_id))
            self._transferred += 1
        else:
            self._failed += 1

        # Log progress periodically
        done = self._transferred + self._failed
        if done % 1000 == 0:
            logger.info(
                f"Progress: {done:,} processed "
                f"({self._transferred:,} OK, {self._failed:,} failed) | "
                f"{self._tracker.completed_count:,}/{self._total:,} total complete"
            )

    async def run(self) -> None:
        """Main entry point — loads pending work and runs the transfer loop."""
        pending = self._load_pending_photos()

        if pending.is_empty():
            logger.info("All photos already transferred — nothing to do")
            return

        logger.info(
            f"Starting transfer of {len(pending):,} photos "
            f"with concurrency={self._concurrency}"
        )

        try:
            async with (
                aiohttp.ClientSession() as http_session,
                GCSAsyncStorage() as gcs_client,
            ):
                tasks = [
                    self._transfer_with_tracking(
                        row["photo_id"],
                        row["extension"],
                        http_session,
                        gcs_client,
                    )
                    for row in pending.iter_rows(named=True)
                ]
                await asyncio.gather(*tasks)
        finally:
            self._tracker.close()

        logger.info(
            f"Transfer complete: {self._transferred:,} succeeded, "
            f"{self._failed:,} failed"
        )
