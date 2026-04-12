"""
Download and parse the LILA Community Fish Detection Dataset.

Dataset: gs://public-datasets-lila/community-fish-detection-dataset
- ~1.9M images/frames as JPEGs
- ~935K bounding box annotations (category_id=1, "fish")
- ~1.2M negative frames (category_id=0, "empty")
- COCO JSON format with is_train field for location-based train/val split
- 17 source sub-datasets with severe domain imbalance

Pipeline:
    1. Download COCO JSON from GCS (one-time)
    2. Clean annotations: split bbox, filter invalid, tag positive/negative
    3. Domain-balanced sampling: cap overrepresented sources, balance pos/neg
    4. Download selected images from GCS
"""

import asyncio
import json
from pathlib import Path

import aiohttp
import polars as pl
from google.api_core.exceptions import ClientError, ServerError, ServiceUnavailable, TooManyRequests, GoogleAPICallError
from gcloud.aio.storage import Storage as GCSAsyncStorage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

from sqlalchemy.orm import sessionmaker

from .photo_transfer import TransferProgressTracker, _retry_logic, _log_retry as _log_retry_transfer
from ..config import _get_logger, GCSConfig

def _log_retry(retry_state):
    logger = _get_logger("LilaDataset")
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed: "
        f"{retry_state.outcome.exception()}. "
        f"Waiting {retry_state.next_action.sleep:.1f}s before retry."
    )

def _retry_logic(exc: BaseException) -> bool:
    """Retry logic for aiohttp requests."""
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in (429, 500, 502, 503, 504)

    if isinstance(exc, aiohttp.ClientConnectorError):
        return True

    if isinstance(exc, (ServiceUnavailable, TooManyRequests, GoogleAPICallError)):
        return True

    return False

class LilaDataset:

    # LILA public GCS bucket served over HTTPS — no auth needed
    LILA_BASE_URL = "https://storage.googleapis.com/public-datasets-lila/community-fish-detection-dataset"

    def __init__(self, gcs, data_path: str, gcs_config: GCSConfig,
                 session_factory: sessionmaker, concurrency: int = 50):
        self.gcs_client = gcs.get_gcs_client()
        self.logger = _get_logger("LilaDataset")
        self.ann_out_dir = Path(__file__).parents[2] / "data" / "metadata" / "lila"
        self.gcs_bucket = "public-datasets-lila"
        self.gcs_prefix = "community-fish-detection-dataset"
        self.data_path = data_path
        self.bucket = self.gcs_client.bucket(self.gcs_bucket)
        self._session_factory = session_factory
        self._tracker = TransferProgressTracker(
            data_path=self.data_path,
            source="lila",
            session_factory=session_factory,
        )
        self._gcs_config = gcs_config
        self._concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

        self._transferred: int = 0
        self._failed: int = 0
        self._total: int = 0

    @retry(retry=(retry_if_exception_type(ClientError) |
                  retry_if_exception_type(ServerError)),
           wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           before_sleep=_log_retry)
    def _download_coco_json(self):
        """Download the COCO JSON annotation file from GCS."""
        json_path = self.ann_out_dir / "community_fish_detection_dataset.json"
        if json_path.exists():
            self.logger.info("Annotations file already exists at %s", json_path)
            return

        self.ann_out_dir.mkdir(parents=True, exist_ok=True)
        gcs_json_path = f"{self.gcs_prefix}/community_fish_detection_dataset.json.zip"

        self.logger.info("Downloading annotations from %s ...", gcs_json_path)
        self.bucket.blob(gcs_json_path).download_to_filename(str(json_path))
        self.logger.info("Downloaded annotations to %s", json_path)

    def _load_and_clean(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load COCO JSON and return cleaned (images_df, annotations_df).

        Cleaning steps:
            - Parse images list into DataFrame
            - Parse annotations, split bbox into x/y/w/h columns
            - Filter annotations with zero-area bounding boxes
            - Tag images as positive (has >=1 fish bbox) or negative (empty)

        Returns:
            images_df: columns [id, file_name, width, height, is_train, dataset, has_fish]
            annotations_df: columns [id, image_id, category_id, x, y, w, h]
                            (only category_id=1 annotations with valid bboxes)
        """
        json_path = self.ann_out_dir / "community_fish_detection_dataset.json"
        self.logger.info("Loading COCO JSON from %s ...", json_path)

        with open(json_path, "r") as f:
            coco = json.load(f)

        # Parse images — extract only the columns we need.
        # Some sources include extra fields (habitat_type, visibility, etc.)
        # which cause schema inference failures if included.
        images_raw = [
            {
                "id": img["id"],
                "file_name": img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "is_train": img["is_train"],
                "dataset": img["dataset"],
            }
            for img in coco["images"]
        ]
        images_df = pl.DataFrame(images_raw)

        # Parse annotations — split bbox, keep only fish annotations with valid boxes.
        # Build as column lists (not list-of-dicts) because annotation IDs have
        # mixed types (int and str) across sources, which breaks Polars inference.
        ann_ids, img_ids, cats = [], [], []
        xs, ys, ws, hs = [], [], [], []
        for ann in coco["annotations"]:
            if ann.get("category_id") != 1 or "bbox" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            ann_ids.append(str(ann["id"]))
            img_ids.append(str(ann["image_id"]))
            cats.append(ann["category_id"])
            xs.append(x)
            ys.append(y)
            ws.append(w)
            hs.append(h)

        annotations_df = pl.DataFrame({
            "ann_id": pl.Series(ann_ids, dtype=pl.Utf8),
            "image_id": pl.Series(img_ids, dtype=pl.Utf8),
            "category_id": pl.Series(cats, dtype=pl.Int64),
            "x": pl.Series(xs, dtype=pl.Float64),
            "y": pl.Series(ys, dtype=pl.Float64),
            "w": pl.Series(ws, dtype=pl.Float64),
            "h": pl.Series(hs, dtype=pl.Float64),
        })

        # Tag images: has_fish = True if image has at least one valid fish bbox
        positive_ids = annotations_df["image_id"].unique()
        images_df = images_df.with_columns(
            pl.col("id").is_in(positive_ids).alias("has_fish")
        )

        pos_count = images_df.filter(pl.col("has_fish")).height
        neg_count = images_df.filter(~pl.col("has_fish")).height

        self.logger.info(
            f"Cleaned dataset: {images_df.height:,} images "
            f"({pos_count:,} positive, {neg_count:,} negative), "
            f"{annotations_df.height:,} bounding boxes"
        )

        return images_df, annotations_df

    @staticmethod
    def _stratified_sample(
        df: pl.DataFrame, n: int, seed: int
    ) -> list[pl.DataFrame]:
        """Sample n rows from df, preserving the natural is_train ratio.

        Splits df into train/val groups and allocates n proportionally.
        Returns a list of DataFrames (one per non-empty split) to be
        appended to sampled_parts.
        """
        if n == 0:
            return []
        if n >= df.height:
            return [df]

        train_df = df.filter(pl.col("is_train"))
        val_df = df.filter(~pl.col("is_train"))

        # Proportional allocation — give rounding remainder to the larger split
        train_ratio = train_df.height / df.height if df.height > 0 else 0.0
        n_train = round(n * train_ratio)
        n_val = n - n_train

        # Clamp to available counts and redistribute overflow
        n_train = min(n_train, train_df.height)
        n_val = min(n_val, val_df.height)
        # If one split couldn't fill its allocation, give remainder to the other
        shortfall = n - (n_train + n_val)
        if shortfall > 0:
            extra_train = min(shortfall, train_df.height - n_train)
            n_train += extra_train
            shortfall -= extra_train
            n_val += min(shortfall, val_df.height - n_val)

        parts: list[pl.DataFrame] = []
        if n_train > 0:
            if n_train < train_df.height:
                parts.append(train_df.sample(n=n_train, shuffle=True, seed=seed))
            else:
                parts.append(train_df)
        if n_val > 0:
            if n_val < val_df.height:
                parts.append(val_df.sample(n=n_val, shuffle=True, seed=seed))
            else:
                parts.append(val_df)
        return parts

    def sample_balanced_dataset(
        self,
        images_df: pl.DataFrame,
        total_images: int = 100_000,
        max_pos_per_source: int = 5_000,
        max_neg_proportion: float = 0.125,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Sample a domain-balanced subset of images.

        This method reduces overrepresented source datasets and balances
        the positive:negative ratio within the total_images budget.

        Phase 1: Even positive distribution across sources, greedy negative inclusion.
        Phase 2: Per-image rebalancing to enforce 1:1 pos/neg globally.

        Args:
            images_df: Cleaned images DataFrame from _load_and_clean().
            total_images: Target number of images to include in the sample.
            max_pos_per_source: Max positive images any single source can contribute.
            max_neg_proportion: Max fraction of total_images any single source
                                can contribute as negatives (default 12.5%).
            seed: Random seed for reproducibility.

        Returns:
            Filtered images_df containing only the selected sample.
        """
        sources = images_df["dataset"].unique().to_list()
        num_sources = len(sources)
        self.logger.info(f"Sources ({num_sources}): {sources}")

        max_neg_per_source = int(total_images * max_neg_proportion)

        self.logger.info(
            f"Phase 1 caps — max pos/source: {max_pos_per_source:,}, "
            f"max neg/source: {max_neg_per_source:,}"
        )

        # ── Phase 1: Per-source capping with train/val stratification ────
        sampled_parts: list[pl.DataFrame] = []

        for source in sources:
            source_df = images_df.filter(pl.col("dataset") == source)
            pos_df = source_df.filter(pl.col("has_fish"))
            neg_df = source_df.filter(~pl.col("has_fish"))

            # Sample positives: up to cap, stratified by is_train
            n_pos = min(pos_df.height, max_pos_per_source)
            sampled_parts.extend(
                self._stratified_sample(pos_df, n_pos, seed)
            )

            # Sample negatives: up to cap, stratified by is_train
            n_neg = min(neg_df.height, max_neg_per_source)
            if n_neg > 0:
                sampled_parts.extend(
                    self._stratified_sample(neg_df, n_neg, seed)
                )

            self.logger.info(
                f"  {source}: {n_pos:,} pos, {n_neg:,} neg "
                f"(from {pos_df.height:,} / {neg_df.height:,} available)"
            )

        adj_images_df = pl.concat(sampled_parts)

        # ── Phase 2: Per-image rebalancing ─────────────────────────────
        # Iterate one image at a time, always targeting the most
        # overrepresented source. Removes excess or adds missing
        # positives/negatives until we hit 1:1 pos/neg ratio.
        pos_count = adj_images_df.filter(pl.col("has_fish")).height
        neg_count = adj_images_df.filter(~pl.col("has_fish")).height
        self.logger.info(
            f"After source capping: {adj_images_df.height:,} images "
            f"({pos_count:,} pos, {neg_count:,} neg)"
        )

        max_pos_images = total_images / 2
        ids_in_sample = set(adj_images_df["id"].to_list())

        while pos_count != neg_count:
            # Rank sources by count descending each iteration
            sorted_sources = (
                adj_images_df.group_by("dataset")
                .len()
                .sort("len", descending=True)
            )["dataset"].to_list()

            pos_to_remove_add = max_pos_images - pos_count
            neg_to_remove_add = max_pos_images - neg_count

            if (pos_count + neg_count) % 100 == 0:
                self.logger.debug(
                    f"Rebalancing: {pos_count:,} pos, {neg_count:,} neg "
                    f"(pos delta: {pos_to_remove_add:+,}, neg delta: {neg_to_remove_add:+,})"
                )

            made_progress = False

            # Remove excess positives — try each source biggest-first
            if pos_to_remove_add < 0:
                for source in sorted_sources:
                    candidates = adj_images_df.filter(
                        (pl.col("dataset") == source) & pl.col("has_fish")
                    )
                    if candidates.height == 0:
                        continue
                    removal_id = candidates.sample(n=1)[0, "id"]
                    adj_images_df = adj_images_df.filter(pl.col("id") != removal_id)
                    ids_in_sample.discard(removal_id)
                    pos_count -= 1
                    made_progress = True
                    break

            # Remove excess negatives — try each source biggest-first
            elif neg_to_remove_add < 0:
                for source in sorted_sources:
                    candidates = adj_images_df.filter(
                        (pl.col("dataset") == source) & ~pl.col("has_fish")
                    )
                    if candidates.height == 0:
                        continue
                    removal_id = candidates.sample(n=1)[0, "id"]
                    adj_images_df = adj_images_df.filter(pl.col("id") != removal_id)
                    ids_in_sample.discard(removal_id)
                    neg_count -= 1
                    made_progress = True
                    break

            # Add missing positives — try each source biggest-first
            elif pos_to_remove_add > 0:
                for source in sorted_sources:
                    available = images_df.filter(
                        (pl.col("dataset") == source)
                        & pl.col("has_fish")
                        & ~pl.col("id").is_in(list(ids_in_sample))
                    )
                    if available.height == 0:
                        continue
                    addition = available.sample(n=1)
                    new_id = addition[0, "id"]
                    adj_images_df = pl.concat([adj_images_df, addition])
                    ids_in_sample.add(new_id)
                    pos_count += 1
                    made_progress = True
                    break

            # Add missing negatives — try each source biggest-first
            elif neg_to_remove_add > 0:
                for source in sorted_sources:
                    available = images_df.filter(
                        (pl.col("dataset") == source)
                        & ~pl.col("has_fish")
                        & ~pl.col("id").is_in(list(ids_in_sample))
                    )
                    if available.height == 0:
                        continue
                    addition = available.sample(n=1)
                    new_id = addition[0, "id"]
                    adj_images_df = pl.concat([adj_images_df, addition])
                    ids_in_sample.add(new_id)
                    neg_count += 1
                    made_progress = True
                    break

            if not made_progress:
                self.logger.warning(
                    f"Cannot reach 1:1 balance — no sources can provide "
                    f"the needed class. Stopping at {pos_count:,} pos, {neg_count:,} neg"
                )
                break

        self.logger.info(
            f"After rebalancing: {adj_images_df.height:,} images "
            f"({pos_count:,} pos, {neg_count:,} neg)"
        )

        # ── Summary ──────────────────────────────────────────────────────
        final_pos = adj_images_df.filter(pl.col("has_fish")).height
        final_neg = adj_images_df.filter(~pl.col("has_fish")).height
        self.logger.info(
            f"Final sample: {adj_images_df.height:,} images "
            f"({final_pos:,} pos, {final_neg:,} neg)"
        )

        source_summary = (
            adj_images_df.group_by("dataset")
            .agg(
                pl.len().alias("count"),
                pl.col("has_fish").sum().alias("positive"),
            )
            .sort("count", descending=True)
        )
        self.logger.info(f"Source distribution:\n{source_summary}")

        return adj_images_df

    @retry(retry=retry_if_exception(_retry_logic),
           wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           before_sleep=_log_retry_transfer)
    async def _transfer_single(
        self,
        file_name: str,
        http_session: aiohttp.ClientSession,
        gcs_client: GCSAsyncStorage,
    ) -> bool:
        """Download a single image from LILA and upload to our GCS bucket.

        Streams image bytes through memory — no local disk write.

        Args:
            file_name: LILA image path (e.g. "salmon_cv/frame_00123.jpg")
            http_session: aiohttp session for LILA HTTPS downloads
            gcs_client: async GCS client for uploads to our bucket

        Returns:
            True if transfer succeeded, False otherwise
        """
        try:
            source_url = f"{self.LILA_BASE_URL}/{file_name}"

            response = await http_session.get(source_url)
            response.raise_for_status()
            image_bytes = await response.read()

            bucket_name = self._gcs_config.bucket
            prefix = self._gcs_config.prefixes["object_detection"]
            await gcs_client.upload(bucket_name, f"{prefix}/{file_name}", image_bytes)
            return True

        except aiohttp.ClientResponseError as e:
            self.logger.warning(f"HTTP error for {file_name}: {e.status} {e.message}")
            return False

        except Exception as e:
            self.logger.warning(f"Error transferring {file_name}: {e}")
            return False

    async def _transfer_with_tracking(
        self,
        file_name: str,
        http_session: aiohttp.ClientSession,
        gcs_client: GCSAsyncStorage,
    ) -> None:
        """Wraps _transfer_single with semaphore, progress tracking, and logging."""
        async with self._semaphore:
            success = await self._transfer_single(file_name, http_session, gcs_client)

        if success:
            self._tracker.record(file_name)
            self._transferred += 1
        else:
            self._failed += 1

        done = self._transferred + self._failed
        if done % 1000 == 0:
            self.logger.info(
                f"Progress: {done:,} processed "
                f"({self._transferred:,} OK, {self._failed:,} failed) | "
                f"{self._tracker.completed_count:,}/{self._total:,} total complete"
            )

    async def _run_transfers(self, image_list: list[str]) -> None:
        """Filter already-completed images and run async transfer loop."""
        completed = self._tracker.load()
        pending = [img for img in image_list if img not in completed]

        self._total = len(image_list)

        if not pending:
            self.logger.info("All images already transferred — nothing to do")
            return

        self.logger.info(
            f"Starting transfer of {len(pending):,} images "
            f"({len(completed):,} already done) with concurrency={self._concurrency}"
        )

        try:
            async with (
                aiohttp.ClientSession() as http_session,
                GCSAsyncStorage() as gcs_client,
            ):
                tasks = [
                    self._transfer_with_tracking(file_name, http_session, gcs_client)
                    for file_name in pending
                ]
                await asyncio.gather(*tasks)
        finally:
            self._tracker.close()

        self.logger.info(
            f"Transfer complete: {self._transferred:,} succeeded, "
            f"{self._failed:,} failed"
        )

    def extract_lila_images(self, total_images: int = 100_000):
        """Full pipeline: download JSON → clean → sample → transfer images to GCS."""
        self._download_coco_json()
        images_df, annotations_df = self._load_and_clean()

        sampled = self.sample_balanced_dataset(images_df, total_images=total_images)

        image_list = sampled["file_name"].to_list()
        self.logger.info(f"Transferring {len(image_list):,} images ...")
        asyncio.run(self._run_transfers(image_list))

        return sampled, annotations_df
