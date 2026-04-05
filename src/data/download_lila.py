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

import json
from pathlib import Path

import polars as pl
from google.api_core.exceptions import ClientError, ServerError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from ..config import _get_logger


def _log_retry(retry_state):
    logger = _get_logger("LilaDataset")
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed: "
        f"{retry_state.outcome.exception()}. "
        f"Waiting {retry_state.next_action.sleep:.1f}s before retry."
    )


class LilaDataset:

    def __init__(self, gcs):
        self.gcs_client = gcs.get_gcs_client()
        self.logger = _get_logger("LilaDataset")
        self.ann_out_dir = Path(__file__).parents[2] / "data" / "metadata" / "lila"
        self.img_out_dir = Path(__file__).parents[2] / "data" / "raw" / "lila" / "images"
        self.gcs_bucket = "public-datasets-lila"
        self.gcs_prefix = "community-fish-detection-dataset"
        self.bucket = self.gcs_client.bucket(self.gcs_bucket)

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

    def sample_balanced_dataset(
        self,
        images_df: pl.DataFrame,
        total_images: int = 100_000,
        max_source_proportion: float = 0.15,
        seed: int = 42,
    ) -> pl.DataFrame:
        """Sample a domain-balanced subset of images.

        This method reduces overrepresented source datasets and balances
        the positive:negative ratio within the total_images budget.

        Phase 1: Per-source capping with 50/50 pos/neg split.
        Phase 2: Per-source rebalancing to enforce 1:1 pos/neg globally,
                 adding/removing from the most overrepresented source.

        Args:
            images_df: Cleaned images DataFrame from _load_and_clean().
            total_images: Target number of images to include in the sample.
            max_source_proportion: Max fraction any single source can occupy.
            seed: Random seed for reproducibility.

        Returns:
            Filtered images_df containing only the selected sample.
        """
        # Compute per-source caps
        sources = images_df["dataset"].unique().to_list()
        self.logger.info(f"Sources: {sources}")
        max_images = int(total_images * max_source_proportion)

        # ── Phase 1: Per-source capping ──────────────────────────────────
        sampled_parts: list[pl.DataFrame] = []

        for source in sources:
            # Set a max source count for the source
            source_df = images_df.filter(pl.col("dataset") == source)
            max_source_count = min(source_df.height, max_images)

            # Filter data source to get positive and negative samples
            pos_df = source_df.filter(pl.col("has_fish"))
            neg_df = source_df.filter(~pl.col("has_fish"))

            if source_df.height > max_source_count:
                # Over cap — split budget 50/50 between pos and neg
                max_pos = int(min(max_source_count / 2, pos_df.height))

                sample_pos_df = pos_df.sample(n=max_pos, shuffle=True, seed=seed)

                if neg_df.height > 0:
                    n_neg = int(min(max_pos, neg_df.height))
                    sample_neg_df = neg_df.sample(n=n_neg, shuffle=True, seed=seed)
                    sample_df = pl.concat([sample_pos_df, sample_neg_df])
                    self.logger.info(
                        f"  {source}: sampled {sample_df.height:,} images "
                        f"({sample_pos_df.height:,} pos, {sample_neg_df.height:,} neg)"
                    )
                else:
                    sample_df = sample_pos_df
                    self.logger.info(
                        f"  {source}: sampled {sample_df.height:,} images "
                        f"({sample_pos_df.height:,} pos, no negatives available)"
                    )

                sampled_parts.append(sample_df)

            else:
                # Small source — keep everything
                sampled_parts.append(source_df)
                self.logger.info(
                    f"  {source}: kept all {source_df.height:,} "
                    f"({pos_df.height:,} pos, {neg_df.height:,} neg)"
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
            # Find the source with the most records
            source_most_records = (
                adj_images_df.group_by("dataset")
                .len()
                .sort("len", descending=True)
                .item(0, "dataset")
            )

            pos_to_remove_add = max_pos_images - pos_count
            neg_to_remove_add = max_pos_images - neg_count

            if pos_to_remove_add % 10 == 0:
                self.logger.debug(f"Pos to remove/add: {pos_to_remove_add:,}")
            if neg_to_remove_add % 10 == 0:
                self.logger.debug(f"Neg to remove/add: {neg_to_remove_add:,}")

            # Remove excess positives (one at a time from biggest source)
            if pos_to_remove_add < 0:
                removal_id = (
                    adj_images_df.filter(
                        (pl.col("dataset") == source_most_records) & pl.col("has_fish")
                    )
                    .sample(n=1)
                    .select(pl.col("id"))
                    .item(0)
                )
                adj_images_df = adj_images_df.filter(pl.col("id") != removal_id)
                ids_in_sample.discard(removal_id)
                pos_count -= 1
                continue

            # Remove excess negatives (one at a time from biggest source)
            if neg_to_remove_add < 0:
                removal_id = (
                    adj_images_df.filter(
                        (pl.col("dataset") == source_most_records) & ~pl.col("has_fish")
                    )
                    .sample(n=1)
                    .select(pl.col("id"))
                    .item(0)
                )
                adj_images_df = adj_images_df.filter(pl.col("id") != removal_id)
                ids_in_sample.discard(removal_id)
                neg_count -= 1
                continue

            # Add missing positives (one at a time from biggest source's pool)
            if pos_to_remove_add > 0:
                available = images_df.filter(
                    (pl.col("dataset") == source_most_records)
                    & pl.col("has_fish")
                    & ~pl.col("id").is_in(list(ids_in_sample))
                )
                if available.height > 0:
                    addition = available.sample(n=1)
                    new_id = addition["id"].item(0)
                    adj_images_df = pl.concat([adj_images_df, addition])
                    ids_in_sample.add(new_id)
                    pos_count += 1
                continue

            # Add missing negatives (one at a time from biggest source's pool)
            if neg_to_remove_add > 0:
                available = images_df.filter(
                    (pl.col("dataset") == source_most_records)
                    & ~pl.col("has_fish")
                    & ~pl.col("id").is_in(list(ids_in_sample))
                )
                if available.height > 0:
                    addition = available.sample(n=1)
                    new_id = addition["id"].item(0)
                    adj_images_df = pl.concat([adj_images_df, addition])
                    ids_in_sample.add(new_id)
                    neg_count += 1
                continue

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


    @retry(retry=(retry_if_exception_type(ClientError) |
                  retry_if_exception_type(ServerError)),
           wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           before_sleep=_log_retry)
    def _download_images(self, image_list: list[str]):
        """Download images from GCS by file_name."""
        self.img_out_dir.mkdir(parents=True, exist_ok=True)

        for image in tqdm(image_list, desc="Downloading LILA images"):
            gcs_images_path = f"{self.gcs_prefix}/{image}"
            output_path = self.img_out_dir / Path(image).name

            if output_path.exists():
                continue

            blob = self.bucket.blob(gcs_images_path)
            blob.download_to_filename(str(output_path))

    def extract_lila_images(self, total_images: int = 100_000):
        """Full pipeline: download JSON → clean → sample → download images."""
        self._download_coco_json()
        images_df, annotations_df = self._load_and_clean()

        sampled = self.sample_balanced_dataset(images_df, total_images=total_images)

        image_list = sampled["file_name"].to_list()
        self.logger.info(f"Downloading {len(image_list):,} images ...")
        self._download_images(image_list)

        return sampled, annotations_df
