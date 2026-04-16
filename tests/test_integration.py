"""
Integration tests for etl pipelines against a real Postgres instance.

Prerequisites:
    docker compose -f docker-compose.test.yml up -d
    cd whatsthatfish && .venv/bin/python -m pytest tests/test_integration.py -v

These tests use small fixture parquets (~10-20 rows) and verify actual
SQL behavior: FK constraints, ON CONFLICT upserts, type coercion,
batch boundaries, and crash recovery against real Postgres.
"""

from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
import os

from whatsthatfish.src.etl.download_lila import LilaDataset
from whatsthatfish.src.etl.gcs_client import GCSClient
from whatsthatfish.src.database import LilaAnnotations
from whatsthatfish.src.etl.inaturalist_dataset import INaturalistDataset
from whatsthatfish.src.etl.photo_transfer import TransferProgressTracker, PhotoTransferPipeline
from whatsthatfish.src.database.models import InatTaxa, InatFilteredObservations, SuccessfulUploads, LilaCollectedImages

from whatsthatfish.tests.test_pipeline import tracker_dir


# ─── Helper ───────────────────────────────────────────────────────────


def _build_inat_dataset(session_factory, fixtures_dir: Path) -> INaturalistDataset:
    """Construct an INaturalistDataset pointed at fixture etl."""

    class FakeS3Config:
        bucket = "unused"
        datasets = {}
        output_paths = {}

    ds = INaturalistDataset(
        config=FakeS3Config(),
        session_factory=session_factory,
        data_path=str(fixtures_dir),
    )
    return ds

def _build_lila_dataset(session_factory, fixtures_dir: Path) -> LilaDataset:
    """Construct an INaturalistDataset pointed at fixture etl."""

    class FakeGCSConfig:
        bucket = "unused"
        prefixes = {}

    lila_ds = LilaDataset(
        gcs=GCSClient(FakeGCSConfig()),
        data_path=str(fixtures_dir),
        gcs_config=FakeGCSConfig(),
        session_factory=session_factory,
    )

    return lila_ds

def _build_pt_pipeline(session_factory, tracker_dir: Path, compact_every: int) -> PhotoTransferPipeline:

    class FakeS3Config:
        bucket = "unused"
        datasets = {}
        output_paths = {}

    class FakeGCSConfig:
        bucket = "unused"
        prefixes = {}

    pt = PhotoTransferPipeline(
      gcs_config=FakeGCSConfig(),
      s3_config = FakeS3Config(),
      data_path=str(tracker_dir),
      session_factory=session_factory,
      compact_every=compact_every
    )

    return pt

def _run_full_filter_and_load(ds):
    """Run the filter pipeline and load both taxa and observations."""
    dataset_path, filtered_taxa = ds._build_filtered_dataset(
        taxa_scope=[47178, 196614],
    )
    ds._load_taxa(filtered_taxa)
    count = ds._load_filtered_observations(dataset_path)
    return count, dataset_path





# ─── 1. Taxa upsert ──────────────────────────────────────────────────


class TestTaxaUpsert:
    """Verify taxa load into Postgres and upserts update existing rows."""

    def test_initial_taxa_insert(self, session_factory, fixtures_dir):
        """Load fixture taxa into an empty DB — all active fish taxa should land."""
        ds = _build_inat_dataset(session_factory, fixtures_dir)

        # _build_filtered_dataset produces the filtered taxa DataFrame
        # but we don't need the full pipeline — just run the filter + insert
        taxa_df = pl.read_parquet(fixtures_dir / "taxa.parquet")

        # Apply the same filter as _build_filtered_dataset stage 1
        taxa_scope = [47178, 196614]
        id_match = pl.col("taxon_id").is_in(taxa_scope)
        ancestry_match = pl.lit(False)
        for tid in taxa_scope:
            pattern = rf"(^|/){tid}($|/)"
            ancestry_match = ancestry_match | pl.col("ancestry").str.contains(pattern)

        filtered_taxa = taxa_df.filter((id_match | ancestry_match) & pl.col("active"))

        # Insert taxa
        count = ds._load_taxa(filtered_taxa)

        # Verify: 9 active fish taxa (Actinopterygii lineage + Chondrichthyes lineage)
        # Excluded: 3099 (inactive), 47158/5001 (insects), 3/6001 (birds)
        assert count == 9

        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(InatTaxa)
            ).scalar()
            assert db_count == 9

            # Spot-check a specific taxon
            shark = session.execute(
                select(InatTaxa).where(InatTaxa.taxon_id == 4001)
            ).scalar_one()
            assert shark.name == "Carcharhinus melanopterus"
            assert shark.rank == "species"
            assert shark.active is True

    def test_upsert_updates_existing_taxa(self, session_factory, fixtures_dir):
        """Insert taxa, then re-insert with a changed name — verify the update."""
        ds = _build_inat_dataset(session_factory, fixtures_dir)

        taxa_df = pl.read_parquet(fixtures_dir / "taxa.parquet")
        taxa_scope = [47178, 196614]
        id_match = pl.col("taxon_id").is_in(taxa_scope)
        ancestry_match = pl.lit(False)
        for tid in taxa_scope:
            pattern = rf"(^|/){tid}($|/)"
            ancestry_match = ancestry_match | pl.col("ancestry").str.contains(pattern)

        filtered_taxa = taxa_df.filter((id_match | ancestry_match) & pl.col("active"))

        # First insert
        ds._load_taxa(filtered_taxa)

        # Modify a name and re-upsert
        updated_taxa = filtered_taxa.with_columns(
            pl.when(pl.col("taxon_id") == 3001)
            .then(pl.lit("Thalassoma lunare (updated)"))
            .otherwise(pl.col("name"))
            .alias("name")
        )
        ds._load_taxa(updated_taxa)

        # Verify: still 9 rows (no duplicates), name updated
        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(InatTaxa)
            ).scalar()
            assert db_count == 9

            updated = session.execute(
                select(InatTaxa).where(InatTaxa.taxon_id == 3001)
            ).scalar_one()
            assert updated.name == "Thalassoma lunare (updated)"


# ─── 2. Filtered observations insert with FK ─────────────────────────

class TestFilteredObservationsInsert:
    """Verify observations insert respects FK to taxa and counts are correct."""

    def test_observations_insert_with_fk(self, session_factory, fixtures_dir):
        """Full pipeline: filter fixtures → insert taxa → insert observations."""
        ds = _build_inat_dataset(session_factory, fixtures_dir)
        count, _ = _run_full_filter_and_load(ds)

        # 7 photos survive: 6 research-grade fish observations, one obs has 2 photos
        assert count == 7

        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(InatFilteredObservations)
            ).scalar()
            assert db_count == 7

            # Verify FK relationship works — join back to taxa
            row = session.execute(
                select(InatFilteredObservations)
                .where(InatFilteredObservations.photo_id == 10006)
            ).scalar_one()
            assert row.taxon_id == 4001  # shark observation

            # Verify the taxon relationship loads
            assert row.taxon.name == "Carcharhinus melanopterus"

    def test_casual_and_non_fish_filtered_out(self, session_factory, fixtures_dir):
        """Casual-quality and non-fish photos should not appear in the DB."""
        ds = _build_inat_dataset(session_factory, fixtures_dir)
        _, _ = _run_full_filter_and_load(ds)

        with session_factory() as session:
            # photo_id 10008 is from a casual-quality observation
            casual = session.execute(
                select(InatFilteredObservations)
                .where(InatFilteredObservations.photo_id == 10008)
            ).scalar_one_or_none()
            assert casual is None

            # photo_id 10009 is from an insect observation
            insect = session.execute(
                select(InatFilteredObservations)
                .where(InatFilteredObservations.photo_id == 10009)
            ).scalar_one_or_none()
            assert insect is None


# ─── 3. Anti-join idempotency ─────────────────────────────────────────


class TestIdempotency:
    def test_second_run_inserts_zero(self, session_factory, fixtures_dir):
        """Running the pipeline twice should be idempotent."""

        ds = _build_inat_dataset(session_factory, fixtures_dir)
        first_count, _ = _run_full_filter_and_load(ds)
        assert first_count == 7

        second_count, _ = _run_full_filter_and_load(ds)
        assert second_count == 0

        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(InatFilteredObservations)
            ).scalar()
            assert db_count == 7



# ─── 4. Batch boundary correctness ───────────────────────────────────

class TestBatchBoundary:
    def test_small_batch_no_rows_lost(self, session_factory, fixtures_dir):
        """Tiny max_params forces many batches — verify nothing dropped."""

        inat_ds = _build_inat_dataset(session_factory, fixtures_dir)
        lila_ds = _build_lila_dataset(session_factory, fixtures_dir)

        annotations_df = pl.read_parquet(fixtures_dir / "annotations.parquet")
        collected_images_df = pl.read_parquet(fixtures_dir / "collected_images.parquet")


        lila_ds._load_collected_images(collected_images_df, max_params=30)
        with session_factory() as session:
            db_count_img = session.execute(
                select(func.count()).select_from(LilaCollectedImages)
            ).scalar()
            assert db_count_img == 9


        lila_ds._load_annotations(annotations_df, max_params=30)
        with session_factory() as session:
            db_count_ann = session.execute(
                select(func.count()).select_from(LilaAnnotations)
            ).scalar()
            assert db_count_ann == 9

        count, _ = _run_full_filter_and_load(inat_ds)
        assert count == 7


# ─── 5. WAL crash recovery against real Postgres ─────────────────────

class TestWALCrashRecovery:
    def test_wal_replay_against_postgres(self, session_factory, fixtures_dir, tracker_dir: Path):
        """Crash recovery with real Postgres backend."""

        if os.path.exists(fixtures_dir / "inat_transfer_progress_wal.csv"):
            os.remove(fixtures_dir / "inat_transfer_progress_wal.csv")

        photos = pl.read_parquet(fixtures_dir / "photos.parquet").select('photo_id')
        pt_pipeline = _build_pt_pipeline(session_factory, tracker_dir, compact_every=1000)
        pt_pipeline._tracker.load()


        pt_pipeline._tracker.record(photos[0].item())
        pt_pipeline._tracker.record(photos[1].item())

        pt_pipeline._tracker.compact()
        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(SuccessfulUploads)
            ).scalar()
            assert db_count == 2



        pt_pipeline._tracker.record(photos[2].item())
        assert pt_pipeline._tracker.completed_count == 3

        pt_pipeline_1 = _build_pt_pipeline(session_factory, tracker_dir, compact_every=1000)
        pt_pipeline_1._tracker.load()


        pt_pipeline_1._tracker.record(photos[3].item())
        assert pt_pipeline_1._tracker.completed_count == 4

        pt_pipeline_1._tracker.compact()
        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(SuccessfulUploads)
            ).scalar()
            assert db_count == 4

        pt_pipeline_1._tracker.close()

# ─── 6. LILA annotations + images FK ─────────────────────────────────



class TestLilaFKConstraint:
    def test_annotations_fail_without_images(self, session_factory, fixtures_dir):
        """Inserting annotations before images should raise IntegrityError (FK violation)."""
        lila_ds = _build_lila_dataset(session_factory, fixtures_dir)
        annotations_df = pl.read_parquet(fixtures_dir / "annotations.parquet")

        # Annotations reference image_id → collected_images.id
        # With no images in the DB, Postgres should reject the insert
        with pytest.raises(IntegrityError):
            lila_ds._load_annotations(annotations_df)

        # Verify nothing leaked through — table should be empty
        with session_factory() as session:
            db_count = session.execute(
                select(func.count()).select_from(LilaAnnotations)
            ).scalar()
            assert db_count == 0

    def test_images_then_annotations_succeeds(self, session_factory, fixtures_dir):
        """Happy path: load images first, then annotations — all rows land."""
        lila_ds = _build_lila_dataset(session_factory, fixtures_dir)
        images_df = pl.read_parquet(fixtures_dir / "collected_images.parquet")
        annotations_df = pl.read_parquet(fixtures_dir / "annotations.parquet")

        # Images first (parent table)
        lila_ds._load_collected_images(images_df)

        with session_factory() as session:
            img_count = session.execute(
                select(func.count()).select_from(LilaCollectedImages)
            ).scalar()
            assert img_count == 9

        # Annotations second (child table with FK)
        lila_ds._load_annotations(annotations_df)

        with session_factory() as session:
            ann_count = session.execute(
                select(func.count()).select_from(LilaAnnotations)
            ).scalar()
            assert ann_count == 9

            # Spot-check: frame_00100 should have 2 annotations
            frame_100_anns = session.execute(
                select(LilaAnnotations)
                .where(LilaAnnotations.image_id == "salmon_cv/frame_00100.jpg")
            ).scalars().all()
            assert len(frame_100_anns) == 2

            # Verify relationship traversal: annotation → collected_image
            ann = frame_100_anns[0]
            assert ann.collected_images.dataset == "salmon_cv"
            assert ann.collected_images.is_train is True

    def test_negative_frames_have_no_annotations(self, session_factory, fixtures_dir):
        """Negative frames (images with no fish) should exist without annotations."""
        lila_ds = _build_lila_dataset(session_factory, fixtures_dir)
        images_df = pl.read_parquet(fixtures_dir / "collected_images.parquet")
        annotations_df = pl.read_parquet(fixtures_dir / "annotations.parquet")

        lila_ds._load_collected_images(images_df)
        lila_ds._load_annotations(annotations_df)

        with session_factory() as session:
            # These two images are negative frames — no annotations reference them
            neg_frame_1 = session.execute(
                select(LilaCollectedImages)
                .where(LilaCollectedImages.id == "salmon_cv/frame_00400.jpg")
            ).scalar_one()
            assert len(neg_frame_1.annotations) == 0

            neg_frame_2 = session.execute(
                select(LilaCollectedImages)
                .where(LilaCollectedImages.id == "brackish/seq01_000300.jpg")
            ).scalar_one()
            assert len(neg_frame_2.annotations) == 0

            # Confirm these images still exist in the DB (not filtered out)
            assert neg_frame_1.dataset == "salmon_cv"
            assert neg_frame_2.dataset == "brackish"
