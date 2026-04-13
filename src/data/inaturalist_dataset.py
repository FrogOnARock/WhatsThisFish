import os
import shutil
import tarfile

import boto3 as b3
import polars as pl
from botocore import UNSIGNED
from botocore.client import Config
from dotenv import load_dotenv
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

from ..config import _get_logger
from ..database.models import InatFilteredObservations, InatTaxa
from ..retry import s3_retry, db_retry

load_dotenv()


class INaturalistDataset:

    def __init__(self, config,
                 session_factory: sessionmaker,
                 data_path: str,):
        self.s3_config = config
        self.logger = _get_logger("INaturalistDataset")
        self.s3_client = b3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket_name = self.s3_config.bucket
        self._session_factory = session_factory
        self.data_path = data_path

    @s3_retry
    def retrieve_s3_data_by_blob(
            self,
            key: str = None,
            output_raw: str = None,
            output_ext: str = None,
            **kwargs) -> str:

        """
        Retrieves the data from AWS S3 bucket
        :param bucket_name: refers to the bucket name we're retrieving data from (example: ml-inat-competition-datasets)
        :param key: refers to the key or the dataset we're leveraging (example: 2021/val.json.tar.gz)
        :param kwargs: any additional arguments for retrieving data from S3 bucket (ex., {"year", "type", etc.}
        :return: returns the data from the S3 bucket
        """

        if os.path.exists(output_ext):
            if os.path.isdir(output_ext):
                shutil.rmtree(output_ext)
            else:
                os.remove(output_ext)

        bucket_name = self.bucket_name
        obj = self.s3_client.head_object(Bucket=bucket_name, Key=key)
        size = obj["ContentLength"]
        self.logger.debug(f"Size of {key} is {size} bytes")

        last_reported = 0
        progress_bytes = 0
        def progress(chunk):
            nonlocal progress_bytes, last_reported
            progress_bytes += chunk
            pct = int((progress_bytes / size) * 100)
            if pct >= last_reported + 10:
                last_reported = pct
                self.logger.debug(f"Downloaded {pct}%")

        self.s3_client.download_file(bucket_name, key, output_raw, Callback=progress)
        self.logger.info(f"Downloaded {key} to {output_raw}")

        if str(output_raw).endswith(".tar.gz"):
            self.logger.info(f"Extracting {key} to {output_ext}")
            with tarfile.open(output_raw, "r:*") as tar:
                tar.extractall(path=output_ext)
            self.logger.info(f"Extracted {key}, removing tar file.")
            os.remove(output_raw)

        if str(output_raw).endswith(".csv.gz"):
            if key == "taxa.csv.gz":
                schema_overrides = {"rank_level": pl.Float64}
            else:
                schema_overrides = None
            self.logger.info(f"Converting {key} to parquet at {output_ext}")
            pl.scan_csv(output_raw, separator='\t', ignore_errors=True, schema_overrides=schema_overrides).sink_parquet(output_ext)
            self.logger.info(f"Converted {key}, removing csv file.")
            os.remove(output_raw)

    def retrieve_s3_data_by_bucket(self, data_path: str) -> None:
        self.logger.info("Retrieving data from S3 bucket")
        for key, value in self.s3_config.datasets.items():
            output_path = self.s3_config.output_paths[key]
            self.logger.info(f"Retrieving data from S3 bucket for: {key}")
            path = str(data_path)
            self.retrieve_s3_data_by_blob(key=value, output_raw=path + "/" + value, output_ext=path + "/" + output_path)
            self.logger.info(f"Data retrieved from S3 bucket for: {key}")


    def _build_filtered_dataset(
        self,
        taxa_scope: list[int],
        output_name: str = "dataset.parquet",
    ) -> tuple[str, pl.DataFrame]:
        """Filter taxa → observations → photos and sink to a single output parquet.

        Three-stage streaming pipeline to avoid OOM on large parquets:
          1. Eagerly filter taxa (small, ~1.6M rows) → extract taxon_id series
          2. Lazily filter observations by taxon_id + quality → sink intermediate
          3. Stream-join photos against intermediate → sink final output

        Returns:
            Tuple of (path to output parquet, filtered taxa DataFrame)
        """
        taxa_path = os.path.join(self.data_path, "taxa.parquet")
        obs_path = os.path.join(self.data_path, "observations.parquet")
        photos_path = os.path.join(self.data_path, "photos.parquet")
        obs_intermediate_path = os.path.join(self.data_path, "_obs_intermediate.parquet")
        output_path = os.path.join(self.data_path, output_name)

        # --- 1. Filter taxa (small table, safe to collect) ---
        self.logger.info(f"Filtering taxa for taxon_ids={taxa_scope}")
        taxa = pl.read_parquet(taxa_path)

        id_match = pl.col("taxon_id").is_in(taxa_scope)
        ancestry_match = pl.lit(False)
        for tid in taxa_scope:
            pattern = rf"(^|/){tid}($|/)"
            ancestry_match = ancestry_match | pl.col("ancestry").str.contains(pattern)

        filtered_taxa = taxa.filter((id_match | ancestry_match) & pl.col("active"))
        taxon_id_series = filtered_taxa.get_column("taxon_id").implode()
        self.logger.info(f"Found {filtered_taxa.shape[0]:,} active taxa")
        del taxa

        # --- 2. Filter observations → sink to intermediate parquet ---
        self.logger.info("Sinking filtered observations to intermediate parquet")
        (
            pl.scan_parquet(obs_path)
            .filter(
                pl.col("taxon_id").is_in(taxon_id_series)
                & (pl.col("quality_grade") == "research")
            )
            .select(
                "observation_uuid", "observer_id", "latitude",
                "longitude", "observed_on", "taxon_id",
            )
            .sink_parquet(obs_intermediate_path)
        )
        del taxon_id_series

        obs_count = pl.scan_parquet(obs_intermediate_path).select(pl.len()).collect().item()
        self.logger.info(f"Intermediate observations: {obs_count:,} rows")

        # --- 3. Stream-join photos against intermediate observations → sink ---
        self.logger.info("Joining photos to observations (streaming)")
        obs_lazy = pl.scan_parquet(obs_intermediate_path)
        photos_lazy = pl.scan_parquet(photos_path)

        (
            photos_lazy
            .join(obs_lazy, on="observation_uuid", how="inner")
            .select(
                "photo_uuid", "photo_id", "observation_uuid",
                "extension", "license", "width", "height", "position",
                "observer_id", "latitude", "longitude", "observed_on", "taxon_id",
            )
            .sink_parquet(output_path)
        )

        os.remove(obs_intermediate_path)

        count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        self.logger.info(f"Dataset complete: {count:,} photo records in {output_path}")
        return output_path, filtered_taxa

    @db_retry
    def _load_taxa(
        self,
        filtered_taxa: pl.DataFrame,
        max_params: int = 65535,
    ) -> int:
        """Upsert filtered taxa into inat_taxa.

        Uses ON CONFLICT DO UPDATE so that name/rank/ancestry changes
        in the source parquet are reflected in Postgres.

        Args:
            filtered_taxa: DataFrame with columns matching InatTaxa
            max_params: Postgres bind parameter limit

        Returns:
            Total rows upserted
        """
        # Select only the columns that map to InatTaxa
        taxa_cols = ["taxon_id", "ancestry", "rank_level", "rank", "name", "active"]
        df = filtered_taxa.select(taxa_cols)

        self.logger.info(f"Upserting {len(df):,} taxa rows")

        upserted = 0
        batch_size = max_params // len(taxa_cols)
        for batch_start in range(0, len(df), batch_size):
            batch = df.slice(batch_start, batch_size).to_dicts()
            stmt = insert(InatTaxa).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["taxon_id"],
                set_={
                    "ancestry": stmt.excluded.ancestry,
                    "rank_level": stmt.excluded.rank_level,
                    "rank": stmt.excluded.rank,
                    "name": stmt.excluded.name,
                    "active": stmt.excluded.active,
                },
            )
            with self._session_factory() as session:
                session.execute(stmt)
                session.commit()
            upserted += len(batch)

        self.logger.info(f"Taxa upsert complete: {upserted:,} rows")
        return upserted

    @db_retry
    def _get_existing_photo_uuids(self) -> set[str]:
        """Pull existing photo_uuids from Postgres for anti-join."""
        with self._session_factory() as session:
            rows = session.execute(
                select(InatFilteredObservations.photo_uuid)
            ).scalars().all()
        self.logger.info(f"Found {len(rows):,} existing photo records in DB")
        return set(rows)

    def _load_filtered_observations(
        self,
        dataset_path: str,
        max_params: int = 65535,
    ) -> int:
        """Anti-join against existing DB rows and bulk insert new records.

        Args:
            dataset_path: path to the filtered dataset parquet
            batch_size: rows per INSERT + COMMIT cycle

        Returns:
            Total rows inserted
        """
        existing_uuids = self._get_existing_photo_uuids()

        # ~1.5M rows — the filtered dataset fits comfortably in memory
        df = pl.read_parquet(dataset_path)
        total_scanned = len(df)
        self.logger.info(f"Read {total_scanned:,} rows from {dataset_path}")

        # Anti-join: only keep rows whose photo_uuid is NOT already in the DB
        if existing_uuids:
            df = df.filter(~pl.col("photo_uuid").is_in(existing_uuids))

        new_rows = len(df)
        if new_rows == 0:
            self.logger.info("No new rows to insert — database is up to date")
            return 0

        self.logger.info(f"Inserting {new_rows:,} new rows ({total_scanned - new_rows:,} already in DB)")

        inserted = 0
        batch_size = (max_params / len(df.columns)) - 1000
        batch_size = int(batch_size)
        for batch_increment in range(0, len(df), batch_size):

            if inserted > 0:
                self.logger.info(f"Inserted {inserted} rows. Rows left to insert {len(df) - inserted}.")

            batch = df.slice(batch_increment, batch_size).to_dicts()
            with self._session_factory() as session:
                session.execute(insert(InatFilteredObservations), batch)
                session.commit()
            inserted += len(batch)
            self.logger.info(f"Inserted {inserted:,} rows ({len(batch):,} in this batch)")

        return inserted

    def run(self, taxa: list[int] | None = None):
        """Full pipeline: S3 download → parquet → filter + join → bulk insert."""
        if taxa is None:
            taxa = [47178, 196614]

        self.logger.info("Starting iNaturalist dataset pipeline")
        self.retrieve_s3_data_by_bucket(self.data_path)

        dataset_path, filtered_taxa = self._build_filtered_dataset(taxa_scope=taxa)
        self._load_taxa(filtered_taxa)
        count = self._load_filtered_observations(dataset_path)
        self.logger.info(f"Pipeline complete: {count:,} new rows inserted")

