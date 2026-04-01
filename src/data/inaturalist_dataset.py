import boto3 as b3
import polars as pl
from aiohttp import ConnectionTimeoutError
from botocore import UNSIGNED
from botocore.client import Config
import os
import shutil
from dotenv import load_dotenv
from ..config import _get_logger
import tarfile
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from botocore.exceptions import ClientError, ConnectionError, ConnectionClosedError
load_dotenv()


def log_retry(retry_state):
    logger = _get_logger("INaturalistDataset")
    logger.warning(
        f"Attempt {retry_state.attempt_number} failed: "
        f"{retry_state.outcome.exception()}. "
        f"Waiting {retry_state.next_action.sleep:.1f}s before retry."
    )


class INaturalistDataset:

    def __init__(self, config):
        self.s3_config = config
        self.logger = _get_logger("INaturalistDataset")
        self.s3_client = b3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket_name = self.s3_config.bucket


    @retry(retry=(retry_if_exception_type(ClientError) |
                  retry_if_exception_type(ConnectionError) |
                  retry_if_exception_type(ConnectionTimeoutError) |
                  retry_if_exception_type(ConnectionClosedError) |
                  retry_if_exception_type(tarfile.ReadError)),
           wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           before_sleep=log_retry)
    def retrieve_s3_data_by_blob(
            self,
            key: str = None,
            output_raw: str = None,
            output_ext: str = None,
            **kwargs):

        """
        Retrieves the data from AWS S3 bucket
        :param bucket_name: refers to the bucket name we're retrieving data from (example: ml-inat-competition-datasets)
        :param key: refers to the key or the dataset we're leveraging (example: 2021/val.json.tar.gz)
        :param kwargs: any additional arguments for retrieving data from S3 bucket (ex., {"year", "type", etc.}
        :return: returns the data from the S3 bucket
        """

        try:

            if os.path.exists(output_ext):
                if os.path.isdir(output_ext):
                    shutil.rmtree(output_ext)
                else:
                    os.remove(output_ext)

            bucket_name = self.bucket_name
            object = self.s3_client.head_object(Bucket=bucket_name, Key=key)
            size = object["ContentLength"]
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


        except (ClientError, ConnectionError, ConnectionTimeoutError, ConnectionClosedError) as e:
            raise e

        except tarfile.ReadError as e:
            raise e

        except Exception as e:
            raise e

    async def retrieve_s3_data_by_bucket(self, data_path: str):

        try:
            self.logger.info("Retrieving data from S3 bucket")
            for key, value in self.s3_config.datasets.items():

                # Find output path for each dataset
                output_path = self.s3_config.output_paths[key]
                self.logger.info(f"Retrieving data from S3 bucket for: {key}")
                path = str(data_path)
                self.retrieve_s3_data_by_blob(key=value, output_raw=path + "/" + value, output_ext=path + "/" + output_path)
                self.logger.info(f"Data retrieved from S3 bucket for: {key}")

        except ValueError as e:
            raise e

    def build_filtered_dataset(
        self,
        data_path: str,
        taxon_ids: int | list[int],
        output_name: str = "dataset.parquet",
    ) -> str:
        """
        Builds a single filtered parquet containing photo metadata joined with
        observation data for research-grade observations of the specified taxa.

        Pipeline stages are sunk to disk between steps to avoid holding large
        intermediate results in memory:
          1. Filter taxa (small, collected in memory)
          2. Filter observations → sink to intermediate parquet
          3. Stream-join photos against intermediate observations → sink to output

        Output columns: photo_uuid, photo_id, observation_uuid, extension,
            license, width, height, position, observer_id, latitude, longitude,
            observed_on, taxon_id

        :param data_path: path to the data directory containing taxa/observations/photos parquets
        :param taxon_ids: a single taxon_id or list of taxon_ids to filter by
        :param output_name: filename for the output parquet
        :return: path to the output parquet
        """
        if isinstance(taxon_ids, int):
            taxon_ids = [taxon_ids]

        taxa_path = os.path.join(data_path, "taxa.parquet")
        obs_path = os.path.join(data_path, "observations.parquet")
        photos_path = os.path.join(data_path, "photos.parquet")
        obs_intermediate_path = os.path.join(data_path, "_obs_intermediate.parquet")
        output_path = os.path.join(data_path, output_name)

        # --- 1. Filter taxa (small table ~1.6M rows, safe to collect) ---
        self.logger.info(f"Filtering taxa for taxon_ids={taxon_ids}")
        taxa = pl.read_parquet(taxa_path)

        id_match = pl.col("taxon_id").is_in(taxon_ids)
        ancestry_match = pl.lit(False)
        for tid in taxon_ids:
            pattern = rf"(^|/){tid}($|/)"
            ancestry_match = ancestry_match | pl.col("ancestry").str.contains(pattern)

        filtered_taxa = taxa.filter((id_match | ancestry_match) & pl.col("active"))
        taxon_id_series = filtered_taxa.get_column("taxon_id").implode()
        self.logger.info(f"Found {filtered_taxa.shape[0]} active taxa")
        del taxa, filtered_taxa

        # --- 2. Filter observations → sink to intermediate parquet ---
        self.logger.info("Sinking filtered observations to intermediate parquet")
        (
            pl.scan_parquet(obs_path)
            .filter(
                pl.col("taxon_id").is_in(taxon_id_series)
                & (pl.col("quality_grade") == "research")
            )
            .select(
                "observation_uuid",
                "observer_id",
                "latitude",
                "longitude",
                "observed_on",
                "taxon_id",
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
                "photo_uuid",
                "photo_id",
                "observation_uuid",
                "extension",
                "license",
                "width",
                "height",
                "position",
                "observer_id",
                "latitude",
                "longitude",
                "observed_on",
                "taxon_id",
            )
            .sink_parquet(output_path)
        )

        # Clean up intermediate file
        os.remove(obs_intermediate_path)

        count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
        self.logger.info(f"Dataset complete: {count:,} photo records in {output_path}")
        return output_path



