import boto3 as b3
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
import polars
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
            print(f"Size of {key} is {size} bytes")

            last_reported = 0
            progress_bytes = 0
            def progress(chunk):
                nonlocal progress_bytes, last_reported
                progress_bytes += chunk
                pct = int((progress_bytes / size) * 100)
                if pct >= last_reported + 10:
                    last_reported = pct
                    print(f"Downloaded {pct}%")

            self.s3_client.download_file(bucket_name, key, output_raw, Callback=progress)

            self.logger.info(f"Downloaded {key} to {output_raw}")

            if str(output_raw).endswith(".tar.gz"):
                self.logger.info(f"Extracting {key} to {output_ext}")
                with tarfile.open(output_raw, "r:*") as tar:
                    tar.extractall(path=output_ext)
                self.logger.info(f"Extracted {key}, removing tar file.")
                os.remove(output_raw)

            if str(output_raw).endswith(".csv.gz"):
                self.logger.info(f"Converting {key} to parquet at {output_ext}")
                polars.scan_csv(output_raw).sink_parquet(output_ext)
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

    # def filter_images_by_taxon(self, taxon_id: int):






