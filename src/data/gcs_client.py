import os

from dotenv import load_dotenv
from google.cloud import storage as gcs
from google.oauth2 import service_account

from ..config import _get_logger
from ..retry import gcs_retry

load_dotenv()


class GCSClient:
    def __init__(self,
                 config):
        self.key_path = os.environ.get("GCS_SECRET")
        self.config = config
        self.logger = _get_logger("GCSClient")


    @gcs_retry
    def get_gcs_client(self) -> gcs.Client:
        """
        Create an authenticated GCS client.

        Priority:
            1. Explicit key path passed as argument
            2. GCS_SERVICE_ACCOUNT_KEY environment variable
            3. GOOGLE_APPLICATION_CREDENTIALS environment variable
            4. Default credentials (gcloud auth application-default login)
        """
        # Check explicit path or env var
        key_file = self.key_path

        if key_file:
            if not os.path.exists(key_file):
                raise FileNotFoundError(f"Service account key not found: {key_file}")

            credentials = service_account.Credentials.from_service_account_file(
                key_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            client = gcs.Client(credentials=credentials, project=credentials.project_id)
            print(f"  Authenticated via service account: {credentials.service_account_email}")
            return client

        # Fall back to default credentials
        client = gcs.Client()
        print("Authenticated via default credentials")
        return client


    def gcs_upload(self,
                   dataset_dir: str,
                   bucket_name: str,
                   prefix: str):

        """
        For a given bucket name and prefix, upload all files in the dataset_dir to the bucket.
        :param dataset_dir: The directory containing the extracted image data.
        :param bucket_name: The bucket on GCS we're uploading to.
        :param prefix: The prefix to use to retrieve already uploaded data to ensure we don't overwrite.
        :return: True if all files were uploaded successfully, False otherwise.
        """

        if not os.path.exists(dataset_dir):
            return False, "Dataset directory does not exist"


        # Get all image files in the dataset_dir
        image_files = set(os.listdir(dataset_dir))


        # Instantiate a GCS client
        storage_client = self.get_gcs_client()
        bucket=storage_client.bucket(bucket_name)

        # Get existing blobs in the bucket
        blob_list = storage_client.list_blobs(bucket_name, delimiter='/', prefix=prefix)

        existing_blobs = []
        # Iterate and get blob data
        for blob in blob_list:
            existing_blobs.append(blob.name)

        # If no blobs exist, upload all files, otherwise only upload new files
        if len(existing_blobs) == 0:
            for image in image_files:
                blob = bucket.blob(f"{prefix}/{image}")
                blob.upload_from_filename(os.path.join(dataset_dir, image))
        else:
            for image in image_files:
                if image in existing_blobs:
                    continue
                else:
                    print(f"Uploading {image} to bucket {bucket_name}")
                    blob = bucket.blob(f"{prefix}/{image}")
                    blob.upload_from_filename(os.path.join(dataset_dir, image))

        return True, "All files uploaded successfully"
