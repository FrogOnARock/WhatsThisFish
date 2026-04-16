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
        """Upload all files in dataset_dir to GCS, skipping those already present.

        Args:
            dataset_dir: The directory containing images to upload.
            bucket_name: The GCS bucket to upload to.
            prefix: The GCS prefix (e.g. "training/") used for both
                    upload paths and listing existing blobs.

        Returns:
            (True, message) on success, (False, message) on failure.
        """
        if not os.path.exists(dataset_dir):
            return False, "Dataset directory does not exist"

        image_files = set(os.listdir(dataset_dir))

        storage_client = self.get_gcs_client()
        bucket = storage_client.bucket(bucket_name)

        # List existing blobs under this prefix — returns full paths like "prefix/image.jpg"
        existing_blobs = {
            blob.name
            for blob in storage_client.list_blobs(bucket_name, prefix=prefix)
        }

        # Build the set of files that need uploading by comparing
        # the prefixed blob name against what's already in the bucket
        to_upload = [
            image for image in image_files
            if f"{prefix}/{image}" not in existing_blobs
        ]

        self.logger.info(
            f"Upload: {len(to_upload)} new, "
            f"{len(image_files) - len(to_upload)} already present"
        )

        for image in to_upload:
            blob = bucket.blob(f"{prefix}/{image}")
            blob.upload_from_filename(os.path.join(dataset_dir, image))

        return True, f"Uploaded {len(to_upload)} files"
