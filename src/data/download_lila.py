"""
Download and parse the LILA Community Fish Detection Dataset.

Dataset: gs://public-datasets-lila/community-fish-detection-dataset
- ~1.9M images/frames as JPEGs
- ~935K bounding box annotations
- COCO JSON format with single class: fish (category_id: 1)
- is_train field for location-based train/val split
- 17 source sub-datasets

Supports partial download by source dataset name for development.
Includes COCO JSON → YOLO format conversion.
"""

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
from google.cloud import storage as gcs
from google.oauth2 import service_account
import polars as pl
import gcsfs
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ClientError, ServerError
from tqdm import tqdm
from ..config import _get_logger

def log_retry(retry_state):
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
           retry_error_callback=log_retry)
    def _download_coco_json(self):
        """Download the COCO JSON annotation file from GCS."""
        json_path = self.ann_out_dir / "annotations.json.zip"
        if json_path.exists():
            self.logger.info("Annotations file already exists at %s", self.ann_out_dir)

        gcs_json_path = f"{self.gcs_prefix}/community_fish_detection_dataset.json.zip"
        self.logger.info("Downloading annotations from %s ...", gcs_json_path)
        self.bucket.blob(gcs_json_path).download_to_filename(self.ann_out_dir)
        self.logger.info("Downloaded annotations to %s", self.ann_out_dir)



    @retry(retry=(retry_if_exception_type(ClientError) |
                  retry_if_exception_type(ServerError)),
           wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5),
           retry_error_callback=log_retry)
    def _download_images(self, image_list: list[str]):

        try:
            for image in tqdm(image_list):
                """Download the images from GCS."""
                gcs_images_path = f"{self.gcs_prefix}/{image}"
                self.logger.info("Downloading images from %s ...", gcs_images_path)

                bucket = self.gcs_client.bucket(self.gcs_bucket)
                blob = bucket.blob(gcs_images_path)
                blob.download_to_filename(self.img_out_dir / blob.name)

            self.logger.info("Downloaded images to %s", self.img_out_dir)

        except (ClientError, ServerError) as e:
            raise e

        except Exception as e:
            raise e


    def _clean_data_for_bounding_boxes(self):
        """Upon some basic discovery I noticed there was images that were missing bounding boxes.
        So this section is to identify images with missing bounding boxes or missing categories and remove prior to image retrieval
        No sense it retaining these images in the dataset."""

        json_path = self.ann_out_dir / "community_fish_detection_dataset.json"
        with open(json_path, "r") as f:
            annotations = json.load(f)

        images = annotations["images"]
        annotations = annotations["annotations"]
        cleaned = {}
        for col in annotations[0].keys():
            if col != "bbox":
                cleaned[col] = []

        # Ensure that we have the necessary columns
        cleaned["x"] = []
        cleaned["y"] = []
        cleaned["w"] = []
        cleaned["h"] = []

        bbox_missing = []

        for ann in annotations:
            if "bbox" not in ann:
                bbox_missing.append(ann["image_id"])
                continue
            ann["x"], ann["y"], ann["w"], ann["h"] = ann["bbox"]
            del ann["bbox"]
            for col in ann:
                cleaned[col].append(ann[col])

        images_df = pl.from_dict(images)
        annotations_df = pl.from_dict(cleaned)


        images_df = images_df.filter(~pl.col("id").is_in(bbox_missing))
        annotations_images_df = annotations_df.join(images_df, left_on="image_id", right_on="id", how="inner")

        return annotations_images_df["file_name"].to_list()


    async def extract_lila_images(self):
        """Extract images from GCS and save to local directory."""

        try:
            self._download_coco_json()
            image_list = self._clean_data_for_bounding_boxes()
            self._download_images(image_list)

        except Exception as e:
            raise e


















