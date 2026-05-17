import io
import os

import numpy as np
from torch.utils.data import Dataset
from sqlalchemy import select
from PIL import Image
import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from ..database.models import LilaImageQuality, LilaYolo
from ..config import get_config, _bucket
from ..database.config import get_session_factory
from ..retry import transfer_retry


class ObjectDetectionDataset(Dataset):

    def __init__(self, split: str = "train", transforms=None, max_samples: int = None):
        self.session = get_session_factory()
        self.session_factory = self.session()
        self.gcs_config = get_config().gcs
        self.gcs_prefix = self.gcs_config.prefixes.get("gcs_object_detection")
        self.split = True if split == "train" else False
        self.transform = transforms

        query = (
            select(LilaImageQuality.file_name, LilaImageQuality.uiqm, LilaYolo.annotation)
            .join(LilaYolo, LilaImageQuality.file_name == LilaYolo.file_name)
            .where(LilaYolo.annotation[0]["is_train"].as_boolean() == self.split)
        )
        if max_samples is not None:
            query = query.limit(max_samples)

        self.data = self.session_factory.execute(query).all()

    @property
    def labels(self):
        return [
                {
                    "bboxes": np.array([
                        [ann["norm_center_x"], ann["norm_center_y"], ann["norm_width"], ann["norm_height"]]
                        if ann["class_id"] == 0
                        else [0, 0, 0, 0]
                        for ann in record.annotation
                    ]),
                    "cls": np.array([ann["class_id"] for ann in record.annotation])
                }
                for record in self.data
            ]

    def __len__(self):
        return len(self.data)


    @transfer_retry
    def __getitem__(self, idx):

        record = self.data[idx]
        labels = [
            [
            ann["class_id"],
            ann["norm_center_x"],
            ann["norm_center_y"],
            ann["norm_width"],
            ann["norm_height"]
            ]
            for ann in record.annotation if ann["class_id"] == 0
        ] # (n, 5) for each of the bounding boxes

        label_tensor = torch.zeros((0, 5), dtype=torch.float32) if not labels else torch.tensor(labels, dtype=torch.float32)

        filename = record.file_name
        blob = _bucket.blob(self.gcs_prefix + "/" + filename)
        image_pil = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")
        W_img, H_img = image_pil.size  # PIL size is (W, H)

        abs_boxes = label_tensor[:, 1:5] * torch.tensor([W_img, H_img, W_img, H_img])
        boxes = BoundingBoxes(
            abs_boxes,
            format=BoundingBoxFormat.CXCYWH, canvas_size=(H_img, W_img)
        )

        image_tensor, boxes = self.transform(image_pil, boxes)
        H_out, W_out = image_tensor.shape[-2], image_tensor.shape[-1]
        norm_boxes = boxes / torch.tensor([W_out, H_out, W_out, H_out])
        label_tensor = torch.cat([label_tensor[:, 0:1], norm_boxes], dim=1)

        return image_tensor, label_tensor, filename