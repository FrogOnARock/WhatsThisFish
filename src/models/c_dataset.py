import io

from PIL.Image import Image
import torch
from torch.utils.data import Dataset
from sqlalchemy import select

from ..database.config import get_session_factory
from ..database.models import InatClassificationDataset
from ..config import _bucket, get_config

class ClassificationDataset(Dataset):
    def __init__(self,
                 split: str = "train",
                 transforms = None,
                 ):

        self.transforms = transforms
        self.gcs_config = get_config().gcs
        self.gcs_prefix = self.gcs_config.prefixes.get("gcs_train")
        self.session_factory = get_session_factory()
        self.session = self.session_factory()
        self.split = 0 if split == "train" else 1
        self.data = self.session.execute(
            select(InatClassificationDataset.photo_uuid, InatClassificationDataset.filename,
                   InatClassificationDataset.uiqm, InatClassificationDataset.subfamily, InatClassificationDataset.genus,
                   InatClassificationDataset.species, InatClassificationDataset.proposed_bbox).where(InatClassificationDataset.train == self.split)
        ).all()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        record = self.data[idx]
        filename = record.filename
        bounding_box = record.proposed_bbox
        blob = _bucket.blob(self.gcs_prefix + "/" + filename)
        image_pil = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")

        W_img, H_img = image_pil.shape[:-1]
        bbox = bounding_box * [W_img, H_img, W_img, H_img]
        X = bbox[0] - (bbox[2] / 2)
        Y = bbox[1] - (bbox[3] / 2)

        img = image_pil.resize(size=(X, Y))
        label = [
            record.subfamily,
            record.genus,
            record.species
        ]
        label_tensor = torch.tensor(label, dtype=torch.int)
        img_tensor, label_tensor = self.transforms(img, label_tensor)

        return img_tensor, label_tensor



class UltraClassificationDataset(Dataset)
    def __init__(self,
                 split: str = "train",
                 transforms=None,
                 ):
        self.transforms = transforms
        self.session_factory = get_session_factory()
        self.session = self.session_factory()
        self.split = 0 if split == "train" else 1
        self.data = self.session.execute(
            select(InatClassificationDataset.photo_uuid, InatClassificationDataset.filename,
                   InatClassificationDataset.uiqm, InatClassificationDataset.subfamily, InatClassificationDataset.genus,
                   InatClassificationDataset.species).where(InatClassificationDataset.train == self.split)
        ).all()

    @property
    def __labels__(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):



