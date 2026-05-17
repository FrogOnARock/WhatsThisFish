from copy import copy

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

from .od_dataset import ObjectDetectionDataset
from ..config import init_gcs_worker


def object_detection_collate(original_batch):

    img = torch.stack([item[0] for item in original_batch])
    batch_idx = []
    cls = []
    bboxes = []

    for idx, item in enumerate(original_batch):
        label_tensors = item[1]
        if label_tensors.shape[0] > 0:
            batch_idx.append(torch.full((label_tensors.shape[0],), idx))
            cls.append(label_tensors[:, 0:1])
            bboxes.append(label_tensors[:, 1:5])

    return {
        "img": img,
        "batch_idx": torch.cat(batch_idx, 0) if batch_idx else torch.zeros(0),
        "cls": torch.cat(cls, 0) if cls else torch.zeros((0, 1)),
        "bboxes": torch.cat(bboxes, 0) if bboxes else torch.zeros((0, 4)),
        "im_file": [item[2] for item in original_batch],
        "ori_shape": [tuple(img.shape[1:]) for img in img],
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0)) for _ in original_batch]
    }

class ODDataLoader(DataLoader):
    def reset(self):
        pass  # Ultralytics calls this between epochs; standard DataLoader re-iterates naturally


def od_dataloader(mode: str, batch_size: int = 16, max_samples: int = None):
    base_transform = [
        v2.Resize(size=(640, 640)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]

    if mode == "train":
        to_insert = [
            v2.ColorJitter(brightness=0.4, saturation=0.8, hue=0.015),
            v2.RandomHorizontalFlip(),
            v2.ScaleJitter(target_size=(640, 640), scale_range=(0.5, 2.0)),
        ]
        base_transform = to_insert + base_transform
        transform = v2.Compose(base_transform)
        od_dataset = ObjectDetectionDataset(transforms=transform, split=mode, max_samples=max_samples)
        sampler = WeightedRandomSampler(
            [max(row.uiqm, 1e-6) for row in od_dataset.data],
            len(od_dataset),
            replacement=True
        )
    else:
        transform = v2.Compose(base_transform)
        od_dataset = ObjectDetectionDataset(transforms=transform, split=mode, max_samples=max_samples)
        sampler = None


    dataloader = DataLoader(
        dataset=od_dataset,
        sampler=sampler,
        shuffle=False,
        collate_fn=object_detection_collate,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=init_gcs_worker,
        prefetch_factor=2
    )
    dataloader.reset = lambda: None
    return dataloader


class CustomDetectionValidator(DetectionValidator):
    def preprocess(self, batch):
        # images are already [0,1] from ToTensor(); scale up so parent's /255 restores [0,1]
        batch["img"] = batch["img"] * 255
        return super().preprocess(batch)


class CustomDetectionTrainer(DetectionTrainer):
    max_samples: int = None

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        return od_dataloader(mode=mode, batch_size=batch_size, max_samples=self.max_samples)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
        batch["cls"] = batch["cls"].to(self.device)
        batch["bboxes"] = batch["bboxes"].to(self.device)
        batch["batch_idx"] = batch["batch_idx"].to(self.device)
        return batch

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
