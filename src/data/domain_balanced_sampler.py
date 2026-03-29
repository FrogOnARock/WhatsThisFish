"""
Domain-balanced sampler for the LILA dataset.

The LILA dataset has severe source imbalance (e.g., Salmon Computer Vision
has ~532K frames, ~28% of the dataset). This sampler gives each of the 17
source datasets equal sampling probability per epoch.
"""

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler


class DomainBalancedSampler(torch.utils.data.Sampler):
    """Each source dataset gets equal probability regardless of size.

    Args:
        coco_json_path: Path to parsed COCO JSON with source_dataset fields
        image_ids: Ordered list of image IDs matching dataset indices
        num_samples: Number of samples per epoch (default: len(dataset))
    """

    def __init__(self, coco_json_path: str, image_ids: list[int],
                 num_samples: int | None = None):
        with open(coco_json_path) as f:
            coco = json.load(f)

        id_to_source = {}
        for img in coco["images"]:
            id_to_source[img["id"]] = img.get("source_dataset", "unknown")

        # Count images per source
        source_counts = Counter()
        sources_per_idx = []
        for img_id in image_ids:
            source = id_to_source.get(img_id, "unknown")
            source_counts[source] += 1
            sources_per_idx.append(source)

        # Compute weight per image: 1 / count(source_dataset)
        weights = []
        for source in sources_per_idx:
            weights.append(1.0 / source_counts[source])

        self.num_samples = num_samples or len(image_ids)
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(
            self.weights, self.num_samples, replacement=True
        ).tolist())

    def __len__(self):
        return self.num_samples
