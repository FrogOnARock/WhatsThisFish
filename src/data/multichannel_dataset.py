"""
Multi-channel dataset loader for 5-channel YOLO training.

Extends the standard Ultralytics dataset to produce 5-channel tensors:
  [RGB(3) + gradient_magnitude(1) + local_contrast_norm(1)]

Extra channels are computed on-the-fly AFTER augmentations are applied
to the RGB image, ensuring consistency between RGB and derived channels.
"""

import cv2
import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset

from whatsthatfish.src.preprocessing.gradient_map import compute_gradient_magnitude
from whatsthatfish.src.preprocessing.local_contrast_norm import local_contrast_normalize


class MultiChannelYOLODataset(YOLODataset):
    """YOLO dataset that produces 5-channel tensors.

    Augmentation order:
        1. Ultralytics applies spatial + color augmentations to RGB
        2. We intercept the augmented RGB image
        3. Compute gradient magnitude and LCN from augmented RGB
        4. Stack into 5-channel tensor
    """

    def __init__(self, *args, lcn_kernel_size: int = 31, **kwargs):
        super().__init__(*args, **kwargs)
        self.lcn_kernel_size = lcn_kernel_size

    def __getitem__(self, index):
        """Override to add gradient and LCN channels after augmentation."""
        data = super().__getitem__(index)

        # The parent returns a dict with 'img' as a (C, H, W) tensor
        img_tensor = data["img"]  # (3, H, W), float32, [0, 1] or [0, 255]

        # Convert to HWC BGR for our preprocessing functions
        if isinstance(img_tensor, torch.Tensor):
            img_np = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        else:
            img_np = img_tensor

        # Determine scale — Ultralytics typically uses [0, 255] uint8-like
        if img_np.max() <= 1.0:
            img_bgr = (img_np[:, :, ::-1] * 255).astype(np.uint8)
        else:
            img_bgr = img_np[:, :, ::-1].astype(np.uint8)

        # Compute extra channels from augmented image
        grad = compute_gradient_magnitude(img_bgr)  # (H, W), [0, 1]
        lcn = local_contrast_normalize(img_bgr, kernel_size=self.lcn_kernel_size)  # (H, W), [0, 1]

        # Stack: convert back to the same scale as original tensor
        if isinstance(img_tensor, torch.Tensor):
            grad_t = torch.from_numpy(grad).unsqueeze(0)  # (1, H, W)
            lcn_t = torch.from_numpy(lcn).unsqueeze(0)    # (1, H, W)

            # Scale extra channels to match img_tensor range
            if img_tensor.max() > 1.0:
                grad_t = grad_t * 255.0
                lcn_t = lcn_t * 255.0

            data["img"] = torch.cat([img_tensor, grad_t, lcn_t], dim=0)  # (5, H, W)
        else:
            grad_exp = grad[:, :, np.newaxis]
            lcn_exp = lcn[:, :, np.newaxis]
            if img_np.max() > 1.0:
                grad_exp = grad_exp * 255.0
                lcn_exp = lcn_exp * 255.0
            data["img"] = np.concatenate([img_np, grad_exp, lcn_exp], axis=2)

        return data
