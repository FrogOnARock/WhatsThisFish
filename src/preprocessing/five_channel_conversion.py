import cv2
import numpy as np
import torch
from PIL import Image
from .lcn_gradient_map import gradient_map, local_contrast_normalization

class AddMultiChannel:
    """Torchvision-compatible transform: PIL Image -> (5, H, W) float32 tensor.

    Channel layout:
        0-2 : RGB, normalized to [0.0, 1.0]
          3 : Scharr gradient magnitude (grayscale), normalized to [0.0, 1.0]
          4 : Local contrast normalization on grayscale, normalized to [0.0, 1.0]

    Usage:
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            AddMultiChannel(),
        ])
        tensor = transform(pil_image) # shape: (5, 640, 640)
    """

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)  # (H, W, 3) RGB
        return self.compute_channels(arr)

    def compute_channels(self, arr: np.ndarray) -> torch.Tensor:
        """Build the 5-channel tensor from a (H, W, 3) uint8 RGB array.
        """
        rgb = arr[..., :3] / 255
        img_gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        img_gray_float = img_gray.astype(np.float64)

        s_gradient = gradient_map(img_gray_float) / 255 #convert to [0, 1] scale
        lcn = local_contrast_normalization(img_gray_float) / 255

        stacked_channels = np.concatenate([rgb, s_gradient[..., np.newaxis], lcn[..., np.newaxis]], axis=2)
        return torch.from_numpy(stacked_channels.transpose(2, 0, 1)).float()