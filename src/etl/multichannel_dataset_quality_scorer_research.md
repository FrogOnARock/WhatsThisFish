# Research Topics for `multichannel_dataset.py` + `quality_scorer.py`

**Code Context**: A YOLO-based fish identification pipeline that (1) extends `YOLODataset` to inject gradient-magnitude and local-contrast-normalisation channels at `__getitem__` time, and (2) scores underwater image quality via UIQM (colorfulness + sharpness + contrast) and Shannon entropy to gate selective enhancement. Both files sit in the same `src/data/` layer and cooperate to ensure the model only trains on meaningfully preprocessed images.

---

## Research Topics:

### 1. `Vectorizing Block-Loop Operations in NumPy with stride_tricks`

**Summary**: Both `_compute_uism` and `_compute_uiconm` in `quality_scorer.py` walk every non-overlapping 8×8 block of a 2-D image using nested Python `for` loops. On a 640×640 image that is ~6,400 iterations per call, and `compute_uiqm` calls both — meaning the inner Python interpreter overhead fires ~12,800 times before a single gradient is computed. NumPy's `stride_tricks` module lets you reshape the array so that block-axis operations become fully vectorized array operations, removing the loop entirely.

**How it applies**: `_compute_uism` extracts `block.max()` and `block.min()` per 8×8 patch and accumulates `20 * log(bmax / bmin)`. `_compute_uiconm` does the same on a grayscale image. Both are textbook candidates for the reshape-then-reduce pattern: reshape `(H, W)` into `(H//B, B, W//B, B)`, transpose to `(H//B, W//B, B, B)`, then call `.max()` and `.min()` along the last two axes in one shot.

**Usage patterns**:
- Offline quality pre-scoring of large datasets (thousands of images before training starts)
- Real-time quality gating inside a DataLoader worker where latency matters
- Any sliding or non-overlapping window statistic: max, min, mean, std, entropy

**How to implement**:
1. Crop the image to a multiple of the block size: `h_crop = (h // B) * B`.
2. Reshape: `blocks = arr[:h_crop, :w_crop].reshape(h_crop // B, B, w_crop // B, B)`.
3. Transpose so block pixels are in the last two axes: `blocks = blocks.transpose(0, 2, 1, 3)` giving shape `(nH, nW, B, B)`.
4. Compute `bmax = blocks.max(axis=(-2, -1))` and `bmin = blocks.min(axis=(-2, -1))` — both are `(nH, nW)` arrays.
5. Apply the log ratio with a mask: `valid = (bmin > 0) & (bmax > 0)`, then `np.log(bmax[valid] / bmin[valid]).mean()`.

**Code snippet example**:
```python
import numpy as np

def _compute_uiconm_vectorized(img_bgr: np.ndarray, block_size: int = 8) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size

    # Shape: (nH, nW, B, B)
    blocks = (gray[:h_crop, :w_crop]
              .reshape(h_crop // block_size, block_size,
                       w_crop // block_size, block_size)
              .transpose(0, 2, 1, 3))

    bmax = blocks.max(axis=(-2, -1))   # (nH, nW)
    bmin = blocks.min(axis=(-2, -1))   # (nH, nW)

    valid = (bmin > 0) & (bmax > 0)
    if not valid.any():
        return 0.0

    ratios = np.log(bmax[valid] / bmin[valid] + 1e-8)
    return float(ratios.mean())
```

---

### 2. `Decoupling Channel-Injection from __getitem__ with a PyTorch Transform`

**Summary**: `MultiChannelYOLODataset.__getitem__` currently bundles RGB→BGR conversion, scale detection, gradient computation, LCN computation, and tensor re-stacking directly inside the data-loading hot path. PyTorch's transform API (both `torchvision.transforms` and the newer `torchvision.transforms.v2` / `albumentations` style) was designed precisely to keep this logic composable, cacheable, and testable outside of the dataset class.

**How it applies**: The current approach has two risks: (1) every DataLoader worker re-runs the full conversion + channel math on every call to `__getitem__`, which cannot be cached; (2) the scale-detection heuristic `if img_np.max() <= 1.0` is fragile — a very dark image with all pixels under 1.0 in a `[0, 255]` tensor would be mis-scaled. Wrapping the channel injection in a callable `Transform` object isolates both concerns and makes the heuristic testable in isolation.

**Usage patterns**:
- Composing with `torchvision.transforms.v2.Compose` so the channel step slots into an existing augmentation pipeline
- Writing a unit test that feeds a known synthetic tensor and asserts the output shape is `(5, H, W)`
- Swapping gradient/LCN for other feature channels (e.g., depth maps) without touching the dataset class

**How to implement**:
1. Create a `AddGradientLCNChannels` class with `__call__(self, img_tensor) -> torch.Tensor`.
2. Move all the BGR conversion, scale detection, `compute_gradient_magnitude`, and `local_contrast_normalize` calls into that class.
3. Pass an instance as the `transforms` argument when constructing the dataset, or apply it at the end of the existing transform chain.
4. In `__getitem__`, replace the entire block after `data = super().__getitem__(index)` with `data["img"] = self.extra_channel_transform(data["img"])`.

**Code snippet example**:
```python
import torch
import numpy as np
import cv2
from whatsthatfish.src.preprocessing.gradient_map import compute_gradient_magnitude
from whatsthatfish.src.preprocessing.local_contrast_norm import local_contrast_normalize


class AddGradientLCNChannels:
    """Callable transform: (3, H, W) tensor -> (5, H, W) tensor."""

    def __init__(self, lcn_kernel_size: int = 31):
        self.lcn_kernel_size = lcn_kernel_size

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # Determine range before any conversion
        scaled_to_255 = img_tensor.max() > 1.0

        img_np = img_tensor.permute(1, 2, 0).numpy()  # HWC, RGB
        img_bgr = img_np[:, :, ::-1]
        if not scaled_to_255:
            img_bgr = (img_bgr * 255).astype(np.uint8)
        else:
            img_bgr = img_bgr.astype(np.uint8)

        grad = compute_gradient_magnitude(img_bgr)          # (H, W), [0, 1]
        lcn  = local_contrast_normalize(img_bgr,
                                        kernel_size=self.lcn_kernel_size)  # (H, W), [0, 1]

        grad_t = torch.from_numpy(grad).unsqueeze(0)
        lcn_t  = torch.from_numpy(lcn).unsqueeze(0)

        if scaled_to_255:
            grad_t = grad_t * 255.0
            lcn_t  = lcn_t  * 255.0

        return torch.cat([img_tensor, grad_t, lcn_t], dim=0)  # (5, H, W)

    def __repr__(self):
        return f"{self.__class__.__name__}(lcn_kernel_size={self.lcn_kernel_size})"


# Usage in dataset construction:
# transform = AddGradientLCNChannels(lcn_kernel_size=31)
# dataset = MultiChannelYOLODataset(..., transforms=transform)
```

---

### 3. `Using UIQM as a Curriculum Learning Signal Rather Than a Hard Gate`

**Summary**: `classify_image_quality` maps continuous UIQM and entropy scores to one of three discrete buckets (`bad_quality`, `hard_example`, `good`) using fixed thresholds. The current design feeds this classification back to a "selective enhancement pipeline," but modern training practice suggests that the raw continuous scores — rather than their discretised labels — are more valuable as a *curriculum learning* signal: feeding harder or lower-quality samples at a controlled rate as training progresses, rather than deciding up-front whether to enhance or skip them.

**How it applies**: Because `compute_uiqm` and `compute_entropy` already return `float` values, the scores can be stored in a sidecar file (e.g., a Parquet table indexed by image path) during a one-time pre-scoring pass. The DataLoader sampler can then weight sampling probability by quality tier or by inverse-UIQM, ensuring the model sees degraded-but-learnable examples at the right stage of training without discarding them entirely.

**Usage patterns**:
- Curriculum learning: start training on `good` images, progressively introduce `hard_example` and `bad_quality` samples as validation loss plateaus
- Weighted random sampling: assign `WeightedRandomSampler` weights inversely proportional to UIQM so rare hard examples are oversampled
- Dataset pre-filtering: use UIQM + entropy as a fast quality gate to discard genuinely corrupted images (sensor noise, total occlusion) before annotation

**How to implement**:
1. Run a one-time scoring pass: iterate all images, compute `(uiqm, entropy, blur)`, save to `quality_scores.parquet` with columns `[path, uiqm, entropy, blur, label]`.
2. In the DataLoader, load the parquet and build a `weights` array: `weights = 1.0 / (uiqm_scores + epsilon)` or a tier-based mapping.
3. Pass to `torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)`.
4. For curriculum learning, use a `Callback` (Lightning) or manual epoch hook to swap the sampler weights at a milestone epoch.

**Code snippet example**:

```python
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from pathlib import Path
from whatsthatfish.src.etl.quality_scorer import compute_uiqm, compute_entropy
import cv2
import numpy as np


def build_quality_weights(image_paths: list[Path],
                          epsilon: float = 0.5) -> torch.Tensor:
    """Score every image and return inverse-UIQM sampling weights."""
    scores = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            scores.append(epsilon)
            continue
        uiqm = compute_uiqm(img)
        scores.append(max(uiqm, 0.0) + epsilon)

    weights = 1.0 / np.array(scores, dtype=np.float32)
    weights /= weights.sum()  # normalise to a probability distribution
    return torch.from_numpy(weights)


def make_quality_weighted_sampler(image_paths: list[Path]) -> WeightedRandomSampler:
    weights = build_quality_weights(image_paths)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(image_paths),
        replacement=True,
    )


# Curriculum variant: at epoch N, mix weights toward uniform
def curriculum_weights(base_weights: torch.Tensor,
                       epoch: int,
                       warmup_epochs: int = 10) -> torch.Tensor:
    alpha = min(epoch / warmup_epochs, 1.0)  # 0 → 1 over warmup
    uniform = torch.ones_like(base_weights) / len(base_weights)
    return (1 - alpha) * base_weights + alpha * uniform
```

---

Sources:
- [Writing Custom Datasets, DataLoaders and Transforms - PyTorch Tutorials](https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [torch.utils.data - PyTorch Documentation](https://docs.pytorch.org/docs/main/data.html)
- [NumPy Optimization: Vectorization and Broadcasting - Paperspace Blog](https://blog.paperspace.com/numpy-optimization-vectorization-and-broadcasting/)
- [Look Ma, No for Loops: Array Programming With NumPy - Real Python](https://realpython.com/numpy-array-programming/)
- [Underwater Image Quality Evaluation: A Comprehensive Review - IET Image Processing 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70068)
- [Underwater Image Quality Assessment: A Perceptual Framework Guided by Physical Imaging - arXiv](https://arxiv.org/html/2412.15527v1)
