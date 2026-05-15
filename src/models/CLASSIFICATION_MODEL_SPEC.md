# Classification Model — Implementation Spec

## Overview

5-channel YOLO11-cls fine-tune on ~450K iNaturalist cropped images, 1,500 species.
Input: RGB + Scharr gradient + LCN (5 channels). Geographical train/val split.

---

## 1. ClassificationDataset

**File:** `models/cls_dataset.py`

### Responsibilities
- Query iNat classification images from Postgres, filtered by `is_train` flag (geographical split — see §6)
- Stream images from GCS on `__getitem__`
- Apply 5-channel transform pipeline (RGB + Scharr + LCN)
- UIQM-weighted sampling support (same pattern as `ObjectDetectionDataset`)

### Schema
Query joins:
- `inat_filtered_observations` — `photo_uuid`, `taxon_id`, `is_train`
- `inat_image_quality` — `uiqm` (filter: `inat_clip_context.is_underwater = 1`)
- `inat_clip_context` — underwater filter

### Transform pipeline (train)
```
PIL Image (RGB)
    → Resize(240, 240)
    → RandomHorizontalFlip(p=0.5)
    → ColorJitter(brightness=0.4, saturation=0.7, hue=0.015)
    → ToTensor / ToDtype float32 [0,1]           ← RGB tensor (3, 240, 240)
    → ScharrGradient()                            ← appends channel 4
    → LocalContrastNormalization()                ← appends channel 5
    → output: (5, 240, 240)
```

Val transform: drop RandomHorizontalFlip + ColorJitter, keep rest.

### Notes
- `ScharrGradient` and `LocalContrastNormalization` already implemented in
  `preprocessing/five_channel_conversion.py` — adapt as `nn.Module` transforms
- Cap 300 images/taxon already applied upstream in S3→GCS transfer

---

## 2. Collate Function

**File:** `models/cls_dataloader.py`

Ultralytics classification batch format:
```python
{
    "img":   torch.Tensor,  # (B, 5, 240, 240)
    "cls":   torch.Tensor,  # (B,) integer class indices
}
```

Unlike detection, no `batch_idx` / `bboxes` needed. Simpler than OD collate.

---

## 3. DataLoader

**File:** `models/cls_dataloader.py`

```python
def cls_dataloader(mode: str, batch_size: int = 32, max_samples: int = None):
    ...
    if mode == "train":
        sampler = WeightedRandomSampler(uiqm_weights, len(dataset), replacement=True)
    else:
        sampler = None
    return DataLoader(dataset, sampler=sampler, collate_fn=cls_collate,
                      batch_size=batch_size, num_workers=8,
                      pin_memory=True, worker_init_fn=init_gcs_worker,
                      prefetch_factor=4)
```

---

## 4. CustomClassificationTrainer

**File:** `models/cls_dataloader.py` (or `training/cls_training.py`)

### 4a. `get_dataloader`
Same pattern as OD — ignore `dataset_path`, call `cls_dataloader(mode, batch_size)`.

### 4b. `preprocess_batch`
```python
def preprocess_batch(self, batch):
    batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
    batch["cls"] = batch["cls"].to(self.device)
    return batch
```
No `/255` — `ToDtype(scale=True)` upstream already normalizes.

### 4c. `get_model` — 5-channel first conv extension
```python
def get_model(self, cfg=None, weights=None, verbose=True):
    model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
    conv = model.model[0].conv          # first Conv2d layer
    old_weight = conv.weight.data       # (out, 3, kH, kW)
    mean_weight = old_weight.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
    new_weight = torch.cat([old_weight, mean_weight, mean_weight], dim=1)
    conv.weight = nn.Parameter(new_weight)
    conv.in_channels = 5
    # freeze everything except layer 0 for warmup
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("model.0.")
    return model
```

### 4d. `on_train_epoch_start` — warmup unfreeze
```python
def on_train_epoch_start(self):
    if self.epoch == self.args.warmup_epochs:
        for param in self.model.parameters():
            param.requires_grad = True
        # add newly unfrozen params at reduced LR to preserve layer 0 momentum
        self.optimizer.add_param_group({
            "params": [p for n, p in self.model.named_parameters()
                       if not n.startswith("model.0.")],
            "lr": self.args.lr0 * 0.1,
        })
```

---

## 5. CustomClassificationValidator

Same `/255` issue applies. Override `preprocess`:
```python
class CustomClassificationValidator(ClassificationValidator):
    def preprocess(self, batch):
        batch["img"] = batch["img"] * 255
        return super().preprocess(batch)
```

Wire via `get_validator()` on the trainer.

---

## 6. Geographical Train/Val Split

**Goal:** val set contains observations from geographical regions not seen during training,
testing true generalization rather than memorization of location-specific visual patterns
(water clarity, reef type, lighting).

**Approach:**
- Use `latitude`/`longitude` from `inat_filtered_observations`
- Assign observations to geographic grid cells (e.g., 10° × 10° bins)
- Hold out N grid cells as val — select cells that are:
  - Geographically diverse (not all one ocean)
  - Represent ~20% of observations
  - Have reasonable species coverage (avoid cells with only 1-2 species)
- Set `is_train = False` for all observations in held-out cells

**Alternative:** ocean-basin split (Pacific / Atlantic / Indian / etc.) — simpler,
more interpretable, naturally diverse.

**TODO:** Implement split assignment script before populating `is_train` column.

---

## 7. Open Questions

- [ ] Center crop vs full image for classification input (discussed — center crop for now,
      upgrade to detector-generated crops post-OD training)
- [ ] `imgsz` for classification: 224 (YOLO default) vs 240 — confirm with Ultralytics
      ClassificationTrainer
- [ ] `warmup_epochs` value — CLAUDE.md mentions warmup phase but no specific count;
      3-5 epochs is typical
- [ ] Whether to use YOLO11-cls or CLIP fine-tune as the classification backbone
