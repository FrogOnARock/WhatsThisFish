# YOLO11l Detection Training — Diagnostic Report

**Run:** `runs/detect/train` | **Epochs completed:** 34 / 50 | **Date:** 2026-05-13

---

## Observed Symptoms

### 1. Zero detection metrics across all 34 epochs

```
metrics/precision(B) = 0.0  (all epochs)
metrics/recall(B)    = 0.0  (all epochs)
metrics/mAP50(B)     = 0.0  (all epochs)
metrics/mAP50-95(B)  = 0.0  (all epochs)
```

The model has never made a single TP prediction according to Ultralytics' evaluator,
despite the training loss showing clear learning.

### 2. Training loss is decreasing (model IS learning)

| Epoch | box_loss | cls_loss | dfl_loss |
|-------|----------|----------|----------|
| 1     | 1.500    | 1.675    | 1.464    |
| 17    | 0.940    | 0.653    | 1.109    |
| 34    | 0.648    | 0.396    | 0.944    |

All three training losses are on a consistent downward trajectory.

### 3. Validation loss is diverging and steadily increasing

| Epoch | val/box_loss | val/cls_loss | val/dfl_loss |
|-------|-------------|--------------|--------------|
| 1     | 3.397        | 6.260        | 3.333        |
| 17    | 3.613        | 7.687        | 3.734        |
| 34    | 3.533        | 7.931        | 3.537        |

`val/cls_loss` at epoch 1 (6.26) is already **3.7× higher** than train/cls_loss (1.67)
at the same epoch — indicating a distribution mismatch from the very first validation pass,
not a gradual overfit.

---

## Root Causes

### Root Cause #1 — Double normalization on validation (zero mAP cause)

**This is almost certainly the cause of all-zero metrics.**

During training, `CustomDetectionTrainer.preprocess_batch` is called on each batch.
Our override correctly skips the `/255` division because `ToTensor()` already normalizes
images to `[0, 1]`:

```python
def preprocess_batch(self, batch):
    batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
    # deliberately no /255 — ToTensor() upstream already normalizes
```

However, during validation Ultralytics does **not** call `trainer.preprocess_batch`.
Instead, `DetectionTrainer.validate()` delegates to `self.validator`, which has its own
`DetectionValidator.preprocess()` method that unconditionally divides by 255:

```python
# Ultralytics DetectionValidator.preprocess (not overridden)
batch["img"] = batch["img"].to(self.device, non_blocking=True).float()
batch["img"] /= 255   # <-- applied to images already in [0, 1]
```

Result: validation images land in `[0, ~0.004]` — effectively black. The model, trained
on properly normalized `[0, 1]` images, produces no valid detections on near-black input.
This explains zero precision, zero recall, and the immediate high `val/cls_loss` at epoch 1.

**Fix:** Override `get_validator()` in `CustomDetectionTrainer` and patch the validator's
`preprocess` to skip the `/255` division.

---

### Root Cause #2 — WeightedRandomSampler applied to validation

`od_dataloader(mode)` builds a `WeightedRandomSampler` regardless of mode. For validation,
this means:
- Each epoch samples a different UIQM-biased subset of the held-out set (not the full set)
- Metrics are computed on a moving target, making trends uninterpretable
- High-quality images are overrepresented; hard/noisy val images may never be seen

This does not cause zero mAP by itself but makes all validation metrics unreliable.

**Fix:** Skip the sampler when `mode != "train"`. Use `shuffle=False` and iterate the
full val set sequentially.

---

### Root Cause #3 — No augmentation applied during training

`args.yaml` shows Ultralytics augmentation flags set (mosaic=1.0, randaugment, HSV jitter,
fliplr=0.5). However, these augmentations are implemented inside Ultralytics' own
`YOLODataset` class, which we bypass entirely with our custom `DataLoader`.

Our training pipeline currently applies only `Resize([640, 640])` + `ToTensor()`. The model
is training on clean, unaugmented images — increasing overfitting risk and reducing
generalization.

**Fix:** Either re-implement the key augmentations (mosaic, fliplr, HSV jitter) in our
`ObjectDetectionDataset.__getitem__` or via `torchvision.transforms`, or wrap Ultralytics'
augmentation pipeline selectively.

---

### Root Cause #4 — Hardcoded batch size in dataloader

`od_dataloader` hardcodes `batch_size=8` regardless of the `batch_size` argument passed
by the trainer. Ultralytics calls `get_dataloader` with `batch_size * 2` for val (to use
larger val batches for speed). This parameter is silently ignored.

Minor issue relative to the above but worth fixing for correctness.

---

## Priority Order for Fixes

1. **Override validator preprocess** — unblocks all metrics immediately (highest impact)
2. **Remove sampler from val** — makes validation metrics reliable
3. **Wire through batch_size param** — correctness fix
4. **Add augmentation** — generalization / overfitting mitigation (larger scope)

---

## What "Good" Looks Like

Per `CLAUDE.md` evaluation targets:
- mAP@0.5 ≥ 0.75
- mAP@0.5:0.95 ≥ 0.50
- Recall@0.5 ≥ 0.90 at conf=0.15
