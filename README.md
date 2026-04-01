# WhatsThisFish

Fish identification tool for divers. Uses iNaturalist and LILA datasets to train a YOLO11-based detection and classification model, enhanced with custom underwater image preprocessing channels.

## Overview

WhatsThisFish detects and identifies fish species in underwater images and video. It combines two large-scale open datasets — iNaturalist (3.57M photos, 15,564 taxa) and the LILA Community Fish Detection Dataset (~1.9M images) — with a modified YOLO11 model that processes 5-channel input (RGB + gradient magnitude + local contrast normalization) to improve detection of camouflaged species in complex reef environments.

## Project Structure

```
whatsthatfish/
├── src/
│   ├── data/                  # Data pipeline
│   │   ├── inaturalist_dataset.py   # iNaturalist S3 ingestion (taxa, observations, photos)
│   │   ├── download_lila.py         # LILA fish detection dataset (COCO → YOLO format)
│   │   ├── gcs_client.py            # GCS upload client for whats-that-fish bucket
│   │   ├── factory.py               # Async data extraction orchestration
│   │   ├── quality_scorer.py        # UIQM underwater image quality metrics
│   │   ├── domain_balanced_sampler.py  # Balances 17 LILA source datasets per epoch
│   │   └── multichannel_dataset.py  # 5-channel YOLODataset (RGB + gradient + LCN)
│   ├── preprocessing/         # Image channel computation
│   │   ├── gradient_map.py          # Scharr-based gradient magnitude
│   │   ├── local_contrast_norm.py   # Gaussian local contrast normalization
│   │   └── channel_composer.py      # Composes 5-channel tensors from BGR input
│   ├── model/
│   │   └── yolo_multichannel.py     # YOLO11 modified for 5-channel input
│   ├── training/
│   │   └── train.py                 # Three-stage transfer learning pipeline
│   ├── inference/
│   │   ├── detect_images.py         # Batch image detection
│   │   ├── detect_video.py          # Frame extraction + per-frame detection
│   │   └── crop_export.py           # Best-frame selection and crop export
│   └── evaluation/
│       └── evaluate.py              # mAP, recall, cross-domain generalization
├── data/                      # Local parquet files and image data
├── tests/
└── pyproject.toml
```

## Architecture

### Data Pipeline

- **iNaturalist S3** — Downloads taxonomy, observations, and photo metadata from the public `inaturalist-open-data` bucket. Converts CSV to Parquet and builds filtered datasets using lazy streaming (`pl.scan_parquet()` + `sink_parquet()`) to handle 400M+ row tables without OOM.
- **LILA** — Downloads the Community Fish Detection Dataset, validates bounding boxes, and converts COCO JSON annotations to YOLO format.
- **GCS** — Uploads preprocessed datasets to the `whats-that-fish` bucket (prefixes: `training/`, `validation/`, `contributions/`).

### Preprocessing (5-Channel Input)

Standard RGB is augmented with two additional channels to improve detection of camouflaged fish:

| Channel | Method | Purpose |
|---------|--------|---------|
| R, G, B | Original image | Color and texture information |
| Gradient | Scharr operator | Highlights object edges and boundaries |
| LCN | Local contrast normalization (Gaussian, ~31px) | Reveals subtle local anomalies against busy backgrounds |

### Model

Modified YOLO11 that accepts 5-channel input by replacing the first convolutional layer. Pretrained RGB weights are preserved; extra channels are initialized from the RGB mean plus small noise for stable fine-tuning.

### Training (Three-Stage Transfer Learning)

1. **Stage 1** — COCO pretraining (general object detection)
2. **Stage 2** — LILA fine-tuning with backbone frozen (fish detection domain)
3. **Stage 3** — Target domain adaptation with low learning rate

### Quality Scoring

Images are scored using UIQM (Underwater Image Quality Measure), entropy, and blur metrics, then classified as `bad_quality`, `hard_example`, or `good` for selective enhancement and curriculum training.

### Evaluation Targets

| Metric | Target |
|--------|--------|
| mAP@0.5 | >= 0.75 |
| mAP@0.5:0.95 | >= 0.50 |
| Recall@0.5 | >= 0.90 |

## Taxa Scope

- **Actinopterygii** (47178) — ray-finned fishes: ~50K taxa
- **Chondrichthyes** (196614) — sharks, rays, chimaeras: ~2K taxa
- Filtered to active taxa only: **43,991 taxa**, of which **15,564 have research-grade observations**

## Dataset

- **3.57M photos** across 2.36M research-grade observations
- **79.4% CC-BY-NC** licensed (project is non-commercial)
- Domain-balanced sampling across 17 LILA source datasets

## Setup

Requires Python >= 3.12.

```bash
cd whatsthatfish
pip install -e .
```

## Current Status

### Completed

- iNaturalist S3 data ingestion pipeline (taxa, observations, photos parquets)
- Taxonomy filtering and dataset construction (3.57M photos, 15,564 taxa)
- LILA fish detection dataset downloader with COCO-to-YOLO conversion
- GCS client with auth and upload capability
- UIQM underwater image quality scorer
- 5-channel preprocessing pipeline (gradient + LCN)
- Modified YOLO11 for 5-channel input
- Three-stage transfer learning pipeline
- Domain-balanced sampling across LILA sources
- Image/video inference and crop export
- Evaluation framework with mAP and cross-domain metrics

### Next Steps

1. S3-to-GCS photo transfer pipeline (async concurrent streaming of 3.57M photos)
2. Quality scorer vectorization (numpy stride_tricks for ~10x speedup)
3. Channel injection refactor (composable transform class)
4. Curriculum sampling with continuous UIQM scores
5. Observation-count-aware sampling strategy for rare taxa

## License

Non-commercial use. Dataset includes CC-BY-NC licensed content from iNaturalist.
