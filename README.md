# WhatsThisFish

Fish identification tool for divers. Uses iNaturalist and LILA datasets to train a YOLO-based detection and classification model, enhanced with custom underwater image preprocessing channels.

## Overview

WhatsThisFish detects and identifies fish species in underwater images and video. It combines two large-scale open datasets — iNaturalist (3.57M photos, 15,564 taxa) and the LILA Community Fish Detection Dataset (~1.9M images) — with a modified YOLO model that processes multi-channel input to improve detection of camouflaged species in complex reef environments.

## Project Structure

```
whatsthatfish/
├── src/
│   ├── config.py                  # App config loader (S3, GCS settings)
│   ├── config/
│   │   └── data_config.yaml       # Data source configuration
│   ├── data/                      # Data pipeline
│   │   ├── factory.py             # Async data extraction orchestration
│   │   ├── inaturalist_dataset.py # iNaturalist S3 ingestion (taxa, observations, photos)
│   │   ├── download_lila.py       # LILA fish detection dataset downloader
│   │   └── gcs_client.py          # GCS upload client
│   ├── database/                  # Database layer
│   │   ├── base.py                # SQLAlchemy declarative base with naming conventions
│   │   ├── config.py              # Sync + async engine/session factories
│   │   └── models.py              # ORM models (inat_taxa, inat_observations, inat_photos)
│   ├── preprocessing/             # Image channel computation (TODO)
│   ├── model/                     # YOLO multi-channel model (TODO)
│   ├── training/                  # Training loop (TODO)
│   ├── inference/                 # Image/video detection (TODO)
│   └── evaluation/                # Model evaluation (TODO)
├── alembic/                       # Database migrations
│   ├── env.py                     # Configured for models + DATABASE_URL
│   └── versions/                  # Migration scripts
├── data/                          # Local parquet files
├── alembic.ini
└── pyproject.toml
```

## Architecture

### Data Pipeline

- **iNaturalist S3** — Downloads taxonomy, observations, and photo metadata from the public `inaturalist-open-data` bucket. Converts CSV to Parquet and builds filtered datasets using lazy streaming (`pl.scan_parquet()` + `sink_parquet()`) to handle 400M+ row tables without OOM.
- **LILA** — Downloads the Community Fish Detection Dataset from GCS, validates bounding boxes against COCO JSON annotations, and filters images missing annotations.
- **GCS** — Uploads datasets to the `whats-that-fish` bucket (prefixes: `training/`, `validation/`, `contributions/`). Supports resume by checking existing blobs before upload.
- **Factory** — Orchestrates data extraction across sources via async entry point.

### Database

PostgreSQL with SQLAlchemy 2.0 ORM and Alembic migrations. Tables are prefixed with `inat_` to separate iNaturalist data from future sources.

| Table | Rows | Purpose |
|-------|------|---------|
| `inat_taxa` | ~44K active | Taxonomy (taxon_id, ancestry, rank, name) |
| `inat_observations` | ~400M | Observations (UUID, location, date, quality grade, taxon FK) |
| `inat_photos` | ~413M | Photo metadata (UUID, dimensions, license, observation FK) |

Both sync (`psycopg2`) and async (`asyncpg`) engines are available. Alembic uses the sync engine for migrations; application code can use async for concurrent batch operations.

### Preprocessing (Multi-Channel Input) — TODO

Standard RGB augmented with additional channels (gradient magnitude, local contrast normalization) to improve detection of camouflaged fish against complex reef backgrounds.

### Model — TODO

Modified YOLO accepting multi-channel input. Pretrained RGB weights preserved; extra channels initialized for stable fine-tuning.

### Training — TODO

Multi-stage transfer learning: COCO pretraining → LILA fish detection fine-tuning → target domain adaptation.

### Quality Scoring — TODO

UIQM (Underwater Image Quality Measure) scoring for selective enhancement and curriculum training.

## Taxa Scope

- **Actinopterygii** (47178) — ray-finned fishes: ~50K taxa
- **Chondrichthyes** (196614) — sharks, rays, chimaeras: ~2K taxa
- Filtered to active taxa only: **43,991 taxa**, of which **15,564 have research-grade observations**

## Dataset

- **3.57M photos** across 2.36M research-grade observations
- **79.4% CC-BY-NC** licensed (project is non-commercial)

## Setup

Requires Python >= 3.12 and PostgreSQL.

```bash
cd whatsthatfish
pip install -e .
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://user:pass@host:port/dbname`) |
| `GCS_SECRET` | Path to GCS service account key (optional, falls back to default credentials) |

### Database Setup

```bash
# Generate migration (already done for initial inat_ tables)
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head
```

## Current Status

### Completed

- iNaturalist S3 data ingestion pipeline (taxa, observations, photos → parquet)
- Taxonomy filtering and dataset construction (3.57M photos, 15,564 taxa)
- LILA fish detection dataset downloader with bounding box validation
- GCS client with auth and resumable uploads
- Async data factory orchestration
- PostgreSQL database layer with SQLAlchemy 2.0 models (sync + async)
- Alembic migration for `inat_taxa`, `inat_observations`, `inat_photos` tables

### Next Steps

1. **S3-to-GCS photo transfer pipeline** — Async concurrent streaming of 3.57M photos from iNaturalist S3 to GCS bucket with resume tracking
2. **Database ingestion** — Bulk load parquet data into PostgreSQL tables
3. **UIQM quality scorer** — Underwater image quality metrics with vectorized computation
4. **Preprocessing pipeline** — Gradient maps, local contrast normalization, channel composition
5. **Multi-channel YOLO model** — Modified YOLO for 5+ channel input
6. **Training pipeline** — Multi-stage transfer learning with curriculum sampling
7. **COCO-to-YOLO conversion** — Convert LILA annotations for YOLO training
8. **Inference pipeline** — Image/video detection and crop export
9. **Evaluation framework** — mAP, recall, cross-domain generalization metrics
10. **Domain-balanced sampling** — Balance across LILA source datasets and handle rare taxa

## License

Non-commercial use. Dataset includes CC-BY-NC licensed content from iNaturalist.
