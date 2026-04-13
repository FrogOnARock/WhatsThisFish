"""
Generate small, deterministic parquet fixtures for integration tests.

Run once to create the fixture files, then check them into the repo.
These mirror the real parquet schemas but with ~20 rows each, including
edge cases for filtering logic.

Usage:
    python -m tests.fixtures.generate_fixtures
"""

import uuid
import polars as pl
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent


def generate_taxa() -> pl.DataFrame:
    """~20 taxa rows covering:
    - Fish taxa under Actinopterygii (47178) ancestry — should be included
    - Fish taxa under Chondrichthyes (196614) ancestry — should be included
    - Non-fish taxa (insects, birds) — should be filtered out
    - Inactive fish taxon — should be filtered out
    """
    rows = [
        # ── Fish: Actinopterygii lineage ──
        # Root: Actinopterygii itself
        (47178, "48460/1/47178", 50.0, "class", "Actinopterygii", True),
        # Order under Actinopterygii
        (1000, "48460/1/47178/1000", 40.0, "order", "Perciformes", True),
        # Family
        (2000, "48460/1/47178/1000/2000", 30.0, "family", "Labridae", True),
        # Species — the kind of row we'll have observations for
        (3001, "48460/1/47178/1000/2000/3001", 10.0, "species", "Thalassoma lunare", True),
        (3002, "48460/1/47178/1000/2000/3002", 10.0, "species", "Coris gaimard", True),
        (3003, "48460/1/47178/1000/2000/3003", 10.0, "species", "Halichoeres hortulanus", True),
        # Inactive fish — should be filtered by active=True
        (3099, "48460/1/47178/1000/2000/3099", 10.0, "species", "Extinct wrasse", False),

        # ── Fish: Chondrichthyes lineage ──
        (196614, "48460/1/196614", 50.0, "class", "Chondrichthyes", True),
        (4000, "48460/1/196614/4000", 30.0, "family", "Carcharhinidae", True),
        (4001, "48460/1/196614/4000/4001", 10.0, "species", "Carcharhinus melanopterus", True),

        # ── Non-fish: should NOT appear after filtering ──
        # Insecta — ancestry does NOT contain 47178 or 196614
        (47158, "48460/1/47158", 50.0, "class", "Insecta", True),
        (5001, "48460/1/47158/5001", 10.0, "species", "Danaus plexippus", True),
        # Aves
        (3, "48460/1/2/355675/3", 50.0, "class", "Aves", True),
        (6001, "48460/1/2/355675/3/6001", 10.0, "species", "Corvus corax", True),
    ]
    return pl.DataFrame(
        rows,
        schema=["taxon_id", "ancestry", "rank_level", "rank", "name", "active"],
        orient="row",
    ).cast({"taxon_id": pl.Int64, "rank_level": pl.Float64})


def generate_observations() -> pl.DataFrame:
    """~15 observation rows:
    - Fish observations with quality_grade=research — should pass filter
    - Fish observation with quality_grade=casual — should be filtered out
    - Non-fish observation — should be filtered out by taxon_id join
    """
    rows = [
        # Fish observations (research grade) — should survive filtering
        ("obs-uuid-001", 100, 21.35, -158.13, "2024-01-15", 3001, "research"),
        ("obs-uuid-002", 101, -23.14, 113.77, "2024-02-20", 3002, "research"),
        ("obs-uuid-003", 102, 35.68, 139.69, "2024-03-10", 3003, "research"),
        ("obs-uuid-004", 103, -8.50, 115.26, "2024-04-05", 3001, "research"),
        ("obs-uuid-005", 104, 26.22, 127.68, "2024-05-12", 4001, "research"),  # shark
        # Duplicate observer, different taxon
        ("obs-uuid-006", 100, 21.36, -158.14, "2024-06-01", 3002, "research"),

        # Fish but casual quality — should be filtered out
        ("obs-uuid-007", 105, 10.00, 20.00, "2024-07-01", 3001, "casual"),

        # Non-fish observation — won't join on taxon_id
        ("obs-uuid-008", 106, 40.71, -74.01, "2024-08-15", 5001, "research"),
        ("obs-uuid-009", 107, 51.51, -0.13, "2024-09-20", 6001, "research"),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "observation_uuid", "observer_id", "latitude",
            "longitude", "observed_on", "taxon_id", "quality_grade",
        ],
        orient="row",
    ).cast({"observer_id": pl.Int64, "taxon_id": pl.Int64})

def generate_photos() -> pl.DataFrame:
    """~12 photo rows linked to observations.
    Multiple photos per observation to test the join fan-out.
    """
    rows = [
        # obs-uuid-001: 2 photos
        (str(uuid.UUID(int=1)), 10001, "obs-uuid-001", "jpg", "cc-by", 1024, 768, 0),
        (str(uuid.UUID(int=2)), 10002, "obs-uuid-001", "jpg", "cc-by", 1024, 768, 1),
        # obs-uuid-002: 1 photo
        (str(uuid.UUID(int=3)), 10003, "obs-uuid-002", "jpeg", "cc-by-nc", 800, 600, 0),
        # obs-uuid-003: 1 photo
        (str(uuid.UUID(int=4)), 10004, "obs-uuid-003", "jpg", "cc0", 2048, 1536, 0),
        # obs-uuid-004: 1 photo
        (str(uuid.UUID(int=5)), 10005, "obs-uuid-004", "png", "cc-by", 640, 480, 0),
        # obs-uuid-005: 1 photo (shark)
        (str(uuid.UUID(int=6)), 10006, "obs-uuid-005", "jpg", "cc-by", 1920, 1080, 0),
        # obs-uuid-006: 1 photo
        (str(uuid.UUID(int=7)), 10007, "obs-uuid-006", "jpg", "cc-by-nc", 1024, 768, 0),

        # obs-uuid-007: casual quality — photo exists but observation gets filtered
        (str(uuid.UUID(int=8)), 10008, "obs-uuid-007", "jpg", "cc-by", 800, 600, 0),
        # obs-uuid-008: insect — photo exists but observation gets filtered
        (str(uuid.UUID(int=9)), 10009, "obs-uuid-008", "jpg", "cc-by", 800, 600, 0),
    ]
    return pl.DataFrame(
        rows,
        schema=[
            "photo_uuid", "photo_id", "observation_uuid",
            "extension", "license", "width", "height", "position",
        ],
        orient="row",
    ).cast({"photo_id": pl.Int64, "width": pl.Int64, "height": pl.Int64, "position": pl.Int64})

def generate_lila_collected_images() -> pl.DataFrame:
    """~9 collected images across 3 LILA datasets covering:
    - Multiple datasets (salmon_cv, deep_fish, brackish) for domain filtering
    - Both is_train=True and is_train=False for train/val split
    - Images WITH annotations (positive frames)
    - Images WITHOUT annotations (negative frames — no fish detected)
    """
    rows = [
        # ── salmon_cv: dominant source in real etl ──
        ("salmon_cv/frame_00100.jpg", "salmon_cv", True),
        ("salmon_cv/frame_00200.jpg", "salmon_cv", True),
        ("salmon_cv/frame_00300.jpg", "salmon_cv", False),  # val split
        # Negative frame — no annotations will reference this
        ("salmon_cv/frame_00400.jpg", "salmon_cv", True),

        # ── deep_fish: smaller source ──
        ("deep_fish/img_0010.jpg", "deep_fish", True),
        ("deep_fish/img_0020.jpg", "deep_fish", False),  # val split

        # ── brackish: another source ──
        ("brackish/seq01_000100.jpg", "brackish", True),
        ("brackish/seq01_000200.jpg", "brackish", True),
        # Negative frame
        ("brackish/seq01_000300.jpg", "brackish", False),
    ]
    return pl.DataFrame(
        rows,
        schema=["file_name", "dataset", "is_train"],
        orient="row",
    )


def generate_lila_annotations() -> pl.DataFrame:
    """~10 annotations linked to collected images covering:
    - Single annotation per image (1 fish)
    - Multiple annotations per image (2 bboxes — e.g. school of fish)
    - Different category_ids (fish vs non-fish classes)
    - Normalized bbox coordinates (0-1 range, COCO-style xywh)

    NOTE: salmon_cv/frame_00400 and brackish/seq01_000300 have NO annotations
    (negative frames). Tests should verify these images exist without annotations.
    """
    rows = [
        # salmon_cv/frame_00100: 2 annotations (school of fish)
        (1, "salmon_cv/frame_00100.jpg", "1", 0.10, 0.20, 0.30, 0.25),
        (2, "salmon_cv/frame_00100.jpg", "1", 0.55, 0.40, 0.20, 0.15),

        # salmon_cv/frame_00200: 1 annotation
        (3, "salmon_cv/frame_00200.jpg", "1", 0.30, 0.35, 0.40, 0.30),

        # salmon_cv/frame_00300: 1 annotation (val split image)
        (4, "salmon_cv/frame_00300.jpg", "1", 0.15, 0.10, 0.25, 0.20),

        # deep_fish/img_0010: 1 annotation
        (5, "deep_fish/img_0010.jpg", "1", 0.45, 0.50, 0.35, 0.28),

        # deep_fish/img_0020: 2 annotations
        (6, "deep_fish/img_0020.jpg", "1", 0.05, 0.15, 0.20, 0.18),
        (7, "deep_fish/img_0020.jpg", "2", 0.60, 0.55, 0.15, 0.12),  # different category

        # brackish/seq01_000100: 1 annotation
        (8, "brackish/seq01_000100.jpg", "1", 0.25, 0.30, 0.30, 0.22),

        # brackish/seq01_000200: 1 annotation
        (9, "brackish/seq01_000200.jpg", "1", 0.40, 0.45, 0.25, 0.20),

        # No annotations for salmon_cv/frame_00400 (negative)
        # No annotations for brackish/seq01_000300 (negative)
    ]
    return pl.DataFrame(
        rows,
        schema=["id", "image_id", "category_id", "x", "y", "w", "h"],
        orient="row",
    ).cast({"id": pl.Int64, "x": pl.Float64, "y": pl.Float64, "w": pl.Float64, "h": pl.Float64})


def main():
    taxa = generate_taxa()
    obs = generate_observations()
    photos = generate_photos()
    images = generate_lila_collected_images()
    annotations = generate_lila_annotations()

    taxa.write_parquet(FIXTURES_DIR / "taxa.parquet")
    obs.write_parquet(FIXTURES_DIR / "observations.parquet")
    photos.write_parquet(FIXTURES_DIR / "photos.parquet")
    images.write_parquet(FIXTURES_DIR / "collected_images.parquet")
    annotations.write_parquet(FIXTURES_DIR / "annotations.parquet")

    print(f"Generated fixtures in {FIXTURES_DIR}:")
    print(f"  taxa.parquet:         {taxa.shape[0]} rows, {taxa.shape[1]} cols")
    print(f"  observations.parquet: {obs.shape[0]} rows, {obs.shape[1]} cols")
    print(f"  photos.parquet:       {photos.shape[0]} rows, {photos.shape[1]} cols")

    # Show expected output after filtering
    # Fish taxa: 47178, 1000, 2000, 3001, 3002, 3003, 196614, 4000, 4001 (9 active)
    # Research fish observations: obs-001 through obs-006 (6 rows)
    # Photos joining those observations: 7 photos
    print(f"\nExpected after pipeline:")
    print(f"  Active fish taxa:     9")
    print(f"  Research-grade obs:   6")
    print(f"  Final photo records:  7")


if __name__ == "__main__":
    main()
