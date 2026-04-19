"""One-time backfill: populate width/height on lila_collected_images from COCO JSON.

Parses the COCO JSON images array, matches by image ID, and batch-updates
rows where width or height IS NULL. Safe to re-run — skips already-populated rows.

Usage:
    DATABASE_URL=postgresql://... python -m whatsthatfish.scripts.backfill_image_dimensions
"""

import json
import logging
from pathlib import Path

from sqlalchemy import text

from whatsthatfish.src.database.config import get_engine
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Both copies are identical; prefer the etl path used by LilaDataset
COCO_JSON_PATH = Path(__file__).parents[1] / "src" / "etl" / "metadata" / "lila" / "community_fish_detection_dataset.json"
BATCH_SIZE = 5_000


def parse_image_dimensions(json_path: Path) -> dict[str, tuple[int, int]]:
    """Extract {image_id: (width, height)} from COCO JSON images array."""
    logger.info("Loading COCO JSON from %s (this may take a moment)...", json_path)
    with open(json_path) as f:
        coco = json.load(f)

    dimensions = {
        str(img["id"]): (img["width"], img["height"])
        for img in coco["images"]
    }
    logger.info("Parsed dimensions for %s images", f"{len(dimensions):,}")
    return dimensions


def backfill(dimensions: dict[str, tuple[int, int]]) -> None:
    """Batch-update lila_collected_images rows where width/height IS NULL."""
    engine = get_engine()

    with engine.connect() as conn:
        null_ids = conn.execute(
            text("SELECT id FROM lila_collected_images WHERE width IS NULL OR height IS NULL")
        ).scalars().all()

    if not null_ids:
        logger.info("No rows need backfilling — all width/height values are populated")
        return

    # Filter to only IDs we have dimensions for
    updates = []
    missing = []
    for img_id in null_ids:
        if img_id in dimensions:
            w, h = dimensions[img_id]
            updates.append({"img_id": img_id, "w": w, "h": h})
        else:
            missing.append(img_id)

    if missing:
        logger.warning(
            "%s rows in DB have no matching COCO JSON entry — skipping: %s",
            f"{len(missing):,}", missing[:10]
        )

    logger.info("Backfilling %s rows in batches of %s", f"{len(updates):,}", f"{BATCH_SIZE:,}")

    updated_total = 0
    with engine.begin() as conn:
        for i in range(0, len(updates), BATCH_SIZE):
            batch = updates[i : i + BATCH_SIZE]
            # executemany-style parameterized UPDATE
            conn.execute(
                text(
                    "UPDATE lila_collected_images "
                    "SET width = :w, height = :h "
                    "WHERE id = :img_id"
                ),
                batch,
            )
            updated_total += len(batch)
            logger.info("Updated %s / %s rows", f"{updated_total:,}", f"{len(updates):,}")

    logger.info("Backfill complete: %s rows updated", f"{updated_total:,}")


def main() -> None:
    if not COCO_JSON_PATH.exists():
        raise FileNotFoundError(
            f"COCO JSON not found at {COCO_JSON_PATH}. "
            f"Run LilaDataset._download_coco_json() first."
        )

    dimensions = parse_image_dimensions(COCO_JSON_PATH)
    backfill(dimensions)


if __name__ == "__main__":
    main()
