import os
from enum import Enum
from pathlib import Path
from sqlalchemy import select, insert
from sqlalchemy.orm import sessionmaker
import polars as pl
from dotenv import load_dotenv
from ..retry import db_retry
from ..database.models import LilaAnnotations, LilaCollectedImages, LilaYolo
from ..config import _get_logger, get_model_config
from ..database.config import get_session_factory

load_dotenv()
logger = _get_logger("AnnotationConverter")

class RuntimeConfig(str, Enum):
    CONVERT = "convert"
    WRITE_TEXT = "write_text"
    ALL = "all"

class AnnotationConverter:
    def __init__(self,
                 session_factory: sessionmaker):
        self._session_factory = session_factory
        self._train_path = Path(__file__).parents[1] / "data" / "yolo" / "train"
        self._test_path = Path(__file__).parents[1] / "data" / "yolo" / "test"
        self.config = get_model_config().yolo

    @db_retry
    def _retrieve_annotations(self) -> pl.DataFrame:

        with self._session_factory() as session:
            rows = session.execute(
                select(LilaCollectedImages.file_name, LilaCollectedImages.is_train,
                       LilaCollectedImages.width, LilaCollectedImages.height,
                       LilaAnnotations.image_id, LilaAnnotations.category_id, LilaAnnotations.x,
                       LilaAnnotations.y, LilaAnnotations.w, LilaAnnotations.h
                       ).outerjoin(LilaAnnotations, LilaAnnotations.image_id == LilaCollectedImages.id)
            )
            df = pl.DataFrame(rows, schema=["file_name", "is_train", "width", "height",
                                            "image_id", "category_id",
                                            "x", "y", "w", "h"])

            return df


    def _convert_annotations(self, df: pl.DataFrame) -> pl.DataFrame:

        df = (
            df
            .with_columns(
                ((pl.col("x") + pl.col("w") / 2) / pl.col("width")).alias("norm_center_x"),
                ((pl.col("y") + pl.col("h") / 2) / pl.col("height")).alias("norm_center_y"),
                (pl.col("w") / pl.col("width")).alias("norm_width"),
                (pl.col("h") / pl.col("height")).alias("norm_height")
            ).drop(["x", "y", "w", "h"])
        )

        return df

    def _check_converted(self):
        with self._session_factory() as session:
            return session.execute(select(LilaYolo.file_name)).scalars().all()

    @db_retry
    def _write_annotations(self,
                           df: pl.DataFrame,
                           batch_size: int = 5_000):

        existing = set(self._check_converted())
        df = df.filter(~pl.col("file_name").is_in(existing))

        if df.height == 0:
            logger.info("No new annotations to insert")
            return

        # Group bbox rows by image into a list of bbox dicts per file_name.
        # Each row in df is one bbox — an image with 3 fish has 3 rows.
        # After grouping, each row is one image with annotation = [bbox, bbox, ...].
        bbox_cols = ["class_id", "norm_center_x", "norm_center_y", "norm_width", "norm_height", "is_train"]
        grouped = (
            df.with_columns(
                pl.when(
                ~(pl.col("category_id") == 1)).then(pl.lit(1))
                .otherwise(pl.lit(0)).alias("class_id"))
            .group_by("file_name")
            .agg(pl.struct(bbox_cols).alias("annotation"))
        )

        rows = [
            {
                "file_name": row["file_name"],
                "annotation": [bbox for bbox in row["annotation"]],
            }
            for row in grouped.iter_rows(named=True)
        ]

        logger.info(f"Writing {len(rows):,} images to lila_yolo")

        inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            with self._session_factory() as session:
                session.execute(insert(LilaYolo), batch)
                session.commit()
            inserted += len(batch)
            logger.info(f"Inserted {inserted:,} / {len(rows):,} images")

    @db_retry
    def _write_text(self):

        with self._session_factory() as session:
            rows = session.execute(select(LilaYolo))
            df = pl.DataFrame(rows, schema=["file_name", "annotation"])


        train_path = self.config.data_paths["obj_training_set"]
        test_path = self.config.data_paths["obj_test_set"]

        if not os.path.exists(train_path):
            Path.mkdir(Path(train_path), exist_ok=True)
        if not os.path.exists(test_path):
            Path.mkdir(Path(test_path), exist_ok=True)


        for row in df.to_dicts():
            file_name = row["file_name"]
            stem = Path(file_name).stem
            annotations = row["annotation"]
            path = train_path if annotations[0]["is_train"] else test_path

            with open(f"{path}{stem}.txt", "w") as f:
                for ann in annotations:
                    if ann["class_id"] == 0:
                        f.write(
                            f"{ann['class_id']} {ann['norm_center_x']} {ann['norm_center_y']} {ann['norm_width']} {ann['norm_height']}\n")
                    else:
                        f.write("")

    def run(self, runtime_config: RuntimeConfig = RuntimeConfig.CONVERT):

        if runtime_config == 'convert':
            df = self._retrieve_annotations()
            df = self._convert_annotations(df)
            self._write_annotations(df)
        elif runtime_config == 'write_text':
            self._write_text()
        else:
            df = self._retrieve_annotations()
            df = self._convert_annotations(df)
            self._write_annotations(df)
            self._write_text()


if __name__ == "__main__":
    session_factory = get_session_factory()
    AnnotationConverter(session_factory).run(runtime_config=RuntimeConfig.ALL)












