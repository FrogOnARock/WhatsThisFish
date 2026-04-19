from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String, func, DateTime, JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql.schema import UniqueConstraint
from whatsthatfish.src.database.base import Base


class InatTaxa(Base):
    __tablename__ = "inat_taxa"

    taxon_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ancestry: Mapped[str | None] = mapped_column(String)
    rank_level: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[str | None] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    filtered_observations: Mapped[list["InatFilteredObservations"]] = relationship(
        back_populates="taxon"
    )

    __table_args__ = (
        Index("ix_inat_taxa_rank_level", "rank_level"),
        Index("ix_inat_taxa_ancestry_active", "ancestry", "active"),
    )


class InatFilteredObservations(Base):
    """Pre-filtered iNaturalist photo records for model training.

    Populated by lazy-scanning local parquets (taxa, observations, photos),
    filtering to in-scope taxa (Actinopterygii + Chondrichthyes),
    research-grade observations, and active taxa, then bulk inserting.
    """
    __tablename__ = "inat_filtered_observations"

    photo_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    photo_id: Mapped[int] = mapped_column(BigInteger)
    observation_uuid: Mapped[str] = mapped_column(String(36))
    observer_id: Mapped[int] = mapped_column(BigInteger)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    taxon_id: Mapped[int] = mapped_column(BigInteger,
                                          ForeignKey("inat_taxa.taxon_id"))
    observed_on: Mapped[str | None] = mapped_column(Date)
    extension: Mapped[str] = mapped_column(String(10))
    license: Mapped[str] = mapped_column(String(20))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)

    taxon: Mapped["InatTaxa | None"] = relationship(back_populates="filtered_observations")

    __table_args__ = (
        Index("ix_inat_filtered_observations_latitude_longitude", "latitude", "longitude"),
        Index("ix_inat_filtered_observations_observed_on", "observed_on"),
        Index("ix_inat_filtered_observations_taxon_id", "taxon_id"),
    )


class LilaAnnotations(Base):
    __tablename__ = "lila_annotations"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    image_id: Mapped[str] = mapped_column(String(255),
                                          ForeignKey("lila_collected_images.id"))
    category_id: Mapped[str] = mapped_column(String(255))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)

    collected_images: Mapped["LilaCollectedImages"] = relationship(back_populates="annotations")

    __table_args__ = (
        Index("ix_lila_annotations_image_id", "image_id"),
        Index("ix_lila_annotations_category_id", "category_id")
    )


class LilaCollectedImages(Base):
    __tablename__ = "lila_collected_images"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    file_name: Mapped[str] = mapped_column(String(255), unique=True)
    dataset: Mapped[str] = mapped_column(String(255))
    is_train: Mapped[bool] = mapped_column(Boolean)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    annotations: Mapped[list["LilaAnnotations"]] = relationship(back_populates="collected_images")

    __table_args__ = (
        Index("ix_lila_collected_images_file_name", "file_name"),
        Index("ix_lila_collected_images_dataset", "dataset")
    )


class LilaYolo(Base):
    __tablename__ = "lila_yolo"

    file_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    annotation: Mapped[dict[str, float]] = mapped_column(JSONB)

    __table_args__ = (
        Index("ix_lila_yolo_file_name", "file_name"),
    )



class SuccessfulUploads(Base):
    __tablename__ = "successful_uploads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    identifier: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(255))
    uploaded_at: Mapped[str] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("identifier", "source", name="uq_identifier_source"),
        Index("ix_successful_uploads_source", "source"),
    )

