from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String, func, DateTime
)
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

    observations: Mapped[list["InatObservation"]] = relationship(
        back_populates="taxon"
    )

    __table_args__ = (
        Index("ix_inat_rank_level", "rank_level"),
        Index("ix_inat_ancestry_active", "ancestry", "active"),
    )


class InatObservation(Base):
    __tablename__ = "inat_observations"

    observation_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    observer_id: Mapped[int | None] = mapped_column(BigInteger)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    positional_accuracy: Mapped[int | None] = mapped_column(Integer)
    taxon_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("inat_taxa.taxon_id")
    )
    quality_grade: Mapped[str | None] = mapped_column(String(20))
    observed_on: Mapped[str | None] = mapped_column(Date)
    anomaly_score: Mapped[float | None] = mapped_column(Float)

    taxon: Mapped["InatTaxa | None"] = relationship(back_populates="observations")
    photos: Mapped[list["InatPhoto"]] = relationship(back_populates="observation")

    __table_args__ = (
        Index("ix_inat_taxa_id_quality", "taxon_id", "quality_grade"),
        Index("ix_nat_observed_on", "observed_on"),
        Index("ix_nat_lat_long", "latitude", "longitude"),
    )


class InatPhoto(Base):
    __tablename__ = "inat_photos"

    photo_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    photo_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    observation_uuid: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("inat_observations.observation_uuid")
    )
    observer_id: Mapped[int | None] = mapped_column(BigInteger)
    extension: Mapped[str | None] = mapped_column(String(10))
    license: Mapped[str | None] = mapped_column(String(20))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)

    observation: Mapped["InatObservation | None"] = relationship(
        back_populates="photos"
    )

    __table_args__ = (
        Index("ix_inat_photos_observation_uuid", "observation_uuid"),
        Index("ix_inat_photos_photo_id", "photo_id"),
    )

class LilaAnnotations(Base):
    __tablename__ = "lila_annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[str] = mapped_column(String(255),
                                          ForeignKey("lila_collected_images.file_name"))
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

    file_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    dataset: Mapped[str] = mapped_column(String(255))
    is_train: Mapped[bool] = mapped_column(Boolean)

    annotations: Mapped[list["LilaAnnotations"]] = relationship(back_populates="collected_images")

    __table_args__ = (
        Index("ix_lila_collected_images_file_name", "file_name"),
        Index("ix_lila_collected_images_dataset", "dataset")
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

class FilteredObservationsView(Base):
    """Read-only ORM mapping for the filtered_observations materialized view.

    Created and refreshed via raw SQL in Alembic migrations — not managed
    by Base.metadata.create_all(). The is_view info flag tells Alembic's
    include_object hook to skip this during autogeneration.
    """
    __tablename__ = "filtered_observations_vw"

    photo_uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    photo_id: Mapped[int] = mapped_column(BigInteger)
    observation_uuid: Mapped[str] = mapped_column(String(36))
    observer_id: Mapped[int] = mapped_column(BigInteger)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    taxon_id: Mapped[int] = mapped_column(BigInteger)
    observed_on: Mapped[str | None] = mapped_column(Date)
    extension: Mapped[str] = mapped_column(String(10))
    license: Mapped[str] = mapped_column(String(20))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    position: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        {"info": {"is_view": True}}
    )