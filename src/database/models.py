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
from .base import Base


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


class InatCaptureContext(Base):
    """Underwater vs above-water classification for iNaturalist images.

    iNat fish observations include underwater dive shots, fishing-deck photos,
    aquarium shots, market photos, and lab specimens. For a dive-companion
    classifier we want to filter to underwater captures only — UIQM does not
    solve this (it actively rewards above-water lighting/color).

    Stores raw per-channel means alongside the derived `is_underwater` verdict
    so the threshold can be re-tuned (or the heuristic upgraded to CLIP) via
    UPDATE statements rather than re-scoring 1M images.

    Score columns are nullable: a row with NULL means/verdict represents an
    image that was attempted but unscoreable (corrupt file, etc.).

    Capture-context filter is applied *before* UIQM filtering during
    per-class downsampling — see preprocessing/working_notes.md.
    """
    __tablename__ = "inat_capture_context"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    mean_r: Mapped[float | None] = mapped_column(Float)
    mean_g: Mapped[float | None] = mapped_column(Float)
    mean_b: Mapped[float | None] = mapped_column(Float)
    stddev: Mapped[float | None] = mapped_column(Float)
    is_underwater: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_inat_capture_context_is_underwater", "is_underwater"),
    )


class InatImageQuality(Base):
    """UIQM quality scores for iNaturalist images.

    Mirrors LilaImageQuality structurally so cross-source distribution
    queries (UNION ALL with a source literal) are straightforward, but
    keyed on photo_uuid since that is the natural identifier in the
    iNat source table.

    Score columns are nullable: a row with NULL scores represents a
    photo that was attempted but unscoreable (corrupt file, too small,
    etc.) — distinct from "not yet scored," which means no row exists.
    """
    __tablename__ = "inat_image_quality"

    photo_uuid: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("inat_filtered_observations.photo_uuid"),
        primary_key=True,
    )
    uicm: Mapped[float | None] = mapped_column(Float)
    uism: Mapped[float | None] = mapped_column(Float)
    uiconm: Mapped[float | None] = mapped_column(Float)
    uiqm: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_inat_image_quality_uiqm", "uiqm"),
    )


class LilaImageQuality(Base):
    """UIQM quality scores for LILA images.

    Populated by a one-time scoring pass over locally available LILA images.
    Sub-scores (uicm, uism, uiconm) are kept alongside the composite so that
    threshold experiments can re-weight components without re-scoring.

    Score columns are nullable: a row with NULL scores represents an image
    that was attempted but unscoreable (corrupt file, too small, etc.) —
    distinct from "not yet scored," which means no row exists.
    """
    __tablename__ = "lila_image_quality"

    file_name: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("lila_collected_images.file_name"),
        primary_key=True,
    )
    uicm: Mapped[float | None] = mapped_column(Float)
    uism: Mapped[float | None] = mapped_column(Float)
    uiconm: Mapped[float | None] = mapped_column(Float)
    uiqm: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        Index("ix_lila_image_quality_uiqm", "uiqm"),
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

