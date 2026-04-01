from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
