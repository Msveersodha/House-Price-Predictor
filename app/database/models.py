"""ORM model for the prediction history table.

The model has 8 fixed numeric features, so each one gets its own typed column
(easy to query, index and constrain) instead of a JSON blob. Column names are
exactly the model's feature names, which removes any name-mapping bugs.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Float, Index, Integer, MetaData
from sqlalchemy.dialects.mssql import DATETIME2
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit constraint names (SQL Server needs unique, stable names).
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _utcnow() -> datetime:
    """Naive UTC timestamp (DATETIME2 has no timezone)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        # Loose sanity checks only; strict domain validation lives in the API schema.
        CheckConstraint("MedInc > 0", name="medinc_positive"),
        CheckConstraint("HouseAge >= 0", name="houseage_nonneg"),
        CheckConstraint("AveRooms > 0", name="averooms_positive"),
        CheckConstraint("AveBedrms > 0", name="avebedrms_positive"),
        CheckConstraint("Population > 0", name="population_positive"),
        CheckConstraint("AveOccup > 0", name="aveoccup_positive"),
        CheckConstraint("Latitude BETWEEN -90 AND 90", name="latitude_range"),
        CheckConstraint("Longitude BETWEEN -180 AND 180", name="longitude_range"),
        Index("ix_predictions_created_at", "created_at"),
    )

    # IDENTITY(1,1) on SQL Server
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Input features (names match the trained pipeline's feature_names_in_)
    MedInc: Mapped[float] = mapped_column(Float, nullable=False)
    HouseAge: Mapped[float] = mapped_column(Float, nullable=False)
    AveRooms: Mapped[float] = mapped_column(Float, nullable=False)
    AveBedrms: Mapped[float] = mapped_column(Float, nullable=False)
    Population: Mapped[float] = mapped_column(Float, nullable=False)
    AveOccup: Mapped[float] = mapped_column(Float, nullable=False)
    Latitude: Mapped[float] = mapped_column(Float, nullable=False)
    Longitude: Mapped[float] = mapped_column(Float, nullable=False)

    # Raw model output (units of $100,000) and the converted price
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_price: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime().with_variant(DATETIME2(3), "mssql"),
        nullable=False,
        default=_utcnow,
    )
