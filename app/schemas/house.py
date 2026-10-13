"""Pydantic schemas.

The 8 features below were read from the saved pipeline's ``feature_names_in_``
(see inspect_model.py). They are the California Housing block-group features.
Bounds are deliberately a little wider than the real data ranges, to reject
nonsense (negative income, Latitude 500, ...) without rejecting valid edge cases.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """Request body for POST /predict. Unknown or missing fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",  # unexpected fields -> 422
        strict=True,  # "3.5" (a string) is rejected; ints are accepted for floats
    )

    MedInc: float = Field(
        ..., gt=0, le=20, allow_inf_nan=False,
        title="Median income",
        description="Median income of the block group, in tens of thousands of dollars (3.5 = $35,000).",
        examples=[8.3252],
    )
    HouseAge: float = Field(
        ..., ge=0, le=100, allow_inf_nan=False,
        title="House age",
        description="Median house age in the block group, in years.",
        examples=[41],
    )
    AveRooms: float = Field(
        ..., gt=0, le=200, allow_inf_nan=False,
        title="Average rooms",
        description="Average number of rooms per household.",
        examples=[6.9841],
    )
    AveBedrms: float = Field(
        ..., gt=0, le=50, allow_inf_nan=False,
        title="Average bedrooms",
        description="Average number of bedrooms per household.",
        examples=[1.0238],
    )
    Population: float = Field(
        ..., gt=0, le=50_000, allow_inf_nan=False,
        title="Population",
        description="Number of people living in the block group.",
        examples=[322],
    )
    AveOccup: float = Field(
        ..., gt=0, le=2_000, allow_inf_nan=False,
        title="Average occupancy",
        description="Average number of household members.",
        examples=[2.5556],
    )
    Latitude: float = Field(
        ..., ge=32, le=43, allow_inf_nan=False,
        title="Latitude",
        description="Block group latitude. The model was trained on California only (about 32.5 to 42.0).",
        examples=[37.88],
    )
    Longitude: float = Field(
        ..., ge=-125, le=-114, allow_inf_nan=False,
        title="Longitude",
        description="Block group longitude. California only (about -124.4 to -114.3).",
        examples=[-122.23],
    )


# Order matters: the model is fed columns in exactly this order.
FEATURE_NAMES: tuple[str, ...] = tuple(HouseFeatures.model_fields)


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_price: float = Field(..., description="Estimated price in `currency`.")
    currency: str
    created_at: datetime
    features: HouseFeatures

    @classmethod
    def from_record(cls, record: Any, currency: str) -> "PredictionResponse":
        created = record.created_at
        if created.tzinfo is None:  # stored as naive UTC
            created = created.replace(tzinfo=timezone.utc)
        # model_construct: do not re-validate old rows against today's bounds.
        features = HouseFeatures.model_construct(
            **{name: getattr(record, name) for name in FEATURE_NAMES}
        )
        return cls(
            prediction_id=record.id,
            predicted_price=record.predicted_price,
            currency=currency,
            created_at=created,
            features=features,
        )


class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    total: int
    limit: int
    offset: int


class FeatureInfo(BaseModel):
    name: str
    title: str
    description: str
    minimum: float | None = None
    exclusive_minimum: float | None = None
    maximum: float | None = None
    example: float | None = None


class ModelInfoResponse(BaseModel):
    currency: str
    price_unit_multiplier: float
    features: list[FeatureInfo]


def describe_features() -> list[FeatureInfo]:
    """Feature metadata derived from the schema above (single source of truth)."""
    props = HouseFeatures.model_json_schema()["properties"]
    out: list[FeatureInfo] = []
    for name in FEATURE_NAMES:
        p = props[name]
        examples = p.get("examples") or [None]
        out.append(
            FeatureInfo(
                name=name,
                title=p.get("title", name),
                description=p.get("description", ""),
                minimum=p.get("minimum"),
                exclusive_minimum=p.get("exclusiveMinimum"),
                maximum=p.get("maximum"),
                example=examples[0],
            )
        )
    return out
