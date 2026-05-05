from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ChallengeWeightsResponse(BaseModel):
    challenge_slug: str
    epoch: int | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChallengeWeightsResult(BaseModel):
    slug: str
    emission_percent: float
    weights: dict[str, float] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None


class FinalWeights(BaseModel):
    uids: list[int]
    weights: list[float]
    hotkey_weights: dict[str, float] = Field(default_factory=dict)
