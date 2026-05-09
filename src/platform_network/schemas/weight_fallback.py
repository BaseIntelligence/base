from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class SignedWeightsPayload(BaseModel):
    challenge_slug: str | None = None
    epoch: int | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SignedWeightsResponse(BaseModel):
    payload: SignedWeightsPayload
    signature: str
