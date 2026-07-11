"""Canonical challenge SDK wire schemas."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from base.schemas.assignment import (
    AssignmentProgressRequest,
    AssignmentProgressResponse,
    AssignmentPullResponse,
    AssignmentResultRequest,
    AssignmentResultResponse,
    AssignmentView,
)
from base.schemas.worker import ExecutionProof, ProviderInfo, WorkerSignature


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_StrictModel):
    status: str = "ok"
    slug: str
    version: str


class VersionResponse(_StrictModel):
    distribution_name: str
    artifact_version: str
    release_id: str
    api_version: str
    challenge_version: str
    sdk_contract_version: str
    sdk_version: str
    role: str = "challenge"
    capabilities: list[str] = Field(default_factory=list)


class WeightsResponse(_StrictModel):
    challenge_slug: str
    epoch: int | None = None
    weights: dict[str, float]
    metadata: dict[str, str] = Field(
        default_factory=lambda: {"computed_at": datetime.now(UTC).isoformat()}
    )


__all__ = [
    "AssignmentProgressRequest",
    "AssignmentProgressResponse",
    "AssignmentPullResponse",
    "AssignmentResultRequest",
    "AssignmentResultResponse",
    "AssignmentView",
    "ExecutionProof",
    "HealthResponse",
    "ProviderInfo",
    "VersionResponse",
    "WeightsResponse",
    "WorkerSignature",
]
