"""API schemas for the assignment coordination plane (pull/progress/result)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AssignmentView(BaseModel):
    """Public view of a coordinated work-unit assignment for a validator."""

    id: str
    challenge_slug: str
    work_unit_id: str
    submission_ref: str
    payload: dict[str, Any] = Field(default_factory=dict)
    required_capability: str
    status: str
    attempt_count: int
    max_attempts: int
    deadline_at: datetime | None = None
    last_progress_at: datetime | None = None
    checkpoint_ref: str | None = None


class AssignmentPullResponse(BaseModel):
    """Response for ``POST /v1/assignments/pull``."""

    assignments: list[AssignmentView] = Field(default_factory=list)


class AssignmentProgressRequest(BaseModel):
    """Body for ``POST /v1/assignments/{id}/progress``."""

    checkpoint_ref: str | None = None
    meta: dict[str, Any] | None = None


class AssignmentProgressResponse(BaseModel):
    """Response for a successful progress heartbeat."""

    status: str
    deadline_at: datetime | None = None
    last_progress_at: datetime | None = None
    checkpoint_ref: str | None = None


class AssignmentResultRequest(BaseModel):
    """Body for ``POST /v1/assignments/{id}/result``."""

    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    checkpoint_ref: str | None = None


class AssignmentResultResponse(BaseModel):
    """Response for a result post (idempotent when already terminal)."""

    status: str
    result_ref: str | None = None
    idempotent: bool = False
