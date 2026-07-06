"""API schemas for the master worker coordination plane (architecture.md 3.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkerFaultView(BaseModel):
    """A fault attributed to a worker, surfaced read-only in the fleet view."""

    work_unit_id: str
    challenge_slug: str | None = None
    detail: str | None = None
    created_at: datetime


class WorkerView(BaseModel):
    """Fleet view of a worker: status, owner, provider, last-seen, faults."""

    worker_id: str
    worker_pubkey: str
    miner_hotkey: str
    provider: str
    provider_instance_ref: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    status: str
    last_heartbeat_at: datetime | None = None
    created_at: datetime
    faults: list[WorkerFaultView] = Field(default_factory=list)


class WorkerRegisterRequest(BaseModel):
    """Body for ``POST /v1/workers/register``.

    ``binding_signature`` is the miner's sr25519 signature over
    ``worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}``.
    """

    worker_pubkey: str
    miner_hotkey: str
    binding_signature: str
    nonce: str
    provider: str
    provider_instance_ref: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["gpu"])
    last_seen_meta: dict[str, Any] | None = None


class WorkerRegisterResponse(BaseModel):
    """Response for a successful worker registration."""

    worker: WorkerView
    heartbeat_ttl_seconds: int


class WorkerHeartbeatRequest(BaseModel):
    """Body for ``POST /v1/workers/{worker_id}/heartbeat``."""

    last_seen_meta: dict[str, Any] | None = None


class WorkerHeartbeatResponse(BaseModel):
    """Response for a successful worker heartbeat."""

    status: str
    now: datetime


class WorkerListResponse(BaseModel):
    """Response for ``GET /v1/workers`` and ``GET /v1/workers/active``."""

    workers: list[WorkerView] = Field(default_factory=list)
