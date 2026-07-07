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


class WorkerReplicaView(BaseModel):
    """One worker replica of a gpu unit, surfaced in the unit-status view."""

    worker_id: str
    miner_hotkey: str
    status: str
    manifest_sha256: str | None = None
    has_proof: bool = False


class WorkerAuditUnitView(BaseModel):
    """The validator AUDIT unit linked to a disputed gpu unit.

    ``outcome`` is the audit's terminal state: ``pending`` (not yet resolved),
    ``mismatch-resolved`` (resolved with a fault attributed to the divergent
    worker(s)), or ``passed`` (resolved with no fault).
    """

    work_unit_id: str
    executor_kind: str
    outcome: str


class WorkerUnitStatusView(BaseModel):
    """Per primary gpu unit status for operator dispute discovery (VAL-CROSS-011).

    Exposes the unit id + status (INCLUDING ``disputed``), its replicas
    (worker/owner/manifest/proof-presence), and, when disputed, the linked
    validator audit unit with its terminal outcome -- so the full dispute ->
    audit -> invalidation -> fault chain is reconstructable via APIs alone.
    """

    work_unit_id: str
    challenge_slug: str
    submission_ref: str
    status: str
    replicas: list[WorkerReplicaView] = Field(default_factory=list)
    audit: WorkerAuditUnitView | None = None


class WorkerUnitStatusListResponse(BaseModel):
    """Response for ``GET /v1/workers/units``."""

    units: list[WorkerUnitStatusView] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    """Provider/pod identity carried by an ExecutionProof (architecture 3.4)."""

    name: str
    executor_id: str | None = None
    pod_id: str | None = None
    miner_hotkey: str | None = None


class WorkerSignature(BaseModel):
    """The worker's sr25519 signature binding a manifest hash to a work unit."""

    worker_pubkey: str
    sig: str


class ExecutionProof(BaseModel):
    """Proof envelope attached to every worker result (architecture 3.4).

    Tier 0 (mandatory, all backends) carries the deterministic
    ``manifest_sha256`` plus the worker's sr25519 ``worker_signature`` over the
    pinned message (``sha256(f"{manifest_sha256}:{unit_id}")``). Tier 1 adds
    ``image_digest`` + a populated ``provider`` block; tier 2 adds a non-null
    ``attestation``. The base worker plane emits tier 0; prism fills the higher
    tiers.
    """

    version: int = 1
    tier: int = 0
    manifest_sha256: str
    image_digest: str | None = None
    provider: ProviderInfo | None = None
    worker_signature: WorkerSignature
    attestation: dict[str, Any] | None = None
