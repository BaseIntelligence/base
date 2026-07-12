from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

MASTER_WEIGHTS_FRESHNESS_SECONDS = 720


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


class SourceOutcome(BaseModel):
    """Versioned per-challenge source outcome recorded at epoch seal/withhold."""

    challenge_slug: str
    outcome: str
    reason_code: str
    snapshot_id: str | None = None
    payload_digest: str | None = None
    revision: int | None = None


class SourceRef(BaseModel):
    """Provenance reference to a durable raw-weight snapshot."""

    challenge_slug: str
    snapshot_id: str
    payload_digest: str
    outcome: str = "accepted"


class MasterWeightsResponse(BaseModel):
    """Canonical final vector served to validators from durable storage.

    New fields are optional with defaults so existing test fixtures continue to
    construct responses. Production seal paths always populate provenance.
    """

    protocol_version: str = "1.0"
    vector_id: str | None = None
    vector_digest: str | None = None
    epoch: int | None = None
    revision: int = 1
    netuid: int
    chain_endpoint: str
    uids: list[int]
    weights: list[float]
    hotkey_weights: dict[str, float] = Field(default_factory=dict)
    chain_domain_bytes: str | None = None
    computed_at: datetime
    expires_at: datetime
    source_challenges: list[ChallengeWeightsResult] = Field(default_factory=list)
    source_snapshots: list[SourceRef] = Field(default_factory=list)
    source_outcomes: list[SourceOutcome] = Field(default_factory=list)
    emission_policy_version: str | None = None
    emission_shares: dict[str, float] = Field(default_factory=dict)
    burn_policy_version: str | None = None
    mapping_policy_version: str | None = None
    metagraph_identity: dict[str, Any] = Field(default_factory=dict)
    metagraph_hash: str | None = None
    metagraph_block: int | None = None
    burn_outcome: bool | None = None
    metagraph_updated_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_not_expired(cls, value: datetime) -> datetime:
        if value <= datetime.now(UTC):
            raise ValueError("expires_at must be in the future")
        return value


class ValidatorSubmissionObservationRequest(BaseModel):
    """Validator-reported chain outcome for an immutable master vector.

    Non-authoritative: the master stores the observation without claiming chain
    finality and never performs set_weights.
    """

    vector_id: str
    vector_digest: str
    netuid: int
    chain_endpoint: str = ""
    outcome: str
    attempt: int = Field(ge=1)
    error_code: str | None = None
    observed_at: datetime | None = None

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, value: str) -> str:
        allowed = {"accepted", "rejected", "unknown", "retry_exhausted", "superseded"}
        cleaned = str(value).strip().lower()
        if cleaned not in allowed:
            raise ValueError(f"outcome must be one of {sorted(allowed)}")
        return cleaned

    @field_validator("vector_id", "vector_digest")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned


class ValidatorSubmissionObservationResponse(BaseModel):
    observation_id: str
    validator_hotkey: str
    vector_id: str
    vector_digest: str
    outcome: str
    attempt: int
    created_at: datetime
    idempotent: bool = False
