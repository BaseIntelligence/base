"""Canonical, strict challenge SDK wire schemas."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from math import isfinite
from re import fullmatch
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorResponse(_StrictModel):
    code: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    detail: str = Field(min_length=1, max_length=512)
    correlation_id: str = Field(min_length=1, max_length=128)
    expected: str | None = None
    actual: str | None = None


class AssignmentView(_StrictModel):
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    assignment_id: str = Field(min_length=1)
    work_unit_id: str = Field(min_length=1)
    submission_ref: str = Field(min_length=1)
    challenge_slug: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_capability: str = Field(min_length=1)
    revision: StrictInt = Field(ge=1)
    attempt: StrictInt = Field(ge=1)
    status: Literal["assigned", "running", "completed", "failed", "expired", "disputed"]
    lease_deadline: datetime | None = None
    checkpoint_ref: str | None = None


class EmptyObject(_StrictModel):
    """Canonical empty JSON request body."""


class ValidatorView(_StrictModel):
    hotkey: str = Field(min_length=1)
    uid: StrictInt | None = None
    status: str = Field(min_length=1)
    capabilities: tuple[str, ...] = ()
    subscriptions: tuple[str, ...] = ()
    version: str | None = None
    registered_at: datetime
    last_heartbeat_at: datetime | None = None
    last_seen_meta: dict[str, Any] = Field(default_factory=dict)


class ValidatorRegisterRequest(_StrictModel):
    capabilities: tuple[str, ...] = ("cpu",)
    version: str | None = None
    last_seen_meta: dict[str, Any] | None = None


class ValidatorRegisterResponse(_StrictModel):
    validator: ValidatorView
    heartbeat_interval_seconds: StrictInt = Field(ge=1)


class ValidatorHeartbeatRequest(_StrictModel):
    last_seen_meta: dict[str, Any] | None = None


class ValidatorHeartbeatResponse(_StrictModel):
    status: str = Field(min_length=1)
    now: datetime


class AssignmentPullResponse(_StrictModel):
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    assignments: tuple[AssignmentView, ...] = ()


_LEGACY_GATEWAY_KEYS = frozenset(
    {
        "gateway_token",
        "gateway_url",
        "gateway_base_url",
        "BASE_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN_FILE",
        "BASE_LLM_GATEWAY_URL",
        "PRISM_GATEWAY_TOKEN",
        "PRISM_GATEWAY_TOKEN_FILE",
        "PRISM_LLM_GATEWAY_URL",
        "llm_gateway_url",
        "llm_gateway_token",
        "llm_gateway_token_file",
        "gateway",
        "llm_provider",
        "llm",
    }
)

_LLM_PROVIDER_NESTED_KEYS = frozenset(
    {
        "gateway_token",
        "gateway_url",
        "api_key",
        "base_url",
        "model",
        "openai_api_key",
        "openrouter_api_key",
        "token",
        "token_file",
    }
)


def _reject_legacy_gateway_mapping(value: Any, *, label: str) -> Any:
    if not isinstance(value, dict):
        return value
    hits: list[str] = []

    def is_legacy_key(key_str: str) -> bool:
        return (
            key_str in _LEGACY_GATEWAY_KEYS
            or key_str.upper() in _LEGACY_GATEWAY_KEYS
            or key_str.startswith("gateway_")
            or key_str.startswith("GATEWAY_")
            or key_str.startswith("llm_gateway")
            or key_str.startswith("LLM_GATEWAY")
        )

    def is_llm_provider_bag(nested: Any) -> bool:
        if not isinstance(nested, dict):
            return False
        return any(
            str(nested_key) in _LLM_PROVIDER_NESTED_KEYS
            or str(nested_key).startswith("gateway_")
            or str(nested_key).startswith("llm_gateway")
            for nested_key in nested
        )

    def walk(node: dict[str, Any], path: str) -> None:
        for key, nested in node.items():
            key_str = str(key)
            here = f"{path}.{key_str}" if path else key_str
            if is_legacy_key(key_str):
                hits.append(here)
                continue
            if key_str in {"provider", "Provider"} and is_llm_provider_bag(nested):
                hits.append(here)
                continue
            if isinstance(nested, dict):
                walk(nested, here)

    walk(value, "")
    if hits:
        raise ValueError(
            f"unsupported removed LLM gateway fields in {label}: "
            + ", ".join(sorted(hits))
        )
    return value


class AssignmentProgressRequest(_StrictModel):
    checkpoint_ref: str | None = None
    meta: dict[str, Any] | None = None

    @field_validator("meta")
    @classmethod
    def reject_gateway_meta(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        return _reject_legacy_gateway_mapping(value, label="meta")

    @model_validator(mode="before")
    @classmethod
    def reject_top_level_gateway_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            top = {
                key: value
                for key, value in data.items()
                if key not in {"checkpoint_ref", "meta"}
            }
            _reject_legacy_gateway_mapping(top, label="progress request")
        return data


class AssignmentProgressResponse(_StrictModel):
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    status: Literal["running"]
    lease_deadline: datetime | None = None
    last_progress_at: datetime
    checkpoint_ref: str | None = None


class AssignmentResultRequest(_StrictModel):
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    success: StrictBool
    payload: dict[str, Any] = Field(default_factory=dict)
    checkpoint_ref: str | None = None
    proof: dict[str, Any] | None = None

    @field_validator("payload")
    @classmethod
    def reject_gateway_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_legacy_gateway_mapping(value, label="result payload")

    @field_validator("proof")
    @classmethod
    def reject_gateway_proof(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return value
        return _reject_legacy_gateway_mapping(value, label="result proof")

    @model_validator(mode="before")
    @classmethod
    def reject_top_level_gateway_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            top = {
                key: value
                for key, value in data.items()
                if key
                not in {"api_version", "success", "payload", "checkpoint_ref", "proof"}
            }
            _reject_legacy_gateway_mapping(top, label="result request")
        return data


class AssignmentResultResponse(_StrictModel):
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    status: Literal["completed", "failed"]
    result_ref: str = Field(min_length=1)
    idempotent: bool = False


class ProviderInfo(_StrictModel):
    name: str = Field(min_length=1)
    executor_id: str | None = None
    pod_id: str | None = None
    miner_hotkey: str | None = None


class WorkerSignature(_StrictModel):
    worker_pubkey: str = Field(min_length=1)
    sig: str = Field(min_length=1)


class ExecutionProof(_StrictModel):
    version: Literal[1]
    tier: Literal[0, 1, 2]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    provider: ProviderInfo | None = None
    worker_signature: WorkerSignature
    attestation: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_tier_evidence(self) -> ExecutionProof:
        if self.tier >= 1 and self.image_digest is None:
            raise ValueError("tier 1 or 2 requires image_digest")
        if self.tier == 2 and self.attestation is None:
            raise ValueError("tier 2 requires attestation")
        return self


class RawWeightPushRequest(_StrictModel):
    """Authenticated challenge hotkey-weight snapshot."""

    protocol_version: str = Field(pattern=r"^1\.\d+$")
    challenge_slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    epoch: StrictInt = Field(ge=0)
    revision: StrictInt = Field(ge=1)
    computed_at: datetime
    expires_at: datetime
    nonce: str = Field(min_length=1, max_length=256)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    # Empty maps are rejected at schema (no accidental zero-emission contribution).
    # All-zero maps with at least one hotkey key are accepted as explicit zero sources.
    weights: dict[str, StrictFloat] = Field(min_length=1)

    @field_validator("computed_at", "expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("raw weight timestamps must include a timezone")
        return value

    @field_validator("epoch", "revision")
    @classmethod
    def reject_boolean_integer(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("boolean is not an integer")
        return value

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        for hotkey, weight in value.items():
            if not isinstance(hotkey, str) or not fullmatch(
                r"[1-9A-HJ-NP-Za-km-z]{3,64}", hotkey
            ):
                raise ValueError("hotkey keys must be syntactically valid hotkeys")
            if not any(char.isalpha() for char in hotkey):
                raise ValueError("hotkey keys must not be UID-only values")
            if not isinstance(weight, float) or not isfinite(weight) or weight < 0:
                raise ValueError("weights must be finite non-negative floats")
        return value

    @model_validator(mode="after")
    def validate_freshness_and_digest(self) -> RawWeightPushRequest:
        # Structural ordering only. Receipt-time freshness is
        # ``raw-weight-freshness.v1`` on the master
        # (computed_at - skew <= receipt < expires_at + skew). Do not reject
        # delayed-in-window deliveries solely because wall-clock now has passed
        # expires_at relative to the client parse time (VAL-WEIGHT-019).
        if self.expires_at <= self.computed_at:
            raise ValueError("expires_at must be after computed_at")
        canonical_payload = self.model_dump(
            mode="json",
            exclude={"payload_digest"},
        )
        if self.payload_digest != self.compute_digest(canonical_payload):
            raise ValueError("payload_digest does not match canonical payload")
        return self

    @staticmethod
    def canonicalize(payload: dict[str, Any]) -> bytes:
        normalized = dict(payload)
        for field_name in ("computed_at", "expires_at"):
            value = normalized.get(field_name)
            if isinstance(value, str):
                try:
                    normalized[field_name] = datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=lambda value: (
                value.astimezone(UTC).isoformat().replace("+00:00", "Z")
                if isinstance(value, datetime) and value.tzinfo is not None
                else value.isoformat()
                if isinstance(value, datetime)
                else value
            ),
        ).encode("utf-8")

    @classmethod
    def compute_digest(cls, payload: dict[str, Any]) -> str:
        return hashlib.sha256(cls.canonicalize(payload)).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self.canonicalize(self.model_dump())


class RawWeightPushAcknowledgement(_StrictModel):
    protocol_version: str = Field(pattern=r"^1\.\d+$")
    challenge_slug: str = Field(min_length=1)
    epoch: StrictInt = Field(ge=0)
    revision: StrictInt = Field(ge=1)
    snapshot_id: str = Field(min_length=1)
    payload_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted: Literal[True] = True
    idempotent: bool = False


class ExternalResultEnvelope(_StrictModel):
    api_version: str = Field(pattern=r"^1\.\d+$")
    work_unit_id: str = Field(min_length=1)
    assignment_id: str = Field(min_length=1)
    submission_ref: str = Field(min_length=1)
    challenge_slug: str = Field(min_length=1)
    result: dict[str, Any]
    proof: ExecutionProof


class ExternalResultResponse(_StrictModel):
    status: Literal["accepted", "conflict", "rejected"]
    work_unit_id: str = Field(min_length=1)
    submission_id: str = Field(min_length=1)
    claimed_tier: StrictInt = Field(ge=0, le=2)
    effective_tier: StrictInt = Field(ge=0, le=2)
    tier_downgraded: StrictBool
    finalized: StrictBool
    submission_status: str | None = None
    reason: str | None = None
    idempotent: StrictBool
    audit_sampled: StrictBool | None = None
    audit_unit_id: str | None = None


class WorkUnitFoldRequest(_StrictModel):
    job_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=256)


class HealthCheck(_StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    status: Literal["ok", "degraded", "unhealthy"]
    required: StrictBool = True


class HealthResponse(_StrictModel):
    status: Literal["ok", "degraded", "unhealthy"] = "ok"
    slug: str
    version: str
    role: Literal["master", "validator", "challenge", "worker"] = "challenge"
    ready: StrictBool = True
    capabilities: tuple[str, ...] = ()
    checks: tuple[HealthCheck, ...] = ()

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("capabilities must be unique")
        return value

    @model_validator(mode="after")
    def validate_health_contract(self) -> HealthResponse:
        _validate_role_capabilities(self.role, self.capabilities)
        if len(self.checks) != len({check.name for check in self.checks}):
            raise ValueError("health check names must be unique")
        required_healthy = all(
            check.status == "ok" for check in self.checks if check.required
        )
        if self.ready != required_healthy:
            raise ValueError("ready must match mandatory health check state")
        expected_status = (
            "unhealthy"
            if not required_healthy
            else "degraded"
            if any(check.status != "ok" for check in self.checks)
            else "ok"
        )
        if self.status != expected_status:
            raise ValueError("status must match health check state")
        return self


class VersionResponse(_StrictModel):
    distribution_name: str = Field(min_length=1)
    artifact_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    release_id: str = Field(min_length=1)
    api_version: str = Field(pattern=r"^\d+\.\d+$")
    challenge_slug: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    challenge_version: str = Field(min_length=1)
    sdk_contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    sdk_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    role: Literal["master", "validator", "challenge", "worker"] = "challenge"
    capabilities: tuple[str, ...] = ()

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("capabilities must be unique")
        return value

    @model_validator(mode="after")
    def validate_role_capabilities(self) -> VersionResponse:
        _validate_role_capabilities(self.role, self.capabilities)
        if self.role == "challenge" and self.challenge_slug is None:
            raise ValueError("challenge version responses require challenge_slug")
        if self.role != "challenge" and self.challenge_slug is not None:
            raise ValueError("only challenge responses may include challenge_slug")
        return self


class RuntimeStatusResponse(_StrictModel):
    health: HealthResponse
    version: VersionResponse

    @model_validator(mode="after")
    def validate_shared_identity(self) -> RuntimeStatusResponse:
        if self.health.role != self.version.role:
            raise ValueError("health and version roles must match")
        if self.health.version != self.version.challenge_version:
            raise ValueError("health and version runtime versions must match")
        if self.health.capabilities != self.version.capabilities:
            raise ValueError("health and version capabilities must match")
        return self


def _validate_role_capabilities(role: str, capabilities: tuple[str, ...]) -> None:
    from .roles import ROLE_REGISTRY, RoleContractError

    for token in capabilities:
        try:
            owner = ROLE_REGISTRY.get(token).role.value
        except RoleContractError as exc:
            raise ValueError("capabilities contain unknown tokens") from exc
        if owner != role:
            raise ValueError("capabilities contain tokens owned by another role")


class WeightsResponse(_StrictModel):
    challenge_slug: str
    epoch: int | None = Field(default=None, ge=0)
    weights: dict[str, float]
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def finite_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not isinstance(item, float) or not isfinite(item) or item < 0
            for item in value.values()
        ):
            raise ValueError("weights must be finite non-negative floats")
        return value


__all__ = [
    "AssignmentProgressRequest",
    "AssignmentProgressResponse",
    "AssignmentPullResponse",
    "AssignmentResultRequest",
    "AssignmentResultResponse",
    "AssignmentView",
    "EmptyObject",
    "ExternalResultEnvelope",
    "ExternalResultResponse",
    "ErrorResponse",
    "ExecutionProof",
    "HealthCheck",
    "HealthResponse",
    "ProviderInfo",
    "RawWeightPushAcknowledgement",
    "RawWeightPushRequest",
    "RuntimeStatusResponse",
    "ValidatorHeartbeatRequest",
    "ValidatorHeartbeatResponse",
    "ValidatorRegisterRequest",
    "ValidatorRegisterResponse",
    "ValidatorView",
    "VersionResponse",
    "WeightsResponse",
    "WorkerSignature",
    "WorkUnitFoldRequest",
]
