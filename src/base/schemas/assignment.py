"""API schemas for the assignment coordination plane (pull/progress/result)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from base.challenge_sdk.version import API_VERSION

# Residual LLM-gateway keys that must never be accepted on progress/result
# wire input (VAL-GATE-017). Nested bags named ``gateway`` are also rejected.
# Bare compute-provider proof metadata (ExecutionProof.provider) stays allowed
# when it does not carry gateway URL/token fields.
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


def _is_legacy_key(key: str) -> bool:
    return (
        key in _LEGACY_GATEWAY_KEYS
        or key.upper() in _LEGACY_GATEWAY_KEYS
        or key.startswith("gateway_")
        or key.startswith("GATEWAY_")
        or key.startswith("llm_gateway")
        or key.startswith("LLM_GATEWAY")
    )


def _is_llm_provider_bag(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(
        str(nested_key) in _LLM_PROVIDER_NESTED_KEYS
        or str(nested_key).startswith("gateway_")
        or str(nested_key).startswith("llm_gateway")
        for nested_key in value
    )


def _find_legacy_gateway_keys(value: Any, *, path: str = "") -> list[str]:
    """Return dotted paths of residual gateway/LLM-provider keys in a mapping."""

    hits: list[str] = []
    if not isinstance(value, dict):
        return hits
    for key, nested in value.items():
        key_str = str(key)
        here = f"{path}.{key_str}" if path else key_str
        if _is_legacy_key(key_str):
            hits.append(here)
            continue
        if key_str in {"provider", "Provider"} and _is_llm_provider_bag(nested):
            hits.append(here)
            continue
        if isinstance(nested, dict):
            hits.extend(_find_legacy_gateway_keys(nested, path=here))
    return hits


def _reject_legacy_gateway_fields(value: Any, *, label: str) -> Any:
    hits = _find_legacy_gateway_keys(value)
    if hits:
        raise ValueError(
            f"unsupported removed LLM gateway fields in {label}: "
            + ", ".join(sorted(hits))
        )
    return value


def canonicalize_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for digest comparison."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def compute_payload_digest(payload: Any) -> str:
    """SHA-256 hex digest of the canonical payload map."""

    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def compute_result_digest(
    *,
    success: bool,
    payload: Any,
    checkpoint_ref: str | None = None,
    proof: Any | None = None,
) -> str:
    """SHA-256 hex digest binding terminal result identity for retry comparison."""

    envelope = {
        "success": bool(success),
        "payload": payload if payload is not None else {},
        "checkpoint_ref": checkpoint_ref,
        "proof": proof if proof is not None else None,
    }
    return hashlib.sha256(canonicalize_json(envelope)).hexdigest()


class AssignmentView(BaseModel):
    """Public view of a coordinated work-unit assignment for a validator."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Canonical wire fields (VAL-SDK-045).
    api_version: str = Field(default=API_VERSION, pattern=r"^\d+\.\d+$")
    assignment_id: str = Field(validation_alias="id")
    work_unit_id: str
    submission_ref: str
    challenge_slug: str
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_digest: str = Field(default="", pattern=r"^([0-9a-f]{64})?$")
    required_capability: str
    revision: int = Field(default=1, ge=1)
    attempt: int = Field(default=1, ge=0, validation_alias="attempt_count")
    status: str
    lease_deadline: datetime | None = Field(
        default=None, validation_alias="deadline_at"
    )
    checkpoint_ref: str | None = None
    # Retained for existing operator/admin-facing consumers.
    max_attempts: int = 3
    last_progress_at: datetime | None = None

    @property
    def id(self) -> str:
        """Compatibility alias used by existing agent runtime code."""

        return self.assignment_id

    @property
    def attempt_count(self) -> int:
        """Compatibility alias used by existing agent runtime code."""

        return self.attempt

    @property
    def deadline_at(self) -> datetime | None:
        """Compatibility alias used by existing agent runtime code."""

        return self.lease_deadline

    @model_validator(mode="after")
    def ensure_payload_digest(self) -> AssignmentView:
        if not self.payload_digest:
            object.__setattr__(
                self, "payload_digest", compute_payload_digest(self.payload)
            )
        return self


class AssignmentPullResponse(BaseModel):
    """Response for ``POST /v1/assignments/pull``."""

    api_version: str = Field(default=API_VERSION, pattern=r"^\d+\.\d+$")
    assignments: list[AssignmentView] = Field(default_factory=list)


class AssignmentProgressRequest(BaseModel):
    """Body for ``POST /v1/assignments/{id}/progress``."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_ref: str | None = None
    meta: dict[str, Any] | None = None

    @field_validator("meta")
    @classmethod
    def reject_gateway_meta(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        return _reject_legacy_gateway_fields(value, label="meta")

    @model_validator(mode="before")
    @classmethod
    def reject_top_level_gateway_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            top = {
                key: value
                for key, value in data.items()
                if key not in {"checkpoint_ref", "meta"}
            }
            _reject_legacy_gateway_fields(top, label="progress request")
        return data


class AssignmentProgressResponse(BaseModel):
    """Response for a successful progress heartbeat."""

    api_version: str = Field(default=API_VERSION, pattern=r"^\d+\.\d+$")
    status: str
    lease_deadline: datetime | None = Field(
        default=None, validation_alias="deadline_at"
    )
    last_progress_at: datetime | None = None
    checkpoint_ref: str | None = None

    @property
    def deadline_at(self) -> datetime | None:
        return self.lease_deadline


class AssignmentResultRequest(BaseModel):
    """Body for ``POST /v1/assignments/{id}/result``."""

    model_config = ConfigDict(extra="forbid")

    api_version: str = Field(default=API_VERSION, pattern=r"^\d+\.\d+$")
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    checkpoint_ref: str | None = None
    proof: dict[str, Any] | None = None

    @field_validator("payload")
    @classmethod
    def reject_gateway_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_legacy_gateway_fields(value, label="result payload")

    @field_validator("proof")
    @classmethod
    def reject_gateway_proof(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if value is None:
            return value
        return _reject_legacy_gateway_fields(value, label="result proof")

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
            _reject_legacy_gateway_fields(top, label="result request")
        return data


class AssignmentResultResponse(BaseModel):
    """Response for a result post (idempotent when already terminal)."""

    api_version: str = Field(default=API_VERSION, pattern=r"^\d+\.\d+$")
    status: str
    result_ref: str | None = None
    idempotent: bool = False
