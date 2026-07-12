"""API schemas for the assignment coordination plane (pull/progress/result)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    status: str
    deadline_at: datetime | None = None
    last_progress_at: datetime | None = None
    checkpoint_ref: str | None = None


class AssignmentResultRequest(BaseModel):
    """Body for ``POST /v1/assignments/{id}/result``."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    checkpoint_ref: str | None = None

    @field_validator("payload")
    @classmethod
    def reject_gateway_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _reject_legacy_gateway_fields(value, label="result payload")

    @model_validator(mode="before")
    @classmethod
    def reject_top_level_gateway_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            top = {
                key: value
                for key, value in data.items()
                if key not in {"success", "payload", "checkpoint_ref"}
            }
            _reject_legacy_gateway_fields(top, label="result request")
        return data


class AssignmentResultResponse(BaseModel):
    """Response for a result post (idempotent when already terminal)."""

    status: str
    result_ref: str | None = None
    idempotent: bool = False
