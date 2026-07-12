"""Release-owned canonical API and CLI contract manifest.

The manifest is data rather than a second router implementation.  It provides
one machine-readable inventory for operators and validation tooling to compare
against the generated FastAPI and Typer surfaces.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str
    error_envelope: Mapping[str, str]
    routes: tuple[Mapping[str, Any], ...]
    cli: tuple[Mapping[str, Any], ...]

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "error_envelope", _freeze(self.error_envelope))
        object.__setattr__(self, "routes", tuple(_freeze(item) for item in self.routes))
        object.__setattr__(self, "cli", tuple(_freeze(item) for item in self.cli))

    def digest(self) -> str:
        payload = _thaw(self.model_dump(mode="python"))
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class _FrozenDict(dict[str, Any]):
    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("manifest mappings are immutable")

    __setitem__ = __delitem__ = clear = setdefault = update = _immutable

    def pop(self, *_args: Any, **_kwargs: Any) -> Any:
        self._immutable()

    def popitem(self) -> tuple[str, Any]:
        self._immutable()
        raise AssertionError


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


API_MANIFEST = ApiManifest(
    version="1",
    error_envelope=MappingProxyType(
        {
            "code": "stable machine-readable error code",
            "detail": "safe human-readable diagnostic",
            "correlation_id": "request correlation identifier",
            "expected": "expected version or value when applicable",
            "actual": "received version or value when applicable",
        }
    ),
    routes=(
        {
            "method": "GET",
            "path": "/health",
            "media_type": "application/json",
            "schema": "HealthResponse",
            "status": [200, 503],
            "auth": "none",
            "roles": ["master", "validator", "challenge", "worker"],
        },
        {
            "method": "GET",
            "path": "/version",
            "media_type": "application/json",
            "schema": "VersionResponse",
            "status": [200, 503],
            "auth": "none",
            "roles": ["master", "validator", "challenge", "worker"],
        },
        {
            "method": "GET",
            "path": "/v1/weights/latest",
            "media_type": "application/json",
            "schema": "MasterWeightsResponse",
            "status": [200, 404, 502, 503],
            "auth": "none",
            "roles": ["master", "validator"],
        },
        {
            "method": "POST",
            "path": "/v1/validators/register",
            "media_type": "application/json",
            "schema": "ValidatorRegisterRequest",
            "response_schema": "ValidatorRegisterResponse",
            "status": [200, 401, 403, 409, 422],
            "auth": "validator-signature",
            "roles": ["validator"],
        },
        {
            "method": "POST",
            "path": "/v1/validators/heartbeat",
            "media_type": "application/json",
            "schema": "ValidatorHeartbeatRequest",
            "response_schema": "ValidatorHeartbeatResponse",
            "status": [200, 401, 403, 409, 422],
            "auth": "validator-signature",
            "roles": ["validator"],
        },
        {
            "method": "POST",
            "path": "/v1/assignments/pull",
            "media_type": "application/json",
            "schema": "EmptyObject",
            "response_schema": "AssignmentPullResponse",
            "status": [200, 401, 403, 409, 422],
            "auth": "validator-signature",
            "roles": ["validator"],
        },
        {
            "method": "POST",
            "path": "/v1/assignments/{id}/progress",
            "media_type": "application/json",
            "schema": "AssignmentProgressRequest",
            "response_schema": "AssignmentProgressResponse",
            "status": [200, 401, 403, 404, 409, 422],
            "auth": "validator-signature",
            "roles": ["validator"],
        },
        {
            "method": "POST",
            "path": "/v1/assignments/{id}/result",
            "media_type": "application/json",
            "schema": "AssignmentResultRequest",
            "response_schema": "AssignmentResultResponse",
            "status": [200, 401, 403, 404, 409, 422],
            "auth": "validator-signature",
            "roles": ["validator"],
        },
        {
            "method": "POST",
            "path": "/internal/v1/work_units/result",
            "media_type": "application/json",
            "schema": "ExternalResultEnvelope",
            "response_schema": "ExternalResultResponse",
            "status": [200, 401, 403, 409, 422, 503],
            "auth": "challenge-auth",
            "roles": ["challenge"],
        },
    ),
    cli=(
        {
            "name": "base master",
            "role": "master",
            "json": True,
            "exit_codes": {"success": 0, "config": 2, "auth": 3, "conflict": 4},
        },
        {
            "name": "base validator",
            "role": "validator",
            "json": True,
            "exit_codes": {"success": 0, "config": 2, "auth": 3, "conflict": 4},
        },
        {
            "name": "base challenge",
            "role": "challenge",
            "json": True,
            "exit_codes": {"success": 0, "config": 2, "auth": 3, "conflict": 4},
        },
        {
            "name": "base worker",
            "role": "worker",
            "json": True,
            "exit_codes": {"success": 0, "config": 2, "auth": 3, "conflict": 4},
        },
        {
            "name": "base validator set-weights",
            "role": "validator",
            "json": True,
            "exit_codes": {"success": 0, "config": 2, "auth": 3, "conflict": 4},
            "gated": True,
        },
    ),
)
API_MANIFEST_DIGEST = API_MANIFEST.digest()

__all__ = ["API_MANIFEST", "API_MANIFEST_DIGEST", "ApiManifest"]
