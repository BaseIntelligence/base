"""HTTP SidecarAttestor — BASE dials recipe listen ``POST /v1/sidecar/attest``.

Transport only: POST nonce+phase, parse signed wire, extract digest. Does not
consume nonces or verify HMAC (callers use ``base.attestation.payload``).
Fail-closed to typed ``ConstationFailCode`` outcomes for the runner.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import httpx

from base.compute.constation_types import ConstationFailCode

logger = logging.getLogger(__name__)

ATTEST_PATH: Final[str] = "/v1/sidecar/attest"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 5.0
_VALID_PHASES: Final[frozenset[str]] = frozenset({"start", "interval", "end"})


class SidecarAttestPhase(StrEnum):
    """Phases accepted by recipe listen mode."""

    START = "start"
    INTERVAL = "interval"
    END = "end"


@dataclass(frozen=True, slots=True)
class SidecarAttestHit:
    """Successful transport parse of a sidecar attestation wire body."""

    digest: str
    nonce: str
    pod_id: str
    phase: str
    signature: str
    algorithm: str
    schema_version: str
    sealed_manifest_hashes: Mapping[str, str]
    wire: Mapping[str, Any]

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class SidecarAttestMiss:
    """Fail-closed transport or parse outcome."""

    reason: ConstationFailCode
    detail: str | None = None

    def __bool__(self) -> bool:
        return False


SidecarAttestResult = SidecarAttestHit | SidecarAttestMiss


@dataclass(frozen=True, slots=True)
class HttpSidecarAttestor:
    """Async HTTP client for recipe sidecar listen mode (dial-out).

    ``base_url`` is the resolved origin (see ``resolve_sidecar_base_url``).
    ``timeout_seconds`` bounds connect+read (must be positive).
    ``transport`` is optional for tests (respx / MockTransport).
    """

    base_url: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    transport: httpx.AsyncBaseTransport | None = None

    def __post_init__(self) -> None:
        url = self.base_url.strip().rstrip("/")
        if not url:
            raise ValueError("base_url must be a non-empty HTTP(S) origin")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds!r}"
            )
        object.__setattr__(self, "base_url", url)

    async def attest(self, *, nonce: str, phase: str) -> SidecarAttestResult:
        """POST attest path; extract payload digest. Does not consume nonce."""
        try:
            nonce_s = _require_nonblank("nonce", nonce)
            phase_s = _normalize_phase(phase)
        except ValueError as exc:
            return SidecarAttestMiss(
                reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
                detail=str(exc),
            )

        url = f"{self.base_url}{ATTEST_PATH}"
        body = {"nonce": nonce_s, "phase": phase_s}
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                transport=self.transport,
            ) as client:
                response = await client.post(url, json=body)
        except httpx.TimeoutException as exc:
            logger.warning(
                "sidecar attest timeout url=%s phase=%s err=%s",
                url,
                phase_s,
                type(exc).__name__,
            )
            return SidecarAttestMiss(
                reason=ConstationFailCode.NETWORK_PARTITION,
                detail=f"timeout phase={phase_s}",
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "sidecar attest transport failed url=%s phase=%s err=%s",
                url,
                phase_s,
                type(exc).__name__,
            )
            return SidecarAttestMiss(
                reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
                detail=f"transport phase={phase_s} err={type(exc).__name__}",
            )

        return _parse_response(response, requested_phase=phase_s)


def _parse_response(
    response: httpx.Response,
    *,
    requested_phase: str,
) -> SidecarAttestResult:
    if response.status_code != 200:
        if 400 <= response.status_code < 500:
            return SidecarAttestMiss(
                reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
                detail=f"http_status={response.status_code}",
            )
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
            detail=f"http_status={response.status_code}",
        )
    try:
        raw: object = response.json()
    except ValueError:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="body_not_json",
        )
    if not isinstance(raw, dict):
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="body_not_object",
        )
    return _parse_wire(raw, requested_phase=requested_phase)


def _parse_wire(
    wire: Mapping[str, Any],
    *,
    requested_phase: str,
) -> SidecarAttestResult:
    payload_raw = wire.get("payload")
    if not isinstance(payload_raw, Mapping):
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="missing_payload",
        )

    digest = _optional_nonblank_str(payload_raw.get("digest"))
    if digest is None:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="missing_digest",
        )
    nonce = _optional_nonblank_str(payload_raw.get("nonce"))
    if nonce is None:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="missing_nonce",
        )
    pod_id = _optional_nonblank_str(payload_raw.get("pod_id"))
    if pod_id is None:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="missing_pod_id",
        )
    signature = _optional_nonblank_str(wire.get("signature"))
    if signature is None:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="missing_signature",
        )

    algorithm = _optional_nonblank_str(wire.get("algorithm")) or "hmac-sha256"
    schema_version = (
        _optional_nonblank_str(wire.get("schema_version"))
        or "prism_attestation_payload.v1"
    )
    phase_raw = wire.get("phase", requested_phase)
    if isinstance(phase_raw, str):
        phase_opt = _optional_nonblank_str(phase_raw)
        phase = phase_opt.lower() if phase_opt is not None else requested_phase
    else:
        phase = requested_phase
    if phase not in _VALID_PHASES:
        phase = requested_phase

    hashes = _parse_sealed_hashes(payload_raw.get("sealed_manifest_hashes"))
    if hashes is None:
        return SidecarAttestMiss(
            reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
            detail="invalid_sealed_manifest_hashes",
        )

    return SidecarAttestHit(
        digest=digest,
        nonce=nonce,
        pod_id=pod_id,
        phase=phase,
        signature=signature,
        algorithm=algorithm,
        schema_version=schema_version,
        sealed_manifest_hashes=hashes,
        wire=dict(wire),
    )


def _parse_sealed_hashes(raw: object) -> dict[str, str] | None:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        return None
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
        path, digest = key.strip(), value.strip()
        if not path or not digest:
            return None
        out[path] = digest
    return out


def _normalize_phase(phase: str) -> str:
    if not isinstance(phase, str):
        raise ValueError("phase must be a string")
    key = phase.strip().lower()
    if key not in _VALID_PHASES:
        raise ValueError(f"invalid phase: {phase!r}")
    return key


def _require_nonblank(field: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} must be non-empty")
    return cleaned


def _optional_nonblank_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


__all__ = [
    "ATTEST_PATH",
    "DEFAULT_TIMEOUT_SECONDS",
    "HttpSidecarAttestor",
    "SidecarAttestHit",
    "SidecarAttestMiss",
    "SidecarAttestPhase",
    "SidecarAttestResult",
]
