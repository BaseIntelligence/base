"""Attested telemetry-session handshake for mid-run progress ingest.

Phase B of the progress stack: after Bearer ``EVAL_RUN_TOKEN`` auth, the runner
opens a short-lived session by signing
``ac-telemetry-session:v1|{eval_run_id}|{instance_id}|{nonce}|{timestamp}``
with its hotkey. Progress POSTs must present the issued session id via
``X-Telemetry-Session``.

Trust boundary: tamper-evidence only. Sessions never carry score material and
never accept a mnemonic — only ``hotkey_ss58`` + signature.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any

from agent_challenge.auth.security import verify_substrate_signature

TELEMETRY_SESSION_HEADER = "X-Telemetry-Session"
CANONICAL_PREFIX = "ac-telemetry-session:v1"
DEFAULT_SESSION_TTL = timedelta(hours=1)
_ACTIVE_EVAL_PHASES = frozenset({"eval_prepared", "eval_running", "eval_verifying"})
_SESSION_REQUIRED_FIELDS = (
    "schema_version",
    "eval_run_id",
    "instance_id",
    "hotkey_ss58",
    "nonce",
    "timestamp",
    "signature",
)
_SESSION_FORBIDDEN_FIELDS = frozenset(
    {
        "mnemonic",
        "seed",
        "private_key",
        "secret",
        "RUNNER_HOTKEY_MNEMONIC",
    }
)


class TelemetrySessionError(ValueError):
    """Schema, auth, or lifecycle failure for a telemetry session."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class TelemetrySession:
    """In-process attested session record (tamper-evidence only)."""

    session_id: str
    eval_run_id: str
    instance_id: str
    hotkey_ss58: str
    expires_at: datetime
    closed: bool = False


_LOCK = Lock()
_SESSIONS: dict[str, TelemetrySession] = {}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_timestamp(raw: str) -> datetime:
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise TelemetrySessionError(
            "telemetry session timestamp is invalid",
            code="invalid_telemetry_session",
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _canonical_message(
    *,
    eval_run_id: str,
    instance_id: str,
    nonce: str,
    timestamp: str,
) -> str:
    return f"{CANONICAL_PREFIX}|{eval_run_id}|{instance_id}|{nonce}|{timestamp}"


def validate_telemetry_session_request(
    value: Mapping[str, Any],
    *,
    eval_run_id: str,
) -> dict[str, str]:
    """Validate a closed telemetry-session open body (no mnemonic)."""

    if not isinstance(value, Mapping):
        raise TelemetrySessionError(
            "telemetry session body must be an object",
            code="invalid_telemetry_session",
        )
    keys = set(value)
    forbidden = sorted(keys & _SESSION_FORBIDDEN_FIELDS)
    if forbidden or any(isinstance(k, str) and "mnemonic" in k.lower() for k in keys):
        raise TelemetrySessionError(
            "telemetry session forbids mnemonic/signing-secret fields",
            code="invalid_telemetry_session",
        )
    missing = [name for name in _SESSION_REQUIRED_FIELDS if name not in keys]
    unknown = sorted(keys - set(_SESSION_REQUIRED_FIELDS))
    if missing or unknown:
        raise TelemetrySessionError(
            f"telemetry session has invalid fields: missing={missing}, unknown={unknown}",
            code="invalid_telemetry_session",
        )
    if value.get("schema_version") != 1:
        raise TelemetrySessionError(
            "telemetry session schema_version must be 1",
            code="invalid_telemetry_session",
        )
    body_run_id = value.get("eval_run_id")
    if not isinstance(body_run_id, str) or body_run_id != eval_run_id:
        raise TelemetrySessionError(
            "telemetry session eval_run_id does not match route",
            code="invalid_telemetry_session",
        )
    out: dict[str, str] = {}
    for name in (
        "eval_run_id",
        "instance_id",
        "hotkey_ss58",
        "nonce",
        "timestamp",
        "signature",
    ):
        raw = value.get(name)
        if not isinstance(raw, str) or not raw.strip():
            raise TelemetrySessionError(
                f"telemetry session {name} must be a non-empty string",
                code="invalid_telemetry_session",
            )
        out[name] = raw.strip()
    # Timestamp must parse; skew is not enforced beyond parseability for tests.
    _parse_timestamp(out["timestamp"])
    return out


def open_telemetry_session(
    *,
    eval_run_id: str,
    eval_run_phase: str,
    body: Mapping[str, Any],
    ttl: timedelta = DEFAULT_SESSION_TTL,
) -> dict[str, str]:
    """Open a hotkey-attested session. Returns ``{session_id, expires_at}``."""

    if eval_run_phase not in _ACTIVE_EVAL_PHASES:
        raise TelemetrySessionError(
            "eval run is terminal; telemetry session forbidden",
            code="eval_run_terminal",
        )
    validated = validate_telemetry_session_request(body, eval_run_id=eval_run_id)
    message = _canonical_message(
        eval_run_id=validated["eval_run_id"],
        instance_id=validated["instance_id"],
        nonce=validated["nonce"],
        timestamp=validated["timestamp"],
    )
    if not verify_substrate_signature(
        validated["hotkey_ss58"],
        message,
        validated["signature"],
    ):
        raise TelemetrySessionError(
            "telemetry session signature invalid",
            code="invalid_telemetry_signature",
        )
    session_id = f"ts_{secrets.token_urlsafe(24)}"
    expires_at = _utcnow() + ttl
    record = TelemetrySession(
        session_id=session_id,
        eval_run_id=eval_run_id,
        instance_id=validated["instance_id"],
        hotkey_ss58=validated["hotkey_ss58"],
        expires_at=expires_at,
        closed=False,
    )
    with _LOCK:
        _SESSIONS[session_id] = record
    return {
        "session_id": session_id,
        "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
    }


def require_telemetry_session(
    session_id: str | None,
    *,
    eval_run_id: str,
) -> TelemetrySession:
    """Validate ``X-Telemetry-Session`` for a progress POST."""

    if session_id is None or not str(session_id).strip():
        raise TelemetrySessionError(
            "telemetry session header required",
            code="telemetry_session_required",
        )
    sid = str(session_id).strip()
    with _LOCK:
        record = _SESSIONS.get(sid)
    if record is None:
        raise TelemetrySessionError(
            "telemetry session unknown",
            code="telemetry_session_unknown",
        )
    if record.eval_run_id != eval_run_id:
        raise TelemetrySessionError(
            "telemetry session does not match eval run",
            code="invalid_telemetry_session",
        )
    if record.closed:
        raise TelemetrySessionError(
            "telemetry session closed",
            code="telemetry_session_closed",
        )
    if record.expires_at <= _utcnow():
        raise TelemetrySessionError(
            "telemetry session expired",
            code="telemetry_session_expired",
        )
    return record


async def close_telemetry_session(session_id: str) -> None:
    """Mark a session closed (test + lifecycle helper)."""

    with _LOCK:
        record = _SESSIONS.get(session_id)
        if record is None:
            return
        _SESSIONS[session_id] = TelemetrySession(
            session_id=record.session_id,
            eval_run_id=record.eval_run_id,
            instance_id=record.instance_id,
            hotkey_ss58=record.hotkey_ss58,
            expires_at=record.expires_at,
            closed=True,
        )


async def expire_telemetry_session(session_id: str) -> None:
    """Force-expire a session by setting expires_at in the past."""

    with _LOCK:
        record = _SESSIONS.get(session_id)
        if record is None:
            return
        _SESSIONS[session_id] = TelemetrySession(
            session_id=record.session_id,
            eval_run_id=record.eval_run_id,
            instance_id=record.instance_id,
            hotkey_ss58=record.hotkey_ss58,
            expires_at=_utcnow() - timedelta(seconds=1),
            closed=record.closed,
        )


def reset_telemetry_sessions_for_tests() -> None:
    """Clear in-process session store (tests only)."""

    with _LOCK:
        _SESSIONS.clear()


__all__ = [
    "CANONICAL_PREFIX",
    "DEFAULT_SESSION_TTL",
    "TELEMETRY_SESSION_HEADER",
    "TelemetrySession",
    "TelemetrySessionError",
    "close_telemetry_session",
    "expire_telemetry_session",
    "open_telemetry_session",
    "require_telemetry_session",
    "reset_telemetry_sessions_for_tests",
    "validate_telemetry_session_request",
]
