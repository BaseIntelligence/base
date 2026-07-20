"""Public unauthenticated TEE math projection for joinbase inspection.

Maps a durable review envelope + verification outcome to the architecture §5.1
safe field list only. Never exposes nonce plaintext, tokens, capabilities,
evidence bodies, model IO, or encryption key material.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from .report import (
    MAX_REVIEW_EVENT_LOG_ENTRIES,
    MAX_REVIEW_QUOTE_BYTES,
    REVIEW_REPORT_DOMAIN,
    review_report_data_preimage,
)

# Public response caps (fail-closed truncation, not silent invent).
MAX_PUBLIC_TEE_QUOTE_BYTES = MAX_REVIEW_QUOTE_BYTES
MAX_PUBLIC_TEE_EVENT_LOG_ENTRIES = min(256, MAX_REVIEW_EVENT_LOG_ENTRIES)
MAX_PUBLIC_TEE_EVENT_LOG_BYTES = 262_144

# Architecture §5.1 top-level allowlist when available:true.
PUBLIC_TEE_TOP_LEVEL_ALLOWLIST = frozenset(
    {
        "available",
        "submission_id",
        "domain",
        "review_digest",
        "report_data_hex",
        "report_data_preimage",
        "measurement",
        "tdx_quote_hex",
        "event_log",
        "verification_outcome",
        "quote_fingerprint_sha256",
        "agent_hash",
        "zip_sha256",
        "verdict",
        "assignment_digest",
        "session_id",
        "assignment_id",
    }
)

PUBLIC_TEE_MEASUREMENT_KEYS = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "rtmr3",
    "compose_hash",
    "os_image_hash",
    "key_provider",
    "vm_shape",
)

PUBLIC_TEE_OUTCOME_KEYS = (
    "status",
    "measurement_allowlisted",
    "report_data_matched",
    "verified_at_ms",
    "reason_code",
)

# Hard deny substrings / keys that must never appear in serialized public bodies.
PUBLIC_TEE_DENY_KEYS = frozenset(
    {
        "review_nonce",
        "session_token",
        "session_token_sha256",
        "capability",
        "capabilities",
        "bearer",
        "authorization",
        "assignment_bearer",
        "review_session_token",
        "encrypted_env",
        "evidence_objects",
        "request_body",
        "response_body",
        "planned_request",
        "transport_observation",
        "encryption_key",
        "private_key",
        "mnemonic",
        "openrouter_api_key",
        "phala_cloud_api_key",
        "key_file",
        "KEY_FILE",
        "challenge_token",
        "shared_token",
        "wallet",
    }
)

PUBLIC_TEE_DENY_SUBSTRINGS = (
    "sk-",
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "mnemonic",
    "OPENROUTER_API_KEY",
    "PHALA_CLOUD_API_KEY",
)

# Align with status dual-flag verified verdict mapping: envelope alone (written
# during review_verifying, or parked as verifier_unavailable) is NOT enough for
# available:true. Prefer public projection and/or a terminal verified outcome.
VERIFIED_PUBLIC_TEE_OUTCOME_STATUSES = frozenset(
    {
        "verified_allow",
        "verified_reject",
        "verified_escalate",
    }
)


def public_tee_unavailable() -> dict[str, Any]:
    """Locked closed form when no authorizing/current verified report exists."""

    return {"available": False}


def public_tee_assignment_qualifies(
    *,
    envelope_json: str | None,
    outcome_json: str | None = None,
    public_projection_json: str | None = None,
) -> bool:
    """True when durable assignment material is eligible for public TEE math.

    Matches status ``report_available`` intent: a durable envelope is required,
    plus either a miner-facing public projection or a terminal verified_*
    verification_outcome status. Envelope-only (pre-verify) and
    ``verifier_unavailable`` must fail closed.
    """

    if not isinstance(envelope_json, str) or not envelope_json.strip():
        return False
    if isinstance(public_projection_json, str) and public_projection_json.strip():
        return True
    outcome = _loads_object(outcome_json)
    if not isinstance(outcome, Mapping):
        return False
    status = outcome.get("status")
    return isinstance(status, str) and status in VERIFIED_PUBLIC_TEE_OUTCOME_STATUSES


def _has_verified_public_signal(
    *,
    verification_outcome: Mapping[str, Any] | None,
    public_projection: Mapping[str, Any] | None,
) -> bool:
    """Builder-side gate: projection and/or verified_* outcome required."""

    if isinstance(public_projection, Mapping):
        return True
    if not isinstance(verification_outcome, Mapping):
        return False
    status = verification_outcome.get("status")
    return isinstance(status, str) and status in VERIFIED_PUBLIC_TEE_OUTCOME_STATUSES


def build_public_tee_math(
    *,
    submission_id: int | str,
    envelope: Mapping[str, Any] | None,
    verification_outcome: Mapping[str, Any] | None = None,
    public_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project stored envelope (+ optional outcome/projection) to safe public math.

    Returns exactly ``{"available": false}`` when the envelope is missing,
    structurally insufficient, or not yet verified (no public projection and no
    verified_* outcome). Never invents measurement/quote material.
    """

    if not isinstance(envelope, Mapping):
        return public_tee_unavailable()

    # Envelope is written before verification (review_verifying) and may park
    # verifier_unavailable without a public projection. Refuse available:true
    # unless status.report_available-class signals are present.
    if not _has_verified_public_signal(
        verification_outcome=verification_outcome,
        public_projection=public_projection,
    ):
        return public_tee_unavailable()

    attestation = envelope.get("attestation")
    if not isinstance(attestation, Mapping):
        return public_tee_unavailable()

    review_digest = envelope.get("review_digest")
    report_data_hex = envelope.get("report_data_hex")
    domain = envelope.get("domain") or REVIEW_REPORT_DOMAIN
    if not isinstance(review_digest, str) or not review_digest:
        return public_tee_unavailable()
    if not isinstance(report_data_hex, str) or not report_data_hex:
        return public_tee_unavailable()

    measurement = _public_measurement(attestation.get("measurement"))
    tdx_quote_hex = _public_quote_hex(attestation.get("tdx_quote_hex"))
    event_log = _public_event_log(attestation.get("event_log"))
    preimage = _public_report_data_preimage(envelope)
    outcome = _public_verification_outcome(verification_outcome)
    quote_fp = _quote_fingerprint(tdx_quote_hex, public_projection)

    payload: dict[str, Any] = {
        "available": True,
        "submission_id": _coerce_submission_id(submission_id, envelope),
        "domain": str(domain),
        "review_digest": review_digest,
        "report_data_hex": report_data_hex,
    }
    if preimage is not None:
        payload["report_data_preimage"] = preimage
    if measurement is not None:
        payload["measurement"] = measurement
    if tdx_quote_hex is not None:
        payload["tdx_quote_hex"] = tdx_quote_hex
    if event_log is not None:
        payload["event_log"] = event_log
    if outcome is not None:
        payload["verification_outcome"] = outcome
    if quote_fp is not None:
        payload["quote_fingerprint_sha256"] = quote_fp

    # Optional cross-check digests from the existing miner public projection.
    if isinstance(public_projection, Mapping):
        for key in (
            "agent_hash",
            "zip_sha256",
            "verdict",
            "assignment_digest",
            "session_id",
            "assignment_id",
        ):
            value = public_projection.get(key)
            if isinstance(value, str) and value:
                payload[key] = value
            elif isinstance(value, list) and key == "reason_codes":
                continue

    assert_public_tee_safe(payload)
    return payload


def build_public_tee_math_from_assignment(
    *,
    submission_id: int | str,
    envelope_json: str | None,
    outcome_json: str | None = None,
    public_projection_json: str | None = None,
) -> dict[str, Any]:
    """Convenience loader from durable ReviewAssignment JSON columns."""

    if not public_tee_assignment_qualifies(
        envelope_json=envelope_json,
        outcome_json=outcome_json,
        public_projection_json=public_projection_json,
    ):
        return public_tee_unavailable()
    envelope = _loads_object(envelope_json)
    if envelope is None:
        return public_tee_unavailable()
    return build_public_tee_math(
        submission_id=submission_id,
        envelope=envelope,
        verification_outcome=_loads_object(outcome_json),
        public_projection=_loads_object(public_projection_json),
    )


def assert_public_tee_safe(payload: Mapping[str, Any]) -> None:
    """Raise ValueError if a public payload violates the deny-list / allowlist."""

    if payload.get("available") is False:
        if set(payload.keys()) != {"available"}:
            raise ValueError("unavailable public tee payload must be exactly {available:false}")
        return
    if payload.get("available") is not True:
        raise ValueError("public tee payload must set available true|false")

    unknown = set(payload.keys()) - PUBLIC_TEE_TOP_LEVEL_ALLOWLIST
    if unknown:
        raise ValueError(f"public tee payload has unknown keys: {sorted(unknown)}")

    serialized = json.dumps(payload, sort_keys=True, default=str)
    lowered = serialized.lower()
    for key in PUBLIC_TEE_DENY_KEYS:
        if _has_exact_json_key(serialized, key):
            raise ValueError(f"public tee payload exposes denied key: {key}")
    for marker in PUBLIC_TEE_DENY_SUBSTRINGS:
        if marker.lower() in lowered:
            raise ValueError(f"public tee payload contains denied substring: {marker}")

    preimage = payload.get("report_data_preimage")
    if isinstance(preimage, Mapping) and "review_nonce" in preimage:
        raise ValueError("public tee report_data_preimage must not include raw review_nonce")


def _has_exact_json_key(serialized: str, key: str) -> bool:
    """True when ``key`` appears as a JSON object key (not a longer key prefix).

    ``review_nonce_sha256`` must not match the deny key ``review_nonce`` because
    the needle requires the exact ``"{key}":`` boundary.
    """

    return f'"{key}":' in serialized


def _loads_object(raw: str | None) -> dict[str, Any] | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _coerce_submission_id(submission_id: int | str, envelope: Mapping[str, Any]) -> int | str:
    core = envelope.get("review_core")
    if isinstance(core, Mapping):
        core_id = core.get("submission_id")
        if isinstance(core_id, int) and not isinstance(core_id, bool):
            return core_id
        if isinstance(core_id, str) and core_id.isdigit():
            try:
                return int(core_id)
            except ValueError:
                return core_id
    if isinstance(submission_id, int) and not isinstance(submission_id, bool):
        return submission_id
    if isinstance(submission_id, str) and submission_id.isdigit():
        return int(submission_id)
    return submission_id


def _public_measurement(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, str] = {}
    for key in PUBLIC_TEE_MEASUREMENT_KEYS:
        item = value.get(key)
        if isinstance(item, str) and item:
            out[key] = item
    return out or None


def _public_quote_hex(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    # Lowercase even-length hex only; drop malformed rather than invent.
    cleaned = value.strip().lower()
    if len(cleaned) % 2 or any(c not in "0123456789abcdef" for c in cleaned):
        return None
    max_chars = MAX_PUBLIC_TEE_QUOTE_BYTES * 2
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def _public_event_log(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or not value:
        return None
    capped: list[dict[str, Any]] = []
    total_bytes = 0
    for item in value:
        if len(capped) >= MAX_PUBLIC_TEE_EVENT_LOG_ENTRIES:
            break
        if not isinstance(item, Mapping):
            continue
        # Copy only bounded public event fields; drop unexpected blobs.
        entry: dict[str, Any] = {}
        for key in ("event", "event_type", "digest", "event_payload", "imr", "seq"):
            if key not in item:
                continue
            val = item[key]
            if isinstance(val, (str, int)) and not isinstance(val, bool):
                entry[key] = val
        if not entry:
            continue
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if total_bytes + len(encoded) > MAX_PUBLIC_TEE_EVENT_LOG_BYTES:
            break
        total_bytes += len(encoded)
        capped.append(entry)
    return capped or None


def _public_report_data_preimage(envelope: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build inspectable preimage without raw review_nonce."""

    core = envelope.get("review_core")
    if not isinstance(core, Mapping):
        return None
    try:
        raw = review_report_data_preimage(core)
    except Exception:
        # Fail closed: omit preimage rather than leak partial core fields.
        return None
    nonce = raw.get("review_nonce")
    nonce_hash = None
    if isinstance(nonce, str) and nonce:
        nonce_hash = sha256(nonce.encode("utf-8")).hexdigest()
    public = {
        "domain": raw.get("domain"),
        "schema_version": raw.get("schema_version"),
        "review_digest": raw.get("review_digest"),
        "session_id": raw.get("session_id"),
        "issued_at_ms": raw.get("issued_at_ms"),
        "received_at_ms": raw.get("received_at_ms"),
    }
    if nonce_hash is not None:
        public["review_nonce_sha256"] = nonce_hash
    # Drop any Nones so FE does not see placeholder invent.
    return {k: v for k, v in public.items() if v is not None}


def _public_verification_outcome(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in PUBLIC_TEE_OUTCOME_KEYS:
        if key not in value:
            continue
        item = value[key]
        if key in {"measurement_allowlisted", "report_data_matched"}:
            if isinstance(item, bool):
                out[key] = item
        elif key == "verified_at_ms":
            if item is None or (isinstance(item, int) and not isinstance(item, bool)):
                out[key] = item
        elif isinstance(item, str):
            out[key] = item
    return out or None


def _quote_fingerprint(
    quote_hex: str | None,
    public_projection: Mapping[str, Any] | None,
) -> str | None:
    if isinstance(public_projection, Mapping):
        existing = public_projection.get("quote_fingerprint_sha256")
        if isinstance(existing, str) and len(existing) == 64:
            return existing
    if not quote_hex:
        return None
    try:
        return sha256(bytes.fromhex(quote_hex)).hexdigest()
    except ValueError:
        return None
