"""Attestation payload schema and HMAC-SHA256 sign/verify.

Mechanism 1+6 surface of prism-lium image attestation: the in-image sidecar
signs ``(nonce, digest, pod_id, variant, sealed_manifest_hashes,
build_secret_response)``. **BASE holds the verify key.** The sidecar holds
sign material derived from the per-build secret (see ``derive_attestation_key``).

Honesty constraints (binding — do not weaken in callers):

* A valid signature proves **only** that an entity holding the in-image secret
  responded. It is **not** a hardware root of trust.
* A valid signature is **never sufficient alone for tier elevation** (B3).
  Tier 1 is granted only by the later ``constation_ok`` predicate, which
  requires allowlist hit, nonce lifecycle, signature, sealed-manifest match,
  corroboration, and continuous constation together.
* This module deliberately exposes **no** tier-grant API.

Crypto: HMAC-SHA256 over a canonical JSON encoding (stdlib only — no extra
deps). Symmetric: the verify key BASE stores is the same derived key the
sidecar uses to sign.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal, assert_never

SCHEMA_VERSION: Final[str] = "prism_attestation_payload.v1"
ALGORITHM: Final[str] = "hmac-sha256"

# Domain-separated key derivation label (not a password; not a TEE claim).
_KEY_DERIVE_INFO: Final[bytes] = b"prism-lium-attestation-hmac-v1"

_IMAGE_DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]+$")

Variant = Literal["cpu", "cuda"]
VALID_VARIANTS: Final[frozenset[str]] = frozenset({"cpu", "cuda"})


class AttestationVerifyReason(StrEnum):
    """Machine-consumed verify outcomes. Callers must not string-match prose."""

    OK = "ok"
    SIGNATURE_MISMATCH = "signature_mismatch"
    INVALID_PAYLOAD = "invalid_payload"
    UNSUPPORTED_ALGORITHM = "unsupported_algorithm"
    UNSUPPORTED_SCHEMA = "unsupported_schema"
    EMPTY_KEY = "empty_key"


class AttestationPayloadError(ValueError):
    """Payload failed structural validation before signing/verifying."""


class AttestationVerifyError(Exception):
    """Raised when ``raise_on_failure=True`` and verification does not pass."""

    def __init__(self, message: str, *, reason: AttestationVerifyReason) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class AttestationPayload:
    """Unsigned attestation fields the sidecar binds under HMAC.

    Field meanings (tamper-evidence only — not hardware attestation):

    * ``nonce`` — BASE-issued single-use challenge id (opaque string).
    * ``digest`` — container image content digest ``sha256:<64-hex>``.
    * ``pod_id`` — Lium (or equivalent) pod identifier the run claims.
    * ``variant`` — ``cpu`` or ``cuda`` image variant.
    * ``sealed_manifest_hashes`` — path → lowercase SHA-256 hex of sealed files.
    * ``build_secret_response`` — response proving possession of the in-image
      build secret for this nonce (opaque lowercase hex; typically
      HMAC-SHA256(build_secret, nonce)).
    """

    nonce: str
    digest: str
    pod_id: str
    variant: Variant
    sealed_manifest_hashes: Mapping[str, str]
    build_secret_response: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "nonce", _require_nonempty_str("nonce", self.nonce))
        object.__setattr__(self, "digest", _require_image_digest(self.digest))
        object.__setattr__(self, "pod_id", _require_nonempty_str("pod_id", self.pod_id))
        object.__setattr__(self, "variant", _require_variant(self.variant))
        object.__setattr__(
            self,
            "sealed_manifest_hashes",
            _require_manifest_hashes(self.sealed_manifest_hashes),
        )
        object.__setattr__(
            self,
            "build_secret_response",
            _require_hex_str("build_secret_response", self.build_secret_response),
        )


@dataclass(frozen=True, slots=True)
class SignedAttestation:
    """Payload plus HMAC-SHA256 signature hex. BASE verifies with its key."""

    payload: AttestationPayload
    signature: str
    algorithm: str = ALGORITHM
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "signature", _require_hex_str("signature", self.signature)
        )
        object.__setattr__(
            self, "algorithm", _require_nonempty_str("algorithm", self.algorithm)
        )
        object.__setattr__(
            self,
            "schema_version",
            _require_nonempty_str("schema_version", self.schema_version),
        )


@dataclass(frozen=True, slots=True)
class AttestationVerifyResult:
    """Structured verify outcome. ``ok`` never implies tier elevation."""

    ok: bool
    reason: AttestationVerifyReason
    payload: AttestationPayload | None = None


def derive_attestation_key(build_secret: bytes | str) -> bytes:
    """Derive the HMAC key BASE stores for verify and the sidecar uses to sign.

    The raw build secret is never used directly as the HMAC key. Derivation is
    domain-separated HMAC-SHA256 so accidental reuse of the secret bytes as a
    different protocol key does not collide.

    Returns 32 bytes. Empty input is rejected.
    """
    secret = (
        build_secret.encode("utf-8")
        if isinstance(build_secret, str)
        else bytes(build_secret)
    )
    if not secret:
        raise AttestationPayloadError("build_secret must be non-empty")
    return hmac.new(_KEY_DERIVE_INFO, secret, hashlib.sha256).digest()


def compute_build_secret_response(*, build_secret: bytes | str, nonce: str) -> str:
    """Compute the ``build_secret_response`` field for a challenge nonce.

    ``HMAC-SHA256(build_secret, nonce_utf8)`` hex. Proves possession of the
    in-image secret for this nonce; still not a hardware root of trust.
    """
    secret = (
        build_secret.encode("utf-8")
        if isinstance(build_secret, str)
        else bytes(build_secret)
    )
    if not secret:
        raise AttestationPayloadError("build_secret must be non-empty")
    nonce_s = _require_nonempty_str("nonce", nonce)
    return hmac.new(secret, nonce_s.encode("utf-8"), hashlib.sha256).hexdigest()


def canonical_payload_bytes(payload: AttestationPayload) -> bytes:
    """Deterministic UTF-8 JSON bytes covered by the HMAC.

    Keys sorted; ``sealed_manifest_hashes`` sorted by path. Separators compact.
    Schema version and algorithm are **not** inside the payload body — they
    travel beside the signature on ``SignedAttestation``.
    """
    body: dict[str, Any] = {
        "build_secret_response": payload.build_secret_response,
        "digest": payload.digest,
        "nonce": payload.nonce,
        "pod_id": payload.pod_id,
        "sealed_manifest_hashes": {
            path: payload.sealed_manifest_hashes[path]
            for path in sorted(payload.sealed_manifest_hashes)
        },
        "variant": payload.variant,
    }
    raw = json.dumps(
        body, separators=(",", ":"), sort_keys=True, ensure_ascii=True
    )
    return raw.encode("utf-8")


def sign_attestation_payload(
    payload: AttestationPayload,
    *,
    signing_key: bytes,
) -> SignedAttestation:
    """Sign ``payload`` with the sidecar signing key (HMAC-SHA256).

    BASE holds the matching verify key (same bytes for HMAC). A successful
    later verify proves only that an entity holding the in-image secret
    responded — never sufficient alone for tier elevation (B3).
    """
    key = _require_key(signing_key)
    digest = hmac.new(key, canonical_payload_bytes(payload), hashlib.sha256).hexdigest()
    return SignedAttestation(
        payload=payload,
        signature=digest,
        algorithm=ALGORITHM,
        schema_version=SCHEMA_VERSION,
    )


def verify_attestation_payload(
    signed: SignedAttestation,
    *,
    verify_key: bytes,
    raise_on_failure: bool = False,
) -> AttestationVerifyResult:
    """Verify a signed attestation with the BASE-held verify key.

    A valid signature proves only that an entity holding the in-image secret
    responded. It is not a hardware root of trust and is never sufficient
    alone for tier elevation (B3). Callers must still run ``constation_ok``.

    Tampering any signed field (or using the wrong key) yields
    ``SIGNATURE_MISMATCH``.
    """
    result = _verify(signed, verify_key=verify_key)
    if raise_on_failure and not result.ok:
        raise AttestationVerifyError(
            f"attestation verify failed: {result.reason.value}",
            reason=result.reason,
        )
    return result


def _verify(signed: SignedAttestation, *, verify_key: bytes) -> AttestationVerifyResult:
    if not verify_key:
        return AttestationVerifyResult(
            ok=False, reason=AttestationVerifyReason.EMPTY_KEY
        )
    if signed.algorithm != ALGORITHM:
        return AttestationVerifyResult(
            ok=False, reason=AttestationVerifyReason.UNSUPPORTED_ALGORITHM
        )
    if signed.schema_version != SCHEMA_VERSION:
        return AttestationVerifyResult(
            ok=False, reason=AttestationVerifyReason.UNSUPPORTED_SCHEMA
        )
    try:
        # Re-validate structure (defense in depth if constructed without __post_init__).
        payload = signed.payload
        expected = hmac.new(
            verify_key, canonical_payload_bytes(payload), hashlib.sha256
        ).hexdigest()
    except (AttestationPayloadError, TypeError, ValueError):
        return AttestationVerifyResult(
            ok=False, reason=AttestationVerifyReason.INVALID_PAYLOAD
        )

    if not hmac.compare_digest(expected, signed.signature.lower()):
        return AttestationVerifyResult(
            ok=False, reason=AttestationVerifyReason.SIGNATURE_MISMATCH
        )
    return AttestationVerifyResult(
        ok=True, reason=AttestationVerifyReason.OK, payload=payload
    )


def _require_key(key: bytes) -> bytes:
    if not isinstance(key, (bytes, bytearray)):
        raise AttestationPayloadError("signing/verify key must be bytes")
    raw = bytes(key)
    if not raw:
        raise AttestationPayloadError("signing/verify key must be non-empty")
    return raw


def _require_nonempty_str(field: str, value: str) -> str:
    if not isinstance(value, str):
        raise AttestationPayloadError(f"{field} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise AttestationPayloadError(f"{field} must be non-empty")
    return cleaned


def _require_image_digest(value: str) -> str:
    cleaned = _require_nonempty_str("digest", value).lower()
    if not _IMAGE_DIGEST_RE.fullmatch(cleaned):
        raise AttestationPayloadError(
            f"digest must be sha256:<64 lowercase hex>, got {value!r}"
        )
    return cleaned


def _require_variant(value: Variant | str) -> Variant:
    if not isinstance(value, str):
        raise AttestationPayloadError(f"variant must be str, got {type(value)!r}")
    cleaned = value.strip().lower()
    if cleaned not in VALID_VARIANTS:
        raise AttestationPayloadError(
            f"variant must be one of {sorted(VALID_VARIANTS)}, got {value!r}"
        )
    if cleaned == "cpu":
        return "cpu"
    if cleaned == "cuda":
        return "cuda"
    assert_never(cleaned)  # type: ignore[arg-type]


def _require_hex_str(field: str, value: str) -> str:
    cleaned = _require_nonempty_str(field, value).lower()
    if not _HEX_RE.fullmatch(cleaned) or len(cleaned) % 2 != 0:
        raise AttestationPayloadError(f"{field} must be even-length lowercase hex")
    return cleaned


def _require_manifest_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AttestationPayloadError("sealed_manifest_hashes must be a mapping")
    if not value:
        raise AttestationPayloadError("sealed_manifest_hashes must be non-empty")
    out: dict[str, str] = {}
    for raw_path, raw_hash in value.items():
        path = _require_nonempty_str("sealed_manifest_hashes path", str(raw_path))
        digest = _require_nonempty_str(
            "sealed_manifest_hashes value", str(raw_hash)
        ).lower()
        if not _HEX64_RE.fullmatch(digest):
            raise AttestationPayloadError(
                f"sealed_manifest_hashes[{path!r}] must be 64-char lowercase hex"
            )
        if path in out:
            raise AttestationPayloadError(f"duplicate sealed path: {path!r}")
        out[path] = digest
    return out


__all__ = [
    "ALGORITHM",
    "SCHEMA_VERSION",
    "AttestationPayload",
    "AttestationPayloadError",
    "AttestationVerifyError",
    "AttestationVerifyReason",
    "AttestationVerifyResult",
    "SignedAttestation",
    "VALID_VARIANTS",
    "Variant",
    "canonical_payload_bytes",
    "compute_build_secret_response",
    "derive_attestation_key",
    "sign_attestation_payload",
    "verify_attestation_payload",
]
