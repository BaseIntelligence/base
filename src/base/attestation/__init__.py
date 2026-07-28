"""Prism-lium image attestation protocol (pure crypto + schema).

See ``payload`` module docstring: signature proves only that an entity holding
the in-image secret responded — not a hardware root of trust, and never
sufficient alone for tier elevation (B3).
"""

from base.attestation.payload import (
    ALGORITHM,
    SCHEMA_VERSION,
    VALID_VARIANTS,
    AttestationPayload,
    AttestationPayloadError,
    AttestationVerifyError,
    AttestationVerifyReason,
    AttestationVerifyResult,
    SignedAttestation,
    canonical_payload_bytes,
    compute_build_secret_response,
    derive_attestation_key,
    sign_attestation_payload,
    verify_attestation_payload,
)

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
    "canonical_payload_bytes",
    "compute_build_secret_response",
    "derive_attestation_key",
    "sign_attestation_payload",
    "verify_attestation_payload",
]
