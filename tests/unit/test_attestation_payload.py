"""TDD tests for attestation payload schema + HMAC-SHA256 verification.

Checkbox 9 (prism-lium-image-attestation): sidecar signs
(nonce, digest, pod_id, variant, sealed_manifest_hashes, build_secret_response).
BASE holds the verify key. A valid signature proves only that an entity holding
the in-image secret responded — not a hardware root of trust, and never
sufficient alone for tier elevation (B3).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from base.attestation.payload import (
    ALGORITHM,
    SCHEMA_VERSION,
    AttestationPayload,
    AttestationVerifyError,
    AttestationVerifyReason,
    SignedAttestation,
    canonical_payload_bytes,
    derive_attestation_key,
    sign_attestation_payload,
    verify_attestation_payload,
)

# Fixed test vectors — must match prism-recipe/tests/test_attestation_payload.py
NONCE = "550e8400-e29b-41d4-a716-446655440000"
DIGEST = "sha256:" + ("ab" * 32)
POD_ID = "pod_test_001"
VARIANT = "cuda"
MANIFEST_HASHES = {
    "src/prism_recipe/harness.py": "a" * 64,
    "src/prism_recipe/gpu_train.py": "b" * 64,
}
BUILD_SECRET = b"unit-test-build-secret-not-for-prod"
BUILD_SECRET_RESPONSE = hmac.new(
    BUILD_SECRET, NONCE.encode("utf-8"), hashlib.sha256
).hexdigest()
VERIFY_KEY = derive_attestation_key(BUILD_SECRET)


def _payload(**overrides: Any) -> AttestationPayload:
    fields: dict[str, Any] = {
        "nonce": NONCE,
        "digest": DIGEST,
        "pod_id": POD_ID,
        "variant": VARIANT,
        "sealed_manifest_hashes": dict(MANIFEST_HASHES),
        "build_secret_response": BUILD_SECRET_RESPONSE,
    }
    fields.update(overrides)
    return AttestationPayload(**fields)


def test_well_formed_signed_payload_verifies() -> None:
    """S1 happy: Given signed payload, When BASE verifies with key, Then ok."""
    payload = _payload()
    signed = sign_attestation_payload(payload, signing_key=VERIFY_KEY)

    result = verify_attestation_payload(signed, verify_key=VERIFY_KEY)

    assert result.ok is True
    assert result.reason is AttestationVerifyReason.OK
    assert result.payload == payload
    assert signed.algorithm == ALGORITHM
    assert signed.schema_version == SCHEMA_VERSION
    assert len(signed.signature) == 64  # sha256 hex


def test_tamper_digest_one_byte_rejected() -> None:
    """S2: Flip one byte of digest after sign → verify fails."""
    signed = sign_attestation_payload(_payload(), signing_key=VERIFY_KEY)
    tampered_digest = "sha256:" + ("ac" + "ab" * 31)  # first byte flipped ab→ac
    assert tampered_digest != DIGEST
    tampered = SignedAttestation(
        payload=_payload(digest=tampered_digest),
        signature=signed.signature,
        algorithm=signed.algorithm,
        schema_version=signed.schema_version,
    )

    result = verify_attestation_payload(tampered, verify_key=VERIFY_KEY)

    assert result.ok is False
    assert result.reason is AttestationVerifyReason.SIGNATURE_MISMATCH


def test_tamper_sealed_manifest_hashes_rejected() -> None:
    """S2b: Flip one byte in a sealed manifest hash → verify fails."""
    signed = sign_attestation_payload(_payload(), signing_key=VERIFY_KEY)
    bad_hashes = dict(MANIFEST_HASHES)
    original = bad_hashes["src/prism_recipe/harness.py"]
    flipped = "0" if original[0] != "0" else "1"
    bad_hashes["src/prism_recipe/harness.py"] = flipped + original[1:]
    tampered = SignedAttestation(
        payload=_payload(sealed_manifest_hashes=bad_hashes),
        signature=signed.signature,
        algorithm=signed.algorithm,
        schema_version=signed.schema_version,
    )

    result = verify_attestation_payload(tampered, verify_key=VERIFY_KEY)

    assert result.ok is False
    assert result.reason is AttestationVerifyReason.SIGNATURE_MISMATCH


@pytest.mark.parametrize(
    "field_name,override",
    [
        ("nonce", {"nonce": "00000000-0000-4000-8000-000000000000"}),
        ("pod_id", {"pod_id": "pod_other"}),
        ("variant", {"variant": "cpu"}),
        (
            "build_secret_response",
            {"build_secret_response": "ff" * 32},
        ),
    ],
)
def test_tamper_any_field_rejected(field_name: str, override: dict[str, Any]) -> None:
    """Any single-field alteration after sign must fail verification."""
    signed = sign_attestation_payload(_payload(), signing_key=VERIFY_KEY)
    tampered = SignedAttestation(
        payload=_payload(**override),
        signature=signed.signature,
        algorithm=signed.algorithm,
        schema_version=signed.schema_version,
    )

    result = verify_attestation_payload(tampered, verify_key=VERIFY_KEY)

    assert result.ok is False, f"tamper of {field_name} must fail"
    assert result.reason is AttestationVerifyReason.SIGNATURE_MISMATCH


def test_wrong_verify_key_rejected() -> None:
    signed = sign_attestation_payload(_payload(), signing_key=VERIFY_KEY)
    other_key = derive_attestation_key(b"different-secret")

    result = verify_attestation_payload(signed, verify_key=other_key)

    assert result.ok is False
    assert result.reason is AttestationVerifyReason.SIGNATURE_MISMATCH


def test_canonical_bytes_are_deterministic_and_order_independent() -> None:
    """Manifest hash map order must not affect the signed bytes."""
    p1 = _payload(
        sealed_manifest_hashes={
            "src/prism_recipe/harness.py": "a" * 64,
            "src/prism_recipe/gpu_train.py": "b" * 64,
        }
    )
    p2 = _payload(
        sealed_manifest_hashes={
            "src/prism_recipe/gpu_train.py": "b" * 64,
            "src/prism_recipe/harness.py": "a" * 64,
        }
    )
    assert canonical_payload_bytes(p1) == canonical_payload_bytes(p2)


def test_derive_attestation_key_is_stable() -> None:
    k1 = derive_attestation_key(BUILD_SECRET)
    k2 = derive_attestation_key(BUILD_SECRET)
    assert k1 == k2
    assert len(k1) == 32
    assert k1 != BUILD_SECRET  # derived, not raw secret


def test_signature_never_grants_tier_api_absent() -> None:
    """B3: module must not expose tier elevation from signature alone."""
    import base.attestation.payload as mod

    forbidden = (
        "effective_tier",
        "grant_tier",
        "elevate_tier",
        "tier_from_signature",
        "constation_ok",
    )
    public = {n for n in dir(mod) if not n.startswith("_")}
    for name in forbidden:
        assert name not in public, f"forbidden tier API leaked: {name}"


def test_verify_raises_optional_strict_mode() -> None:
    signed = sign_attestation_payload(_payload(), signing_key=VERIFY_KEY)
    bad = SignedAttestation(
        payload=_payload(digest="sha256:" + ("00" * 32)),
        signature=signed.signature,
        algorithm=signed.algorithm,
        schema_version=signed.schema_version,
    )
    with pytest.raises(AttestationVerifyError) as excinfo:
        verify_attestation_payload(bad, verify_key=VERIFY_KEY, raise_on_failure=True)
    assert excinfo.value.reason is AttestationVerifyReason.SIGNATURE_MISMATCH


def test_docstring_states_not_hardware_root_and_not_sufficient_for_tier() -> None:
    """Docstrings must state the B3 honesty constraints plainly."""
    import base.attestation.payload as mod

    doc = (mod.__doc__ or "") + (verify_attestation_payload.__doc__ or "")
    lowered = doc.lower()
    assert "entity holding" in lowered or "in-image secret" in lowered
    assert "hardware" in lowered or "root of trust" in lowered
    assert "never sufficient" in lowered or "not sufficient" in lowered
    assert "tier" in lowered


def test_fixed_vector_signature_hex_matches_known_value() -> None:
    """Cross-repo lock: same inputs → same signature hex (prism-recipe twin)."""
    payload = _payload()
    signed = sign_attestation_payload(payload, signing_key=VERIFY_KEY)
    # Recompute expected with stdlib only (characterization of algorithm).
    expected = hmac.new(
        VERIFY_KEY, canonical_payload_bytes(payload), hashlib.sha256
    ).hexdigest()
    assert signed.signature == expected
    # Pin a concrete hex so base and prism-recipe cannot drift silently.
    assert signed.signature == (
        "8eb6bfbed9bec0503597de0ccb4e8293f73936c358f3ac34ca0ffe38dd47b024"
    )
