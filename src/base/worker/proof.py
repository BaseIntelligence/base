"""ExecutionProof construction and verification (architecture.md sec 3.4).

Every worker result carries an ``ExecutionProof`` envelope. The tier-0 core is a
deterministic ``manifest_sha256`` plus the worker's sr25519 signature binding
that hash to the work unit. The signed message format is PINNED identically to
prism (VAL-AGENT-008 / VAL-PRISM-006): the signature is over
``sha256(manifest_sha256 + ":" + unit_id)`` -- the sha256 digest of the UTF-8
bytes of the string ``{manifest_sha256}:{unit_id}`` -- so a proof produced by the
worker plane verifies with the same code as one produced by prism, and a proof
cannot be replayed across units.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from base.challenge_sdk.proof import (
    EXECUTION_PROOF_VERSION,
    execution_proof_signing_payload,
)
from base.schemas.worker import (
    PHALA_TDX_TIER,
    ExecutionProof,
    PhalaAttestation,
    PhalaMeasurement,
    ProviderInfo,
    WorkerSignature,
)
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature
from base.validator.agent.signing import RequestSigner

#: Result-payload key carrying the serialized :class:`ExecutionProof`.
PROOF_PAYLOAD_KEY = "execution_proof"
#: Result-payload key carrying the deterministic prism manifest hash.
MANIFEST_SHA256_PAYLOAD_KEY = "manifest_sha256"

#: Domain-separation tag bound into a Phala attestation's ``report_data``
#: (architecture.md sec 6). Prevents a quote minted for this purpose from being
#: repurposed under a different protocol tag.
PHALA_REPORT_DATA_TAG = "base-agent-challenge-v1"
#: Byte width of the TDX ``report_data`` field a quote carries.
PHALA_REPORT_DATA_BYTES = 64


def build_execution_proof(
    *,
    signer: RequestSigner,
    manifest_sha256: str,
    unit_id: str,
    tier: int | str = 0,
    provider: ProviderInfo | None = None,
    image_digest: str | None = None,
    attestation: dict[str, Any] | None = None,
) -> ExecutionProof:
    """Build and sign an ExecutionProof for ``unit_id`` under ``signer``.

    ``signer`` is the WORKER keypair; its public identity becomes
    ``worker_signature.worker_pubkey`` and it signs the pinned message.
    """

    signature = signer.sign(
        execution_proof_signing_payload(
            manifest_sha256=manifest_sha256, unit_id=unit_id
        )
    )
    return ExecutionProof(
        version=EXECUTION_PROOF_VERSION,
        tier=tier,
        manifest_sha256=manifest_sha256,
        image_digest=image_digest,
        provider=provider,
        worker_signature=WorkerSignature(worker_pubkey=signer.hotkey, sig=signature),
        attestation=attestation,
    )


def verify_execution_proof(
    proof: ExecutionProof,
    *,
    unit_id: str,
    signature_verifier: SignatureVerifier = verify_substrate_signature,
) -> bool:
    """Whether ``proof``'s worker signature verifies for ``unit_id`` (sr25519).

    Rejects a proof presented with a DIFFERENT ``unit_id`` than the one signed,
    so a proof cannot be replayed across units.
    """

    payload = execution_proof_signing_payload(
        manifest_sha256=proof.manifest_sha256, unit_id=unit_id
    )
    return signature_verifier(
        proof.worker_signature.worker_pubkey, payload, proof.worker_signature.sig
    )


def _canonical_measurement_mapping(
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
) -> dict[str, str]:
    """The static, allowlist-pinnable measurement subset (excludes ``rtmr3``)."""

    if isinstance(canonical_measurement, PhalaMeasurement):
        return canonical_measurement.canonical()
    static_fields = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash")
    return {field: str(canonical_measurement[field]) for field in static_fields}


def phala_report_data(
    *,
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str,
) -> bytes:
    """The 32-byte ``report_data`` digest binding a Phala run (architecture sec 6).

    ``SHA256`` over a canonical (sorted-key, compact) JSON preimage of
    ``{tag, canonical_measurement, agent_hash, sorted(task_ids), scores_digest,
    validator_nonce}`` with ``tag == PHALA_REPORT_DATA_TAG``. ``task_ids`` are
    sorted so the binding is order-independent; the measurement contributes only
    its static, pinnable subset (``rtmr3`` is runtime and excluded). Every other
    component is bound, so changing any one changes the digest.

    This is the single source of truth for the derivation shared by the image
    emitter (M1) and the validator/master verifier (M4): both MUST call this
    function rather than re-implementing sec 6.
    """

    preimage = {
        "tag": PHALA_REPORT_DATA_TAG,
        "canonical_measurement": _canonical_measurement_mapping(canonical_measurement),
        "agent_hash": agent_hash,
        "task_ids": sorted(task_ids),
        "scores_digest": scores_digest,
        "validator_nonce": validator_nonce,
    }
    encoded = json.dumps(preimage, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).digest()


def phala_report_data_hex(
    *,
    canonical_measurement: PhalaMeasurement | Mapping[str, Any],
    agent_hash: str,
    task_ids: Iterable[str],
    scores_digest: str,
    validator_nonce: str,
) -> str:
    """``report_data`` as a 64-byte TDX field (128 hex chars, left-aligned).

    The 32-byte :func:`phala_report_data` digest occupies the leading bytes and
    the trailing bytes are zero, matching Phala's observed left-aligned zero-pad
    round-trip for the fixed-width quote field.
    """

    digest = phala_report_data(
        canonical_measurement=canonical_measurement,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=scores_digest,
        validator_nonce=validator_nonce,
    )
    return digest.ljust(PHALA_REPORT_DATA_BYTES, b"\x00").hex()


def build_phala_execution_proof(
    *,
    signer: RequestSigner,
    manifest_sha256: str,
    unit_id: str,
    attestation: PhalaAttestation | Mapping[str, Any],
    provider: ProviderInfo | None = None,
    image_digest: str | None = None,
) -> ExecutionProof:
    """Build a Phala-tier ExecutionProof carrying a TDX attestation payload.

    The envelope keeps the tier-0 worker sr25519 signature over the pinned
    ``sha256(f"{manifest_sha256}:{unit_id}")`` message, so
    :func:`verify_execution_proof` still validates the worker-signature layer;
    ``tier`` is set to :data:`PHALA_TDX_TIER` and ``attestation`` carries the
    serialized :class:`PhalaAttestation`. Cryptographic quote verification is
    layered on top by the validator/master (milestone M4); this helper does not
    fork a parallel envelope.
    """

    payload = (
        attestation
        if isinstance(attestation, PhalaAttestation)
        else PhalaAttestation.model_validate(dict(attestation))
    )
    return build_execution_proof(
        signer=signer,
        manifest_sha256=manifest_sha256,
        unit_id=unit_id,
        tier=PHALA_TDX_TIER,
        provider=provider,
        image_digest=image_digest,
        attestation=payload.model_dump(mode="json"),
    )


__all__ = [
    "EXECUTION_PROOF_VERSION",
    "MANIFEST_SHA256_PAYLOAD_KEY",
    "PHALA_REPORT_DATA_BYTES",
    "PHALA_REPORT_DATA_TAG",
    "PHALA_TDX_TIER",
    "PROOF_PAYLOAD_KEY",
    "build_execution_proof",
    "build_phala_execution_proof",
    "execution_proof_signing_payload",
    "phala_report_data",
    "phala_report_data_hex",
    "verify_execution_proof",
]
