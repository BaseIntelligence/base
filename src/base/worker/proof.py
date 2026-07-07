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
from typing import Any

from base.schemas.worker import ExecutionProof, ProviderInfo, WorkerSignature
from base.security.miner_auth import SignatureVerifier, verify_substrate_signature
from base.validator.agent.signing import RequestSigner

EXECUTION_PROOF_VERSION = 1

#: Result-payload key carrying the serialized :class:`ExecutionProof`.
PROOF_PAYLOAD_KEY = "execution_proof"
#: Result-payload key carrying the deterministic prism manifest hash.
MANIFEST_SHA256_PAYLOAD_KEY = "manifest_sha256"


def execution_proof_signing_payload(*, manifest_sha256: str, unit_id: str) -> bytes:
    """The exact bytes an ExecutionProof signature covers (pinned format).

    ``sha256`` digest of the UTF-8 bytes of ``{manifest_sha256}:{unit_id}``.
    """

    message = f"{manifest_sha256}:{unit_id}".encode()
    return hashlib.sha256(message).digest()


def build_execution_proof(
    *,
    signer: RequestSigner,
    manifest_sha256: str,
    unit_id: str,
    tier: int = 0,
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


__all__ = [
    "EXECUTION_PROOF_VERSION",
    "MANIFEST_SHA256_PAYLOAD_KEY",
    "PROOF_PAYLOAD_KEY",
    "build_execution_proof",
    "execution_proof_signing_payload",
    "verify_execution_proof",
]
