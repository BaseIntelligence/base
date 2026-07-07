"""ExecutionProof construction/verification tests (VAL-AGENT-008).

Uses REAL sr25519 keypairs (bittensor) so the pinned signature format is
exercised end-to-end: the signature is over ``sha256(f"{manifest}:{unit_id}")``
and verifies against the worker pubkey, but not when replayed onto another unit.
"""

from __future__ import annotations

import hashlib

from base.security.miner_auth import verify_substrate_signature
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.proof import (
    EXECUTION_PROOF_VERSION,
    build_execution_proof,
    execution_proof_signing_payload,
    verify_execution_proof,
)

MANIFEST = "a" * 64
UNIT_ID = "submission-123"


def _signer(uri: str = "//WorkerAlice") -> KeypairRequestSigner:
    import bittensor as bt

    return KeypairRequestSigner(bt.Keypair.create_from_uri(uri))


def test_signing_payload_is_sha256_of_pinned_string() -> None:
    payload = execution_proof_signing_payload(manifest_sha256=MANIFEST, unit_id=UNIT_ID)
    assert payload == hashlib.sha256(f"{MANIFEST}:{UNIT_ID}".encode()).digest()


def test_build_execution_proof_is_tier0_with_worker_signature() -> None:
    signer = _signer()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    assert proof.version == EXECUTION_PROOF_VERSION == 1
    assert proof.tier == 0
    assert proof.manifest_sha256 == MANIFEST
    assert proof.attestation is None
    assert proof.worker_signature.worker_pubkey == signer.hotkey


def test_worker_signature_verifies_over_pinned_message() -> None:
    signer = _signer()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is True

    # Independent verification against the pubkey using the shared sr25519 helper.
    payload = execution_proof_signing_payload(manifest_sha256=MANIFEST, unit_id=UNIT_ID)
    assert verify_substrate_signature(
        proof.worker_signature.worker_pubkey, payload, proof.worker_signature.sig
    )


def test_proof_cannot_be_replayed_across_units() -> None:
    signer = _signer()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    assert verify_execution_proof(proof, unit_id="different-unit") is False


def test_tampered_manifest_hash_fails_verification() -> None:
    signer = _signer()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    proof.manifest_sha256 = "b" * 64
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is False


def test_corrupted_signature_fails_verification() -> None:
    signer = _signer()
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id=UNIT_ID
    )
    proof.worker_signature.sig = "0x" + "00" * 64
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is False


def test_tier1_when_image_digest_and_pod_present() -> None:
    from base.schemas.worker import ProviderInfo

    signer = _signer()
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        tier=1,
        image_digest="sha256:" + "c" * 64,
        provider=ProviderInfo(
            name="lium", executor_id="ex-1", pod_id="pod-1", miner_hotkey="miner-H1"
        ),
    )
    assert proof.tier == 1
    assert proof.image_digest == "sha256:" + "c" * 64
    assert proof.provider is not None
    assert proof.provider.pod_id == "pod-1"
    assert verify_execution_proof(proof, unit_id=UNIT_ID) is True
