"""ExecutionProof construction/verification unit tests (VAL-PRISM-002/003/004/005/006/008).

Offline, no GPU: exercises ``prism_challenge.proof`` directly with REAL sr25519 keypairs
(bittensor) so the pinned signature format is verified end-to-end, and pins the canonical manifest
hashing, tier computation, and the held-out-secret security invariant.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from prism_challenge import proof
from prism_challenge.auth import verify_hotkey_signature
from prism_challenge.proof import (
    ExecutionProof,
    build_execution_proof,
    build_execution_proof_from_manifest,
    compute_manifest_sha256,
    verify_execution_proof,
    worker_signer_from_key,
)

UNIT_ID = "submission-abc123"

MANIFEST_A: dict[str, Any] = {
    "schema_version": "prism_run_manifest.v2",
    "run": {"world_size": 1, "nproc_per_node": 1, "device": "cpu"},
    "metrics": {"prequential_bpb": 1.2345, "available_bytes": 4096.0},
    "score": {"final_score": 0.42},
}
MANIFEST_B: dict[str, Any] = {
    "schema_version": "prism_run_manifest.v2",
    "metrics": {"prequential_bpb": 0.9},
    "compute": {"gpu_count": 1, "world_size": 1, "nproc_per_node": 1, "device": "cpu"},
    "artifacts": {"trained_state": "trained_state.pt"},
}


def _signer(uri: str = "//WorkerAlice") -> proof.KeypairWorkerSigner:
    return worker_signer_from_key(uri)


# --- VAL-PRISM-002: canonical manifest hashing (both directions + order-insensitive) ------------


@pytest.mark.parametrize("manifest", [MANIFEST_A, MANIFEST_B])
def test_manifest_sha256_matches_on_disk_bytes_both_ways(
    manifest: dict[str, Any], tmp_path: Path
) -> None:
    signer = _signer()
    # Persist the manifest exactly as the runner/host do (canonical: sort_keys=True, indent=2).
    path = tmp_path / "prism_run_manifest.v2.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    built = build_execution_proof_from_manifest(signer=signer, unit_id=UNIT_ID, manifest_path=path)

    # (i) sha256 of the EXACT on-disk bytes, computed OUTSIDE the proof code path.
    disk_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # (ii) sha256 of json.dumps(json.loads(file), sort_keys=True, indent=2) bytes.
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    canonical_digest = hashlib.sha256(
        json.dumps(reloaded, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()

    assert built.manifest_sha256 == disk_digest
    assert built.manifest_sha256 == canonical_digest
    assert len(built.manifest_sha256) == 64
    assert built.manifest_sha256 == built.manifest_sha256.lower()
    assert all(c in "0123456789abcdef" for c in built.manifest_sha256)


def test_manifest_hash_is_key_order_insensitive() -> None:
    reordered = {key: MANIFEST_A[key] for key in reversed(list(MANIFEST_A))}
    assert list(reordered) != list(MANIFEST_A)
    assert compute_manifest_sha256(reordered) == compute_manifest_sha256(MANIFEST_A)
    signer = _signer()
    p1 = build_execution_proof_from_manifest(signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A)
    p2 = build_execution_proof_from_manifest(signer=signer, unit_id=UNIT_ID, manifest=reordered)
    assert p1.manifest_sha256 == p2.manifest_sha256


def test_manifest_source_is_exactly_one() -> None:
    signer = _signer()
    with pytest.raises(ValueError):
        build_execution_proof_from_manifest(signer=signer, unit_id=UNIT_ID)
    with pytest.raises(ValueError):
        build_execution_proof_from_manifest(
            signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, manifest_bytes=b"{}"
        )


# --- VAL-PRISM-003: tier 0 with no provider metadata -------------------------------------------


def test_tier0_when_no_provider_metadata() -> None:
    signer = _signer()
    digest = compute_manifest_sha256(MANIFEST_A)
    empty_env: dict[str, str] = {}
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=empty_env
    )
    assert p.tier == 0
    assert p.image_digest is None
    assert p.provider is None
    assert p.attestation is None
    assert p.manifest_sha256 == digest


# --- VAL-PRISM-004: tier 1 requires BOTH image digest AND pod metadata --------------------------

_IMAGE_DIGEST = "sha256:" + "c" * 64
_FULL_ENV = {
    proof.PROVIDER_NAME_ENV: "lium",
    proof.EXECUTOR_ID_ENV: "ex-1",
    proof.POD_ID_ENV: "pod-1",
    proof.MINER_HOTKEY_ENV: "miner-H1",
    proof.IMAGE_DIGEST_ENV: _IMAGE_DIGEST,
}


def test_tier1_full_provider_env() -> None:
    signer = _signer()
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=dict(_FULL_ENV)
    )
    assert p.tier == 1
    assert p.image_digest == _IMAGE_DIGEST
    assert p.provider is not None
    assert p.provider.name == "lium"
    assert p.provider.executor_id == "ex-1"
    assert p.provider.pod_id == "pod-1"
    assert p.provider.miner_hotkey == "miner-H1"


def test_tier1_downgrades_to_tier0_without_pod_or_digest() -> None:
    signer = _signer()
    digest_only = {k: v for k, v in _FULL_ENV.items() if k != proof.POD_ID_ENV}
    pod_only = {k: v for k, v in _FULL_ENV.items() if k != proof.IMAGE_DIGEST_ENV}
    p_digest_only = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=digest_only
    )
    p_pod_only = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=pod_only
    )
    assert p_digest_only.tier == 0
    assert p_pod_only.tier == 0


# --- VAL-PRISM-005 / VAL-TEE: tier-2 claim needs closed structured attestation -----------------


def test_opaque_nonempty_attestation_does_not_claim_tier2() -> None:
    """Opaque strings are insufficient: claimed tier stays image/pod tier-1, never 2."""
    signer = _signer()
    attestation = {"tdx_quote_b64": "QUOTE", "gpu_eat_jwt": "JWT"}
    env = {**_FULL_ENV, proof.ATTESTATION_ENV: json.dumps(attestation)}
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=env
    )
    assert p.tier != 2
    assert p.tier == 1
    assert p.attestation == attestation


def test_structured_attestation_claims_tier2_but_is_unverified() -> None:
    signer = _signer()
    attestation = {
        "version": 1,
        "provider": "local_fixture",
        "evidence_type": "prism.tee.v1",
        "tdx_quote_b64": "QUJDRA==",
        "gpu_eat_jwt": "aaa.bbb.ccc",
    }
    env = {**_FULL_ENV, proof.ATTESTATION_ENV: json.dumps(attestation)}
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=env
    )
    # Emission may claim tier 2 for wire compat; effective elevation never exceeds IMAGE_PIN tier-1.
    assert p.tier == 2
    assert p.attestation == attestation


def test_tier_never_2_without_attestation() -> None:
    signer = _signer()
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=dict(_FULL_ENV)
    )
    assert p.attestation is None
    assert p.tier != 2


def test_attestation_object_missing_documented_keys_is_not_tier2() -> None:
    signer = _signer()
    env = {**_FULL_ENV, proof.ATTESTATION_ENV: json.dumps({"unrelated": "x"})}
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=env
    )
    assert p.tier != 2
    assert p.tier == 1  # falls back to the digest+pod tier


# --- VAL-PRISM-006: worker signature binds manifest_sha256 + unit_id ----------------------------


def test_worker_signature_verifies_over_pinned_message() -> None:
    signer = _signer()
    p = build_execution_proof(
        signer=signer, manifest_sha256=compute_manifest_sha256(MANIFEST_A), unit_id=UNIT_ID
    )
    assert verify_execution_proof(p, unit_id=UNIT_ID) is True
    # Independent verifier given only (proof JSON, unit_id) accepts the signature.
    reloaded = ExecutionProof.model_validate_json(p.model_dump_json())
    assert verify_execution_proof(reloaded, unit_id=UNIT_ID) is True
    # Independent raw sr25519 check over the pinned message.
    payload = proof.execution_proof_signing_payload(
        manifest_sha256=p.manifest_sha256, unit_id=UNIT_ID
    )
    assert verify_hotkey_signature(
        p.worker_signature.worker_pubkey, payload, p.worker_signature.sig
    )


def test_signing_payload_is_sha256_of_pinned_string() -> None:
    digest = compute_manifest_sha256(MANIFEST_A)
    payload = proof.execution_proof_signing_payload(manifest_sha256=digest, unit_id=UNIT_ID)
    assert payload == hashlib.sha256(f"{digest}:{UNIT_ID}".encode()).digest()


def test_proof_cannot_be_replayed_across_units() -> None:
    signer = _signer()
    p = build_execution_proof(
        signer=signer, manifest_sha256=compute_manifest_sha256(MANIFEST_A), unit_id=UNIT_ID
    )
    assert verify_execution_proof(p, unit_id="a-different-unit") is False


def test_tampered_hash_or_signature_fails_verification() -> None:
    signer = _signer()
    p = build_execution_proof(
        signer=signer, manifest_sha256=compute_manifest_sha256(MANIFEST_A), unit_id=UNIT_ID
    )
    tampered_hash = p.model_copy(update={"manifest_sha256": "b" * 64})
    assert verify_execution_proof(tampered_hash, unit_id=UNIT_ID) is False
    tampered_sig = p.model_copy(
        update={"worker_signature": p.worker_signature.model_copy(update={"sig": "0x" + "00" * 64})}
    )
    assert verify_execution_proof(tampered_sig, unit_id=UNIT_ID) is False


# --- VAL-PRISM-008: proof construction cannot read held-out secret split config -----------------

# Env keys that would leak master-side secrets if the builder ever read them.
_SECRET_ENV_KEYS = frozenset(
    {
        "PRISM_BASE_EVAL_VAL_DATA_DIR",
        "PRISM_EVAL_VAL_DATA_DIR",
        "PRISM_GATEWAY_TOKEN",
        "BASE_GATEWAY_TOKEN",
        "PRISM_HF_TOKEN",
    }
)
_SECRET_VAL_PATH = "/data/fineweb-edu/val/SECRET-HELDOUT-SPLIT"
_SECRET_LLM_KEY = "sk-SECRET-LLM-PROVIDER-KEY"


class _SecretAccessError(RuntimeError):
    """Raised by the sentinel env when a held-out-secret key is accessed."""


class _SentinelEnv(dict):
    """A mapping that RAISES on any access to a held-out-secret key.

    Proof construction that only reads the non-secret provider allowlist never trips it; a builder
    that reached for the val-split path / LLM key would raise, proving the read path exists.
    """

    def __getitem__(self, key: str) -> Any:
        if key in _SECRET_ENV_KEYS:
            raise _SecretAccessError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key in _SECRET_ENV_KEYS:
            raise _SecretAccessError(key)
        return super().get(key, default)


def test_proof_construction_succeeds_with_heldout_config_unset() -> None:
    # (a) No provider/secret env at all: proof still builds (tier 0).
    signer = _signer()
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env={}
    )
    assert p.version == 1
    assert p.tier == 0


def test_proof_construction_never_touches_secret_env_keys() -> None:
    # (b) A sentinel that raises on secret-key access is armed, yet proof construction succeeds.
    signer = _signer()
    sentinel = _SentinelEnv(
        {
            **_FULL_ENV,
            "PRISM_BASE_EVAL_VAL_DATA_DIR": _SECRET_VAL_PATH,
            "PRISM_GATEWAY_TOKEN": _SECRET_LLM_KEY,
            "PRISM_HF_TOKEN": _SECRET_LLM_KEY,
        }
    )
    # The sentinel is genuinely armed: reaching for a secret key raises.
    with pytest.raises(_SecretAccessError):
        sentinel.get("PRISM_BASE_EVAL_VAL_DATA_DIR")

    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=sentinel
    )
    assert p.tier == 1  # only the non-secret provider allowlist was consulted


def test_serialized_proof_leaks_no_secret_material() -> None:
    # (c) The serialized proof carries no val-split path or LLM key material.
    signer = _signer()
    sentinel = _SentinelEnv(
        {
            **_FULL_ENV,
            "PRISM_BASE_EVAL_VAL_DATA_DIR": _SECRET_VAL_PATH,
            "PRISM_GATEWAY_TOKEN": _SECRET_LLM_KEY,
        }
    )
    p = build_execution_proof_from_manifest(
        signer=signer, unit_id=UNIT_ID, manifest=MANIFEST_A, env=sentinel
    )
    serialized = p.model_dump_json()
    assert _SECRET_VAL_PATH not in serialized
    assert "SECRET-HELDOUT-SPLIT" not in serialized
    assert _SECRET_LLM_KEY not in serialized
    # The proof module only ever reads the non-secret provider allowlist.
    assert not (set(proof.PROVIDER_ENV_KEYS) & _SECRET_ENV_KEYS)
