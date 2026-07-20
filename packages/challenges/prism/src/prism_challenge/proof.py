"""ExecutionProof construction and verification for the prism evaluator (architecture.md 3.4).

The prism evaluator emits an :class:`ExecutionProof` in the work-unit result payload at every
successful finalization. The tier-0 core is the deterministic ``manifest_sha256`` -- the sha256 of
the canonical on-disk bytes of ``prism_run_manifest.v2.json`` (``json.dumps(manifest,
sort_keys=True, indent=2)``, the exact form the runner/host persist) -- plus the worker's sr25519
signature binding that hash to the work unit. The signed message format is PINNED identically to the
base worker plane (VAL-AGENT-008 / VAL-PRISM-006): the signature is over
``sha256(manifest_sha256 + ":" + unit_id)`` -- the sha256 digest of the UTF-8 bytes of the string
``{manifest_sha256}:{unit_id}`` -- so a proof prism emits verifies with the same code as one the
base worker plane emits, and a proof cannot be replayed across units.

Tiers (architecture.md 3.4):

* tier 0 -- mandatory, all backends: canonical manifest hash + worker signature.
* tier 1 -- BOTH a pinned ``image_digest`` AND pod metadata (``provider.pod_id``).
* tier 2 -- CLAIMED when a closed structured attestation is present; EFFECTIVE tier 2 is granted
  only after Prism TEE verification succeeds (LOCAL-FIXTURE PASS today). Opaque non-empty
  ``tdx_quote_b64`` / ``gpu_eat_jwt`` alone never elevate effective tier.

Security invariant (VAL-PRISM-008): proof construction reads ONLY the manifest, the work unit id,
the worker signer, and a FIXED ALLOWLIST of non-secret provider env vars. It never reads the
master-side held-out split configuration (``base_eval_val_data_dir`` / heldout config) or any LLM
provider key -- there is no code path from proof construction to those secrets (this module imports
no settings and touches no filesystem path other than an explicitly supplied manifest file).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, runtime_checkable

from base.challenge_sdk.proof import (
    EXECUTION_PROOF_VERSION,
    ExecutionProof,
    ProviderInfo,
    WorkerSignature,
    execution_proof_signing_payload,
)

from .auth import verify_hotkey_signature

ExecutionProofVersion = Literal[1]
ExecutionProofTier = Literal[0, 1, 2]

#: Result-payload key carrying the serialized :class:`ExecutionProof`. Identical to the base worker
#: plane key so a proof prism emits is passed through unchanged by the base ``WorkerProofExecutor``.
PROOF_PAYLOAD_KEY = "execution_proof"

#: Result-payload key carrying the canonical run manifest (the full ``prism_run_manifest.v2`` dict)
#: so the accepting side can recompute ``manifest_sha256`` from the FORWARDED content and reject a
#: tampered manifest whose bytes no longer hash to the signed digest (VAL-PRISM-007).
MANIFEST_PAYLOAD_KEY = "run_manifest"

# Non-secret provider env vars the worker agent injects for the tier-1/2 fields (architecture 3.5).
PROVIDER_NAME_ENV = "PRISM_PROVIDER_NAME"
EXECUTOR_ID_ENV = "PRISM_EXECUTOR_ID"
POD_ID_ENV = "PRISM_POD_ID"
MINER_HOTKEY_ENV = "PRISM_MINER_HOTKEY"
IMAGE_DIGEST_ENV = "PRISM_IMAGE_DIGEST"
ATTESTATION_ENV = "PRISM_ATTESTATION"

#: The ONLY env vars proof construction ever reads (all non-secret provider provenance).
PROVIDER_ENV_KEYS: tuple[str, ...] = (
    PROVIDER_NAME_ENV,
    EXECUTOR_ID_ENV,
    POD_ID_ENV,
    MINER_HOTKEY_ENV,
    IMAGE_DIGEST_ENV,
    ATTESTATION_ENV,
)

#: Documented attestation payload keys (architecture.md 3.4).
ATTESTATION_KEYS: tuple[str, ...] = ("tdx_quote_b64", "gpu_eat_jwt")


@runtime_checkable
class WorkerSigner(Protocol):
    """Signs the pinned ExecutionProof message with a worker sr25519 key."""

    @property
    def worker_pubkey(self) -> str: ...

    def sign(self, message: bytes) -> str: ...


@dataclass(frozen=True)
class KeypairWorkerSigner:
    """A :class:`WorkerSigner` backed by a substrate (bittensor) keypair."""

    keypair: Any

    @property
    def worker_pubkey(self) -> str:
        return str(self.keypair.ss58_address)

    def sign(self, message: bytes) -> str:
        signature = self.keypair.sign(message)
        if isinstance(signature, bytes | bytearray):
            return "0x" + bytes(signature).hex()
        return str(signature)


def worker_signer_from_key(key: str) -> KeypairWorkerSigner:
    """Build a worker signer from an sr25519 URI (``//Name``), mnemonic, or seed.

    The key is the WORKER's OWN signing key (never a master secret); the worker agent injects it.
    """

    from .keypair import keypair_from_mnemonic, keypair_from_seed, keypair_from_uri

    value = key.strip()
    if value.startswith("//"):
        keypair = keypair_from_uri(value)
    elif " " in value:
        keypair = keypair_from_mnemonic(value)
    else:
        keypair = keypair_from_seed(value)
    return KeypairWorkerSigner(keypair)


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    """The canonical on-disk serialization of a run manifest (``sort_keys=True, indent=2``).

    Key-order-insensitive by construction: two dicts with identical content but different key order
    serialize identically, so the manifest hash is stable regardless of how the manifest is built.
    """

    return json.dumps(manifest, sort_keys=True, indent=2)


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 hex of the canonical manifest bytes (order-insensitive)."""

    return manifest_sha256_from_bytes(canonical_manifest_json(manifest).encode("utf-8"))


def manifest_sha256_from_bytes(raw: bytes) -> str:
    """sha256 hex of raw manifest bytes (use the exact on-disk bytes of the v2 manifest)."""

    return hashlib.sha256(raw).hexdigest()


def read_manifest_sha256(path: str | os.PathLike[str]) -> str:
    """sha256 hex of the exact on-disk bytes of the manifest file at ``path``."""

    return manifest_sha256_from_bytes(Path(path).read_bytes())


def has_attestation(attestation: Any) -> bool:
    """Whether ``attestation`` has both documented attestation components as non-empty strings.

    WARNING: this is ONLY a shape/presence hint for CLAIMED-tier emission compatibility.
    It is NEVER cryptographic verification and NEVER elevates effective tier. Prism has no
    TEE verifier; effective tier is IMAGE_PIN only (max 1) via
    :func:`~prism_challenge.audit.effective_tier`.
    """

    if not isinstance(attestation, Mapping):
        return False
    tdx = attestation.get("tdx_quote_b64")
    gpu = attestation.get("gpu_eat_jwt")
    return (
        isinstance(tdx, str)
        and bool(tdx.strip())
        and isinstance(gpu, str)
        and bool(gpu.strip())
    )


def has_structured_attestation_claim(attestation: Any) -> bool:
    """Whether attestation claims the closed prism.tee.v1 shape (compat claim only; unverified).

    Used only to decide the CLAIMED emission tier for wire compatibility. Effective tier is
    never elevated from this claim (max effective tier is 1 via IMAGE_PIN).
    """

    if not has_attestation(attestation):
        return False
    assert isinstance(attestation, Mapping)
    version = attestation.get("version")
    provider = attestation.get("provider")
    evidence_type = attestation.get("evidence_type")
    return (
        version in (1, "1", "1.0")
        and isinstance(provider, str)
        and provider in {"local_fixture", "lium", "targon"}
        and evidence_type == "prism.tee.v1"
    )


def compute_tier(
    *,
    image_digest: str | None,
    provider: ProviderInfo | None,
    attestation: Any,
) -> ExecutionProofTier:
    """Compute the CLAIMED proof tier from available provenance (architecture 3.4).

    tier 2 may still be *claimed* for a closed structured attestation shape (wire compat;
    never mere non-empty opaque strings). The claimed tier is NOT trusted at verification:
    :func:`~prism_challenge.audit.effective_tier` recomputes from IMAGE_PIN only and never
    elevates above tier 1. tier 1 iff BOTH a pinned image digest AND pod metadata
    (``provider.pod_id``) are present; else tier 0.
    """

    if has_structured_attestation_claim(attestation):
        return 2
    if image_digest and provider is not None and provider.pod_id:
        return 1
    return 0


def provider_from_env(env: Mapping[str, str] | None = None) -> ProviderInfo | None:
    """Read the provider block from the injected provider env (``None`` when no provider name)."""

    env = os.environ if env is None else env
    name = _clean(env.get(PROVIDER_NAME_ENV))
    if not name:
        return None
    return ProviderInfo(
        name=name,
        executor_id=_clean(env.get(EXECUTOR_ID_ENV)),
        pod_id=_clean(env.get(POD_ID_ENV)),
        miner_hotkey=_clean(env.get(MINER_HOTKEY_ENV)),
    )


def image_digest_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """Read the evaluator image digest from the injected provider env."""

    env = os.environ if env is None else env
    return _clean(env.get(IMAGE_DIGEST_ENV))


def attestation_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Read + parse the attestation payload (JSON) from the injected provider env, if any."""

    env = os.environ if env is None else env
    raw = _clean(env.get(ATTESTATION_ENV))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def build_execution_proof(
    *,
    signer: WorkerSigner,
    manifest_sha256: str,
    unit_id: str,
    provider: ProviderInfo | None = None,
    image_digest: str | None = None,
    attestation: dict[str, Any] | None = None,
    tier: ExecutionProofTier | None = None,
) -> ExecutionProof:
    """Build and sign an ExecutionProof binding ``manifest_sha256`` to ``unit_id`` under ``signer``.

    The tier is computed from the provenance unless explicitly overridden. ``signer`` is the WORKER
    keypair; its public identity becomes ``worker_signature.worker_pubkey``.
    """

    effective_tier: ExecutionProofTier = (
        compute_tier(image_digest=image_digest, provider=provider, attestation=attestation)
        if tier is None
        else tier
    )
    signature = signer.sign(
        execution_proof_signing_payload(manifest_sha256=manifest_sha256, unit_id=unit_id)
    )
    return ExecutionProof(
        version=cast(ExecutionProofVersion, EXECUTION_PROOF_VERSION),
        tier=effective_tier,
        manifest_sha256=manifest_sha256,
        image_digest=image_digest,
        provider=provider,
        worker_signature=WorkerSignature(worker_pubkey=signer.worker_pubkey, sig=signature),
        attestation=attestation,
    )


def build_execution_proof_from_manifest(
    *,
    signer: WorkerSigner,
    unit_id: str,
    manifest: Mapping[str, Any] | None = None,
    manifest_bytes: bytes | None = None,
    manifest_path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ExecutionProof:
    """Build a signed proof from a manifest source + the injected provider env.

    Exactly ONE manifest source must be given. Prefer ``manifest_path`` at emission time so the
    hash is taken from the exact on-disk bytes of ``prism_run_manifest.v2.json``. The provider
    provenance is read ONLY from the non-secret provider env allowlist.
    """

    digest = _resolve_manifest_sha256(
        manifest=manifest, manifest_bytes=manifest_bytes, manifest_path=manifest_path
    )
    return build_execution_proof(
        signer=signer,
        manifest_sha256=digest,
        unit_id=unit_id,
        provider=provider_from_env(env),
        image_digest=image_digest_from_env(env),
        attestation=attestation_from_env(env),
    )


def verify_execution_proof(
    proof: ExecutionProof,
    *,
    unit_id: str,
    verify: Any = verify_hotkey_signature,
) -> bool:
    """Whether ``proof``'s worker signature verifies for ``unit_id`` (sr25519, pinned message).

    Rejects a proof presented with a DIFFERENT ``unit_id`` than the one signed, so a proof cannot
    be replayed across units.
    """

    payload = execution_proof_signing_payload(
        manifest_sha256=proof.manifest_sha256, unit_id=unit_id
    )
    return bool(verify(proof.worker_signature.worker_pubkey, payload, proof.worker_signature.sig))


def _resolve_manifest_sha256(
    *,
    manifest: Mapping[str, Any] | None,
    manifest_bytes: bytes | None,
    manifest_path: str | os.PathLike[str] | None,
) -> str:
    provided = [item for item in (manifest, manifest_bytes, manifest_path) if item is not None]
    if len(provided) != 1:
        raise ValueError("exactly one of manifest, manifest_bytes, manifest_path is required")
    if manifest is not None:
        return compute_manifest_sha256(manifest)
    if manifest_bytes is not None:
        return manifest_sha256_from_bytes(manifest_bytes)
    assert manifest_path is not None
    return read_manifest_sha256(manifest_path)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ATTESTATION_ENV",
    "ATTESTATION_KEYS",
    "EXECUTION_PROOF_VERSION",
    "EXECUTOR_ID_ENV",
    "IMAGE_DIGEST_ENV",
    "MINER_HOTKEY_ENV",
    "POD_ID_ENV",
    "MANIFEST_PAYLOAD_KEY",
    "PROOF_PAYLOAD_KEY",
    "PROVIDER_ENV_KEYS",
    "PROVIDER_NAME_ENV",
    "ExecutionProof",
    "KeypairWorkerSigner",
    "ProviderInfo",
    "WorkerSignature",
    "WorkerSigner",
    "attestation_from_env",
    "build_execution_proof",
    "build_execution_proof_from_manifest",
    "canonical_manifest_json",
    "compute_manifest_sha256",
    "compute_tier",
    "execution_proof_signing_payload",
    "has_attestation",
    "has_structured_attestation_claim",
    "image_digest_from_env",
    "manifest_sha256_from_bytes",
    "provider_from_env",
    "read_manifest_sha256",
    "verify_execution_proof",
    "worker_signer_from_key",
]
