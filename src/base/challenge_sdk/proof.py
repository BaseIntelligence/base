"""Ordinary execution-proof types and canonical signing bytes."""

from __future__ import annotations

import hashlib

from .schemas import ExecutionProof, ProviderInfo, WorkerSignature

EXECUTION_PROOF_VERSION = 1


def execution_proof_signing_payload(*, manifest_sha256: str, unit_id: str) -> bytes:
    """Return the bytes signed for a manifest and work-unit binding."""

    return hashlib.sha256(f"{manifest_sha256}:{unit_id}".encode()).digest()


__all__ = [
    "EXECUTION_PROOF_VERSION",
    "ExecutionProof",
    "ProviderInfo",
    "WorkerSignature",
    "execution_proof_signing_payload",
]
