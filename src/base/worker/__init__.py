"""Miner-funded GPU worker agent runtime (architecture.md sec 3.2).

A long-running client that enrolls with the master under a miner-signed binding,
heartbeats, pulls gpu work units, executes them on its own broker via the shared
:class:`AssignmentExecutor` seam, and posts results carrying an ``ExecutionProof``
envelope. It authenticates as its worker keypair, never as a validator permit.
"""

from __future__ import annotations

from base.worker.coordination_client import (
    WorkerCoordinationClient,
    WorkerCoordinationClientError,
)
from base.worker.executor import (
    StubManifestExecutor,
    WorkerProofError,
    WorkerProofExecutor,
    WorkerProvenance,
)
from base.worker.proof import (
    EXECUTION_PROOF_VERSION,
    MANIFEST_SHA256_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
    execution_proof_signing_payload,
    verify_execution_proof,
)
from base.worker.runtime import (
    AgentCycleSummary,
    BackoffPolicy,
    WorkerAgent,
    WorkerBinding,
)

__all__ = [
    "EXECUTION_PROOF_VERSION",
    "MANIFEST_SHA256_PAYLOAD_KEY",
    "PROOF_PAYLOAD_KEY",
    "AgentCycleSummary",
    "BackoffPolicy",
    "StubManifestExecutor",
    "WorkerAgent",
    "WorkerBinding",
    "WorkerCoordinationClient",
    "WorkerCoordinationClientError",
    "WorkerProofError",
    "WorkerProofExecutor",
    "WorkerProvenance",
    "build_execution_proof",
    "execution_proof_signing_payload",
    "verify_execution_proof",
]
