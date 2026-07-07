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
from base.worker.deploy import (
    LOCAL_PROVIDER,
    PROVIDER_KEY_ENV,
    SUPPORTED_PROVIDERS,
    MissingProviderKeyError,
    NoOfferWithinBudgetError,
    UnsupportedProviderError,
    WorkerDeployError,
    build_signed_binding,
    build_worker_pod_env,
    normalize_provider,
    plan_provider_deployment,
    rank_worker_offers,
    require_provider_api_key,
    select_worker_offer,
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
    "LOCAL_PROVIDER",
    "MANIFEST_SHA256_PAYLOAD_KEY",
    "PROOF_PAYLOAD_KEY",
    "PROVIDER_KEY_ENV",
    "SUPPORTED_PROVIDERS",
    "AgentCycleSummary",
    "BackoffPolicy",
    "MissingProviderKeyError",
    "NoOfferWithinBudgetError",
    "StubManifestExecutor",
    "UnsupportedProviderError",
    "WorkerAgent",
    "WorkerBinding",
    "WorkerCoordinationClient",
    "WorkerCoordinationClientError",
    "WorkerDeployError",
    "WorkerProofError",
    "WorkerProofExecutor",
    "WorkerProvenance",
    "build_execution_proof",
    "build_signed_binding",
    "build_worker_pod_env",
    "execution_proof_signing_payload",
    "normalize_provider",
    "plan_provider_deployment",
    "rank_worker_offers",
    "require_provider_api_key",
    "select_worker_offer",
    "verify_execution_proof",
]
