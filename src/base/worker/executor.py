"""Worker-plane execution: guarantee an ExecutionProof on every result.

The worker agent reuses the validator :class:`AssignmentExecutor` seam to run
work on its OWN local broker. :class:`WorkerProofExecutor` wraps any such
executor so every SUCCESSFUL result carries an ``ExecutionProof`` envelope: it
passes an executor-emitted proof through unchanged, and otherwise builds and
signs a tier-0 proof from the run's ``manifest_sha256`` and the worker's static
provenance. Failed results (e.g. an unreachable broker) pass through untouched --
a failed unit has no deterministic manifest to bind.

:class:`StubManifestExecutor` is a CPU-only reference executor (no GPU, no
container) that deterministically "evaluates" a unit into a fixed manifest hash;
it backs the worker unit tests and ``base worker deploy --provider local``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from base.schemas.worker import ProviderInfo
from base.validator.agent.executor import (
    AssignmentContext,
    AssignmentExecutor,
    ExecutionResult,
    ProgressCallback,
)
from base.validator.agent.signing import RequestSigner
from base.worker.proof import (
    MANIFEST_SHA256_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    build_execution_proof,
)


class WorkerProofError(RuntimeError):
    """A successful result lacked the manifest needed to build a proof."""


@dataclass(frozen=True)
class WorkerProvenance:
    """Static provider/pod identity stamped into every proof a worker emits.

    ``image_digest`` + ``pod_id`` together promote a proof to tier 1 (architecture
    sec 3.4); the base worker plane usually has neither, so proofs stay tier 0.
    """

    provider_name: str
    miner_hotkey: str
    executor_id: str | None = None
    pod_id: str | None = None
    image_digest: str | None = None

    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            name=self.provider_name,
            executor_id=self.executor_id,
            pod_id=self.pod_id,
            miner_hotkey=self.miner_hotkey,
        )

    def tier(self) -> int:
        return 1 if self.image_digest and self.pod_id else 0


class WorkerProofExecutor:
    """Wrap an executor so every successful result carries an ExecutionProof."""

    def __init__(
        self,
        inner: AssignmentExecutor,
        *,
        signer: RequestSigner,
        provenance: WorkerProvenance,
    ) -> None:
        self._inner = inner
        self._signer = signer
        self._provenance = provenance

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        result = await self._inner.execute(context, progress=progress)
        if not result.success:
            return result
        payload = dict(result.payload)
        if PROOF_PAYLOAD_KEY in payload:
            return ExecutionResult(
                success=True,
                payload=payload,
                checkpoint_ref=result.checkpoint_ref,
            )
        manifest_sha256 = payload.get(MANIFEST_SHA256_PAYLOAD_KEY)
        if not manifest_sha256:
            raise WorkerProofError(
                "successful result is missing manifest_sha256; cannot build "
                "an ExecutionProof"
            )
        proof = build_execution_proof(
            signer=self._signer,
            manifest_sha256=str(manifest_sha256),
            unit_id=context.assignment.work_unit_id,
            tier=self._provenance.tier(),
            provider=self._provenance.provider_info(),
            image_digest=self._provenance.image_digest,
        )
        payload[PROOF_PAYLOAD_KEY] = proof.model_dump(mode="json")
        return ExecutionResult(
            success=True, payload=payload, checkpoint_ref=result.checkpoint_ref
        )


class StubManifestExecutor:
    """Deterministic CPU stub: 'evaluate' a unit into a fixed manifest hash.

    The hash is a pure function of the work unit id, so two workers replaying the
    same unit agree (reconciliation-friendly). Emits no proof itself -- the
    :class:`WorkerProofExecutor` wrapping it produces the tier-0 envelope.
    """

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        digest = hashlib.sha256(context.assignment.work_unit_id.encode()).hexdigest()
        return ExecutionResult(
            success=True, payload={MANIFEST_SHA256_PAYLOAD_KEY: digest}
        )


__all__ = [
    "StubManifestExecutor",
    "WorkerProofError",
    "WorkerProofExecutor",
    "WorkerProvenance",
]
