"""agent-challenge validator-execution adapter (architecture sec 4, G2).

Dispatches a pulled agent-challenge assignment to the sibling package's
decentralized Terminal-Bench 2.1 ``own_runner`` cycle on the validator's OWN
broker. The sibling package's dispatch entrypoint is imported LAZILY so platform
does not hard-depend on ``agent_challenge``; an unavailable package surfaces a
clear :class:`AssignmentExecutionError` rather than a silent drop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from base.master.replay_audit import (
    REPLAY_AUDIT_REQUEST_KEY,
    ReplayAuditRequest,
    ReplayAuditResult,
    is_replay_assignment_payload,
)
from base.schemas.worker import PHALA_TDX_TIER, ExecutionProof, WorkerSignature
from base.validator.agent.executor import (
    AssignmentContext,
    AssignmentExecutionError,
    ExecutionResult,
    ProgressCallback,
)
from base.validator.agent.signing import RequestSigner
from base.worker.proof import execution_proof_signing_payload

CHALLENGE_SLUG = "agent-challenge"

#: Signature of the sibling package's dispatch entrypoint
#: (``agent_challenge.validator_dispatch.dispatch_assignment``).
DispatchFn = Callable[..., Awaitable[Mapping[str, Any]]]


class AgentChallengeCycleExecutor:
    """Run a pulled agent-challenge assignment via the sibling validator cycle."""

    def __init__(
        self,
        *,
        dispatch: DispatchFn | None = None,
        dispatch_replay: DispatchFn | None = None,
    ) -> None:
        self._dispatch = dispatch
        self._dispatch_replay = dispatch_replay

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        if is_replay_assignment_payload(context.assignment.payload):
            return await self._execute_replay(context)
        dispatch = self._dispatch or _load_dispatch()
        broker = context.broker
        result = await dispatch(
            work_unit_id=context.assignment.work_unit_id,
            payload=dict(context.assignment.payload or {}),
            broker_url=broker.broker_url,
            broker_token=broker.broker_token,
            broker_token_file=broker.broker_token_file,
            broker_allowed_images=tuple(broker.allowed_images),
        )
        return ExecutionResult(success=True, payload=dict(result))

    async def _execute_replay(self, context: AssignmentContext) -> ExecutionResult:
        payload = context.assignment.payload
        raw_request = payload.get(REPLAY_AUDIT_REQUEST_KEY)
        if not isinstance(raw_request, Mapping):
            raise AssignmentExecutionError("replay assignment has no labelled request")
        request = ReplayAuditRequest.from_mapping(raw_request)
        dispatch = self._dispatch_replay or _load_replay_dispatch()
        broker = context.broker
        result = await dispatch(
            request=request.to_dict(),
            work_unit_id=context.assignment.work_unit_id,
            payload=dict(payload),
            broker_url=broker.broker_url,
            broker_token=broker.broker_token,
            broker_token_file=broker.broker_token_file,
            broker_allowed_images=tuple(broker.allowed_images),
        )
        if not isinstance(result, Mapping):
            raise AssignmentExecutionError(
                "replay dispatch returned a non-object result"
            )
        replay_result = ReplayAuditResult.from_mapping(
            result.get("replay_audit_result", result)
        )
        replay_result.validate_against(request)
        return ExecutionResult(
            success=True,
            payload={"replay_audit_result": replay_result.to_dict()},
        )


def _load_dispatch() -> DispatchFn:
    try:
        from agent_challenge.validator_dispatch import dispatch_assignment
    except Exception as exc:  # noqa: BLE001 - surfaced as a dispatch failure
        raise AssignmentExecutionError(
            f"agent-challenge dispatch adapter is unavailable: {exc}"
        ) from exc
    return dispatch_assignment


def _load_replay_dispatch() -> DispatchFn:
    try:
        from agent_challenge.validator_dispatch import dispatch_replay_audit
    except Exception as exc:  # noqa: BLE001 - surfaced as a dispatch failure
        raise AssignmentExecutionError(
            f"agent-challenge replay dispatch adapter is unavailable: {exc}"
        ) from exc
    return dispatch_replay_audit


def rebind_worker_signature(
    proof: ExecutionProof, *, signer: RequestSigner, unit_id: str
) -> ExecutionProof:
    """Rebind a Phala-tier envelope's tier-0 worker signature to ``unit_id``.

    The canonical eval image runs a LEAN CVM image with no bittensor/sr25519
    keypair, so its emitted Phala-tier ``ExecutionProof`` carries only a
    schema-valid PLACEHOLDER ``worker_signature`` (empty pubkey/sig). When the
    validator ingests the attested ``BASE_BENCHMARK_RESULT`` payload it re-signs
    the tier-0 layer over the pinned ``sha256(f"{manifest_sha256}:{unit_id}")``
    message with its OWN signer, so ``verify_execution_proof`` enforces a real
    signature bound to this unit (no cross-unit replay, VAL-VERIFY-013).

    The trust root remains the **attestation** (the hardware-signed TDX quote),
    which the validator verifies cryptographically; this rebind only anchors the
    worker-plane envelope layer to a real key -- it never substitutes for quote
    verification. The attestation payload is carried through unchanged.
    """

    if proof.tier != PHALA_TDX_TIER:
        raise AssignmentExecutionError(
            "rebind_worker_signature only applies to Phala-tier proofs"
        )
    if proof.worker_signature.worker_pubkey != "" or proof.worker_signature.sig != "":
        raise AssignmentExecutionError(
            "rebind_worker_signature requires the exact empty Eval placeholder"
        )
    signature = signer.sign(
        execution_proof_signing_payload(
            manifest_sha256=proof.manifest_sha256, unit_id=unit_id
        )
    )
    return proof.model_copy(
        update={
            "worker_signature": WorkerSignature(
                worker_pubkey=signer.hotkey, sig=signature
            )
        }
    )


__all__ = [
    "CHALLENGE_SLUG",
    "AgentChallengeCycleExecutor",
    "rebind_worker_signature",
]
