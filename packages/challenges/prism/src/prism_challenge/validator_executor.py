"""Decentralized validator execution of an assigned prism work unit.

The assigned (online, gpu) validator pulls its single prism work unit from the master coordination
plane and runs the WHOLE re-execution by dispatching to its OWN broker-backed ``DockerExecutor``
(monkeypatched to the CPU re-exec mock in tests). The master coordinator never invokes the executor
for the prism unit - it only assigns the work (VAL-PRISM-037).

Execution reuses :class:`~prism_challenge.queue.PrismWorker`, whose container path builds the
validator's broker executor and preserves forced-random-init + prequential bits-per-byte scoring.
Driving it through :meth:`PrismWorker.process_submission` (a CAS claim on the specific submission)
makes the loop idempotent and concurrency-safe: re-running a completed/in-flight unit is a no-op
that neither re-dispatches the broker nor mutates the recorded result (VAL-PRISM-002 / 004).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import AuditResolution, is_audit_unit_id, resolve_audit_unit
from .coordination import (
    PRISM_DEFAULT_CONCURRENCY,
    PRISM_WORK_UNIT_CAPABILITY,
    RESUME_CHECKPOINT_PAYLOAD_KEY,
    PrismWorkUnit,
    list_pending_prism_work_units,
    pull_assigned_work_units,
)
from .proof import (
    WorkerSigner,
    build_execution_proof_from_manifest,
    worker_signer_from_key,
)
from .queue import PrismWorker

logger = logging.getLogger(__name__)

#: Replays an audited submission's evaluation and returns its fresh canonical manifest sha256
#: (``None`` on an inconclusive replay failure). Defaults to
#: :meth:`PrismWorker.replay_audit_manifest_sha256`; injectable so tests pin a deterministic hash.
AuditReplayFn = Callable[[str], Awaitable[str | None]]

#: Submission statuses at which a prism work unit is terminal (no re-execution, no re-dispatch).
TERMINAL_SUBMISSION_STATUSES = frozenset({"completed", "failed", "rejected"})


@dataclass(frozen=True)
class PrismWorkUnitExecution:
    """Outcome of executing (or idempotently skipping) one assigned prism work unit."""

    work_unit_id: str
    submission_id: str
    status: str
    #: True when the validator's broker was actually dispatched (False = idempotent no-op).
    executed: bool
    #: True when a fresh result was persisted by this run (False = already terminal).
    posted: bool
    #: The serialized ExecutionProof emitted for this unit's result payload, when the worker plane
    #: is enabled and a fresh successful finalization produced a manifest to bind (else ``None``).
    execution_proof: dict[str, Any] | None = None
    #: The canonical run manifest whose bytes back ``execution_proof.manifest_sha256``, forwarded so
    #: the accepting plane can reject a tampered manifest (VAL-PRISM-007); ``None`` when no proof.
    execution_manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class PrismValidatorCycleSummary:
    """Aggregate of one validator pull/execute/post cycle."""

    pulled: int
    executed: int
    skipped: int
    completed_submissions: tuple[str, ...]
    #: ExecutionProofs emitted this cycle, keyed by ``work_unit_id`` (empty when the plane is off).
    execution_proofs: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Run manifests backing the emitted proofs, keyed by ``work_unit_id`` (empty when off).
    execution_manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Audit resolutions produced this cycle when the worker plane is ON (audit-only cycle); empty
    #: on the legacy flag-off primary path (VAL-FINAL-005).
    audits: tuple[AuditResolution, ...] = ()


async def execute_work_unit(
    worker: PrismWorker,
    unit: PrismWorkUnit,
    *,
    proof_signer: WorkerSigner | None = None,
    proof_env: Mapping[str, str] | None = None,
) -> PrismWorkUnitExecution:
    """Run one assigned prism re-execution on the validator's own broker and report the outcome.

    Idempotent: :meth:`PrismWorker.process_submission` claims the submission only while it is
    pending, so a unit that already reached a terminal state is not re-dispatched and its recorded
    result is left untouched. A reassigned unit carrying ``resume_checkpoint_ref`` in its payload
    resumes from the last public HF checkpoint instead of restarting (VAL-PRISM-023).

    When the worker plane is enabled, a fresh successful finalization also emits an
    :class:`~prism_challenge.proof.ExecutionProof` bound to this unit (architecture.md 3.4).
    """

    resume_ref = unit.payload.get(RESUME_CHECKPOINT_PAYLOAD_KEY)
    result_id = await worker.process_submission(
        unit.submission_id,
        resume_checkpoint_ref=str(resume_ref) if resume_ref else None,
    )
    executed = result_id is not None
    status = await worker.repository.submission_status(unit.submission_id)
    execution_proof: dict[str, Any] | None = None
    execution_manifest: dict[str, Any] | None = None
    if executed and status == "completed":
        execution_proof, execution_manifest = await _emit_execution_proof(
            worker, unit, signer=proof_signer, env=proof_env
        )
    return PrismWorkUnitExecution(
        work_unit_id=unit.work_unit_id,
        submission_id=unit.submission_id,
        status=status or "",
        executed=executed,
        posted=executed,
        execution_proof=execution_proof,
        execution_manifest=execution_manifest,
    )


async def _emit_execution_proof(
    worker: PrismWorker,
    unit: PrismWorkUnit,
    *,
    signer: WorkerSigner | None,
    env: Mapping[str, str] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build the ExecutionProof + its backing manifest for a finalized unit (else ``(None, None)``).

    Gated on ``worker_plane.enabled``. The manifest hash is taken from the exact on-disk bytes of
    the run's ``prism_run_manifest.v2.json``; the provenance comes ONLY from the non-secret provider
    env allowlist. Held-out split config and LLM keys are never read (VAL-PRISM-008). The manifest
    content is returned with the proof so the accepting plane can recompute + verify the digest.
    """

    if not worker.settings.worker_plane.enabled:
        return None, None
    resolved_signer = _resolve_proof_signer(worker, signer)
    if resolved_signer is None:
        logger.warning(
            "worker plane enabled but no signing key configured; no ExecutionProof emitted "
            "for work_unit=%s",
            unit.work_unit_id,
        )
        return None, None
    manifest_path = await worker.repository.latest_run_manifest_path(
        unit.submission_id, worker.execution_backend
    )
    if not manifest_path:
        return None, None
    proof = build_execution_proof_from_manifest(
        signer=resolved_signer,
        unit_id=unit.work_unit_id,
        manifest_path=manifest_path,
        env=os.environ if env is None else env,
    )
    return proof.model_dump(mode="json"), _load_manifest(manifest_path)


def _load_manifest(manifest_path: str) -> dict[str, Any] | None:
    """Load the run manifest JSON from ``manifest_path`` (``None`` when it cannot be parsed)."""

    try:
        parsed = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _resolve_proof_signer(worker: PrismWorker, signer: WorkerSigner | None) -> WorkerSigner | None:
    if signer is not None:
        return signer
    key = worker.settings.worker_plane.signing_key
    if not key:
        return None
    return worker_signer_from_key(key)


async def run_validator_cycle(
    *,
    worker: PrismWorker,
    work_unit_ids: Iterable[str] | None = None,
    capabilities: Iterable[str] = (PRISM_WORK_UNIT_CAPABILITY,),
    in_flight: int | None = None,
    max_concurrency: int = PRISM_DEFAULT_CONCURRENCY,
    proof_signer: WorkerSigner | None = None,
    proof_env: Mapping[str, str] | None = None,
    audit_replay: AuditReplayFn | None = None,
) -> PrismValidatorCycleSummary:
    """Run one decentralized validator cycle.

    This is the autonomous validator entry point (the base validator agent's cycle). With the worker
    plane ON it is AUDIT-ONLY: workers execute the primary gpu submissions (base assignment plane +
    forwarding + light finalization), so the validator cycle NEVER pulls or executes a primary
    submission -- it only pulls the sampled ``audit:`` units, replays each deterministically, and
    resolves them (architecture.md 4; VAL-FINAL-005). With the flag OFF it is the legacy
    primary-execution cycle (:func:`run_primary_execution_cycle`), unchanged (no audit units exist).

    NOTE: worker-plane PRIMARY execution (a miner-funded worker running an assigned gpu unit + the
    ExecutionProof emission that feeds base reconciliation) goes through
    :func:`run_primary_execution_cycle` directly (via ``dispatch_assignment`` routing on the unit
    type), NOT through this flag-gated entry -- so a worker still executes primaries while the flag
    is on, and only the VALIDATOR cycle is audit-only.
    """

    if worker.settings.worker_plane.enabled:
        return await run_validator_audit_cycle(
            worker=worker, work_unit_ids=work_unit_ids, audit_replay=audit_replay
        )
    return await run_primary_execution_cycle(
        worker=worker,
        work_unit_ids=work_unit_ids,
        capabilities=capabilities,
        in_flight=in_flight,
        max_concurrency=max_concurrency,
        proof_signer=proof_signer,
        proof_env=proof_env,
    )


async def run_primary_execution_cycle(
    *,
    worker: PrismWorker,
    work_unit_ids: Iterable[str] | None = None,
    capabilities: Iterable[str] = (PRISM_WORK_UNIT_CAPABILITY,),
    in_flight: int | None = None,
    max_concurrency: int = PRISM_DEFAULT_CONCURRENCY,
    proof_signer: WorkerSigner | None = None,
    proof_env: Mapping[str, str] | None = None,
) -> PrismValidatorCycleSummary:
    """Pull -> execute (own broker) -> post the caller's assigned PRIMARY prism units.

    Pulls the caller's assigned, capability-matched prism units (at most
    ``max_concurrency - in_flight`` of them, so a busy executor runs one submission at a time),
    re-executes each on the caller's own broker, and reports which submissions completed. The pull
    and assignment are execution-free; only :func:`execute_work_unit` dispatches the broker. A
    successful worker-plane finalization emits an ExecutionProof per unit (architecture.md 3.4).

    This is the shared primary-execution path for BOTH a legacy validator (worker plane off) and a
    miner-funded worker running an assigned gpu unit (worker plane on); the audit-only restriction
    lives in :func:`run_validator_cycle`, not here.

    ``in_flight`` defaults to the caller's REAL in-flight draw (the count of currently-running
    submissions) so the concurrency-1 cap is enforced against reality rather than a static zero; a
    caller may override it (e.g. tests pinning a specific value).
    """

    if in_flight is None:
        in_flight = await worker.repository.count_in_flight_submissions()
    units = await list_pending_prism_work_units(worker.repository)
    pulled = pull_assigned_work_units(
        units,
        work_unit_ids=work_unit_ids,
        capabilities=capabilities,
        in_flight=in_flight,
        max_concurrency=max_concurrency,
    )
    executed = 0
    skipped = 0
    completed: list[str] = []
    execution_proofs: dict[str, dict[str, Any]] = {}
    execution_manifests: dict[str, dict[str, Any]] = {}
    for unit in pulled:
        outcome = await execute_work_unit(
            worker, unit, proof_signer=proof_signer, proof_env=proof_env
        )
        if outcome.executed:
            executed += 1
        else:
            skipped += 1
        if outcome.status == "completed":
            completed.append(outcome.submission_id)
        if outcome.execution_proof is not None:
            execution_proofs[outcome.work_unit_id] = outcome.execution_proof
        if outcome.execution_manifest is not None:
            execution_manifests[outcome.work_unit_id] = outcome.execution_manifest
    return PrismValidatorCycleSummary(
        pulled=len(pulled),
        executed=executed,
        skipped=skipped,
        completed_submissions=tuple(completed),
        execution_proofs=execution_proofs,
        execution_manifests=execution_manifests,
    )


async def run_validator_audit_cycle(
    *,
    worker: PrismWorker,
    work_unit_ids: Iterable[str] | None = None,
    audit_replay: AuditReplayFn | None = None,
    claimant: str | None = None,
) -> PrismValidatorCycleSummary:
    """Execute the sampled prism AUDIT units assigned to this validator (architecture.md 3.5).

    Enumerates the pending ``audit:`` units (``list_pending_audit_units``, each id recognised via
    :func:`~prism_challenge.audit.is_audit_unit_id`), restricted to the caller's assigned
    ``work_unit_ids`` when given (``None`` = every pending audit). Each candidate is CLAIMED under a
    lightweight per-audit lease (``repository.claim_audit_unit``) BEFORE it is replayed, so in a
    MULTI-validator deployment each pending audit is replayed by at most one validator instead of
    every validator redundantly replaying the same set (idempotent but wasteful GPU/CPU). An audit
    already claimed by another validator (live lease) is skipped, not replayed. Single-validator
    behaviour is unchanged: the sole validator wins every claim. For each claimed audit, the audited
    submission's evaluation is replayed deterministically to obtain a fresh manifest sha256, and the
    replay result is resolved through :func:`~prism_challenge.audit.resolve_audit_unit` (the
    ``POST /internal/v1/audit_units/{id}/result`` target): a MATCHING hash passes (finalized score
    untouched); a DIVERGENT hash invalidates the score and records a ``worker_fault``; a replay
    failure resolves inconclusive (re-audited within bounds). This cycle NEVER executes a primary
    submission (VAL-FINAL-005). ``audit_replay`` defaults to the real container replay; tests inject
    a deterministic hash. ``claimant`` identifies the lease holder (defaults to a per-cycle id).
    """

    replay = audit_replay if audit_replay is not None else worker.replay_audit_manifest_sha256
    wanted = set(work_unit_ids) if work_unit_ids is not None else None
    lease_seconds = worker.settings.worker_plane.audit_claim_lease_seconds
    who = claimant or uuid4().hex
    pending = await worker.repository.list_pending_audit_units()
    resolutions: list[AuditResolution] = []
    pulled = 0
    executed = 0
    skipped = 0
    for row in pending:
        audit_unit_id = str(row["audit_unit_id"])
        if not is_audit_unit_id(audit_unit_id):
            continue
        if wanted is not None and audit_unit_id not in wanted:
            continue
        if not await worker.repository.claim_audit_unit(
            audit_unit_id, claimant=who, lease_seconds=lease_seconds
        ):
            # Another validator already holds a live claim on this audit: skip it rather than
            # redundantly replay the same deterministic run (each audit is single-consumer).
            skipped += 1
            continue
        pulled += 1
        submission_id = str(row["submission_id"])
        replay_hash: str | None
        error: str | None = None
        try:
            replay_hash = await replay(submission_id)
        except Exception as exc:  # noqa: BLE001 - a replay failure is an inconclusive audit
            logger.warning("audit replay failed for %s: %s", audit_unit_id, exc)
            replay_hash = None
            error = str(exc)
        executed += 1
        resolution = await resolve_audit_unit(
            worker.repository,
            audit_unit_id=audit_unit_id,
            replay_manifest_sha256=replay_hash,
            failed=replay_hash is None,
            error=error,
        )
        resolutions.append(resolution)
    return PrismValidatorCycleSummary(
        pulled=pulled,
        executed=executed,
        skipped=skipped,
        completed_submissions=(),
        audits=tuple(resolutions),
    )
