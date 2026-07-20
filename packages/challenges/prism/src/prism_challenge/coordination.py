"""Coordination-plane integration for prism re-execution (architecture.md sections 3, 4, 7).

The master coordinates evaluation but never executes it. A prism submission becomes EXACTLY ONE
``gpu`` work unit (``work_unit_id == submission_id``); a balanced, capability-aware pass on the
master assigns it to exactly one ONLINE gpu validator with concurrency 1, and that validator runs
the whole re-execution on its OWN broker (see :mod:`prism_challenge.validator_executor`).

This module is the prism half of that contract: it exposes a submission's pending work unit to the
master plane and provides the validator-side ``pull`` view (capability- and concurrency-filtered).
It is deliberately execution-free - it never touches the broker/``DockerExecutor`` - so the
coordinator can enumerate and assign work without invoking any executor (VAL-PRISM-037).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

#: prism re-execution is a GPU re-execution; its single work unit is gpu-capability gated so it is
#: only ever assigned to a validator advertising ``gpu`` (VAL-PRISM-003).
PRISM_WORK_UNIT_CAPABILITY = "gpu"

#: One submission per validator at a time: a busy gpu validator is never handed a second concurrent
#: prism unit (architecture.md sections 3.4, 7; VAL-PRISM-002).
PRISM_DEFAULT_CONCURRENCY = 1

#: Payload key carrying the resume checkpoint ref to a reassigned prism unit. Mirrors the platform
#: coordination contract so a reassigned validator resumes from the last public HF checkpoint rather
#: than restarting (consumed by the resume feature; surfaced here for payload parity).
RESUME_CHECKPOINT_PAYLOAD_KEY = "resume_checkpoint_ref"


class SupportsPendingSubmissions(Protocol):
    """The slice of :class:`~prism_challenge.repository.PrismRepository` this module needs."""

    async def list_pending_submissions(self) -> list[dict[str, object]]: ...

    async def latest_checkpoint_ref(self, submission_id: str) -> str | None: ...


@dataclass(frozen=True)
class PrismWorkUnit:
    """One assignable prism work unit (the whole re-execution of a single submission)."""

    work_unit_id: str
    submission_id: str
    #: The submitting miner's hotkey. The master records this as the assignment's ``submission_ref``
    #: (the prism agent reference); execution itself only needs ``submission_id`` + ``payload``.
    hotkey: str
    required_capability: str = PRISM_WORK_UNIT_CAPABILITY
    payload: dict[str, Any] = field(default_factory=dict)


def prism_work_unit_id(submission_id: str) -> str:
    """Return the stable coordination-plane work-unit id for a prism submission.

    prism produces exactly one work unit per submission, so the work-unit id IS the submission id
    (architecture.md section 3.4; VAL-PRISM-001).
    """

    return str(submission_id)


def capability_can_run(required: str, capabilities: Iterable[str]) -> bool:
    """Whether a validator advertising ``capabilities`` may run a ``required``-capability unit."""

    return required in set(capabilities)


async def list_pending_prism_work_units(
    repository: SupportsPendingSubmissions,
) -> list[PrismWorkUnit]:
    """Expose each submission awaiting re-execution as exactly one pending gpu work unit.

    When a submission already has a published HF checkpoint (a prior attempt persisted one before
    crashing), the unit payload carries it under ``resume_checkpoint_ref`` so the reassigned
    validator resumes from the last public checkpoint rather than restarting (VAL-PRISM-023). A
    first attempt with no published checkpoint carries no resume ref and starts fresh
    (VAL-PRISM-024).
    """

    rows = await repository.list_pending_submissions()
    units: list[PrismWorkUnit] = []
    for row in rows:
        submission_id = str(row["id"])
        payload: dict[str, Any] = {}
        resume_ref = await repository.latest_checkpoint_ref(submission_id)
        if resume_ref:
            payload[RESUME_CHECKPOINT_PAYLOAD_KEY] = resume_ref
        units.append(
            PrismWorkUnit(
                work_unit_id=prism_work_unit_id(submission_id),
                submission_id=submission_id,
                hotkey=str(row.get("hotkey") or ""),
                payload=payload,
            )
        )
    return units


def pull_assigned_work_units(
    units: Sequence[PrismWorkUnit],
    *,
    work_unit_ids: Iterable[str] | None = None,
    capabilities: Iterable[str] = (PRISM_WORK_UNIT_CAPABILITY,),
    in_flight: int = 0,
    max_concurrency: int = PRISM_DEFAULT_CONCURRENCY,
) -> list[PrismWorkUnit]:
    """Return the caller's runnable subset of ``units`` (capability- and concurrency-filtered).

    ``work_unit_ids`` is the subset the master assigned to this validator (``None`` = all exposed
    units, the single-validator case); only those that match the caller's ``capabilities`` are
    eligible, so a cpu-only validator's pull omits the gpu prism unit (VAL-PRISM-003). At most
    ``max_concurrency - in_flight`` units are returned, so a validator already running a prism unit
    (``in_flight >= max_concurrency``) gets nothing more this pull and the rest waits or goes to an
    idle validator (concurrency 1; VAL-PRISM-002).
    """

    caps = set(capabilities)
    eligible = [unit for unit in units if capability_can_run(unit.required_capability, caps)]
    if work_unit_ids is not None:
        wanted = set(work_unit_ids)
        eligible = [unit for unit in eligible if unit.work_unit_id in wanted]
    available = max(0, max_concurrency - max(0, in_flight))
    return eligible[:available]


def work_unit_to_payload(unit: PrismWorkUnit) -> Mapping[str, Any]:
    """Serialise a work unit for the ``/internal/v1/work_units`` coordination-plane response."""

    return {
        "work_unit_id": unit.work_unit_id,
        "submission_id": unit.submission_id,
        # The master (``HttpChallengeWorkSource``) records the prism agent reference under
        # ``submission_ref``; for prism that reference is the submitting miner's hotkey.
        "submission_ref": unit.hotkey,
        "required_capability": unit.required_capability,
        "payload": dict(unit.payload),
    }


#: Executor kind an audit unit is dispatched to: audits ALWAYS run on the authoritative validator
#: replay path (``validator_dispatch``), never on a worker (architecture.md 3.5; VAL-PRISM-012).
AUDIT_EXECUTOR_KIND = "validator"


def audit_work_unit_to_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Serialise a pending audit unit (an ``audit_units`` row joined with the submission hotkey).

    The audit unit carries a work_unit_id DISTINCT from the audited submission's primary unit id
    and is marked for validator execution (``required_capability == "gpu"``, executor kind
    ``validator``) referencing the audited submission id (VAL-PRISM-012).
    """

    submission_id = str(row["submission_id"])
    return {
        "work_unit_id": str(row["audit_unit_id"]),
        "submission_id": submission_id,
        "submission_ref": str(row.get("hotkey") or ""),
        "required_capability": str(row.get("required_capability") or PRISM_WORK_UNIT_CAPABILITY),
        "executor_kind": AUDIT_EXECUTOR_KIND,
        "audit": True,
        "audited_submission_id": submission_id,
        "payload": {"audit": True, "audited_submission_id": submission_id},
    }
