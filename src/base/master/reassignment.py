"""Reassignment pass that ties crash detection to work-unit reassignment.

When a validator misses heartbeats past the timeout (or an assignment lease
expires), its in-flight work must be reverted and reassigned to another online
validator, bounded by ``max_attempts`` so it never loops forever (architecture
sec 4). This module composes the existing building blocks into one pass:

1. :meth:`ValidatorCoordinationService.detect_offline_validators` flips stale
   validators ``online`` -> ``offline`` and records ``crash_detected`` events.
2. :meth:`AssignmentService.reclaim_stale_assignments` reverts the offline /
   past-deadline in-flight units to pending (carrying ``checkpoint_ref`` for
   prism resume) or terminally fails the retry-exhausted ones.
3. :meth:`AssignmentService.assign_pending` reassigns the reverted units to
   another eligible online validator (incrementing ``attempt_count``).
"""

from __future__ import annotations

from dataclasses import dataclass

from base.master.assignment import AssignmentService
from base.master.validator_coordination import ValidatorCoordinationService


@dataclass(frozen=True)
class ReassignmentPassResult:
    """Observable outcome of one reassignment pass."""

    offline: list[str]
    reverted: list[str]
    failed: list[str]
    assigned: dict[str, str]


async def run_reassignment_pass(
    *,
    validator_service: ValidatorCoordinationService,
    assignment_service: AssignmentService,
    seed: int | None = None,
) -> ReassignmentPassResult:
    """Detect crashed validators, reclaim their work, and reassign it.

    Returns the hotkeys newly marked offline, the work units reverted to pending
    and the ones terminally failed (retries exhausted), plus the
    ``{work_unit_id: validator_hotkey}`` mapping assigned this pass.
    """

    offline = await validator_service.detect_offline_validators()
    reclaim = await assignment_service.reclaim_stale_assignments()
    assigned = await assignment_service.assign_pending(seed=seed)
    return ReassignmentPassResult(
        offline=offline,
        reverted=reclaim.reverted,
        failed=reclaim.failed,
        assigned=assigned,
    )
