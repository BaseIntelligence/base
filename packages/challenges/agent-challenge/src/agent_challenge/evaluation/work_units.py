"""Pending work-unit exposure for the master coordination plane.

The central AST + LLM gates are the only entrypoint to assignable work: a
submission's deterministically selected benchmark tasks
(:func:`select_benchmark_tasks`) become work units only after an ``allow``
verdict has created its :class:`EvaluationJob`. A ``reject`` or ``escalate``
verdict produces no job, hence no work units. The coordination plane fans these
pending units (one per not-yet-completed task) out across online validators.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import settings
from ..core.db import database
from ..core.models import AgentSubmission, EvaluationJob
from ..core.statuses import (
    ASSIGNABLE_JOB_STATUSES,
    HALTED_SUBMISSION_STATUSES,
    TERMINAL_TASK_STATUSES,
)
from .benchmarks import benchmark_tasks_from_json

__all__ = [
    "ASSIGNABLE_JOB_STATUSES",
    "HALTED_SUBMISSION_STATUSES",
    "TERMINAL_TASK_STATUSES",
    "WORK_UNIT_CAPABILITY",
    "PendingWorkUnit",
    "assignable_jobs_statement",
    "list_pending_work_units",
    "pending_work_units_for_job",
    "work_unit_id_for",
]

WORK_UNIT_CAPABILITY = "cpu"


@dataclass(frozen=True)
class PendingWorkUnit:
    """One assignable task unit derived from an allowed submission's job."""

    work_unit_id: str
    submission_id: int
    submission_ref: str
    miner_hotkey: str
    job_id: str
    task_id: str
    docker_image: str
    required_capability: str = WORK_UNIT_CAPABILITY


def work_unit_id_for(submission_id: int, task_id: str) -> str:
    """Return the stable coordination-plane work-unit id for a task."""

    return f"{submission_id}:{task_id}"


def pending_work_units_for_job(job: EvaluationJob) -> list[PendingWorkUnit]:
    """Expand a job's selected tasks into not-yet-completed work units."""

    if settings.attested_review_enabled:
        return []
    submission = job.submission
    completed = {
        result.task_id for result in job.task_results if result.status in TERMINAL_TASK_STATUSES
    }
    units: list[PendingWorkUnit] = []
    for task in benchmark_tasks_from_json(job.selected_tasks_json):
        if task.task_id in completed:
            continue
        units.append(
            PendingWorkUnit(
                work_unit_id=work_unit_id_for(submission.id, task.task_id),
                submission_id=submission.id,
                submission_ref=submission.agent_hash,
                miner_hotkey=submission.miner_hotkey,
                job_id=job.job_id,
                task_id=task.task_id,
                docker_image=task.docker_image,
            )
        )
    return units


def assignable_jobs_statement():
    """Select non-terminal jobs of non-halted submissions, eager-loaded."""

    return (
        select(EvaluationJob)
        .join(EvaluationJob.submission)
        .options(
            selectinload(EvaluationJob.submission),
            selectinload(EvaluationJob.task_results),
        )
        .where(EvaluationJob.status.in_(ASSIGNABLE_JOB_STATUSES))
        .where(AgentSubmission.raw_status.not_in(HALTED_SUBMISSION_STATUSES))
        .order_by(EvaluationJob.created_at, EvaluationJob.id)
    )


async def list_pending_work_units(
    session: AsyncSession | None = None,
) -> list[PendingWorkUnit]:
    """Return every pending work unit currently exposed to the coordination plane."""

    if settings.attested_review_enabled:
        return []
    if session is not None:
        return await _collect_pending_work_units(session)
    async with database.session() as owned_session:
        return await _collect_pending_work_units(owned_session)


async def _collect_pending_work_units(session: AsyncSession) -> list[PendingWorkUnit]:
    jobs = (await session.execute(assignable_jobs_statement())).scalars().all()
    units: list[PendingWorkUnit] = []
    for job in jobs:
        units.extend(pending_work_units_for_job(job))
    return units
