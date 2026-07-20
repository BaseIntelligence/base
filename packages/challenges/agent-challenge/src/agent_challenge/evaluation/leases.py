from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.core.models import (
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    TerminalBenchTrial,
)

LeaseTarget = Literal[
    "evaluation_job", "analysis_run", "evaluation_attempt", "terminal_bench_trial"
]


def lease_deadline(*, now: datetime | None = None, lease_seconds: int) -> datetime:
    return (now or datetime.now(UTC)) + timedelta(seconds=lease_seconds)


async def heartbeat_evaluation_job(
    session: AsyncSession,
    job_id: str,
    *,
    lease_owner: str,
    lease_seconds: int,
) -> bool:
    now = datetime.now(UTC)
    result = await session.execute(
        update(EvaluationJob)
        .where(EvaluationJob.job_id == job_id)
        .where(EvaluationJob.status == "running")
        .where(EvaluationJob.lease_owner == lease_owner)
        .values(
            heartbeat_at=now,
            lease_expires_at=lease_deadline(now=now, lease_seconds=lease_seconds),
        )
    )
    return result.rowcount == 1


async def heartbeat_analysis_run(
    session: AsyncSession,
    analysis_run_id: int,
    *,
    lease_owner: str,
    lease_seconds: int,
) -> bool:
    return await _heartbeat_by_id(
        session,
        AnalysisRun,
        analysis_run_id,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
    )


async def heartbeat_evaluation_attempt(
    session: AsyncSession,
    attempt_id: int,
    *,
    lease_owner: str,
    lease_seconds: int,
) -> bool:
    return await _heartbeat_by_id(
        session,
        EvaluationAttempt,
        attempt_id,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
    )


async def heartbeat_terminal_bench_trial(
    session: AsyncSession,
    trial_id: int,
    *,
    lease_owner: str,
    lease_seconds: int,
) -> bool:
    return await _heartbeat_by_id(
        session,
        TerminalBenchTrial,
        trial_id,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
    )


async def _heartbeat_by_id(
    session: AsyncSession,
    model: type[AnalysisRun] | type[EvaluationAttempt] | type[TerminalBenchTrial],
    row_id: int,
    *,
    lease_owner: str,
    lease_seconds: int,
) -> bool:
    now = datetime.now(UTC)
    result = await session.execute(
        update(model)
        .where(model.id == row_id)
        .where(model.status == "running")
        .where(model.lease_owner == lease_owner)
        .values(
            heartbeat_at=now,
            lease_expires_at=lease_deadline(now=now, lease_seconds=lease_seconds),
        )
    )
    return result.rowcount == 1
