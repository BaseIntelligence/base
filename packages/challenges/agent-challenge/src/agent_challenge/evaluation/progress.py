"""Mid-run attested Eval progress ingest (observability only).

The CVM posts per-task phase transitions with the same Bearer ``EVAL_RUN_TOKEN``
used for the final result route. Events land in ``TaskLogEvent`` so the existing
SSE feed surfaces them during the run. This path never mutates scores.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.canonical import eval_wire
from agent_challenge.core.models import EvalRun, TaskLogEvent
from agent_challenge.evaluation.authorization import load_eval_run_plan
from agent_challenge.evaluation.task_events import (
    SAFE_TASK_PHASE_STATUSES,
    record_task_event,
)

PROGRESS_SOURCE = "eval_progress"
MAX_PROGRESS_BODY_BYTES = 16 * 1024


class EvalProgressError(ValueError):
    """Schema or policy failure for a progress ingest request."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


def _metadata_dict(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _existing_progress_event(
    session: AsyncSession,
    *,
    submission_id: int,
    eval_run_id: str,
    task_id: str,
    client_sequence: int,
) -> TaskLogEvent | None:
    rows = await session.scalars(
        select(TaskLogEvent).where(
            TaskLogEvent.submission_id == submission_id,
            TaskLogEvent.task_id == task_id,
            TaskLogEvent.event_type.in_(("task.status", "task.progress")),
        )
    )
    for event in rows:
        meta = _metadata_dict(event.metadata_json)
        if (
            meta.get("source") == PROGRESS_SOURCE
            and meta.get("eval_run_id") == eval_run_id
            and meta.get("client_sequence") == client_sequence
        ):
            return event
    return None


async def _max_client_sequence(
    session: AsyncSession,
    *,
    submission_id: int,
    eval_run_id: str,
    task_id: str,
) -> int:
    rows = await session.scalars(
        select(TaskLogEvent).where(
            TaskLogEvent.submission_id == submission_id,
            TaskLogEvent.task_id == task_id,
            TaskLogEvent.event_type.in_(("task.status", "task.progress")),
        )
    )
    highest = 0
    for event in rows:
        meta = _metadata_dict(event.metadata_json)
        if meta.get("source") != PROGRESS_SOURCE or meta.get("eval_run_id") != eval_run_id:
            continue
        seq = meta.get("client_sequence")
        if isinstance(seq, int) and seq > highest:
            highest = seq
    return highest


def _progress_receipt(
    *,
    eval_run_id: str,
    task_id: str,
    sequence: int,
    event_id: int,
    created: bool,
) -> dict[str, Any]:
    return eval_wire.validate_eval_progress_receipt(
        {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "task_id": task_id,
            "sequence": sequence,
            "event_id": event_id,
            "created": created,
        }
    )


async def process_eval_progress(
    session: AsyncSession,
    *,
    run: EvalRun,
    progress_request: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Validate and record one mid-run progress event.

    Returns ``(receipt, created)``. ``created`` is False on idempotent replay of
    the same ``(eval_run_id, task_id, sequence)``. Never mutates ``EvalRun.score``
    or any score columns.
    """

    # Fail closed if wire taxonomy drifts from the public SSE phase set.
    if SAFE_TASK_PHASE_STATUSES != eval_wire.EVAL_PROGRESS_PHASES:
        raise EvalProgressError(
            "progress phase taxonomy mismatch",
            code="progress_phase_taxonomy",
        )

    try:
        validated = eval_wire.validate_eval_progress_request(progress_request)
    except eval_wire.EvalWireError as exc:
        message = str(exc)
        if "forbids score fields" in message:
            raise EvalProgressError(message, code="progress_score_forbidden") from exc
        if "status is not a safe task phase" in message:
            raise EvalProgressError(message, code="progress_phase_invalid") from exc
        raise EvalProgressError(message, code="progress_invalid") from exc

    if validated["eval_run_id"] != run.eval_run_id:
        raise EvalProgressError(
            "progress run does not match route",
            code="progress_run_mismatch",
        )

    plan = load_eval_run_plan(run)
    if validated["submission_id"] != plan["submission_id"]:
        raise EvalProgressError(
            "progress submission_id does not match eval plan",
            code="progress_submission_mismatch",
        )

    allowed_tasks = {item["task_id"] for item in plan["selected_tasks"]}
    if validated["task_id"] not in allowed_tasks:
        raise EvalProgressError(
            "progress task_id is not in the eval plan",
            code="progress_task_unknown",
        )

    existing = await _existing_progress_event(
        session,
        submission_id=run.submission_id,
        eval_run_id=run.eval_run_id,
        task_id=validated["task_id"],
        client_sequence=validated["sequence"],
    )
    if existing is not None:
        return (
            _progress_receipt(
                eval_run_id=run.eval_run_id,
                task_id=validated["task_id"],
                sequence=validated["sequence"],
                event_id=existing.id,
                created=False,
            ),
            False,
        )

    highest = await _max_client_sequence(
        session,
        submission_id=run.submission_id,
        eval_run_id=run.eval_run_id,
        task_id=validated["task_id"],
    )
    if validated["sequence"] <= highest:
        raise EvalProgressError(
            "progress sequence must be monotone per task",
            code="progress_sequence_stale",
        )

    message = validated["message"]
    if message is None:
        message = f"task {validated['task_id']} {validated['status']}"

    events = await record_task_event(
        session,
        submission_id=run.submission_id,
        job_id=None,
        task_id=validated["task_id"],
        event_type=validated["event_type"],
        message=message,
        progress=validated["progress"],
        status=validated["status"],
        metadata={
            "source": PROGRESS_SOURCE,
            "eval_run_id": run.eval_run_id,
            "client_sequence": validated["sequence"],
            "phase": validated["status"],
        },
    )
    if not events:
        raise EvalProgressError(
            "progress event was not recorded",
            code="progress_not_recorded",
        )
    event = events[0]
    await session.flush()
    return (
        _progress_receipt(
            eval_run_id=run.eval_run_id,
            task_id=validated["task_id"],
            sequence=validated["sequence"],
            event_id=event.id,
            created=True,
        ),
        True,
    )


__all__ = [
    "MAX_PROGRESS_BODY_BYTES",
    "PROGRESS_SOURCE",
    "EvalProgressError",
    "process_eval_progress",
]
