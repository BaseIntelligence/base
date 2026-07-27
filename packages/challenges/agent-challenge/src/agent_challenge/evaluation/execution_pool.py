"""Live execution pool snapshot for master fan-out aggregation.

``GET /v1/execution-pool/live`` returns in-flight EvalRun rows with each run's
latest TaskLogEvent. Observability only — never score / weight fields.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.core.models import EvalRun, TaskLogEvent

# Keep in lockstep with evaluation.authorization._ACTIVE_PHASES and
# evaluation.telemetry_session._ACTIVE_EVAL_PHASES.
ACTIVE_EVAL_PHASES: frozenset[str] = frozenset(
    {"eval_prepared", "eval_running", "eval_verifying"}
)

_SCORE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "score",
        "scores",
        "final_score",
        "raw_score",
        "normalized_score",
        "weight",
        "weights",
        "emission",
        "emission_percent",
        "incentive",
        "passed_tasks",
        "total_tasks",
        "canonical_score_record_json",
        "canonical_score_record_sha256",
    }
)


def _metadata_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.isoformat() + "+00:00"
    return value.isoformat()


def _latest_event_payload(event: TaskLogEvent) -> dict[str, Any]:
    meta = _metadata_dict(event.metadata_json)
    phase = event.status or meta.get("phase")
    client_sequence = meta.get("client_sequence")
    sequence = (
        int(client_sequence)
        if isinstance(client_sequence, int)
        else int(event.sequence)
    )
    payload: dict[str, Any] = {
        "event_type": event.event_type,
        "sequence": sequence,
        "task_id": event.task_id,
        "message": event.message,
        "progress": event.progress,
        "phase": phase,
        "created_at": _iso(event.created_at),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _unit_from_run(
    run: EvalRun, *, latest_event: Mapping[str, Any] | None
) -> dict[str, Any]:
    unit: dict[str, Any] = {
        "unit_id": run.eval_run_id,
        "eval_run_id": run.eval_run_id,
        "submission_id": str(run.submission_id),
        "status": run.phase,
        "phase": run.phase,
        "latest_event": dict(latest_event) if latest_event is not None else None,
    }
    # Defense in depth: never leak score-shaped keys even if a caller mutates.
    for banned in _SCORE_FIELD_NAMES:
        unit.pop(banned, None)
    return unit


async def _latest_events_by_eval_run(
    session: AsyncSession,
    *,
    submission_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Map eval_run_id → latest TaskLogEvent payload for the given submissions."""

    if not submission_ids:
        return {}
    rows = await session.scalars(
        select(TaskLogEvent)
        .where(TaskLogEvent.submission_id.in_(list(submission_ids)))
        .order_by(TaskLogEvent.sequence.desc(), TaskLogEvent.id.desc())
    )
    latest: dict[str, dict[str, Any]] = {}
    for event in rows:
        meta = _metadata_dict(event.metadata_json)
        eval_run_id = meta.get("eval_run_id")
        if not isinstance(eval_run_id, str) or not eval_run_id:
            continue
        if eval_run_id in latest:
            continue
        latest[eval_run_id] = _latest_event_payload(event)
    return latest


async def list_live_execution_units(session: AsyncSession) -> list[dict[str, Any]]:
    """Return in-flight EvalRun units with latest progress event (if any)."""

    runs = list(
        await session.scalars(
            select(EvalRun)
            .where(EvalRun.phase.in_(tuple(ACTIVE_EVAL_PHASES)))
            .order_by(EvalRun.updated_at.desc(), EvalRun.id.desc())
        )
    )
    if not runs:
        return []

    latest_by_run = await _latest_events_by_eval_run(
        session,
        submission_ids=[run.submission_id for run in runs],
    )
    return [
        _unit_from_run(run, latest_event=latest_by_run.get(run.eval_run_id))
        for run in runs
    ]


__all__ = [
    "ACTIVE_EVAL_PHASES",
    "list_live_execution_units",
]
