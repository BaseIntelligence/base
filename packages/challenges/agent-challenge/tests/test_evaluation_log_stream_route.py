"""Phase 4 internal real-time log ingest route.

``POST /internal/v1/evaluations/{attempt_id}/events`` accepts NDJSON log events
from an own_runner job and appends them to the live SSE feed. The route is the
trust boundary, so these tests pin:

* the per-attempt SCOPED-token auth (wrong slug -> 403, wrong/absent token ->
  401, correct scoped token -> 202);
* attribution binding to the attempt's own submission/job/task (never values
  from the request body, which a miner controls);
* score independence (ingesting logs never touches ``attempt.score``);
* input validation (unknown attempt -> 404, non-log/unknown-stream events
  ignored, oversized body -> 413).
"""

from __future__ import annotations

import json

from sqlalchemy import select

from agent_challenge.api import routes
from agent_challenge.evaluation.terminal_bench import TERMINAL_BENCH_EVALUATOR
from agent_challenge.models import (
    AgentSubmission,
    EvaluationAttempt,
    EvaluationJob,
    TaskLogEvent,
)
from agent_challenge.sdk.auth import mint_attempt_stream_token

_SLUG = "agent-challenge"
_SHARED_TOKEN = "test-token"


async def _attempt(session, *, agent_hash: str, task_id: str = "hello-world") -> EvaluationAttempt:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="running",
        selected_tasks_json="[]",
        attempt_count=1,
    )
    session.add(job)
    await session.flush()
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=1,
        task_id=task_id,
        evaluator_name=TERMINAL_BENCH_EVALUATOR,
        status="running",
    )
    session.add(attempt)
    await session.flush()
    return attempt


def _stream_headers(attempt_id: int, *, token: str | None = None, slug: str = _SLUG) -> dict:
    scoped = token if token is not None else mint_attempt_stream_token(_SHARED_TOKEN, attempt_id)
    headers = {"X-Base-Challenge-Slug": slug, "Content-Type": "application/x-ndjson"}
    if scoped:
        headers["Authorization"] = f"Bearer {scoped}"
    return headers


def _ndjson(*events: dict) -> bytes:
    return "\n".join(json.dumps(event) for event in events).encode("utf-8")


async def _events(session, submission_id: int) -> list[TaskLogEvent]:
    rows = await session.execute(
        select(TaskLogEvent)
        .where(TaskLogEvent.submission_id == submission_id)
        .order_by(TaskLogEvent.sequence)
    )
    return list(rows.scalars().all())


async def test_ingest_records_events_bound_to_attempt(client, database_session) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-ok")
        await session.commit()
        attempt_id, submission_id = attempt.id, attempt.submission_id

    body = _ndjson(
        # task_id in the body is attacker-controlled and MUST be ignored.
        {"kind": "log", "stream": "agent", "message": "agent line", "task_id": "SPOOFED"},
        {"kind": "log", "stream": "test_stdout", "message": "1 passed", "trial_name": "t0"},
    )
    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=body,
        headers=_stream_headers(attempt_id),
    )
    assert response.status_code == 202
    assert response.json() == {"recorded": 2}

    async with database_session() as session:
        events = await _events(session, submission_id)
        attempt = await session.get(EvaluationAttempt, attempt_id)

    assert [(event.event_type, event.stream, event.message) for event in events] == [
        ("task.log", "agent", "agent line"),
        ("task.log", "test_stdout", "1 passed"),
    ]
    # Attribution comes from the attempt row, never the request body.
    assert {event.task_id for event in events} == {"hello-world"}
    # Ingesting logs never touches the score.
    assert attempt.score is None


async def test_ingest_ignores_non_log_and_unknown_stream(client, database_session) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-filter")
        await session.commit()
        attempt_id, submission_id = attempt.id, attempt.submission_id

    body = _ndjson(
        {"kind": "progress", "status": "running"},
        {"kind": "log", "stream": "rootkit", "message": "nope"},
        {"kind": "log", "stream": "agent", "message": ""},
        {"kind": "log", "stream": "harness", "message": "kept"},
        {"garbage": True},
    )
    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=body,
        headers=_stream_headers(attempt_id),
    )
    assert response.status_code == 202
    assert response.json() == {"recorded": 1}

    async with database_session() as session:
        events = await _events(session, submission_id)
    assert [(event.stream, event.message) for event in events] == [("harness", "kept")]


async def test_ingest_redacts_builtin_secret_patterns(client, database_session) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-redact")
        await session.commit()
        attempt_id, submission_id = attempt.id, attempt.submission_id

    body = _ndjson(
        {"kind": "log", "stream": "agent", "message": "token Bearer abcdef.GHIJ-klmno here"},
    )
    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=body,
        headers=_stream_headers(attempt_id),
    )
    assert response.status_code == 202
    async with database_session() as session:
        events = await _events(session, submission_id)
    assert "abcdef.GHIJ-klmno" not in events[0].message
    assert "Bearer [REDACTED]" in events[0].message


async def test_ingest_rejects_wrong_slug(client, database_session) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-slug")
        await session.commit()
        attempt_id = attempt.id

    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=_ndjson({"kind": "log", "stream": "agent", "message": "x"}),
        headers=_stream_headers(attempt_id, slug="wrong-challenge"),
    )
    assert response.status_code == 403


async def test_ingest_rejects_wrong_and_missing_token(client, database_session) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-token")
        await session.commit()
        attempt_id = attempt.id

    # A token minted for a DIFFERENT attempt must not work here.
    cross = mint_attempt_stream_token(_SHARED_TOKEN, attempt_id + 999)
    wrong = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=_ndjson({"kind": "log", "stream": "agent", "message": "x"}),
        headers=_stream_headers(attempt_id, token=cross),
    )
    assert wrong.status_code == 401

    missing = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=_ndjson({"kind": "log", "stream": "agent", "message": "x"}),
        headers=_stream_headers(attempt_id, token=""),
    )
    assert missing.status_code == 401


async def test_ingest_unknown_attempt_is_404(client) -> None:
    response = await client.post(
        "/internal/v1/evaluations/99999/events",
        content=_ndjson({"kind": "log", "stream": "agent", "message": "x"}),
        headers=_stream_headers(99999),
    )
    assert response.status_code == 404


async def test_ingest_oversized_body_is_413(client, database_session, monkeypatch) -> None:
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-big")
        await session.commit()
        attempt_id = attempt.id

    monkeypatch.setattr(routes, "MAX_STREAM_EVENTS_BYTES", 16)
    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=_ndjson({"kind": "log", "stream": "agent", "message": "x" * 100}),
        headers=_stream_headers(attempt_id),
    )
    assert response.status_code == 413
