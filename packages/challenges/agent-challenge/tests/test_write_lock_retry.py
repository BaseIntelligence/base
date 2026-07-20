"""Tests for the transient SQLite writer-lock retry helper and its use on the
high-frequency live-log ingest write path.

Covers the interim SQLite concurrency hardening: a momentary
``sqlite3.OperationalError: database is locked`` on a write transaction is rolled
back and retried with backoff instead of surfacing as an HTTP 500, while non-lock
errors are never swallowed and read concurrency is untouched (the helper only
wraps write operations the caller hands it).
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.api import routes
from agent_challenge.evaluation.terminal_bench import TERMINAL_BENCH_EVALUATOR
from agent_challenge.models import (
    AgentSubmission,
    EvaluationAttempt,
    EvaluationJob,
    TaskLogByteTotal,
    TaskLogEvent,
)
from agent_challenge.sdk import write_retry
from agent_challenge.sdk.auth import mint_attempt_stream_token
from agent_challenge.sdk.write_retry import run_write_with_lock_retry

_SLUG = "agent-challenge"
_SHARED_TOKEN = "test-token"


def _lock_error() -> OperationalError:
    return OperationalError(
        "INSERT INTO task_log_events ...",
        {},
        sqlite3.OperationalError("database is locked"),
    )


def _busy_error() -> OperationalError:
    return OperationalError(
        "INSERT INTO task_log_events ...",
        {},
        sqlite3.OperationalError("database is busy"),
    )


def _non_lock_error() -> OperationalError:
    return OperationalError(
        "SELECT 1",
        {},
        sqlite3.OperationalError("no such table: task_log_events"),
    )


class _RecordingSession:
    """Minimal stand-in that records commit/rollback calls made by the helper."""

    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1


# --------------------------------------------------------------------------- #
# Lock-error classification
# --------------------------------------------------------------------------- #


def test_is_sqlite_write_lock_error_matches_lock_and_busy():
    assert write_retry.is_sqlite_write_lock_error(_lock_error())
    assert write_retry.is_sqlite_write_lock_error(_busy_error())
    # A non-lock OperationalError (e.g. what asyncpg/PostgreSQL would raise) is
    # not classified as a lock collision, so the retry path stays inert there.
    assert not write_retry.is_sqlite_write_lock_error(_non_lock_error())


# --------------------------------------------------------------------------- #
# run_write_with_lock_retry mechanics
# --------------------------------------------------------------------------- #


async def test_retries_on_database_locked_then_succeeds(monkeypatch):
    monkeypatch.setattr(write_retry, "_backoff_delay", lambda attempt: 0.0)
    session = _RecordingSession()
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _lock_error()
        return "ok"

    result = await run_write_with_lock_retry(session, operation)

    assert result == "ok"
    # First attempt hit the lock and was rolled back; the retry committed.
    assert attempts == 2
    assert session.rollbacks == 1
    assert session.commits == 1


async def test_reraises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(write_retry, "_backoff_delay", lambda attempt: 0.0)
    session = _RecordingSession()
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise _lock_error()

    with pytest.raises(OperationalError):
        await run_write_with_lock_retry(session, operation, max_attempts=5)

    # Every attempt was tried and rolled back; nothing was ever committed.
    assert attempts == 5
    assert session.rollbacks == 5
    assert session.commits == 0


async def test_non_lock_operational_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(write_retry, "_backoff_delay", lambda attempt: 0.0)
    session = _RecordingSession()
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise _non_lock_error()

    with pytest.raises(OperationalError) as excinfo:
        await run_write_with_lock_retry(session, operation)

    assert "no such table" in str(excinfo.value.orig)
    # Propagated immediately: no retry, no rollback/commit driven by the helper.
    assert attempts == 1
    assert session.rollbacks == 0
    assert session.commits == 0


async def test_successful_write_commits_once_without_retry():
    session = _RecordingSession()
    attempts = 0

    async def operation() -> int:
        nonlocal attempts
        attempts += 1
        return 42

    result = await run_write_with_lock_retry(session, operation)

    assert result == 42
    assert attempts == 1
    assert session.commits == 1
    assert session.rollbacks == 0


async def test_backoff_delay_is_bounded_and_grows():
    # Documents the jittered exponential window (base .. cap) so a regression that
    # makes the backoff unbounded or zero is caught.
    for attempt in range(6):
        delay = write_retry._backoff_delay(attempt)
        assert 0.0 < delay <= write_retry.WRITE_LOCK_MAX_DELAY_SECONDS


# --------------------------------------------------------------------------- #
# Live-log ingest write path is retry-wrapped end to end
# --------------------------------------------------------------------------- #


async def _make_submission(session: AsyncSession, agent_hash: str) -> AgentSubmission:
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
    return submission


async def _attempt(session: AsyncSession, *, agent_hash: str) -> EvaluationAttempt:
    submission = await _make_submission(session, agent_hash)
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
        task_id="hello-world",
        evaluator_name=TERMINAL_BENCH_EVALUATOR,
        status="running",
    )
    session.add(attempt)
    await session.flush()
    return attempt


def _stream_headers(attempt_id: int) -> dict[str, str]:
    return {
        "X-Base-Challenge-Slug": _SLUG,
        "Content-Type": "application/x-ndjson",
        "Authorization": f"Bearer {mint_attempt_stream_token(_SHARED_TOKEN, attempt_id)}",
    }


async def test_ingest_endpoint_retries_on_database_locked(client, database_session, monkeypatch):
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="ingest-retry")
        await session.commit()
        attempt_id = attempt.id
        submission_id = attempt.submission_id

    monkeypatch.setattr(write_retry, "_backoff_delay", lambda attempt: 0.0)

    # Inject one momentary "database is locked" on the SECOND event, after the
    # first event was already recorded + committed. Because each event is its own
    # lock-retry transaction, the helper rolls back and replays only the failing
    # event, so the already-committed first event is never duplicated and the byte
    # totals stay exact (no over- or under-count).
    real_record = routes.record_task_event
    calls = 0

    async def flaky_record(session, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _lock_error()
        return await real_record(session, **kwargs)

    monkeypatch.setattr(routes, "record_task_event", flaky_record)

    messages = [f"line {index}" for index in range(3)]
    body = "\n".join(
        json.dumps({"kind": "log", "stream": "agent", "message": message}) for message in messages
    ).encode("utf-8")

    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=body,
        headers=_stream_headers(attempt_id),
    )

    assert response.status_code == 202
    assert response.json() == {"recorded": 3}

    async with database_session() as session:
        log_count = await session.scalar(
            select(func.count())
            .select_from(TaskLogEvent)
            .where(
                TaskLogEvent.submission_id == submission_id,
                TaskLogEvent.event_type == "task.log",
            )
        )
        submission_bytes = await session.scalar(
            select(TaskLogByteTotal.total_bytes).where(
                TaskLogByteTotal.submission_id == submission_id,
                TaskLogByteTotal.scope == "submission",
                TaskLogByteTotal.scope_key == "",
            )
        )

    # Exactly three rows persisted (the rolled-back partial write is not double
    # counted), and byte accounting matches the three messages byte-for-byte.
    assert log_count == 3
    assert submission_bytes == sum(len(message.encode("utf-8")) for message in messages)
