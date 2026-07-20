"""Regression tests for the async/perf fixes in the combined-worker deployment.

These cover the three fixes that keep one shared event loop + small connection
pool healthy under load:

* Fix #2 -- O(1) durable log-byte accounting (``TaskLogByteTotal`` counters +
  ``message_bytes`` column): byte-exact across separate ingest batches, caps
  enforced identically, one-time backfill correctness, and the new lookup index.
* Fix #2 -- chunked commits during live-log ingest.
* Fix #3 -- SSE generators acquire a *fresh* short-lived session per poll and
  release it between polls (so pooled connections are not pinned for the whole
  stream and are cleaned up on client disconnect).
"""

from __future__ import annotations

import asyncio
import contextlib
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.api import routes
from agent_challenge.evaluation import task_events
from agent_challenge.evaluation.terminal_bench import TERMINAL_BENCH_EVALUATOR
from agent_challenge.models import (
    AgentSubmission,
    EvaluationAttempt,
    EvaluationJob,
    TaskLogByteTotal,
    TaskLogEvent,
)
from agent_challenge.sdk.auth import mint_attempt_stream_token
from agent_challenge.sdk.db import Database

_SLUG = "agent-challenge"
_SHARED_TOKEN = "test-token"


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


async def _byte_total(
    session: AsyncSession, submission_id: int, scope: str, scope_key: str
) -> int | None:
    return await session.scalar(
        select(TaskLogByteTotal.total_bytes).where(
            TaskLogByteTotal.submission_id == submission_id,
            TaskLogByteTotal.scope == scope,
            TaskLogByteTotal.scope_key == scope_key,
        )
    )


# --------------------------------------------------------------------------- #
# Fix #2: byte-exact durable accounting across separate ingest batches
# --------------------------------------------------------------------------- #


async def test_log_byte_totals_are_byte_exact_across_separate_ingest_batches(database_session):
    async with database_session() as session:
        submission = await _make_submission(session, "cross-batch")
        submission_id = submission.id
        await session.commit()

    messages = ["héllo-1", "second line", "x" * 4096, "unicode 世界 🌍", "tail"]
    # Each message is recorded in its OWN session + commit == a separate ingest
    # batch, so the running-total counters must accumulate durably across batches
    # (the pre-fix code re-scanned every prior row instead).
    for message in messages:
        async with database_session() as session:
            await task_events.record_task_event(
                session,
                submission_id=submission_id,
                task_id="task-a",
                task_result_id=41,
                event_type="task.log",
                stream="agent",
                message=message,
            )
            await session.commit()

    expected = sum(len(message.encode("utf-8")) for message in messages)
    async with database_session() as session:
        submission_total = await _byte_total(
            session, submission_id, task_events.LOG_BYTE_SCOPE_SUBMISSION, ""
        )
        task_total = await _byte_total(
            session, submission_id, task_events.LOG_BYTE_SCOPE_TASK, "task-a"
        )
        result_total = await _byte_total(
            session, submission_id, task_events.LOG_BYTE_SCOPE_TASK_RESULT, "41"
        )
        # Independent O(n) full-scan recomputation (the pre-fix definition) proves
        # the O(1) counters are byte-exact.
        scanned = (
            await session.execute(
                select(TaskLogEvent.message_bytes, TaskLogEvent.event_type).where(
                    TaskLogEvent.submission_id == submission_id
                )
            )
        ).all()

    reference = sum(
        message_bytes
        for message_bytes, event_type in scanned
        if event_type not in task_events._NON_COUNTED_EVENT_TYPES
    )
    assert expected > 0
    assert submission_total == expected
    assert task_total == expected
    assert result_total == expected
    assert reference == expected


async def test_log_byte_cap_enforced_identically_across_separate_batches(
    database_session, monkeypatch
):
    monkeypatch.setattr(task_events, "MAX_TASK_EVENT_BYTES", 100)
    monkeypatch.setattr(task_events, "MAX_TASK_LOG_BYTES", 10)
    monkeypatch.setattr(task_events, "MAX_SUBMISSION_LOG_BYTES", 1000)

    async with database_session() as session:
        submission = await _make_submission(session, "cap-batches")
        submission_id = submission.id
        await session.commit()

    async def _record(message: str) -> list[TaskLogEvent]:
        async with database_session() as session:
            events = await task_events.record_task_event(
                session,
                submission_id=submission_id,
                task_id="task-a",
                event_type="task.log",
                stream="agent",
                message=message,
            )
            await session.commit()
            return events

    first = await _record("123456")  # 6 bytes, under the 10-byte task cap
    second = await _record("abcdef")  # would reach 12 -> truncated to 4, cap marker
    third = await _record("ghi")  # fully suppressed once the cap is reached

    assert [event.event_type for event in first] == ["task.log"]
    assert first[0].message == "123456"
    assert first[0].truncated is False

    assert [event.event_type for event in second] == [
        "task.log",
        task_events.TASK_LOG_CAP_EVENT_TYPE,
    ]
    assert second[0].message == "abcd"
    assert second[0].truncated is True

    assert third == []

    async with database_session() as session:
        task_total = await _byte_total(
            session, submission_id, task_events.LOG_BYTE_SCOPE_TASK, "task-a"
        )
        submission_total = await _byte_total(
            session, submission_id, task_events.LOG_BYTE_SCOPE_SUBMISSION, ""
        )
    # Exactly the 10-byte task cap of stored payload bytes, no more, no less.
    assert task_total == 10
    assert submission_total == 10


async def test_task_log_byte_backfill_seeds_counters_and_creates_index(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    legacy_db = Database(f"sqlite+aiosqlite:///{db_path}")
    await legacy_db.init()
    try:
        async with legacy_db.session() as session:
            submission = await _make_submission(session, "legacy")
            submission_id = submission.id
            await session.commit()

        # Simulate pre-fix rows: ``message_bytes`` left at the server default (0)
        # and NO running-total rows -> exactly the state the one-time backfill
        # targets. Includes a non-counted ``task.progress`` row which must be
        # excluded from the counters but still get its ``message_bytes`` filled.
        async with legacy_db.engine.begin() as conn:
            await conn.exec_driver_sql(
                "INSERT INTO task_log_events "
                "(submission_id, task_result_id, task_id, sequence, event_type, "
                "message, metadata_json, created_at) VALUES "
                f"({submission_id}, 7, 'task-a', 1, 'task.log', 'héllo', '{{}}', "
                "CURRENT_TIMESTAMP),"
                f"({submission_id}, 7, 'task-a', 2, 'task.log', 'world!!', '{{}}', "
                "CURRENT_TIMESTAMP),"
                f"({submission_id}, NULL, 'task-a', 3, 'task.progress', 'skip me', "
                "'{{}}', CURRENT_TIMESTAMP)"
            )

        # Re-run the task-log migration; counters are empty so the backfill runs.
        async with legacy_db.engine.begin() as conn:
            await legacy_db._migrate_sqlite_task_log_columns(conn)
            index_names = {
                row[1] for row in await conn.exec_driver_sql("PRAGMA index_list(task_log_events)")
            }

        counted_bytes = len("héllo".encode()) + len(b"world!!")
        async with legacy_db.session() as session:
            rows = (
                await session.execute(
                    select(TaskLogEvent.event_type, TaskLogEvent.message_bytes)
                    .where(TaskLogEvent.submission_id == submission_id)
                    .order_by(TaskLogEvent.sequence)
                )
            ).all()
            submission_total = await _byte_total(
                session, submission_id, task_events.LOG_BYTE_SCOPE_SUBMISSION, ""
            )
            task_total = await _byte_total(
                session, submission_id, task_events.LOG_BYTE_SCOPE_TASK, "task-a"
            )
            result_total = await _byte_total(
                session, submission_id, task_events.LOG_BYTE_SCOPE_TASK_RESULT, "7"
            )

        # message_bytes backfilled byte-exactly for every row (counted or not).
        assert [message_bytes for _event_type, message_bytes in rows] == [
            len("héllo".encode()),
            len(b"world!!"),
            len(b"skip me"),
        ]
        # Counters seeded byte-exactly, excluding the non-counted task.progress row.
        assert submission_total == counted_bytes
        assert task_total == counted_bytes
        assert result_total == counted_bytes
        # The new lookup index for task_result_id fan-out exists.
        assert "ix_task_log_events_task_result_id" in index_names
    finally:
        await legacy_db.close()


# --------------------------------------------------------------------------- #
# Fix #2: chunked commits during live-log ingest
# --------------------------------------------------------------------------- #


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


async def test_live_log_ingest_commits_per_event(client, database_session, monkeypatch):
    async with database_session() as session:
        attempt = await _attempt(session, agent_hash="chunked-commit")
        await session.commit()
        attempt_id = attempt.id

    commit_calls = 0
    real_commit = AsyncSession.commit

    async def counting_commit(self):
        nonlocal commit_calls
        commit_calls += 1
        return await real_commit(self)

    monkeypatch.setattr(AsyncSession, "commit", counting_commit)

    body = "\n".join(
        json.dumps({"kind": "log", "stream": "agent", "message": f"line {index}"})
        for index in range(5)
    ).encode("utf-8")
    response = await client.post(
        f"/internal/v1/evaluations/{attempt_id}/events",
        content=body,
        headers=_stream_headers(attempt_id),
    )

    assert response.status_code == 202
    assert response.json() == {"recorded": 5}
    # Each event is recorded + committed in its own lock-retry transaction, so the
    # SQLite writer lock is released after every event rather than held across the
    # whole (up to 512-event) request. The pre-fix code committed exactly once, so
    # one commit per event (>= 5 here) proves the lock is not held request-wide.
    assert commit_calls >= 5


# --------------------------------------------------------------------------- #
# Fix #3: SSE generators use a fresh short-lived session per poll
# --------------------------------------------------------------------------- #


class _SessionAcquisitionCounter:
    def __init__(self, monkeypatch) -> None:
        self.opened = 0
        self.closed = 0
        self._real = routes.database.session
        monkeypatch.setattr(routes.database, "session", self._counting)

    @contextlib.asynccontextmanager
    async def _counting(self):
        self.opened += 1
        async with self._real() as inner:
            try:
                yield inner
            finally:
                self.closed += 1


async def _drive_stream_polls(stream) -> None:
    pending = asyncio.create_task(anext(stream))
    # Several poll iterations elapse at SSE_POLL_SECONDS (0.01s) each.
    await asyncio.sleep(0.1)
    pending.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await pending
    await stream.aclose()


async def test_task_event_stream_uses_fresh_session_per_poll(database_session, monkeypatch):
    monkeypatch.setattr(routes, "SSE_POLL_SECONDS", 0.01)
    # Suppress heartbeats so anext keeps polling (never returns) while we sample.
    monkeypatch.setattr(routes, "SSE_HEARTBEAT_SECONDS", 1000.0)

    async with database_session() as session:
        submission = await _make_submission(session, "sse-task-poll")
        submission_id = submission.id
        await session.commit()

    counter = _SessionAcquisitionCounter(monkeypatch)
    await _drive_stream_polls(routes._submission_task_event_stream(submission_id, None, 0))

    # A fresh session per poll (connection released between polls, not pinned) ...
    assert counter.opened >= 2
    # ... and every per-poll session was released, including on cancellation.
    assert counter.closed == counter.opened


async def test_submission_event_stream_uses_fresh_session_per_poll(database_session, monkeypatch):
    monkeypatch.setattr(routes, "SSE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(routes, "SSE_HEARTBEAT_SECONDS", 1000.0)

    async with database_session() as session:
        submission = await _make_submission(session, "sse-status-poll")
        submission_id = submission.id
        await session.commit()

    counter = _SessionAcquisitionCounter(monkeypatch)
    await _drive_stream_polls(routes._submission_event_stream(submission_id, None))

    assert counter.opened >= 2
    assert counter.closed == counter.opened
