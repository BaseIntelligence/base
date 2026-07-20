from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from agent_challenge.evaluation import task_events
from agent_challenge.models import AgentSubmission, TaskLogEvent


async def _submission(session, agent_hash: str) -> AgentSubmission:
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


async def _events(session, submission_id: int) -> list[TaskLogEvent]:
    rows = await session.execute(
        select(TaskLogEvent)
        .where(TaskLogEvent.submission_id == submission_id)
        .order_by(TaskLogEvent.sequence)
    )
    return list(rows.scalars().all())


async def test_single_event_over_64kb_truncates_and_marks(database_session):
    async with database_session() as session:
        submission = await _submission(session, "truncate")

        [event] = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            stream="stdout",
            message="x" * (task_events.MAX_TASK_EVENT_BYTES + 10),
        )
        await session.commit()

        assert event.truncated is True
        assert len(event.message.encode("utf-8")) == task_events.MAX_TASK_EVENT_BYTES
        assert event.sequence == 1


async def test_task_cap_marker_emitted_and_further_log_bytes_suppressed(
    database_session,
    monkeypatch,
):
    monkeypatch.setattr(task_events, "MAX_TASK_EVENT_BYTES", 100)
    monkeypatch.setattr(task_events, "MAX_TASK_LOG_BYTES", 10)
    monkeypatch.setattr(task_events, "MAX_SUBMISSION_LOG_BYTES", 1000)

    async with database_session() as session:
        submission = await _submission(session, "task-cap")

        first_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            message="1234567890abcdef",
        )
        second_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            message="suppressed",
        )
        progress_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.progress",
            progress=0.75,
            status="running",
            message="progress survives cap",
        )
        await session.commit()

        rows = await _events(session, submission.id)

        assert [event.event_type for event in first_events] == [
            "task.log",
            "task_log_cap_reached",
        ]
        assert second_events == []
        assert progress_events[0].event_type == "task.progress"
        assert rows[0].message == "1234567890"
        assert rows[0].truncated is True
        assert rows[1].event_type == "task_log_cap_reached"
        assert rows[1].cap_reached is True
        assert rows[2].event_type == "task.progress"
        assert rows[2].message == "progress survives cap"
        assert rows[2].progress == 0.75


async def test_submission_cap_marker_emitted(database_session, monkeypatch):
    monkeypatch.setattr(task_events, "MAX_TASK_EVENT_BYTES", 100)
    monkeypatch.setattr(task_events, "MAX_TASK_LOG_BYTES", 1000)
    monkeypatch.setattr(task_events, "MAX_SUBMISSION_LOG_BYTES", 12)

    async with database_session() as session:
        submission = await _submission(session, "submission-cap")

        first_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            message="abcdefghij",
        )
        second_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-b",
            event_type="task.log",
            message="klmnopqrst",
        )
        progress_events = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-b",
            event_type="task.progress",
            status="running",
        )
        await session.commit()

        rows = await _events(session, submission.id)

        assert [event.event_type for event in first_events] == ["task.log"]
        assert [event.event_type for event in second_events] == [
            "task.log",
            "submission_log_cap_reached",
        ]
        assert second_events[0].message == "kl"
        assert second_events[0].truncated is True
        assert rows[-2].event_type == "submission_log_cap_reached"
        assert rows[-2].cap_reached is True
        assert progress_events[0].event_type == "task.progress"
        assert rows[-1].event_type == "task.progress"


async def test_redaction_removes_api_key_secret(database_session):
    async with database_session() as session:
        submission = await _submission(session, "redaction")

        [event] = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            message="running API_KEY=sk-test-secret-123 now",
        )
        await session.commit()

        assert "sk-test-secret" not in event.message
        assert event.message == "running API_KEY=[REDACTED] now"


async def test_submission_sequence_unique_constraint_enforced(database_session):
    async with database_session() as session:
        submission = await _submission(session, "unique")
        session.add_all(
            [
                TaskLogEvent(
                    submission_id=submission.id,
                    sequence=1,
                    event_type="task.log",
                    message="first",
                ),
                TaskLogEvent(
                    submission_id=submission.id,
                    sequence=1,
                    event_type="task.log",
                    message="duplicate",
                ),
            ]
        )

        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
        else:
            raise AssertionError("duplicate task log sequence was accepted")


async def test_record_task_event_retries_stale_sequence_collision(
    database_session,
    monkeypatch,
):
    original_next_sequence = task_events.next_task_event_sequence
    calls = 0

    async def stale_then_current(session, submission_id: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 1
        return await original_next_sequence(session, submission_id)

    monkeypatch.setattr(task_events, "next_task_event_sequence", stale_then_current)

    async with database_session() as session:
        submission = await _submission(session, "collision-retry")
        session.add(
            TaskLogEvent(
                submission_id=submission.id,
                sequence=1,
                event_type="task.log",
                task_id="task-a",
                message="existing",
            )
        )
        await session.flush()

        [event] = await task_events.record_task_event(
            session,
            submission_id=submission.id,
            task_id="task-a",
            event_type="task.log",
            message="retried",
        )
        await session.commit()

        rows = await _events(session, submission.id)

        assert calls == 2
        assert event.sequence == 2
        assert event.message == "retried"
        assert [(row.sequence, row.message) for row in rows] == [(1, "existing"), (2, "retried")]


async def test_record_task_event_retry_exhaustion_is_bounded(
    database_session,
    monkeypatch,
):
    calls = 0

    async def stale_sequence(session, submission_id: int) -> int:
        nonlocal calls
        calls += 1
        return 1

    monkeypatch.setattr(task_events, "MAX_SEQUENCE_ALLOCATION_RETRIES", 2)
    monkeypatch.setattr(task_events, "next_task_event_sequence", stale_sequence)

    async with database_session() as session:
        submission = await _submission(session, "collision-exhaustion")
        session.add(
            TaskLogEvent(
                submission_id=submission.id,
                sequence=1,
                event_type="task.log",
                task_id="task-a",
                message="existing",
            )
        )
        await session.flush()

        try:
            await task_events.record_task_event(
                session,
                submission_id=submission.id,
                task_id="task-a",
                event_type="task.log",
                message="never persisted",
            )
        except IntegrityError:
            pass
        else:
            raise AssertionError("sequence collision retry exhaustion did not raise")

        rows = await _events(session, submission.id)

        assert calls == 2
        assert [(row.sequence, row.message) for row in rows] == [(1, "existing")]
