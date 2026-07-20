from __future__ import annotations

import asyncio
import json

from sqlalchemy import select
from test_task_event_replay import (
    _assert_task_event_payload_is_public_safe,
    _create_submission_with_events,
)

from agent_challenge.api import routes
from agent_challenge.evaluation import task_events
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.models import AgentSubmission, EvaluationJob

TASK_STREAM_DATA_KEYS = {
    "id",
    "sequence",
    "submission_id",
    "job_id",
    "task_id",
    "version_label",
    "event_type",
    "created_at",
    "stream",
    "message",
    "progress",
    "status",
    "truncated",
    "cap_reached",
    "metadata",
}


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in text.strip().split("\n\n"):
        if not frame or frame.startswith(":"):
            continue
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            if line.startswith(":"):
                continue
            name, value = line.split(": ", 1)
            fields[name] = value
        events.append(
            {
                "id": int(fields["id"]),
                "event": fields["event"],
                "data": json.loads(fields["data"]),
            }
        )
    return events


def _assert_task_stream_event_shape(event: dict[str, object]) -> None:
    data = event["data"]
    assert isinstance(data, dict)
    assert set(data) == TASK_STREAM_DATA_KEYS
    assert event["id"] == data["id"] == data["sequence"]
    assert event["event"] == data["event_type"]
    _assert_task_event_payload_is_public_safe(data)


async def test_task_event_stream_returns_event_stream_and_backlog_in_sequence_order(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-backlog",
            include_completed_event=True,
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == list(range(1, 9))
    assert [event["data"]["sequence"] for event in events] == list(range(1, 9))
    assert [event["event"] for event in events][-1] == "task.completed"
    for event in events:
        _assert_task_stream_event_shape(event)


async def test_task_event_stream_filters_by_stream(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-sse-filter",
            include_completed_event=True,
        )
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events/stream",
        params={"stream": "stdout"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == [2]
    assert {event["data"]["stream"] for event in events} == {"stdout"}
    for event in events:
        _assert_task_stream_event_shape(event)


async def test_task_event_stream_reconnects_after_last_event_id_without_duplicates(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-last-event-id",
            include_completed_event=True,
        )
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events/stream",
        headers={"Last-Event-ID": "5"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == [6, 7, 8]
    assert len({event["id"] for event in events}) == len(events)
    for event in events:
        _assert_task_stream_event_shape(event)


async def test_task_event_stream_cursor_reconnects_and_takes_precedence_over_last_event_id(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-cursor",
            include_completed_event=True,
        )
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events/stream",
        params={"cursor": "6"},
        headers={"Last-Event-ID": "2"},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == [7, 8]
    for event in events:
        _assert_task_stream_event_shape(event)


async def test_task_event_stream_rejects_malformed_negative_and_future_ids(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-bad-cursor",
            include_completed_event=True,
        )
        await session.commit()

    responses = [
        await client.get(f"/submissions/{submission_id}/task-events/stream?cursor=abc"),
        await client.get(f"/submissions/{submission_id}/task-events/stream?cursor=-1"),
        await client.get(f"/submissions/{submission_id}/task-events/stream?cursor=99"),
        await client.get(
            f"/submissions/{submission_id}/task-events/stream",
            headers={"Last-Event-ID": "abc"},
        ),
        await client.get(
            f"/submissions/{submission_id}/task-events/stream",
            headers={"Last-Event-ID": "-1"},
        ),
        await client.get(
            f"/submissions/{submission_id}/task-events/stream",
            headers={"Last-Event-ID": "99"},
        ),
    ]

    for response in responses:
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "task_event_cursor_invalid"
    assert responses[2].json()["detail"]["max_sequence"] == 8
    assert responses[-1].json()["detail"]["max_sequence"] == 8


async def test_task_event_stream_heartbeat_frame_does_not_break_parsing(
    database_session,
    monkeypatch,
):
    monkeypatch.setattr(routes, "SSE_HEARTBEAT_SECONDS", 0.0)
    monkeypatch.setattr(routes, "SSE_POLL_SECONDS", 0.01)
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-heartbeat",
            include_default_events=False,
        )
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        await session.commit()
        stream = routes._submission_task_event_stream(submission_id, submission.version_label, 0)
        frame = await asyncio.wait_for(anext(stream), timeout=1)
        await stream.aclose()

    assert frame == ": heartbeat\n\n"
    assert _parse_sse_events(frame) == []


async def test_task_event_stream_active_stream_receives_new_persisted_event(
    database_session,
    monkeypatch,
):
    monkeypatch.setattr(routes, "SSE_POLL_SECONDS", 0.01)
    async with database_session() as setup_session:
        submission_id = await _create_submission_with_events(
            setup_session,
            agent_hash="stream-live",
            include_default_events=False,
        )
        await setup_session.commit()

    async with database_session() as stream_session:
        submission = await stream_session.get(AgentSubmission, submission_id)
        assert submission is not None
        stream = routes._submission_task_event_stream(submission_id, submission.version_label, 0)
        pending_frame = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.05)
        async with database_session() as writer_session:
            job = await writer_session.scalar(
                select(EvaluationJob).where(EvaluationJob.submission_id == submission_id)
            )
            assert job is not None
            await task_events.record_task_event(
                writer_session,
                submission_id=submission_id,
                job_id=job.id,
                task_id="task-live",
                event_type="task.progress",
                progress=0.25,
                status="running",
                message="live progress",
            )
            await writer_session.commit()
        frame = await asyncio.wait_for(pending_frame, timeout=1)
        await stream.aclose()

    [event] = _parse_sse_events(frame)
    assert event["event"] == "task.progress"
    assert event["id"] == 1
    assert event["data"] == {
        **event["data"],
        "sequence": 1,
        "event_type": "task.progress",
        "task_id": "task-live",
        "progress": 0.25,
        "status": "running",
        "message": "live progress",
    }
    _assert_task_stream_event_shape(event)


async def test_task_event_stream_terminal_events_use_completed_and_failed_event_names(
    client,
    database_session,
):
    async with database_session() as session:
        completed_id = await _create_submission_with_events(
            session,
            agent_hash="stream-terminal-completed",
            include_default_events=False,
        )
        failed_id = await _create_submission_with_events(
            session,
            agent_hash="stream-terminal-failed",
            include_default_events=False,
        )
        completed_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.submission_id == completed_id)
        )
        failed_job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.submission_id == failed_id)
        )
        assert completed_job is not None
        assert failed_job is not None
        await task_events.record_task_event(
            session,
            submission_id=completed_id,
            job_id=completed_job.id,
            task_id="task-terminal",
            event_type="task.completed",
            status="completed",
            message="done",
        )
        await task_events.record_task_event(
            session,
            submission_id=failed_id,
            job_id=failed_job.id,
            task_id="task-terminal",
            event_type="task.failed",
            status="failed",
            message="failed",
        )
        await session.commit()

    completed_response = await client.get(f"/submissions/{completed_id}/task-events/stream")
    failed_response = await client.get(f"/submissions/{failed_id}/task-events/stream")

    assert completed_response.status_code == 200
    assert failed_response.status_code == 200
    completed_events = _parse_sse_events(completed_response.text)
    failed_events = _parse_sse_events(failed_response.text)
    assert completed_events[-1]["event"] == "task.completed"
    assert completed_events[-1]["data"]["event_type"] == "task.completed"
    assert failed_events[-1]["event"] == "task.failed"
    assert failed_events[-1]["data"]["event_type"] == "task.failed"


async def test_task_event_stream_exposes_safe_phase_statuses(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-safe-phases",
            include_default_events=False,
        )
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.submission_id == submission_id)
        )
        assert job is not None
        task = BenchmarkTask(
            task_id="task-stream-phase",
            docker_image="ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
            benchmark="terminal_bench",
        )
        for phase in ("assigned", "starting", "running"):
            await task_events.record_task_phase_event(
                session,
                submission_id=submission_id,
                job_id=job.id,
                task=task,
                phase=phase,
                attempt=1 if phase == "running" else None,
            )
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id=task.task_id,
            event_type="task.completed",
            status="completed",
            message="terminal close",
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events/stream")

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    phase_events = [event for event in events if event["event"] == "task.status"]
    assert [event["data"]["status"] for event in phase_events] == [
        "assigned",
        "starting",
        "running",
    ]
    assert phase_events[-1]["data"]["metadata"] == {
        "attempt": 1,
        "benchmark": "terminal_bench",
        "phase": "running",
    }
    assert events[-1]["event"] == "task.completed"
    for event in events:
        _assert_task_stream_event_shape(event)


async def test_task_event_stream_redaction_and_cap_markers_match_replay(
    client,
    database_session,
    monkeypatch,
):
    monkeypatch.setattr(task_events, "MAX_TASK_EVENT_BYTES", 100)
    monkeypatch.setattr(task_events, "MAX_TASK_LOG_BYTES", 10)
    monkeypatch.setattr(task_events, "MAX_SUBMISSION_LOG_BYTES", 1000)
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="stream-redaction-cap",
            include_default_events=False,
        )
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.submission_id == submission_id)
        )
        assert job is not None
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id="task-cap",
            event_type="task.log",
            message="1234567890abcdef API_KEY=sk-test-secret at /tmp/private-job-dir/stdout.log",
            metadata={
                "stdout_ref": "/tmp/private-job-dir/stdout.log",
                "safe": "Bearer raw-provider-token",
            },
        )
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id="task-cap",
            event_type="task.status",
            status="running",
            message="API_KEY=sk-test-secret at /tmp/private-job-dir/stdout.log",
        )
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            event_type="submission.completed",
            status="completed",
        )
        await session.commit()

    replay = (await client.get(f"/submissions/{submission_id}/task-events?limit=10")).json()
    stream_response = await client.get(f"/submissions/{submission_id}/task-events/stream")

    assert stream_response.status_code == 200
    stream_events = _parse_sse_events(stream_response.text)
    stream_payloads = [event["data"] for event in stream_events]
    for stream_payload, replay_payload in zip(stream_payloads, replay["events"], strict=True):
        without_version = {
            key: value for key, value in stream_payload.items() if key != "version_label"
        }
        assert without_version == replay_payload
    assert [event["event"] for event in stream_events[:2]] == ["task.log", "task_log_cap_reached"]
    assert stream_payloads[0]["truncated"] is True
    assert stream_payloads[0]["metadata"] == {"safe": "Bearer [REDACTED]"}
    assert stream_payloads[1]["cap_reached"] is True
    assert stream_payloads[1]["message"] == ""
    assert "[REDACTED" in json.dumps(stream_payloads, sort_keys=True)
    _assert_task_event_payload_is_public_safe(stream_payloads)
