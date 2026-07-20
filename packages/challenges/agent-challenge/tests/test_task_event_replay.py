from __future__ import annotations

import json
from datetime import UTC, datetime

from sqlalchemy import select

from agent_challenge.api import routes
from agent_challenge.evaluation import task_events
from agent_challenge.evaluation.benchmarks import BenchmarkTask
from agent_challenge.models import AgentSubmission, EvaluationJob, SubmissionFamily
from agent_challenge.submissions.versioning import normalize_submission_name

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
FORBIDDEN_TASK_EVENT_TEXT = (
    "stdout_ref",
    "stderr_ref",
    "logs_ref",
    "/tmp/private-job-dir",
    "Bearer raw-provider-token",
    "sk-test-secret",
    "signature-secret",
    "raw-family-id",
    "normalized_name",
)


async def test_task_event_replay_first_page_returns_bounded_ordered_events(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="replay-page")
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        **payload,
        "submission_id": submission_id,
        "name": "Replay Agent replay-page",
        "agent_hash": "replay-page",
        "display_name": "Replay Agent replay-page",
        "family_id": "family-replay-page-public",
        "version_number": 1,
        "version_label": "v1",
        "version_count": 1,
        "is_latest_version": True,
        "latest_submission_id": submission_id,
        "cursor": 0,
        "next_cursor": 2,
        "limit": 2,
        "has_more": True,
    }
    assert [event["sequence"] for event in payload["events"]] == [1, 2]
    assert [event["event_type"] for event in payload["events"]] == ["task.status", "task.log"]
    assert payload["events"][0] == {
        **payload["events"][0],
        "submission_id": submission_id,
        "job_id": "job-replay-page",
        "task_id": "task-alpha",
        "event_type": "task.status",
        "stream": None,
        "message": "task alpha queued",
        "progress": None,
        "status": "queued",
        "truncated": False,
        "cap_reached": False,
        "metadata": {"phase": "queued"},
    }
    _assert_task_event_payload_is_public_safe(payload)


async def test_task_event_replay_next_page_and_no_duplicate_events(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="replay-pages")
        await session.commit()

    first = (await client.get(f"/submissions/{submission_id}/task-events?limit=3")).json()
    second = (
        await client.get(
            f"/submissions/{submission_id}/task-events",
            params={"cursor": first["next_cursor"], "limit": 3},
        )
    ).json()
    third = (
        await client.get(
            f"/submissions/{submission_id}/task-events",
            params={"cursor": second["next_cursor"], "limit": 3},
        )
    ).json()

    sequences = [event["sequence"] for page in (first, second, third) for event in page["events"]]
    assert sequences == [1, 2, 3, 4, 5, 6, 7]
    assert len(sequences) == len(set(sequences))
    assert first["has_more"] is True
    assert second["has_more"] is True
    assert third["has_more"] is False
    assert third["next_cursor"] == 7


async def test_task_event_replay_cursor_zero_replays_from_beginning(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="cursor-zero")
        await session.commit()

    missing_cursor = await client.get(f"/submissions/{submission_id}/task-events?limit=1")
    zero_cursor = await client.get(f"/submissions/{submission_id}/task-events?cursor=0&limit=1")

    assert missing_cursor.status_code == 200
    assert zero_cursor.status_code == 200
    assert missing_cursor.json()["events"] == zero_cursor.json()["events"]
    assert zero_cursor.json()["cursor"] == 0


async def test_task_event_replay_filters_by_task_id(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="task-filter")
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events",
        params={"task_id": "task-beta", "limit": 10},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["sequence"] for event in events] == [4, 5]
    assert {event["task_id"] for event in events} == {"task-beta"}


async def test_task_event_replay_filters_by_event_type(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="type-filter")
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events",
        params={"event_type": "task.progress", "limit": 10},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["sequence"] for event in events] == [3, 5]
    assert {event["event_type"] for event in events} == {"task.progress"}
    assert events[0]["progress"] == 0.5
    assert events[0]["status"] == "running"


async def test_task_event_replay_filters_by_stream(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="stream-filter")
        await session.commit()

    response = await client.get(
        f"/submissions/{submission_id}/task-events",
        params={"stream": "stderr", "limit": 10},
    )

    assert response.status_code == 200
    events = response.json()["events"]
    assert {event["stream"] for event in events} == {"stderr"}
    assert [event["sequence"] for event in events] == [4]


async def test_task_event_replay_rejects_malformed_negative_and_future_cursor(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="bad-cursor")
        await session.commit()

    malformed = await client.get(f"/submissions/{submission_id}/task-events?cursor=abc")
    negative = await client.get(f"/submissions/{submission_id}/task-events?cursor=-1")
    future = await client.get(f"/submissions/{submission_id}/task-events?cursor=99")

    for response in (malformed, negative, future):
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "task_event_cursor_invalid"
    assert future.json()["detail"]["max_sequence"] == 7


async def test_task_event_replay_valid_current_cursor_returns_empty_page(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="current-cursor")
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?cursor=7")

    assert response.status_code == 200
    assert response.json() == {
        **response.json(),
        "cursor": 7,
        "next_cursor": 7,
        "has_more": False,
        "events": [],
    }


async def test_task_event_replay_active_running_submission(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="active-running",
            raw_status="tb_running",
            effective_status="evaluating",
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert payload["events"][-1]["event_type"] == "task.status"
    assert payload["events"][-1]["status"] == "running"
    assert payload["has_more"] is False


async def test_task_event_replay_completed_submission(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="completed",
            raw_status="tb_completed",
            effective_status="valid",
            include_completed_event=True,
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=20")

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["event_type"] for event in events][-2:] == ["task.status", "task.completed"]
    assert events[-1]["status"] == "completed"


async def test_task_event_replay_cap_markers_serialize_correctly(
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
            agent_hash="cap-marker",
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
            message="1234567890abcdef",
        )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["event_type"] for event in events] == ["task.log", "task_log_cap_reached"]
    assert events[0]["truncated"] is True
    assert events[1]["cap_reached"] is True
    assert events[1]["message"] == ""


async def test_task_event_replay_public_payload_redacts_messages_and_metadata(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(session, agent_hash="redacted")
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert response.status_code == 200
    payload = response.json()
    assert "[REDACTED]" in json.dumps(payload, sort_keys=True)
    assert "[REDACTED_PATH]" in json.dumps(payload, sort_keys=True)
    _assert_task_event_payload_is_public_safe(payload)


async def test_task_event_replay_exposes_only_safe_phase_metadata(client, database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="safe-phases",
            include_default_events=False,
        )
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.submission_id == submission_id)
        )
        assert job is not None
        task = BenchmarkTask(
            task_id="task-safe-phase",
            docker_image="ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
            benchmark="terminal_bench",
        )
        for phase in ("assigned", "starting", "running", "completed", "failed"):
            await task_events.record_task_phase_event(
                session,
                submission_id=submission_id,
                job_id=job.id,
                task=task,
                phase=phase,
                attempt=1 if phase in {"running", "completed", "failed"} else None,
            )
        await session.commit()

    response = await client.get(f"/submissions/{submission_id}/task-events?limit=10")

    assert response.status_code == 200
    events = response.json()["events"]
    assert [event["status"] for event in events] == [
        "assigned",
        "starting",
        "running",
        "completed",
        "failed",
    ]
    assert all(event["event_type"] == "task.status" for event in events)
    assert all(
        set(event["metadata"]).issubset({"phase", "attempt", "benchmark"}) for event in events
    )
    assert events[0]["metadata"] == {"benchmark": "terminal_bench", "phase": "assigned"}
    assert events[-1]["metadata"] == {
        "attempt": 1,
        "benchmark": "terminal_bench",
        "phase": "failed",
    }
    _assert_task_event_payload_is_public_safe(response.json())


async def test_task_phase_contract_uses_latest_safe_status_only(database_session):
    async with database_session() as session:
        submission_id = await _create_submission_with_events(
            session,
            agent_hash="phase-contract",
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
            task_id="task-alpha",
            event_type="task.status",
            status="running",
            metadata={"phase": "running", "attempt": 1, "provider": "platform_sdk"},
        )
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id="task-alpha",
            event_type="task.status",
            status="queued",
            metadata={"phase": "queued", "attempt": 2},
        )
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job.id,
            task_id="task-beta",
            event_type="task.status",
            status="starting",
            metadata={"phase": "starting", "attempt": 0, "job_name": "k8s-job-task8"},
        )
        await session.flush()

        phases = await routes._latest_task_phases_for_job(
            session,
            submission_id=submission_id,
            job_id=job.id,
        )

    payload = [phase.model_dump(mode="json") for phase in phases]
    assert payload == [
        {
            "task_id": "task-alpha",
            "phase": "running",
            "status": "running",
            "updated_at": payload[0]["updated_at"],
            "attempt": 1,
        },
        {
            "task_id": "task-beta",
            "phase": "starting",
            "status": "starting",
            "updated_at": payload[1]["updated_at"],
            "attempt": 0,
        },
    ]
    assert all(
        set(item) == {"task_id", "phase", "status", "updated_at", "attempt"} for item in payload
    )
    _assert_task_event_payload_is_public_safe(payload)


async def _create_submission_with_events(
    session,
    *,
    agent_hash: str,
    raw_status: str = "tb_running",
    effective_status: str = "evaluating",
    include_completed_event: bool = False,
    include_default_events: bool = True,
) -> int:
    family = SubmissionFamily(
        public_family_id=f"family-{agent_hash}-public",
        owner_hotkey=f"miner-{agent_hash}",
        display_name=f"Replay Agent {agent_hash}",
        normalized_name=normalize_submission_name(f"Replay Agent {agent_hash}"),
        version_count=1,
    )
    session.add(family)
    await session.flush()
    submission = AgentSubmission(
        miner_hotkey=f"miner-{agent_hash}",
        name=f"Replay Agent {agent_hash}",
        agent_hash=agent_hash,
        artifact_uri="/tmp/private-job-dir/artifact.zip",
        submission_family_id=family.id,
        version_number=1,
        version_label="v1",
        canonical_artifact_hash=f"canonical-{agent_hash}",
        is_latest_version=True,
        status=raw_status,
        raw_status=raw_status,
        effective_status=effective_status,
        zip_sha256=f"zip-{agent_hash}",
        artifact_path="/tmp/private-job-dir/artifact.zip",
        signature="signature-secret",
        signature_nonce="nonce-secret",
        signature_payload_sha256="payload-secret",
    )
    session.add(submission)
    await session.flush()
    family.latest_submission_id = submission.id
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="running",
        selected_tasks_json="[]",
        logs_ref="/tmp/private-job-dir/job.log",
        created_at=NOW,
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    if include_default_events:
        await _append_default_task_events(
            session,
            submission_id=submission.id,
            job_id=job.id,
            include_completed_event=include_completed_event,
        )
    return submission.id


async def _append_default_task_events(
    session,
    *,
    submission_id: int,
    job_id: int,
    include_completed_event: bool,
) -> None:
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-alpha",
        event_type="task.status",
        status="queued",
        message="task alpha queued",
        metadata={"phase": "queued", "stdout_ref": "/tmp/private-job-dir/stdout.log"},
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-alpha",
        event_type="task.log",
        stream="stdout",
        message="alpha log API_KEY=sk-test-secret-123 at /tmp/private-job-dir/stdout.log",
        metadata={
            "line": 1,
            "nested": {"safe": "visible", "token": "Bearer raw-provider-token"},
            "paths": ["/tmp/private-job-dir/a.txt", "safe-value"],
            "normalized_name": "raw-family-id",
        },
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-alpha",
        event_type="task.progress",
        progress=0.5,
        status="running",
        message="alpha halfway",
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-beta",
        event_type="task.log",
        stream="stderr",
        message="beta warning Bearer raw-provider-token at /tmp/private-job-dir/stderr.log",
        metadata={"canonical_artifact_hash": "canonical-secret", "safe_count": 2},
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-beta",
        event_type="task.progress",
        progress=1.0,
        status="completed",
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        event_type="submission.status",
        status="running",
        message="submission still running",
    )
    await task_events.record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id="task-gamma",
        event_type="task.status",
        status="running",
        message="gamma running",
    )
    if include_completed_event:
        await task_events.record_task_event(
            session,
            submission_id=submission_id,
            job_id=job_id,
            task_id="task-gamma",
            event_type="task.completed",
            status="completed",
            message="gamma completed",
        )


def _assert_task_event_payload_is_public_safe(payload: object) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in FORBIDDEN_TASK_EVENT_TEXT:
        assert forbidden not in serialized
    for forbidden_field in (
        "artifact_path",
        "artifact_uri",
        "canonical_artifact_hash",
        "normalized_name",
        "signature",
        "signature_nonce",
        "signature_payload_sha256",
        "logs_ref",
        "stdout_ref",
        "stderr_ref",
        "raw_artifacts_json",
    ):
        assert forbidden_field not in serialized
