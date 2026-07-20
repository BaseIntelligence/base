from __future__ import annotations

import json

from agent_challenge.models import AgentSubmission
from agent_challenge.submissions.state_machine import transition_submission_status

PUBLIC_SSE_DATA_KEYS = {
    "id",
    "sequence",
    "submission_id",
    "status",
    "public_state",
    "phase",
    "created_at",
}
OPTIONAL_PUBLIC_SSE_DATA_KEYS = {"reason_code", "actor"}
FORBIDDEN_SSE_DATA_KEYS = {
    "metadata",
    "metadata_json",
    "secret",
    "private_path",
    "raw_reason",
    "reason",
    "source",
    "token",
    "lease",
    "lease_owner",
    "broker_ref",
    "from_status",
    "to_status",
    "raw_status",
    "artifact_uri",
    "artifact_path",
    "zip_sha256",
}
FORBIDDEN_SSE_TEXT = (
    "private-source",
    "sk-test-secret",
    "artifact verified",
    'reason":',
    "metadata",
    "secret",
    "private_path",
    "raw_reason",
    "source",
    "token",
    "lease",
    "broker-ref",
)
SAFE_SSE_ACTORS = {"api", "analysis", "worker", "evaluation"}


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in frame.splitlines():
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


def _assert_frontend_safe_sse_data(data: dict[str, object]) -> None:
    allowed_keys = PUBLIC_SSE_DATA_KEYS | OPTIONAL_PUBLIC_SSE_DATA_KEYS
    assert PUBLIC_SSE_DATA_KEYS <= set(data)
    assert set(data) <= allowed_keys
    assert not (set(data) & FORBIDDEN_SSE_DATA_KEYS)
    if "actor" in data:
        assert data["actor"] in SAFE_SSE_ACTORS
    serialized = json.dumps(data, sort_keys=True)
    for forbidden in FORBIDDEN_SSE_TEXT:
        assert forbidden not in serialized


async def _submission_with_statuses(
    session,
    *,
    agent_hash: str,
    statuses: tuple[tuple[str, str, str], ...],
) -> tuple[int, list[int]]:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="received",
        raw_status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()

    event_ids: list[int] = []
    for index, (to_status, actor, reason) in enumerate(statuses):
        kwargs = {"from_status": None} if index == 0 else {}
        event = await transition_submission_status(
            session,
            submission,
            to_status,
            actor=actor,
            reason=reason,
            metadata={
                "private_path": "/tmp/private-source.zip",
                "secret": "sk-test-secret",
                "raw_reason": reason,
                "source": "platform-terminal-bench",
                "token": "submission-token",
                "lease_owner": "worker-a",
                "broker_ref": "broker-ref-123",
            },
            **kwargs,
        )
        event_ids.append(event.id)
    await session.commit()
    return submission.id, event_ids


async def test_submission_events_streams_existing_backlog_in_sequence_order(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id, event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-backlog",
            statuses=(
                ("received", "api", "received"),
                ("upload_verified", "api", "artifact verified"),
                ("rate_limit_reserved", "api", "rate limit reserved"),
                ("analysis_queued", "analysis", "queued"),
                ("analysis_rejected", "worker", "blocking_analysis_rejected"),
            ),
        )

    response = await client.get(f"/submissions/{submission_id}/events")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == event_ids
    assert [event["data"]["sequence"] for event in events] == [1, 2, 3, 4, 5]
    assert [event["data"]["public_state"] for event in events] == [
        "received",
        "queued",
        "queued",
        "queued",
        "invalid",
    ]
    for event in events:
        assert event["event"] == "submission.status"
        _assert_frontend_safe_sse_data(event["data"])
    assert events[-1]["data"] == {
        "id": event_ids[-1],
        "sequence": 5,
        "submission_id": submission_id,
        "status": "invalid",
        "public_state": "invalid",
        "phase": "analysis_complete",
        "created_at": events[-1]["data"]["created_at"],
        "reason_code": "blocking_analysis_rejected",
        "actor": "worker",
    }
    serialized = json.dumps([event["data"] for event in events], sort_keys=True)
    for forbidden in FORBIDDEN_SSE_TEXT:
        assert forbidden not in serialized
    assert "reason_code" not in events[1]["data"]
    assert events[-1]["data"]["reason_code"] == "blocking_analysis_rejected"


async def test_submission_events_reconnect_replays_only_after_last_event_id(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id, event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-reconnect",
            statuses=(
                ("received", "api", "received"),
                ("upload_verified", "api", "artifact verified"),
                ("rate_limit_reserved", "api", "rate limit reserved"),
                ("analysis_queued", "analysis", "queued"),
                ("analysis_rejected", "worker", "blocking_analysis_rejected"),
            ),
        )

    response = await client.get(
        f"/submissions/{submission_id}/events",
        headers={"Last-Event-ID": str(event_ids[2])},
    )

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == event_ids[3:]
    assert [event["data"]["sequence"] for event in events] == [4, 5]
    for event in events:
        _assert_frontend_safe_sse_data(event["data"])


async def test_submission_events_unknown_or_stale_last_event_id_returns_replay_conflict(
    client,
    database_session,
):
    async with database_session() as session:
        other_submission_id, other_event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-other",
            statuses=(("received", "api", "received"), ("cancelled", "api", "cancelled")),
        )
        submission_id, event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-stale",
            statuses=(
                ("received", "api", "received"),
                ("upload_verified", "api", "artifact verified"),
                ("cancelled", "api", "cancelled"),
            ),
        )

    stale_response = await client.get(
        f"/submissions/{submission_id}/events",
        headers={"Last-Event-ID": str(event_ids[0] - 1)},
    )
    foreign_response = await client.get(
        f"/submissions/{submission_id}/events",
        headers={"Last-Event-ID": str(other_event_ids[-1])},
    )

    assert other_submission_id != submission_id
    assert stale_response.status_code == 409
    assert stale_response.json() == {
        "detail": "unknown Last-Event-ID",
        "replay_from": event_ids[0],
    }
    assert foreign_response.status_code == 409
    assert foreign_response.json() == {
        "detail": "unknown Last-Event-ID",
        "replay_from": event_ids[0],
    }


async def test_submission_events_terminal_completed_replays_full_history(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id, event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-completed",
            statuses=(
                ("received", "api", "received"),
                ("upload_verified", "api", "artifact verified"),
                ("rate_limit_reserved", "api", "rate limit reserved"),
                ("analysis_queued", "analysis", "queued"),
                ("ast_running", "worker", "ast started"),
                ("analysis_allowed", "worker", "analysis allowed"),
                ("waiting_miner_env", "worker", "waiting_miner_env"),
                ("tb_queued", "evaluation", "evaluation queued"),
                ("tb_running", "evaluation", "evaluation running"),
                ("tb_completed", "evaluation", "evaluation completed"),
            ),
        )

    response = await client.get(f"/submissions/{submission_id}/events")

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["id"] for event in events] == event_ids
    for event in events:
        _assert_frontend_safe_sse_data(event["data"])
    assert events[-1]["data"]["public_state"] == "valid"
    assert events[-1]["data"]["phase"] == "complete"


async def test_submission_events_latest_event_matches_polling_status(
    client,
    database_session,
):
    async with database_session() as session:
        submission_id, event_ids = await _submission_with_statuses(
            session,
            agent_hash="events-parity",
            statuses=(
                ("received", "api", "received"),
                ("upload_verified", "api", "artifact verified"),
                ("rate_limit_reserved", "api", "rate limit reserved"),
                ("analysis_queued", "analysis", "queued"),
                ("ast_running", "worker", "ast started"),
                ("analysis_allowed", "worker", "analysis allowed"),
                ("waiting_miner_env", "worker", "waiting_miner_env"),
                ("tb_queued", "evaluation", "evaluation queued"),
                ("tb_running", "evaluation", "evaluation running"),
                ("tb_failed_final", "evaluation", "evaluation failed"),
            ),
        )

    status_response = await client.get(f"/submissions/{submission_id}/status")
    events_response = await client.get(f"/submissions/{submission_id}/events")

    assert status_response.status_code == 200
    assert events_response.status_code == 200
    status_payload = status_response.json()
    latest_event = _parse_sse_events(events_response.text)[-1]["data"]
    assert status_payload["last_event_id"] == event_ids[-1]
    _assert_frontend_safe_sse_data(latest_event)
    assert latest_event["id"] == status_payload["last_event_id"]
    assert latest_event["sequence"] == status_payload["last_event_sequence"]
    assert latest_event["public_state"] == status_payload["public_state"]
    assert latest_event["phase"] == status_payload["phase"]
    assert "agent_hash" not in latest_event
    assert "effective_status" not in latest_event
    assert "analyzer" not in latest_event
    assert "evaluation" not in latest_event
    assert "progress" not in latest_event
