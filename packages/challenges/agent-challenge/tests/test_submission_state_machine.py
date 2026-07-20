from __future__ import annotations

import json

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from agent_challenge import submissions
from agent_challenge.models import AgentSubmission, SubmissionStatusEvent
from agent_challenge.submissions import state_machine
from agent_challenge.submissions.state_machine import (
    InvalidSubmissionStatusTransition,
    public_status_for,
    transition_submission_status,
)

assert submissions is not None


async def _submission(session, *, raw_status: str = "received") -> AgentSubmission:
    submission = AgentSubmission(
        miner_hotkey="miner-state-machine",
        name="state-machine-agent",
        agent_hash=f"state-machine-{raw_status}",
        artifact_uri="/tmp/state-machine-agent.zip",
        status=public_status_for(raw_status),
        raw_status=raw_status,
        effective_status=public_status_for(raw_status),
    )
    session.add(submission)
    await session.flush()
    return submission


async def _events(session, submission: AgentSubmission) -> list[SubmissionStatusEvent]:
    return (
        (
            await session.execute(
                select(SubmissionStatusEvent)
                .where(SubmissionStatusEvent.submission_id == submission.id)
                .order_by(SubmissionStatusEvent.sequence)
            )
        )
        .scalars()
        .all()
    )


async def test_valid_submission_state_sequence_appends_ordered_events(database_session):
    async with database_session() as session:
        submission = await _submission(session)

        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="submitted",
            metadata={"source": "test"},
            from_status=None,
        )
        await transition_submission_status(
            session,
            submission,
            "upload_verified",
            actor="artifact",
            reason="zip verified",
            metadata={"sha256": "abc"},
        )
        await transition_submission_status(
            session,
            submission,
            "rate_limit_reserved",
            actor="rate-limit",
            reason="reservation created",
            metadata={"reservation_key": "r1"},
        )
        await transition_submission_status(
            session,
            submission,
            "analysis_queued",
            actor="analysis",
            reason="ready for analysis",
            metadata={"queue": "analysis"},
        )

        events = await _events(session, submission)

    assert [event.sequence for event in events] == [1, 2, 3, 4]
    assert [event.from_status for event in events] == [
        None,
        "received",
        "upload_verified",
        "rate_limit_reserved",
    ]
    assert [event.to_status for event in events] == [
        "received",
        "upload_verified",
        "rate_limit_reserved",
        "analysis_queued",
    ]
    assert events[0].actor == "api"
    assert events[0].reason == "submitted"
    assert json.loads(events[0].metadata_json) == {"source": "test"}
    assert submission.raw_status == "analysis_queued"
    assert submission.status == "queued"
    assert submission.effective_status == "queued"


async def test_invalid_transition_raises_and_appends_no_event(database_session):
    async with database_session() as session:
        submission = await _submission(session, raw_status="analysis_rejected")

        with pytest.raises(InvalidSubmissionStatusTransition) as exc_info:
            await transition_submission_status(
                session,
                submission,
                "tb_queued",
                actor="analysis",
                reason="should not run rejected submissions",
                metadata={"queue": "tb"},
            )

        events = await _events(session, submission)

    assert (
        str(exc_info.value)
        == "invalid submission status transition: 'analysis_rejected' -> 'tb_queued'"
    )
    assert events == []
    assert submission.raw_status == "analysis_rejected"
    assert submission.status == "invalid"
    assert submission.effective_status == "invalid"


async def test_valid_legacy_submission_can_record_durable_tb_completion(database_session):
    async with database_session() as session:
        submission = await _submission(session, raw_status="valid")

        event = await transition_submission_status(
            session,
            submission,
            "tb_completed",
            actor="evaluation",
            reason="evaluation_job_completed",
            metadata={"job_id": "legacy-job", "score": 0.75},
        )

    assert event.from_status == "valid"
    assert event.to_status == "tb_completed"
    assert submission.raw_status == "tb_completed"
    assert submission.status == "valid"
    assert submission.effective_status == "valid"


async def test_waiting_miner_env_allows_later_terminal_bench_queue(database_session):
    async with database_session() as session:
        submission = await _submission(session, raw_status="analysis_allowed")

        await transition_submission_status(
            session,
            submission,
            "waiting_miner_env",
            actor="analysis",
            reason="waiting_miner_env",
        )
        await transition_submission_status(
            session,
            submission,
            "tb_queued",
            actor="evaluation",
            reason="evaluation_job_queued",
        )

    assert submission.raw_status == "tb_queued"
    assert submission.effective_status == "evaluation queued"


async def test_internal_terminal_states_allow_revalidation_requeue(database_session):
    async with database_session() as session:
        completed = await _submission(session, raw_status="tb_completed")

        completed_event = await transition_submission_status(
            session,
            completed,
            "tb_queued",
            actor="evaluation",
            reason="evaluation_job_queued",
        )
        completed_events = await _events(session, completed)

        failed_final = await _submission(session, raw_status="tb_failed_final")

        failed_final_event = await transition_submission_status(
            session,
            failed_final,
            "tb_queued",
            actor="evaluation",
            reason="evaluation_job_queued",
        )
        failed_final_events = await _events(session, failed_final)

    assert completed_event.from_status == "tb_completed"
    assert completed_event.to_status == "tb_queued"
    assert [event.to_status for event in completed_events] == ["tb_queued"]
    assert completed.raw_status == "tb_queued"
    assert completed.effective_status == "evaluation queued"

    assert failed_final_event.from_status == "tb_failed_final"
    assert failed_final_event.to_status == "tb_queued"
    assert [event.to_status for event in failed_final_events] == ["tb_queued"]
    assert failed_final.raw_status == "tb_queued"
    assert failed_final.effective_status == "evaluation queued"


async def test_llm_standby_can_retry_analysis_queue(database_session):
    async with database_session() as session:
        submission = await _submission(session, raw_status="llm_running")

        await transition_submission_status(
            session,
            submission,
            "llm_standby",
            actor="analysis",
            reason="llm_provider_unavailable",
        )
        await transition_submission_status(
            session,
            submission,
            "analysis_queued",
            actor="analysis",
            reason="blocking_analysis_lease_expired",
        )

        events = await _events(session, submission)

    assert [event.to_status for event in events] == ["llm_standby", "analysis_queued"]
    assert submission.raw_status == "analysis_queued"
    assert submission.status == "queued"
    assert submission.effective_status == "queued"


async def test_transition_retries_on_sequence_collision(database_session, monkeypatch):
    async with database_session() as session:
        submission = await _submission(session)
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="submitted",
            from_status=None,
        )

        real_next_sequence = state_machine._next_sequence
        calls = {"n": 0}

        async def flaky_next_sequence(sess, submission_id):
            calls["n"] += 1
            if calls["n"] == 1:
                # Mimic a concurrent writer that grabbed the same MAX(seq)+1=1.
                return 1
            return await real_next_sequence(sess, submission_id)

        monkeypatch.setattr(state_machine, "_next_sequence", flaky_next_sequence)

        event = await transition_submission_status(
            session,
            submission,
            "upload_verified",
            actor="artifact",
            reason="zip verified",
        )

        events = await _events(session, submission)

    assert calls["n"] >= 2
    assert event.sequence == 2
    assert [e.sequence for e in events] == [1, 2]
    assert submission.raw_status == "upload_verified"
    assert submission.status == "queued"
    assert submission.effective_status == "queued"


async def test_collision_retry_preserves_other_pending_changes(database_session, monkeypatch):
    async with database_session() as session:
        submission = await _submission(session)
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="submitted",
            from_status=None,
        )

        other = await _submission(session, raw_status="valid")
        submission.name = "renamed-before-savepoint"

        real_next_sequence = state_machine._next_sequence
        calls = {"n": 0}

        async def flaky_next_sequence(sess, submission_id):
            calls["n"] += 1
            if calls["n"] == 1:
                return 1
            return await real_next_sequence(sess, submission_id)

        monkeypatch.setattr(state_machine, "_next_sequence", flaky_next_sequence)

        event = await transition_submission_status(
            session,
            submission,
            "upload_verified",
            actor="artifact",
            reason="zip verified",
        )

        events = await _events(session, submission)
        await session.refresh(submission)
        await session.refresh(other)

    assert event.sequence == 2
    assert [e.sequence for e in events] == [1, 2]
    assert submission.name == "renamed-before-savepoint"
    assert other.id is not None
    assert other.raw_status == "valid"


async def test_transition_raises_after_retry_exhaustion(database_session, monkeypatch):
    async with database_session() as session:
        submission = await _submission(session)
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="submitted",
            from_status=None,
        )

        calls = {"n": 0}

        async def always_colliding(sess, submission_id):
            calls["n"] += 1
            return 1

        monkeypatch.setattr(state_machine, "_next_sequence", always_colliding)

        with pytest.raises(IntegrityError):
            await transition_submission_status(
                session,
                submission,
                "upload_verified",
                actor="artifact",
                reason="zip verified",
            )

        events = await _events(session, submission)

    assert calls["n"] == state_machine.MAX_SEQUENCE_ALLOCATION_RETRIES
    assert [e.sequence for e in events] == [1]
    assert submission.raw_status == "received"


def test_public_status_mapping_hides_noisy_internal_states() -> None:
    assert public_status_for("ast_running") == "AST review"
    assert public_status_for("llm_running") == "LLM review"
    assert public_status_for("llm_standby") == "LLM standby"
    assert public_status_for("waiting_miner_env") == "Waiting environments"
    assert public_status_for("tb_queued") == "evaluation queued"
    assert public_status_for("tb_running") == "evaluating"
    assert public_status_for("tb_failed_retryable") == "evaluating"
    assert public_status_for("analysis_rejected") == "invalid"
    assert public_status_for("tb_completed") == "valid"
    assert public_status_for("ast_running") != "ast_running"
    assert public_status_for("tb_running") != "tb_running"
