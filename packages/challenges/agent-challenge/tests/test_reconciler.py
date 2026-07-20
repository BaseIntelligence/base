from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from agent_challenge.evaluation import reconciler as reconciler_module
from agent_challenge.evaluation.benchmarks import BenchmarkTask, benchmark_tasks_to_json
from agent_challenge.evaluation.reconciler import run_reconciler_once
from agent_challenge.evaluation.terminal_bench import (
    TERMINAL_BENCH_BASE_SDK_PROVIDER,
    TERMINAL_BENCH_EVALUATOR,
    create_terminal_bench_attempt,
)
from agent_challenge.models import (
    AgentSubmission,
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    ExternalExecutionRef,
    SubmissionStatusEvent,
    TaskLogEvent,
    TaskResult,
    TerminalBenchTrial,
)
from agent_challenge.submissions.state_machine import (
    ensure_submission_status,
    record_initial_status,
    transition_submission_status,
)


async def test_reconciler_requeues_expired_analysis_lease(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    expired_at = datetime.now(UTC) - timedelta(seconds=10)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-analysis-reconciler",
            name="analysis-reconciler-agent",
            agent_hash="analysis-reconciler-hash",
            artifact_uri=str(tmp_path / "agent.zip"),
            raw_status="llm_running",
            status="analysis_running",
            effective_status="analysis_running",
        )
        session.add(submission)
        await session.flush()
        session.add(
            AnalysisRun(
                submission_id=submission.id,
                analyzer_name="blocking_analyzer",
                analyzer_version="test",
                status="running",
                lease_owner="dead-analyzer",
                lease_expires_at=expired_at,
                heartbeat_at=expired_at,
                started_at=expired_at,
            )
        )
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="reconciler-a")
        await session.commit()

    assert summary.analysis_requeued == 1
    async with database_session() as session:
        run = await session.scalar(select(AnalysisRun))
        submission = await session.scalar(select(AgentSubmission))
        event = await session.scalar(select(SubmissionStatusEvent))

    assert run is not None
    assert run.status == "expired_reclaimed"
    assert submission is not None
    assert submission.raw_status == "analysis_queued"
    assert event is not None
    assert event.reason == "blocking_analysis_lease_expired"


async def test_reconciler_gate_noop_on_sqlite(database_session) -> None:
    async with database_session() as session:
        acquired = await reconciler_module._acquire_reconciler_gate(session)

    assert acquired is True


async def test_reconciler_skips_when_gate_not_acquired(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")

    async def _gate_closed(_session) -> bool:
        return False

    monkeypatch.setattr(reconciler_module, "_acquire_reconciler_gate", _gate_closed)

    expired_at = datetime.now(UTC) - timedelta(seconds=10)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="miner-gate-reconciler",
            name="gate-reconciler-agent",
            agent_hash="gate-reconciler-hash",
            artifact_uri=str(tmp_path / "agent.zip"),
            raw_status="llm_running",
            status="analysis_running",
            effective_status="analysis_running",
        )
        session.add(submission)
        await session.flush()
        session.add(
            AnalysisRun(
                submission_id=submission.id,
                analyzer_name="blocking_analyzer",
                analyzer_version="test",
                status="running",
                lease_owner="dead-analyzer",
                lease_expires_at=expired_at,
                heartbeat_at=expired_at,
                started_at=expired_at,
            )
        )
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="reconciler-gated")
        await session.commit()

    assert summary.analysis_requeued == 0
    async with database_session() as session:
        run = await session.scalar(select(AnalysisRun))
        submission = await session.scalar(select(AgentSubmission))
        event = await session.scalar(select(SubmissionStatusEvent))

    assert run is not None
    assert run.status == "running"
    assert submission is not None
    assert submission.raw_status == "llm_running"
    assert event is None


def _parse_sse_events(text: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in frame.splitlines():
            name, value = line.split(": ", 1)
            fields[name] = value
        events.append({"id": int(fields["id"]), "data": json.loads(fields["data"])})
    return events


async def test_reconciler_finalizes_completed_harbor_job_dir_after_worker_restart(
    client,
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, task=task)
        submission_id = submission.id
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_finalized == 1
    async with database_session() as session:
        attempt = (await session.execute(select(EvaluationAttempt))).scalar_one()
        trials = (await session.execute(select(TerminalBenchTrial))).scalars().all()
        refs = (await session.execute(select(ExternalExecutionRef))).scalars().all()
        job = (await session.execute(select(EvaluationJob))).scalar_one()
        task_results = (await session.execute(select(TaskResult))).scalars().all()
        task_log_events = (
            (
                await session.execute(
                    select(TaskLogEvent)
                    .where(TaskLogEvent.submission_id == job.submission_id)
                    .order_by(TaskLogEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    assert attempt.status == "completed"
    assert attempt.score == 1.0
    assert [(trial.task_id, trial.status, trial.score, trial.is_final) for trial in trials] == [
        ("hello-world", "completed", 1.0, 1),
    ]
    assert len(refs) == 2
    assert job.status == "completed"
    assert job.score == 1.0
    assert [(result.task_id, result.status, result.score) for result in task_results] == [
        ("hello-world", "completed", 1.0),
    ]
    assert [event.event_type for event in task_log_events] == [
        "task.progress",
        "task.status",
        "task.progress",
        "task.completed",
    ]
    assert task_log_events[0].task_id == "hello-world"
    assert task_log_events[1].task_id == "hello-world"
    assert task_log_events[1].status == "completed"
    assert task_log_events[1].message == "task hello-world completed"
    assert json.loads(task_log_events[1].metadata_json) == {
        "attempt": 1,
        "benchmark": "terminal_bench",
        "phase": "completed",
    }
    assert task_log_events[-1].status == "completed"
    assert events[-1].to_status == "tb_completed"
    replay = await client.get(f"/submissions/{submission_id}/task-events?limit=10")
    assert replay.status_code == 200
    assert [event["event_type"] for event in replay.json()["events"]] == [
        "task.progress",
        "task.status",
        "task.progress",
        "task.completed",
    ]
    replay_status_event = replay.json()["events"][1]
    assert replay_status_event["task_id"] == "hello-world"
    assert replay_status_event["status"] == "completed"
    assert replay_status_event["message"] == "task hello-world completed"
    assert replay_status_event["metadata"] == {
        "attempt": 1,
        "benchmark": "terminal_bench",
        "phase": "completed",
    }


async def test_reconciler_finalizes_completed_base_sdk_job_dir_provider_neutrally(
    client,
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="platform-sdk-completed",
        )
        submission_id = submission.id
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
            provider=TERMINAL_BENCH_BASE_SDK_PROVIDER,
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_finalized == 1
    async with database_session() as session:
        attempt = (await session.execute(select(EvaluationAttempt))).scalar_one()
        refs = (await session.execute(select(ExternalExecutionRef))).scalars().all()
        job = (await session.execute(select(EvaluationJob))).scalar_one()
        task_results = (await session.execute(select(TaskResult))).scalars().all()
        trials = (await session.execute(select(TerminalBenchTrial))).scalars().all()

    attempt_refs = [ref for ref in refs if ref.terminal_bench_trial_id is None]
    trial_refs = [ref for ref in refs if ref.terminal_bench_trial_id is not None]
    assert attempt.status == "completed"
    assert attempt.score == 1.0
    assert json.loads(attempt.metadata_json)["execution_provider"] == (
        TERMINAL_BENCH_BASE_SDK_PROVIDER
    )
    assert job.status == "completed"
    assert [(result.task_id, result.status, result.score) for result in task_results] == [
        ("hello-world", "completed", 1.0),
    ]
    assert [(trial.task_id, trial.status, trial.score, trial.is_final) for trial in trials] == [
        ("hello-world", "completed", 1.0, 1),
    ]
    assert len(attempt_refs) == 1
    assert len(trial_refs) == 1
    assert attempt_refs[0].provider == TERMINAL_BENCH_BASE_SDK_PROVIDER
    assert attempt_refs[0].status == "completed"
    assert trial_refs[0].provider == "terminal_bench"
    assert trial_refs[0].status == "completed"

    status_response = await client.get(f"/submissions/{submission_id}/status")
    assert status_response.status_code == 200
    public_payload = json.dumps(status_response.json(), sort_keys=True)
    assert TERMINAL_BENCH_BASE_SDK_PROVIDER not in public_payload
    assert plan.job_name not in public_payload
    assert str(plan.job_dir) not in public_payload
    assert "worker-a" not in public_payload


async def test_reconciler_is_idempotent_for_completed_job_dir(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(session, tmp_path, task=task, agent_hash="idem")
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        await session.commit()

    async with database_session() as session:
        first = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()
    async with database_session() as session:
        second = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert first.terminal_bench_finalized == 1
    assert second.terminal_bench_finalized == 0
    async with database_session() as session:
        trials = (await session.execute(select(TerminalBenchTrial))).scalars().all()
        refs = (await session.execute(select(ExternalExecutionRef))).scalars().all()
        events = (await session.execute(select(SubmissionStatusEvent))).scalars().all()
        task_results = (await session.execute(select(TaskResult))).scalars().all()

    assert len(trials) == 1
    assert len(refs) == 2
    assert len(task_results) == 1
    assert [event.to_status for event in events].count("tb_completed") == 1


async def test_reconciler_status_survives_api_restart_through_db_polling_and_sse(
    client,
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="api-restart",
        )
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        submission_id = submission.id
        await session.commit()

    async with database_session() as session:
        await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    status_response = await client.get(f"/submissions/{submission_id}/status")
    events_response = await client.get(f"/submissions/{submission_id}/events")

    assert status_response.status_code == 200
    assert events_response.status_code == 200
    status_payload = status_response.json()
    latest_event = _parse_sse_events(events_response.text)[-1]["data"]
    assert status_payload["public_state"] == "valid"
    assert status_payload["phase"] == "complete"
    assert status_payload["terminal_bench"]["total_trials"] == 1
    assert status_payload["progress"]["terminal_bench_trials"] == 1
    assert latest_event["public_state"] == "valid"
    assert latest_event["id"] == status_payload["last_event_id"]
    serialized = json.dumps(status_payload, sort_keys=True) + events_response.text
    assert "worker-a" not in serialized
    assert "platform-terminal-bench" not in serialized
    assert "base-terminal-bench" not in serialized


async def test_reconciler_requeues_missing_job_dir_once_then_final_at_retry_cap(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    expired = datetime.now(UTC) - timedelta(seconds=30)
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="missing-dir",
        )
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        attempt = await session.get(EvaluationAttempt, plan.attempt_id)
        assert attempt is not None
        attempt.lease_expires_at = expired
        shutil.rmtree(plan.job_dir)
        await session.commit()

    async with database_session() as session:
        retry_summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert retry_summary.terminal_bench_retryable == 1
    async with database_session() as session:
        attempt = (await session.execute(select(EvaluationAttempt))).scalar_one()
        job = (await session.execute(select(EvaluationJob))).scalar_one()
        submission = (await session.execute(select(AgentSubmission))).scalar_one()

    assert attempt.status == "failed_retryable"
    assert attempt.error == "terminal_bench_job_dir_missing"
    assert job.status == "queued"
    assert submission.raw_status == "tb_queued"

    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="missing-dir-final",
            attempt_count=3,
        )
        await _mark_submission_tb_running(session, submission)
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=1
        )
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=2
        )
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        attempt = await session.get(EvaluationAttempt, plan.attempt_id)
        assert attempt is not None
        assert attempt.attempt_number == 3
        attempt.lease_expires_at = expired
        shutil.rmtree(plan.job_dir)
        await session.commit()

    async with database_session() as session:
        final_summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert final_summary.terminal_bench_final_failed == 1
    async with database_session() as session:
        final_attempt = (
            await session.execute(
                select(EvaluationAttempt).where(
                    EvaluationAttempt.submission.has(agent_hash="missing-dir-final"),
                    EvaluationAttempt.status == "failed",
                )
            )
        ).scalar_one()
        final_submission = await session.get(AgentSubmission, final_attempt.submission_id)

    assert final_attempt.status == "failed"
    assert final_attempt.error == "terminal_bench_job_dir_missing"
    assert final_submission is not None
    assert final_submission.raw_status == "tb_failed_final"


async def test_reconciler_handles_missing_broker_reference_without_success_duplicate(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    expired = datetime.now(UTC) - timedelta(seconds=30)
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="missing-ref",
        )
        await _mark_submission_tb_running(session, submission)
        plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        ref = (
            await session.execute(
                select(ExternalExecutionRef).where(
                    ExternalExecutionRef.evaluation_attempt_id == plan.attempt_id,
                    ExternalExecutionRef.provider == "own_runner",
                )
            )
        ).scalar_one()
        await session.delete(ref)
        attempt = await session.get(EvaluationAttempt, plan.attempt_id)
        assert attempt is not None
        attempt.lease_expires_at = expired
        await session.commit()

    async with database_session() as session:
        first = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()
    async with database_session() as session:
        second = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert first.terminal_bench_retryable == 1
    assert second.terminal_bench_retryable == 0
    async with database_session() as session:
        attempt = (await session.execute(select(EvaluationAttempt))).scalar_one()
        trials = (await session.execute(select(TerminalBenchTrial))).scalars().all()
        task_results = (await session.execute(select(TaskResult))).scalars().all()
        submission = (await session.execute(select(AgentSubmission))).scalar_one()

    assert attempt.status == "failed_retryable"
    assert attempt.error == "terminal_bench_broker_ref_missing"
    assert trials == []
    assert task_results == []
    assert submission.raw_status == "tb_queued"


async def test_reconciler_requeues_expired_base_sdk_attempt_then_final_at_retry_cap(
    client,
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    expired = datetime.now(UTC) - timedelta(seconds=30)
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="platform-sdk-stale-retry",
        )
        retry_submission_id = submission.id
        await _mark_submission_tb_running(session, submission)
        retry_plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
            provider=TERMINAL_BENCH_BASE_SDK_PROVIDER,
        )
        attempt = await session.get(EvaluationAttempt, retry_plan.attempt_id)
        assert attempt is not None
        attempt.lease_expires_at = expired
        await session.commit()

    async with database_session() as session:
        retry_summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert retry_summary.stale_terminal_bench_attempts == 1
    assert retry_summary.terminal_bench_retryable == 1
    async with database_session() as session:
        retry_attempt = (await session.execute(select(EvaluationAttempt))).scalar_one()
        retry_job = (await session.execute(select(EvaluationJob))).scalar_one()
        retry_submission = (await session.execute(select(AgentSubmission))).scalar_one()
        retry_ref = (await session.execute(select(ExternalExecutionRef))).scalar_one()

    assert retry_attempt.status == "failed_retryable"
    assert retry_attempt.error == "terminal_bench_lease_expired"
    assert retry_job.status == "queued"
    assert retry_submission.raw_status == "tb_queued"
    assert retry_ref.provider == TERMINAL_BENCH_BASE_SDK_PROVIDER
    assert retry_ref.status == "failed_retryable"

    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="platform-sdk-stale-final",
            attempt_count=3,
        )
        final_submission_id = submission.id
        await _mark_submission_tb_running(session, submission)
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=1
        )
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=2
        )
        final_plan = await create_terminal_bench_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="worker-a",
            provider=TERMINAL_BENCH_BASE_SDK_PROVIDER,
        )
        attempt = await session.get(EvaluationAttempt, final_plan.attempt_id)
        assert attempt is not None
        assert attempt.attempt_number == 3
        attempt.lease_expires_at = expired
        await session.commit()

    async with database_session() as session:
        final_summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert final_summary.stale_terminal_bench_attempts == 1
    assert final_summary.terminal_bench_final_failed == 1
    async with database_session() as session:
        final_attempt = (
            await session.execute(
                select(EvaluationAttempt).where(
                    EvaluationAttempt.submission.has(agent_hash="platform-sdk-stale-final"),
                    EvaluationAttempt.status == "failed",
                )
            )
        ).scalar_one()
        final_job = await session.get(EvaluationJob, final_attempt.job_id)
        final_submission = await session.get(AgentSubmission, final_attempt.submission_id)
        final_ref = (
            await session.execute(
                select(ExternalExecutionRef).where(
                    ExternalExecutionRef.evaluation_attempt_id == final_attempt.id
                )
            )
        ).scalar_one()

    assert final_attempt.status == "failed"
    assert final_attempt.error == "terminal_bench_lease_expired"
    assert final_job is not None
    assert final_job.status == "error"
    assert final_submission is not None
    assert final_submission.raw_status == "tb_failed_final"
    assert final_ref.provider == TERMINAL_BENCH_BASE_SDK_PROVIDER
    assert final_ref.status == "failed"

    retry_status = await client.get(f"/submissions/{retry_submission_id}/status")
    final_status = await client.get(f"/submissions/{final_submission_id}/status")
    assert retry_status.status_code == 200
    assert final_status.status_code == 200
    public_payload = json.dumps(
        {"retry": retry_status.json(), "final": final_status.json()},
        sort_keys=True,
    )
    assert TERMINAL_BENCH_BASE_SDK_PROVIDER not in public_payload
    assert retry_plan.job_name not in public_payload
    assert final_plan.job_name not in public_payload
    assert str(retry_plan.job_dir) not in public_payload
    assert str(final_plan.job_dir) not in public_payload
    assert "worker-a" not in public_payload


async def test_reconciler_supersedes_legacy_null_task_attempts_at_retry_cap(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    expired = datetime.now(UTC) - timedelta(seconds=30)
    task = _terminal_bench_task()
    async with database_session() as session:
        submission, job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="legacy-null-final",
            attempt_count=3,
        )
        await _mark_submission_tb_running(session, submission)
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=1, task_id=None
        )
        await _insert_prior_tb_attempt(
            session, submission=submission, job=job, task=task, attempt_number=2, task_id=None
        )
        third = await _insert_prior_tb_attempt(
            session,
            submission=submission,
            job=job,
            task=task,
            attempt_number=3,
            task_id=None,
            status="running",
        )
        third.lease_owner = "worker-a"
        third.lease_expires_at = expired
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_final_failed == 1
    assert summary.terminal_bench_retryable == 0
    async with database_session() as session:
        final_attempt = (
            await session.execute(
                select(EvaluationAttempt).where(
                    EvaluationAttempt.submission.has(agent_hash="legacy-null-final"),
                    EvaluationAttempt.status == "failed",
                )
            )
        ).scalar_one()
        final_job = await session.get(EvaluationJob, final_attempt.job_id)
        final_submission = await session.get(AgentSubmission, final_attempt.submission_id)
        prior_statuses = (
            (
                await session.execute(
                    select(EvaluationAttempt.status).where(
                        EvaluationAttempt.submission.has(agent_hash="legacy-null-final"),
                        EvaluationAttempt.attempt_number < 3,
                    )
                )
            )
            .scalars()
            .all()
        )

    assert final_attempt.task_id is None
    assert final_job is not None
    assert final_job.status == "error"
    assert final_submission is not None
    assert final_submission.raw_status == "tb_failed_final"
    assert set(prior_statuses) == {"failed_retryable"}


async def test_reconciler_ignores_retryable_attempt_when_submission_terminal(
    database_session,
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.evaluation.reconciler.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.terminal_bench.settings.artifact_root",
        str(tmp_path),
    )
    task = _terminal_bench_task()
    expected_public_statuses = {"tb_failed_final": "error", "tb_completed": "valid"}
    expected_job_statuses = {"tb_failed_final": "error", "tb_completed": "completed"}

    async with database_session() as session:
        for terminal_status, public_status in expected_public_statuses.items():
            submission, job = await _submission_and_job(
                session,
                tmp_path,
                task=task,
                agent_hash=f"terminal-{terminal_status}",
            )
            await _mark_submission_tb_running(session, submission)
            plan = await create_terminal_bench_attempt(
                session,
                submission=submission,
                job=job,
                task=task,
                command=("bash", "-lc", "harbor run"),
                lease_owner="worker-a",
                provider=TERMINAL_BENCH_BASE_SDK_PROVIDER,
            )
            attempt = await session.get(EvaluationAttempt, plan.attempt_id)
            assert attempt is not None
            attempt.status = "failed_retryable"
            attempt.error = "terminal_bench_lease_expired"
            await transition_submission_status(
                session,
                submission,
                terminal_status,
                actor="worker-a",
                reason="terminal_fixture",
            )
            job.status = expected_job_statuses[terminal_status]
            job.error = f"preserve-{terminal_status}-error"
            job.last_error = f"preserve-{terminal_status}-last-error"
            job.finished_at = datetime.now(UTC)
            assert submission.status == public_status
        await session.commit()

    async with database_session() as session:
        summary = await run_reconciler_once(session, lease_owner="recover-a")
        await session.commit()

    assert summary.terminal_bench_retryable == 0
    async with database_session() as session:
        for terminal_status, public_status in expected_public_statuses.items():
            row = (
                await session.execute(
                    select(AgentSubmission, EvaluationJob)
                    .join(EvaluationJob, EvaluationJob.submission_id == AgentSubmission.id)
                    .where(AgentSubmission.agent_hash == f"terminal-{terminal_status}")
                )
            ).one()
            submission, job = row

            assert submission.raw_status == terminal_status
            assert submission.status == public_status
            assert submission.effective_status == public_status
            assert job.status == expected_job_statuses[terminal_status]
            assert job.error == f"preserve-{terminal_status}-error"
            assert job.last_error == f"preserve-{terminal_status}-last-error"
            assert job.lease_owner == "worker-a"
            assert job.finished_at is not None


async def _submission_and_job(
    session,
    tmp_path: Path,
    *,
    task: BenchmarkTask,
    agent_hash: str = "recovered",
    attempt_count: int = 1,
) -> tuple[AgentSubmission, EvaluationJob]:
    agent_dir = tmp_path / f"agent-{agent_hash}"
    agent_dir.mkdir(exist_ok=True)
    submission = AgentSubmission(
        miner_hotkey=f"miner-{agent_hash}",
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=str(agent_dir),
        raw_status="received",
        status="received",
        effective_status="received",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="running",
        selected_tasks_json=benchmark_tasks_to_json([task]),
        total_tasks=1,
        attempt_count=attempt_count,
        lease_owner="worker-a",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        heartbeat_at=datetime.now(UTC),
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission, job


async def _mark_submission_tb_running(session, submission: AgentSubmission) -> None:
    await record_initial_status(session, submission, actor="api", reason="submission_received")
    await transition_submission_status(
        session,
        submission,
        "upload_verified",
        actor="api",
        reason="submission_upload_verified",
    )
    await transition_submission_status(
        session,
        submission,
        "rate_limit_reserved",
        actor="api",
        reason="submission_rate_limit_reserved",
    )
    await transition_submission_status(
        session,
        submission,
        "analysis_queued",
        actor="api",
        reason="blocking_analysis_queued",
    )
    await transition_submission_status(
        session,
        submission,
        "ast_running",
        actor="analysis",
        reason="blocking_analysis_claimed",
    )
    await transition_submission_status(
        session,
        submission,
        "analysis_allowed",
        actor="analysis",
        reason="blocking_analysis_allowed",
    )
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
    await transition_submission_status(
        session,
        submission,
        "tb_running",
        actor="worker-a",
        reason="evaluation_job_claimed",
    )


_UNSET_TASK_ID = object()


async def _insert_prior_tb_attempt(
    session,
    *,
    submission: AgentSubmission,
    job: EvaluationJob,
    task: BenchmarkTask,
    attempt_number: int,
    status: str = "failed_retryable",
    task_id=_UNSET_TASK_ID,
) -> EvaluationAttempt:
    now = datetime.now(UTC)
    resolved_task_id = task.task_id if task_id is _UNSET_TASK_ID else task_id
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=attempt_number,
        task_id=resolved_task_id,
        evaluator_name=TERMINAL_BENCH_EVALUATOR,
        status=status,
        score=0.0,
        error="terminal_bench_lease_expired",
        metadata_json=json.dumps({"task_id": task.task_id}, sort_keys=True),
        started_at=now - timedelta(seconds=100 - attempt_number),
        finished_at=now,
    )
    session.add(attempt)
    await session.flush()
    return attempt


def _terminal_bench_task() -> BenchmarkTask:
    return BenchmarkTask(
        task_id="hello-world",
        docker_image="ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        benchmark="terminal_bench",
        metadata={"task_id": "hello-world"},
    )


def _write_trial(trial_dir: Path, task_id: str, score: float) -> None:
    trial_dir.mkdir(parents=True)
    (trial_dir / "stdout.txt").write_text("stdout", encoding="utf-8")
    (trial_dir / "stderr.txt").write_text("stderr", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "trial_name": trial_dir.name,
                "status": "completed",
                "score": score,
                "stdout_path": "stdout.txt",
                "stderr_path": "stderr.txt",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


async def test_reconciler_does_not_revert_requeued_submission(
    database_session,
    tmp_path,
) -> None:
    task = _terminal_bench_task()
    # Cycle 1: drive the submission to tb_failed_final via a final failed attempt
    # tied to its current (soon-to-be-old) evaluation job.
    async with database_session() as session:
        submission, old_job = await _submission_and_job(
            session,
            tmp_path,
            task=task,
            agent_hash="reeval-guard",
        )
        await _mark_submission_tb_running(session, submission)
        await _insert_prior_tb_attempt(
            session,
            submission=submission,
            job=old_job,
            task=task,
            attempt_number=3,
            status="failed",
        )
        await session.commit()

    async with database_session() as session:
        await run_reconciler_once(session, lease_owner="rec-cycle1")
        await session.commit()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission.raw_status == "tb_failed_final"

    # Re-queue: a fresh job supersedes the old cycle and the submission returns to
    # tb_queued (mirrors create_evaluation_job's tb_failed_final -> tb_queued path).
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        new_job = EvaluationJob(
            job_id="job-reeval-guard-new",
            submission_id=submission.id,
            status="queued",
            selected_tasks_json=benchmark_tasks_to_json([task]),
            total_tasks=1,
        )
        session.add(new_job)
        await session.flush()
        submission.latest_evaluation_job_id = new_job.id
        await ensure_submission_status(
            session,
            submission,
            "tb_queued",
            actor="evaluation",
            reason="evaluation_job_queued",
            metadata={"job_id": new_job.job_id},
        )
        await session.commit()
        new_job_id = new_job.id

    # The reconciler must NOT apply the prior cycle's terminal failure to the
    # freshly re-queued submission; without the guard it walks tb_queued ->
    # tb_running -> tb_failed_retryable -> tb_failed_final and deadlocks re-eval.
    async with database_session() as session:
        await run_reconciler_once(session, lease_owner="rec-cycle2")
        await session.commit()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission.raw_status == "tb_queued"
        new_job = await session.get(EvaluationJob, new_job_id)
        assert new_job.status == "queued"
