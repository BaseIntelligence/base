from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent_challenge.db import Base
from agent_challenge.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    AnalyzerReport,
    EvaluationAttempt,
    EvaluationJob,
    ExternalExecutionRef,
    LlmVerdict,
    OwnerActionAudit,
    PythonAstFeature,
    RateLimitReservation,
    RequestNonce,
    RulesBundle,
    SimilarityMatch,
    SubmissionArtifact,
    SubmissionStatusEvent,
    TaskLogEvent,
    TerminalBenchTrial,
)


@pytest.fixture
async def model_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


async def test_create_all_master_schema():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        table_names = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
    await engine.dispose()

    assert {
        "agent_submissions",
        "submission_families",
        "evaluation_jobs",
        "task_results",
        "task_log_events",
        "request_nonces",
        "owner_action_audit",
        "rules_bundles",
        "analyzer_reports",
        "submission_artifacts",
        "submission_status_events",
        "rate_limit_reservations",
        "analysis_runs",
        "python_ast_features",
        "similarity_matches",
        "llm_verdicts",
        "evaluation_attempts",
        "terminal_bench_trials",
        "external_execution_refs",
        "admin_review_decisions",
        "review_sessions",
        "review_rules_snapshots",
        "review_assignments",
        "review_nonces",
        "review_operator_approvals",
    } <= table_names


async def test_request_nonce_unique_per_hotkey_and_nonce(model_session):
    expires_at = datetime.now(UTC) + timedelta(minutes=5)
    model_session.add_all(
        [
            RequestNonce(hotkey="hotkey-a", nonce="nonce-1", expires_at=expires_at),
            RequestNonce(hotkey="hotkey-b", nonce="nonce-1", expires_at=expires_at),
        ]
    )
    await model_session.commit()

    model_session.add(RequestNonce(hotkey="hotkey-a", nonce="nonce-1", expires_at=expires_at))
    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_owner_audit_append_only(model_session):
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey",
        name="legacy-name",
        agent_hash="hash-a",
        artifact_uri="/tmp/legacy-artifact",
        agent_name="agent-a",
        zip_sha256="a" * 64,
        zip_size_bytes=123,
        artifact_path="/tmp/artifacts/a.zip",
        raw_status="suspicious",
        effective_status="suspicious",
    )
    model_session.add(submission)
    await model_session.flush()

    first = OwnerActionAudit(
        submission_id=submission.id,
        owner_hotkey="owner-hotkey",
        action="override",
        reason="manual review",
        request_hash="1" * 64,
        nonce="nonce-1",
        signature="signature-1",
        request_timestamp="2026-05-22T00:00:00Z",
        before_effective_status="suspicious",
        after_effective_status="overridden_valid",
    )
    submission.effective_status = "overridden_valid"
    second = OwnerActionAudit(
        submission_id=submission.id,
        owner_hotkey="owner-hotkey",
        action="suspicious",
        reason="new evidence",
        request_hash="2" * 64,
        nonce="nonce-2",
        signature="signature-2",
        request_timestamp="2026-05-22T00:01:00Z",
        before_effective_status="overridden_valid",
        after_effective_status="suspicious",
    )
    submission.effective_status = "suspicious"
    model_session.add_all([first, second])
    await model_session.commit()

    rows = list(
        (
            await model_session.execute(
                select(OwnerActionAudit).order_by(OwnerActionAudit.created_at, OwnerActionAudit.id)
            )
        )
        .scalars()
        .all()
    )

    assert len(rows) == 2
    assert rows[0].request_hash == "1" * 64
    assert rows[0].after_effective_status == "overridden_valid"
    assert rows[1].request_hash == "2" * 64
    assert rows[0].id != rows[1].id


async def test_job_links_rules_bundle_and_analyzer_report_metadata(model_session):
    rules = RulesBundle(
        rules_version="f" * 64,
        files_json='[".rules/acceptance.md"]',
        policy_text="accept submissions that satisfy the policy",
    )
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey",
        name="legacy-name",
        agent_hash="hash-b",
        artifact_uri="/tmp/legacy-artifact",
        agent_name="agent-b",
        zip_sha256="b" * 64,
        zip_size_bytes=456,
        artifact_path="/tmp/artifacts/b.zip",
        raw_status="queued",
        effective_status="queued",
    )
    model_session.add_all([rules, submission])
    await model_session.flush()

    job = EvaluationJob(
        job_id="job-b",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json="[]",
        triggered_by_hotkey="owner-hotkey",
        trigger_reason="revalidate",
        rules_version=rules.rules_version,
        image_digest="sha256:" + "c" * 64,
        container_config_json='{"network":"none"}',
        verdict="valid",
        reason_codes_json='["rules_passed"]',
        logs_ref="logs/job-b.txt",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )
    model_session.add(job)
    await model_session.flush()
    submission.latest_evaluation_job_id = job.id
    report = AnalyzerReport(
        job_id=job.id,
        rules_version=rules.rules_version,
        verdict="valid",
        reason_codes_json='["rules_passed"]',
        report_json='{"overall_verdict":"valid"}',
        logs_ref="logs/job-b.txt",
    )
    model_session.add(report)
    await model_session.commit()
    await model_session.refresh(submission, attribute_names=["latest_evaluation_job"])
    await model_session.refresh(job, attribute_names=["analyzer_reports", "submission"])

    assert submission.latest_evaluation_job == job
    assert job.submission == submission
    assert job.rules_version == rules.rules_version
    assert job.analyzer_reports[0].verdict == "valid"
    assert job.analyzer_reports[0].reason_codes_json == '["rules_passed"]'


async def test_durable_submission_evaluation_models_round_trip(model_session):
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey-c",
        name="legacy-name-c",
        agent_hash="hash-c",
        artifact_uri="/tmp/legacy-artifact-c",
        raw_status="received",
        effective_status="received",
    )
    model_session.add(submission)
    await model_session.flush()

    artifact = SubmissionArtifact(
        submission_id=submission.id,
        artifact_kind="source_zip",
        uri="file:///tmp/artifacts/c.zip",
        sha256="c" * 64,
        size_bytes=789,
        metadata_json='{"content_type":"application/zip"}',
    )
    job = EvaluationJob(
        job_id="job-c",
        submission_id=submission.id,
        status="running",
        selected_tasks_json='["task-c"]',
    )
    model_session.add_all([artifact, job])
    await model_session.flush()

    analysis_run = AnalysisRun(
        submission_id=submission.id,
        job_id=job.id,
        analyzer_name="python-static",
        status="completed",
        verdict="valid",
        input_artifact_id=artifact.id,
        report_json='{"files":1}',
    )
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=1,
        evaluator_name="terminal-bench",
        status="running",
        started_at=now,
    )
    model_session.add_all(
        [
            analysis_run,
            attempt,
            SubmissionStatusEvent(
                submission_id=submission.id,
                sequence=1,
                to_status="received",
                actor="api",
            ),
            RateLimitReservation(
                hotkey="miner-hotkey-c",
                limit_key="submit",
                window_start=now,
                window_seconds=60,
                reservation_key="request-c",
                expires_at=now + timedelta(minutes=1),
            ),
            AdminReviewDecision(
                submission_id=submission.id,
                reviewer_hotkey="owner-hotkey",
                decision="approve",
                after_effective_status="valid",
            ),
        ]
    )
    await model_session.flush()

    trial = TerminalBenchTrial(
        evaluation_attempt_id=attempt.id,
        task_id="task-c",
        trial_name="trial-c",
        trial_number=1,
        job_dir="/tmp/jobs/job-c",
        job_name="job-c-task-c-1",
        status="completed",
        score=1.0,
        is_final=1,
        raw_artifacts_json='{"result":"artifact.json"}',
    )
    model_session.add(trial)
    await model_session.flush()

    model_session.add_all(
        [
            PythonAstFeature(
                analysis_run_id=analysis_run.id,
                file_path="agent.py",
                feature_key="module:agent",
                feature_type="module",
                feature_value="agent",
            ),
            SimilarityMatch(
                analysis_run_id=analysis_run.id,
                source_submission_id=submission.id,
                matched_artifact_uri="corpus://baseline",
                match_kind="ast",
                score=0.25,
            ),
            LlmVerdict(
                analysis_run_id=analysis_run.id,
                reviewer_name="reviewer-a",
                model_name="model-a",
                verdict="valid",
            ),
            ExternalExecutionRef(
                evaluation_attempt_id=attempt.id,
                terminal_bench_trial_id=trial.id,
                provider="terminal-bench",
                external_id="external-c",
                status="completed",
                job_dir="/tmp/jobs/job-c",
                job_name="job-c-task-c-1",
                raw_payload_json='{"exit_code":0}',
            ),
        ]
    )
    await model_session.commit()
    await model_session.refresh(
        submission,
        attribute_names=[
            "artifacts",
            "status_events",
            "analysis_runs",
            "evaluation_attempts",
            "admin_review_decisions",
        ],
    )
    await model_session.refresh(
        analysis_run,
        attribute_names=["python_ast_features", "similarity_matches", "llm_verdicts"],
    )
    await model_session.refresh(
        attempt,
        attribute_names=["terminal_bench_trials", "external_execution_refs"],
    )

    assert submission.artifacts[0].metadata_json == '{"content_type":"application/zip"}'
    assert submission.status_events[0].to_status == "received"
    assert analysis_run.python_ast_features[0].feature_key == "module:agent"
    assert analysis_run.similarity_matches[0].matched_artifact_uri == "corpus://baseline"
    assert analysis_run.llm_verdicts[0].verdict == "valid"
    assert attempt.terminal_bench_trials[0].job_dir == "/tmp/jobs/job-c"
    assert attempt.external_execution_refs[0].external_id == "external-c"
    assert submission.admin_review_decisions[0].decision == "approve"
    assert await model_session.scalar(select(func.count(RateLimitReservation.id))) == 1


async def test_status_events_and_rate_limit_reservations_are_unique(model_session):
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey-d",
        name="legacy-name-d",
        agent_hash="hash-d",
        artifact_uri="/tmp/legacy-artifact-d",
    )
    model_session.add(submission)
    await model_session.flush()
    submission_id = submission.id
    model_session.add_all(
        [
            SubmissionStatusEvent(
                submission_id=submission_id,
                sequence=1,
                to_status="received",
            ),
            TaskLogEvent(
                submission_id=submission_id,
                sequence=1,
                event_type="task.log",
                task_id="task-d",
                message="first log",
            ),
            RateLimitReservation(
                hotkey="miner-hotkey-d",
                limit_key="submit",
                window_start=now,
                window_seconds=60,
                reservation_key="request-1",
                expires_at=now + timedelta(minutes=1),
            ),
        ]
    )
    await model_session.commit()

    model_session.add(
        SubmissionStatusEvent(
            submission_id=submission_id,
            sequence=1,
            to_status="queued",
        )
    )
    with pytest.raises(IntegrityError):
        await model_session.commit()
    await model_session.rollback()

    model_session.add(
        TaskLogEvent(
            submission_id=submission_id,
            sequence=1,
            event_type="task.log",
            task_id="task-d",
            message="duplicate log sequence",
        )
    )
    with pytest.raises(IntegrityError):
        await model_session.commit()
    await model_session.rollback()

    model_session.add(
        RateLimitReservation(
            hotkey="miner-hotkey-d",
            limit_key="submit",
            window_start=now,
            window_seconds=60,
            reservation_key="request-1",
            expires_at=now + timedelta(minutes=1),
        )
    )
    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_durable_schema_models_store_recovery_metadata(model_session):
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey-c",
        name="agent-c",
        agent_hash="hash-c",
        artifact_uri="/tmp/agent-c.zip",
    )
    model_session.add(submission)
    await model_session.flush()

    artifact = SubmissionArtifact(
        submission_id=submission.id,
        artifact_kind="source_zip",
        uri="/tmp/agent-c.zip",
        sha256="c" * 64,
        size_bytes=123,
        metadata_json='{"agent":"c"}',
    )
    job = EvaluationJob(
        job_id="job-c",
        submission_id=submission.id,
        status="running",
        selected_tasks_json='["task-c"]',
    )
    model_session.add_all([artifact, job])
    await model_session.flush()

    analysis_run = AnalysisRun(
        submission_id=submission.id,
        job_id=job.id,
        analyzer_name="python_ast",
        analyzer_version="1",
        status="completed",
        verdict="valid",
        input_artifact_id=artifact.id,
        report_json='{"files":1}',
    )
    attempt = EvaluationAttempt(
        submission_id=submission.id,
        job_id=job.id,
        attempt_number=1,
        evaluator_name="terminal_bench",
        status="completed",
        score=0.5,
    )
    model_session.add_all(
        [
            analysis_run,
            attempt,
            SubmissionStatusEvent(
                submission_id=submission.id,
                sequence=1,
                to_status="received",
                actor="api",
            ),
            RateLimitReservation(
                hotkey="miner-hotkey-c",
                limit_key="submit",
                window_start=now,
                window_seconds=60,
                reservation_key="request-c",
                expires_at=now + timedelta(minutes=1),
            ),
            AdminReviewDecision(
                submission_id=submission.id,
                reviewer_hotkey="owner-hotkey",
                decision="accepted",
            ),
        ]
    )
    await model_session.flush()

    trial = TerminalBenchTrial(
        evaluation_attempt_id=attempt.id,
        task_id="task-c",
        trial_name="trial-c",
        trial_number=1,
        job_dir="/tmp/jobs/job-c",
        job_name="job-c-task-c-1",
        status="completed",
        score=0.5,
        is_final=1,
        raw_artifacts_json='{"result":"results.json"}',
    )
    model_session.add(trial)
    await model_session.flush()

    model_session.add_all(
        [
            PythonAstFeature(
                analysis_run_id=analysis_run.id,
                file_path="agent.py",
                feature_key="agent.py:imports",
                feature_type="imports",
                feature_value='["os"]',
            ),
            SimilarityMatch(
                analysis_run_id=analysis_run.id,
                source_submission_id=submission.id,
                matched_artifact_uri="corpus://agent.py",
                match_kind="ast",
                score=0.25,
            ),
            LlmVerdict(
                analysis_run_id=analysis_run.id,
                reviewer_name="llm-reviewer",
                model_name="review-model",
                verdict="valid",
            ),
            ExternalExecutionRef(
                evaluation_attempt_id=attempt.id,
                terminal_bench_trial_id=trial.id,
                provider="terminal_bench",
                external_id="external-job-c",
                status="completed",
                job_dir=trial.job_dir,
                job_name=trial.job_name,
                raw_payload_json='{"state":"done"}',
            ),
        ]
    )
    await model_session.commit()
    await model_session.refresh(
        submission,
        attribute_names=[
            "artifacts",
            "status_events",
            "analysis_runs",
            "evaluation_attempts",
            "admin_review_decisions",
        ],
    )
    await model_session.refresh(
        analysis_run,
        attribute_names=["python_ast_features", "similarity_matches", "llm_verdicts"],
    )
    await model_session.refresh(
        attempt,
        attribute_names=["terminal_bench_trials", "external_execution_refs"],
    )

    assert submission.artifacts[0].metadata_json == '{"agent":"c"}'
    assert submission.status_events[0].sequence == 1
    assert analysis_run.python_ast_features[0].feature_key == "agent.py:imports"
    assert analysis_run.similarity_matches[0].matched_artifact_uri == "corpus://agent.py"
    assert analysis_run.llm_verdicts[0].verdict == "valid"
    assert attempt.terminal_bench_trials[0].job_dir == "/tmp/jobs/job-c"
    assert attempt.external_execution_refs[0].external_id == "external-job-c"
    assert submission.admin_review_decisions[0].decision == "accepted"
    assert await model_session.scalar(select(func.count(RateLimitReservation.id))) == 1


async def test_status_events_and_rate_limit_reservations_have_window_constraints(model_session):
    now = datetime.now(UTC)
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey-d",
        name="agent-d",
        agent_hash="hash-d",
        artifact_uri="/tmp/agent-d.zip",
    )
    model_session.add(submission)
    await model_session.flush()
    model_session.add(
        SubmissionStatusEvent(
            submission_id=submission.id,
            sequence=1,
            to_status="received",
        )
    )
    await model_session.commit()

    model_session.add(
        SubmissionStatusEvent(
            submission_id=submission.id,
            sequence=1,
            to_status="queued",
        )
    )
    with pytest.raises(IntegrityError):
        await model_session.commit()
    await model_session.rollback()

    model_session.add_all(
        [
            RateLimitReservation(
                hotkey="miner-hotkey-d",
                limit_key="submit",
                window_start=now,
                window_seconds=60,
                reservation_key="request-d",
            ),
            RateLimitReservation(
                hotkey="miner-hotkey-d",
                limit_key="submit",
                window_start=now,
                window_seconds=60,
                reservation_key="request-d",
            ),
        ]
    )
    with pytest.raises(IntegrityError):
        await model_session.commit()
