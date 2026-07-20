from __future__ import annotations

import base64
import hashlib
import io
import itertools
import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _routing import public_route_paths
from sqlalchemy import func, select, text

from agent_challenge import routes
from agent_challenge.analyzer.lifecycle import run_analysis_for_submission
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    LlmReviewOutcome,
    SubmitVerdictArgs,
    build_llm_verdict_row,
)
from agent_challenge.app import app
from agent_challenge.evaluation.benchmarks import benchmark_tasks_from_json
from agent_challenge.evaluation.reconciler import run_reconciler_once
from agent_challenge.evaluation.runner import (
    claim_next_evaluation_job_for_worker,
    enqueue_evaluation_job_for_submission,
)
from agent_challenge.evaluation.terminal_bench import create_terminal_bench_attempt
from agent_challenge.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    EvaluationAttempt,
    EvaluationJob,
    ExternalExecutionRef,
    LlmVerdict,
    PythonAstFeature,
    SimilarityMatch,
    SubmissionStatusEvent,
    TaskResult,
    TerminalBenchTrial,
)
from agent_challenge.sdk.auth import build_internal_auth_dependency
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.db import Database
from agent_challenge.security import SignedRequestAuth
from agent_challenge.weights import get_weights

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
REAL_HARBOR_ENV = "AGENT_CHALLENGE_RUN_REAL_HARBOR"
AGENT_SOURCE_COUNTER = itertools.count(1)
ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


@dataclass
class SignedAuthState:
    hotkey: str = "signed-miner-hotkey"
    nonce: str = "signed-nonce-1"


@dataclass
class OwnerAuthState:
    calls: int = 0


@pytest.fixture
def signed_submission_override() -> SignedAuthState:
    state = SignedAuthState()

    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey=state.hotkey,
            signature="test-signature",
            nonce=state.nonce,
            timestamp=NOW.isoformat(),
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield state
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


@pytest.fixture
def owner_auth_override() -> OwnerAuthState:
    state = OwnerAuthState()

    async def authenticate() -> SignedRequestAuth:
        state.calls += 1
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature=f"owner-signature-{state.calls}",
            nonce=f"owner-nonce-{state.calls}",
            timestamp=NOW.isoformat(),
            body_sha256=hashlib.sha256(f"owner-body-{state.calls}".encode()).hexdigest(),
            canonical_request=f"owner-request-{state.calls}",
        )

    app.dependency_overrides[routes.owner_signed_auth] = authenticate
    yield state
    app.dependency_overrides.pop(routes.owner_signed_auth, None)


class StaticReviewProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        tool_choice: str,
        timeout_seconds: float,
    ) -> None:
        raise AssertionError("network-backed reviewer provider must not be called in e2e tests")


class StaticReviewer:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def review(self, *, analysis_run_id, manifest, read_session, similarity_evidence):
        self.calls += 1
        verdict = SubmitVerdictArgs(
            verdict=self.verdict,
            confidence=0.93,
            rationale=f"mock {self.verdict}",
            evidence_paths=["agent.py"],
            similarity_assessment=json.dumps(list(similarity_evidence), sort_keys=True),
            policy_flags=[f"mock_{self.verdict}"],
        )
        transcript: dict[str, list[object]] = {
            "attempts": [],
            "file_reads": [],
            "provider_responses": [],
            "tool_calls": [],
        }
        row = build_llm_verdict_row(
            analysis_run_id=analysis_run_id,
            provider=StaticReviewProvider(),
            verdict=verdict,
            transcript=transcript,
            manifest=manifest,
            similarity_evidence=list(similarity_evidence),
        )
        return LlmReviewOutcome(verdict=verdict, llm_verdict_row=row, transcript=transcript)


async def test_signed_allow_lifecycle_recovers_terminal_bench_and_scores_weight(
    client,
    database_session,
    monkeypatch,
    signed_submission_override: SignedAuthState,
    tmp_path,
) -> None:
    _configure_master_terminal_bench(monkeypatch, tmp_path)
    signed_submission_override.hotkey = "allow-hotkey"
    signed_submission_override.nonce = "allow-nonce"
    archive_bytes = build_zip({"agent.py": "def solve(value):\n    return value + 1\n"})

    response = await client.post(
        "/submissions",
        json={
            "name": "allow-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )
    assert response.status_code == 201
    submission_id = response.json()["submission_id"]
    assert response.json()["status"] == "queued"

    reviewer = StaticReviewer("allow")
    async with database_session() as session:
        summary = await run_analysis_for_submission(
            session,
            submission_id,
            actor="analysis-worker",
            reviewer=reviewer,
        )
        await session.commit()
    assert summary.verdict == "allow"
    assert summary.evaluation_job_id is not None
    assert reviewer.calls == 1

    first_recovery, second_recovery = await _complete_queued_terminal_bench_via_reconciler(
        database_session,
        submission_id=submission_id,
        task_id="hello-world",
        score=1.0,
    )

    assert first_recovery.terminal_bench_finalized == 1
    assert second_recovery.terminal_bench_finalized == 0
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        counts = await _submission_counts(session, submission_id)
        event_statuses = (
            (
                await session.execute(
                    select(SubmissionStatusEvent.to_status)
                    .where(SubmissionStatusEvent.submission_id == submission_id)
                    .order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )
        ast_count = await session.scalar(select(func.count(PythonAstFeature.id)))
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))

    assert submission is not None
    assert submission.raw_status == "tb_completed"
    assert submission.effective_status == "valid"
    assert counts == {
        "jobs": 1,
        "attempts": 1,
        "trials": 1,
        "external_refs": 2,
        "task_results": 1,
        "tb_completed_events": 1,
    }
    assert event_statuses == [
        "received",
        "upload_verified",
        "rate_limit_reserved",
        "analysis_queued",
        "ast_running",
        "llm_running",
        "analysis_allowed",
        "waiting_miner_env",
        "tb_queued",
        "tb_running",
        "tb_completed",
    ]
    assert ast_count and ast_count > 0
    assert llm_count == 1

    status_response = await client.get(f"/submissions/{submission_id}/status")
    events_response = await client.get(f"/submissions/{submission_id}/events")
    assert status_response.status_code == 200
    assert events_response.status_code == 200
    status_payload = status_response.json()
    latest_event = parse_sse_events(events_response.text)[-1]["data"]
    assert status_payload["public_state"] == "valid"
    assert status_payload["phase"] == "complete"
    assert status_payload["progress"]["terminal_bench_trials"] == 1
    assert latest_event["public_state"] == status_payload["public_state"]
    assert latest_event["phase"] == status_payload["phase"]
    assert latest_event["id"] == status_payload["last_event_id"]
    assert await get_weights() == {"allow-hotkey": 1.0}


async def test_recovery_lifecycle_idempotently_finalizes_completed_fixture_with_terminal_sse(
    client,
    database_session,
    monkeypatch,
    signed_submission_override: SignedAuthState,
    tmp_path,
) -> None:
    _configure_master_terminal_bench(monkeypatch, tmp_path)
    signed_submission_override.hotkey = "recovery-hotkey"
    signed_submission_override.nonce = "recovery-nonce"
    submission_id = await _submit_and_allow(
        client,
        database_session,
        reviewer=StaticReviewer("allow"),
    )

    async with database_session() as session:
        job = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="dead-worker",
            lease_seconds=1,
        )
        assert job is not None
        assert job.submission_id == submission_id
        await session.refresh(job, attribute_names=["submission"])
        task = benchmark_tasks_from_json(job.selected_tasks_json)[0]
        plan = await create_terminal_bench_attempt(
            session,
            submission=job.submission,
            job=job,
            task=task,
            command=("bash", "-lc", "harbor run"),
            lease_owner="dead-worker",
            lease_seconds=1,
        )
        attempt = await session.get(EvaluationAttempt, plan.attempt_id)
        assert attempt is not None
        attempt.lease_expires_at = datetime.now(UTC) - timedelta(seconds=10)
        _write_trial(plan.job_dir / "trials" / "trial-one", "hello-world", 1.0)
        await session.commit()

    async with database_session() as session:
        first = await run_reconciler_once(session, lease_owner="recover-worker")
        await session.commit()
    async with database_session() as session:
        second = await run_reconciler_once(session, lease_owner="recover-worker")
        await session.commit()

    assert first.terminal_bench_finalized == 1
    assert second.terminal_bench_finalized == 0
    async with database_session() as session:
        counts = await _submission_counts(session, submission_id)
        event = await session.scalar(
            select(SubmissionStatusEvent)
            .where(SubmissionStatusEvent.submission_id == submission_id)
            .order_by(SubmissionStatusEvent.sequence.desc())
            .limit(1)
        )
    assert counts == {
        "jobs": 1,
        "attempts": 1,
        "trials": 1,
        "external_refs": 2,
        "task_results": 1,
        "tb_completed_events": 1,
    }
    assert event is not None
    assert event.to_status == "tb_completed"

    events_response = await client.get(f"/submissions/{submission_id}/events")
    assert events_response.status_code == 200
    latest_event = parse_sse_events(events_response.text)[-1]
    assert latest_event["event"] == "submission.status"
    assert latest_event["data"]["public_state"] == "valid"
    assert latest_event["data"]["phase"] == "complete"
    assert await get_weights() == {"recovery-hotkey": 1.0}


async def test_admin_escalation_allow_reject_and_rerun_preserve_evidence_and_weight_gate(
    client,
    database_session,
    monkeypatch,
    signed_submission_override: SignedAuthState,
    owner_auth_override: OwnerAuthState,
    tmp_path,
) -> None:
    _configure_master_terminal_bench(monkeypatch, tmp_path)
    signed_submission_override.hotkey = "admin-allow-hotkey"
    signed_submission_override.nonce = "admin-allow-nonce"
    allow_submission_id = await _submit_and_analyze(
        client,
        database_session,
        reviewer=StaticReviewer("escalate"),
    )
    allow_evidence = await _analysis_evidence(database_session, allow_submission_id)

    allow_response = await client.post(
        f"/owner/submissions/{allow_submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "review cleared"},
    )
    assert allow_response.status_code == 200
    assert allow_response.json()["status"] == "waiting_miner_env"
    assert allow_response.json()["job_id"] is None
    assert await get_weights() == {}

    async with database_session() as session:
        allow_job_count = await session.scalar(
            select(func.count(EvaluationJob.id)).where(
                EvaluationJob.submission_id == allow_submission_id
            )
        )
        allow_attempt_count = await session.scalar(
            select(func.count(EvaluationAttempt.id)).where(
                EvaluationAttempt.submission_id == allow_submission_id
            )
        )
    assert allow_job_count == 0
    assert allow_attempt_count == 0
    assert await _analysis_evidence(database_session, allow_submission_id) == allow_evidence

    async with database_session() as session:
        submission = await session.get(AgentSubmission, allow_submission_id)
        assert submission is not None
        job = await enqueue_evaluation_job_for_submission(
            session,
            submission,
            confirmed_miner_env=True,
        )
        assert job is not None
        await session.commit()

    await _complete_queued_terminal_bench_via_reconciler(
        database_session,
        submission_id=allow_submission_id,
        task_id="hello-world",
        score=1.0,
    )
    assert await get_weights() == {"admin-allow-hotkey": 1.0}

    signed_submission_override.hotkey = "admin-rerun-hotkey"
    signed_submission_override.nonce = "admin-rerun-nonce"
    rerun_submission_id = await _submit_and_analyze(
        client,
        database_session,
        reviewer=StaticReviewer("escalate"),
    )
    rerun_evidence = await _analysis_evidence(database_session, rerun_submission_id)
    rerun_response = await client.post(
        f"/owner/submissions/{rerun_submission_id}/admin-escalation",
        json={"decision": "admin_request_rerun", "reason": "fresh pass required"},
    )
    assert rerun_response.status_code == 200
    assert rerun_response.json()["status"] == "analysis_queued"
    assert rerun_response.json()["job_id"] is None
    assert await _analysis_evidence(database_session, rerun_submission_id) == rerun_evidence

    signed_submission_override.hotkey = "admin-reject-hotkey"
    signed_submission_override.nonce = "admin-reject-nonce"
    reject_submission_id = await _submit_and_analyze(
        client,
        database_session,
        reviewer=StaticReviewer("escalate"),
    )
    reject_evidence = await _analysis_evidence(database_session, reject_submission_id)
    reject_response = await client.post(
        f"/owner/submissions/{reject_submission_id}/admin-escalation",
        json={"decision": "admin_reject", "reason": "policy violation confirmed"},
    )
    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "analysis_rejected"
    assert reject_response.json()["job_id"] is None
    async with database_session() as session:
        reject_job_count = await session.scalar(
            select(func.count(EvaluationJob.id)).where(
                EvaluationJob.submission_id == reject_submission_id
            )
        )
        reject_attempt_count = await session.scalar(
            select(func.count(EvaluationAttempt.id)).where(
                EvaluationAttempt.submission_id == reject_submission_id
            )
        )
        decisions = (
            (await session.execute(select(AdminReviewDecision).order_by(AdminReviewDecision.id)))
            .scalars()
            .all()
        )
    assert reject_job_count == 0
    assert reject_attempt_count == 0
    assert await _analysis_evidence(database_session, reject_submission_id) == reject_evidence
    assert [decision.decision for decision in decisions] == [
        "pending_analysis_review",
        "admin_allow",
        "pending_analysis_review",
        "admin_request_rerun",
        "pending_analysis_review",
        "admin_reject",
    ]
    assert owner_auth_override.calls == 3
    assert await get_weights() == {"admin-allow-hotkey": 1.0}


async def test_platform_contract_regressions_are_covered_locally(
    client,
    internal_headers,
    monkeypatch,
    tmp_path,
) -> None:
    # Signed upload persistence and replay rejection are exercised by this E2E suite plus
    # tests/test_submissions_signed.py and tests/test_signed_auth.py in the required command.
    # Full BASE proxy enforcement lives in source_challenges; this repo verifies the
    # challenge-side route annotations and internal bearer+slug contract locally.
    # BASE proxy blocklist semantics live in source_challenges; this repo exposes the
    # challenge-side contract by decorating only public routes and leaving owner/internal routes
    # undiscoverable to the proxy.
    public_paths = public_route_paths(app)
    assert "/submissions" in public_paths
    assert "/submissions/{submission_id}/events" in public_paths
    assert "/internal/v1/get_weights" not in public_paths
    assert "/owner/submissions/{submission_id}/admin-escalation" not in public_paths

    missing_auth = await client.get("/internal/v1/get_weights")
    wrong_slug = await client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer test-token", "X-Base-Challenge-Slug": "wrong"},
    )
    wrong_token = await client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer wrong", "X-Base-Challenge-Slug": "agent-challenge"},
    )
    valid = await client.get("/internal/v1/get_weights", headers=internal_headers)
    assert missing_auth.status_code == 403
    assert wrong_slug.status_code == 403
    assert wrong_token.status_code == 401
    assert valid.status_code == 200
    assert valid.json()["challenge_slug"] == "agent-challenge"

    shared_token_file = tmp_path / "shared-token"
    shared_token_file.write_text("file-token\n", encoding="utf-8")
    auth_dependency = build_internal_auth_dependency(
        ChallengeSettings(shared_token=None, shared_token_file=str(shared_token_file))
    )
    await auth_dependency(
        authorization="Bearer file-token",
        x_base_challenge_slug="agent-challenge",
    )

    database_path = tmp_path / "challenge-contract.sqlite3"
    database = Database(f"sqlite+aiosqlite:///{database_path}")
    try:
        await database.init()
        async with database.engine.connect() as connection:
            table_names = {
                row[0]
                for row in (
                    await connection.execute(
                        text("select name from sqlite_master where type='table'")
                    )
                )
            }
    finally:
        await database.close()
    assert {
        "agent_submissions",
        "submission_status_events",
        "evaluation_attempts",
        "terminal_bench_trials",
        "external_execution_refs",
    }.issubset(table_names)

    broker_token_file = tmp_path / "broker-token"
    broker_token_file.write_text("broker-token-from-file\n", encoding="utf-8")
    for settings_path in (
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.benchmarks.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.docker_backend", "broker")
        monkeypatch.setattr(f"{settings_path}.docker_enabled", True)
        monkeypatch.setattr(f"{settings_path}.docker_broker_url", "https://platform-broker.test")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token", None)
        monkeypatch.setattr(f"{settings_path}.docker_broker_token_file", str(broker_token_file))
    from agent_challenge.evaluation.runner import validate_terminal_bench_broker_readiness

    validate_terminal_bench_broker_readiness()
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.docker_broker_token_file", None)
    with pytest.raises(RuntimeError) as exc_info:
        validate_terminal_bench_broker_readiness()
    assert "CHALLENGE_DOCKER_BROKER_TOKEN" in str(exc_info.value)
    assert "broker-token-from-file" not in str(exc_info.value)


@pytest.mark.skipif(
    os.getenv(REAL_HARBOR_ENV) != "1",
    reason=f"set {REAL_HARBOR_ENV}=1 to run a live Harbor smoke outside the default suite",
)
async def test_optional_real_harbor_smoke_is_explicitly_gated() -> None:
    assert os.getenv(REAL_HARBOR_ENV) == "1"


async def _submit_and_allow(client, database_session, *, reviewer: StaticReviewer) -> int:
    submission_id = await _submit_and_analyze(client, database_session, reviewer=reviewer)
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        assert submission.raw_status == "tb_queued"
    return submission_id


async def _submit_and_analyze(client, database_session, *, reviewer: StaticReviewer) -> int:
    source_id = next(AGENT_SOURCE_COUNTER)
    response = await client.post(
        "/submissions",
        json={
            "name": f"agent-{source_id}",
            "artifact_zip_base64": base64.b64encode(
                build_zip({"agent.py": f"def solve():\n    return {source_id}\n"})
            ).decode("ascii"),
        },
    )
    assert response.status_code == 201
    submission_id = response.json()["submission_id"]
    async with database_session() as session:
        await run_analysis_for_submission(
            session,
            submission_id,
            actor="analysis-worker",
            reviewer=reviewer,
        )
        await session.commit()
    return submission_id


async def _complete_queued_terminal_bench_via_reconciler(
    database_session,
    *,
    submission_id: int,
    task_id: str,
    score: float,
):
    async with database_session() as session:
        job = await claim_next_evaluation_job_for_worker(
            session,
            lease_owner="dead-worker",
            lease_seconds=1,
        )
        assert job is not None
        assert job.submission_id == submission_id
        await session.refresh(job, attribute_names=["submission"])
        tasks = benchmark_tasks_from_json(job.selected_tasks_json)
        assert [task.task_id for task in tasks] == [task_id]
        plan = await create_terminal_bench_attempt(
            session,
            submission=job.submission,
            job=job,
            task=tasks[0],
            command=("bash", "-lc", "harbor run"),
            lease_owner="dead-worker",
            lease_seconds=1,
        )
        _write_trial(plan.job_dir / "trials" / "trial-one", task_id, score)
        await session.commit()

    async with database_session() as session:
        first = await run_reconciler_once(session, lease_owner="recover-worker")
        await session.commit()
    async with database_session() as session:
        second = await run_reconciler_once(session, lease_owner="recover-worker")
        await session.commit()
    return first, second


async def _submission_counts(session, submission_id: int) -> dict[str, int]:
    job_ids = (
        (
            await session.execute(
                select(EvaluationJob.id).where(EvaluationJob.submission_id == submission_id)
            )
        )
        .scalars()
        .all()
    )
    attempt_ids = (
        (
            await session.execute(
                select(EvaluationAttempt.id).where(EvaluationAttempt.submission_id == submission_id)
            )
        )
        .scalars()
        .all()
    )
    return {
        "jobs": len(job_ids),
        "attempts": len(attempt_ids),
        "trials": await _count_for_ids(
            session,
            TerminalBenchTrial.evaluation_attempt_id,
            attempt_ids,
        ),
        "external_refs": await _count_for_ids(
            session,
            ExternalExecutionRef.evaluation_attempt_id,
            attempt_ids,
        ),
        "task_results": await _count_for_ids(session, TaskResult.job_id, job_ids),
        "tb_completed_events": int(
            await session.scalar(
                select(func.count(SubmissionStatusEvent.id)).where(
                    SubmissionStatusEvent.submission_id == submission_id,
                    SubmissionStatusEvent.to_status == "tb_completed",
                )
            )
            or 0
        ),
    }


async def _count_for_ids(session, column, ids: list[int]) -> int:
    if not ids:
        return 0
    return int(await session.scalar(select(func.count()).where(column.in_(ids))) or 0)


async def _analysis_evidence(database_session, submission_id: int) -> dict[str, object]:
    async with database_session() as session:
        analysis = await session.scalar(
            select(AnalysisRun).where(AnalysisRun.submission_id == submission_id)
        )
        assert analysis is not None
        llm = await session.scalar(
            select(LlmVerdict).where(LlmVerdict.analysis_run_id == analysis.id)
        )
        matches = (
            (
                await session.execute(
                    select(SimilarityMatch).where(SimilarityMatch.analysis_run_id == analysis.id)
                )
            )
            .scalars()
            .all()
        )
    assert llm is not None
    return {
        "analysis_id": analysis.id,
        "analysis_status": analysis.status,
        "analysis_verdict": analysis.verdict,
        "analysis_report_json": analysis.report_json,
        "llm_verdict": llm.verdict,
        "llm_raw_request_json": llm.raw_request_json,
        "llm_raw_response_json": llm.raw_response_json,
        "match_count": len(matches),
    }


def _configure_master_terminal_bench(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for settings_path in (
        "agent_challenge.api.routes.settings",
        "agent_challenge.analyzer.lifecycle.settings",
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.reconciler.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.validator_role", "master")
        monkeypatch.setattr(f"{settings_path}.artifact_root", str(tmp_path / "artifacts"))
        monkeypatch.setattr(f"{settings_path}.benchmark_backend", "terminal_bench")
        monkeypatch.setattr(f"{settings_path}.terminal_bench_task_ids", ("hello-world",))
        monkeypatch.setattr(f"{settings_path}.evaluation_task_count", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_concurrency", 1)
        monkeypatch.setattr(
            f"{settings_path}.harbor_runner_image",
            "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        )


def build_zip(files: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_files = {"agent.py": ENTRYPOINT_SOURCE, **files}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_files.items():
            if filename == "agent.py":
                contents = agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


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


def parse_sse_events(text_body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for frame in text_body.strip().split("\n\n"):
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
