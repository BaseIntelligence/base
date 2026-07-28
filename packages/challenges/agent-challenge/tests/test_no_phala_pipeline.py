"""NO_PHALA full offline pipeline: analysis → benchmark → score → weight push.

Covers the master host path when attestation flags are off (required by
NO_PHALA contradiction). Does not touch attested gates.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from base.challenge_sdk.roles import Role, activate_role
from base.challenge_sdk.schemas import RawWeightPushRequest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.analyzer.lifecycle import run_next_analysis
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    LlmReviewOutcome,
    SubmitVerdictArgs,
    build_llm_verdict_row,
)
from agent_challenge.app import app
from agent_challenge.evaluation.no_phala import (
    ATTESTATION_STATUS_UNATTESTED,
    EXECUTION_MODE_NO_PHALA_HOST,
    NO_PHALA_ENV,
)
from agent_challenge.evaluation.raw_weight_push import RawWeightPushClient
from agent_challenge.evaluation.runner import create_evaluation_job, run_evaluation_job
from agent_challenge.evaluation.weights import get_weights
from agent_challenge.models import (
    AgentSubmission,
    AnalysisRun,
    EvaluationJob,
    LlmVerdict,
    PythonAstFeature,
    SubmissionStatusEvent,
)
from agent_challenge.sdk.db import Database
from agent_challenge.sdk.executors import DockerRunResult
from agent_challenge.security import SignedRequestAuth
from agent_challenge.swe_forge import SweForgeTask

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
WINNER_HOTKEY = "5CwinnerHK1"
ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


class StaticReviewProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def complete(self, **kwargs: Any) -> None:
        raise AssertionError("network LLM must not be called")


class StaticReviewer:
    def __init__(self, verdict: str = "allow") -> None:
        self.verdict = verdict
        self.calls = 0

    def review(self, *, analysis_run_id, manifest, read_session, similarity_evidence):
        self.calls += 1
        verdict = SubmitVerdictArgs(
            verdict=self.verdict,
            confidence=0.91,
            rationale=f"mock {self.verdict}",
            evidence_paths=["agent.py"],
            similarity_assessment="[]",
            policy_flags=[f"mock_{self.verdict}"],
        )
        row = build_llm_verdict_row(
            analysis_run_id=analysis_run_id,
            provider=StaticReviewProvider(),
            verdict=verdict,
            transcript={
                "attempts": [],
                "file_reads": [],
                "provider_responses": [],
                "tool_calls": [],
            },
            manifest=manifest,
            similarity_evidence=list(similarity_evidence),
        )
        return LlmReviewOutcome(verdict=verdict, llm_verdict_row=row, transcript={})


class FakeExecutor:
    def run(self, spec, timeout_seconds: int) -> DockerRunResult:
        return DockerRunResult(
            container_name="fake",
            stdout=f"ran {spec.labels.get('base.task', 'task')}",
            stderr="",
            returncode=0,
        )


class ValidReport:
    rules_version = "rules-test"
    overall_verdict = "valid"
    reason_codes = ["rules_passed"]

    def to_dict(self) -> dict[str, object]:
        return {
            "rules_version": self.rules_version,
            "overall_verdict": self.overall_verdict,
            "reason_codes": self.reason_codes,
        }


@pytest.fixture
def signed_submission_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey=WINNER_HOTKEY,
            signature="test-signature",
            nonce="test-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


@pytest.fixture(autouse=True)
def _no_phala_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NO_PHALA_ENV, raising=False)
    monkeypatch.delenv("CHALLENGE_NO_PHALA", raising=False)
    monkeypatch.setenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", "false")
    monkeypatch.setenv("CHALLENGE_ATTESTED_REVIEW_ENABLED", "false")


def _enable_no_phala(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NO_PHALA_ENV, "true")
    monkeypatch.setenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", "false")
    monkeypatch.setenv("CHALLENGE_ATTESTED_REVIEW_ENABLED", "false")


def _agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


def build_zip(files: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_files = {"agent.py": ENTRYPOINT_SOURCE, **files}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_files.items():
            if filename == "agent.py":
                contents = _agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


async def submit_agent(client, files: dict[str, str | bytes]):
    archive_bytes = build_zip(files)
    return await client.post(
        "/submissions",
        json={
            "name": "no-phala-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )


def configure_master(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.attested_review_enabled", False)
    monkeypatch.setattr("agent_challenge.api.routes.settings.phala_attestation_enabled", False)
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])


def _wire_offline_eval(monkeypatch: pytest.MonkeyPatch, *, task_count: int = 2) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [
            SweForgeTask(
                task_id=f"task-{i}",
                docker_image=f"baseintelligence/swe-forge:task-{i}",
            )
            for i in range(task_count)
        ],
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "swe_forge",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.evaluation_task_count", task_count
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_concurrency", 1)
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.run_rules_analyzer",
        lambda _workspace, *, reviewer=None: ValidReport(),
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.evaluation_task_count",
        task_count,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.phala_attestation_enabled",
        False,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        True,
    )


@pytest.mark.asyncio
async def test_no_phala_analysis_allow_enqueues_tb_job(
    client,
    database_session,
    monkeypatch: pytest.MonkeyPatch,
    signed_submission_override,
    tmp_path: Path,
) -> None:
    """Given NO_PHALA + empty env confirmed, When analysis allows, Then tb_queued + job."""

    _enable_no_phala(monkeypatch)
    configure_master(monkeypatch, tmp_path)
    response = await submit_agent(
        client, {"agent.py": "def solve(value):\n    return value + 1\n"}
    )
    assert response.status_code in {200, 201}, response.text

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        submission.env_confirmed_empty = True
        summary = await run_next_analysis(
            session,
            lease_owner="no-phala-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()
        await session.refresh(submission)

        assert summary is not None
        assert summary.verdict == "allow"
        assert summary.evaluation_job_id is not None
        assert submission.raw_status == "tb_queued"
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        assert job_count == 1
        ast_count = await session.scalar(select(func.count(PythonAstFeature.id)))
        assert ast_count and ast_count > 0
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))
        assert llm_count == 1
        analysis_count = await session.scalar(select(func.count(AnalysisRun.id)))
        assert analysis_count == 1


@pytest.mark.asyncio
async def test_no_phala_job_completion_marks_unattested_metadata(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Given NO_PHALA, When TB eval job completes, Then status event carries unattested tags."""

    _enable_no_phala(monkeypatch)
    _wire_offline_eval(monkeypatch, task_count=2)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=WINNER_HOTKEY,
            name="np-complete",
            agent_hash="np-complete-hash",
            artifact_uri=str(agent_dir),
            status="waiting_miner_env",
            raw_status="waiting_miner_env",
            effective_status="Waiting environments",
            env_confirmed_empty=True,
            submitted_at=NOW,
            created_at=NOW,
        )
        session.add(submission)
        await session.flush()
        job = await create_evaluation_job(session, submission, confirmed_miner_env=True)
        assert submission.raw_status == "tb_queued"
        summary = await run_evaluation_job(session, job.job_id, executor=FakeExecutor())
        await session.commit()
        await session.refresh(submission)
        await session.refresh(job)

        assert summary.status == "completed", job.error
        assert summary.score == 1.0
        assert job.status == "completed"
        assert submission.raw_status == "tb_completed"

        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent)
                    .where(SubmissionStatusEvent.submission_id == submission.id)
                    .order_by(SubmissionStatusEvent.id)
                )
            )
            .scalars()
            .all()
        )
        completed = [e for e in events if e.to_status == "tb_completed"]
        assert completed, "expected tb_completed status event"
        meta = json.loads(completed[-1].metadata_json or "{}")
        assert meta.get("attested") is False
        assert meta.get("attestation_status") == ATTESTATION_STATUS_UNATTESTED
        assert meta.get("execution_mode") == EXECUTION_MODE_NO_PHALA_HOST
        assert meta.get("score") == 1.0


@pytest.mark.asyncio
async def test_no_phala_completed_job_enters_weights(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given full-task completed job under flags-off, When get_weights, Then hotkey present."""

    _enable_no_phala(monkeypatch)
    required = 2
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.evaluation_task_count",
        required,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.phala_attestation_enabled",
        False,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        True,
    )

    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=WINNER_HOTKEY,
            name="np-weights",
            agent_hash="np-weights-hash",
            artifact_uri="/tmp/np-weights.zip",
            status="tb_completed",
            raw_status="tb_completed",
            effective_status="valid",
            submitted_at=NOW,
            created_at=NOW,
        )
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="job-np-weights",
            submission_id=submission.id,
            status="completed",
            selected_tasks_json="[]",
            score=0.85,
            passed_tasks=required,
            total_tasks=required,
            verdict="valid",
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        await session.commit()

    weights = await get_weights()
    assert weights == {WINNER_HOTKEY: 0.85}


@pytest.mark.asyncio
async def test_no_phala_weight_push_fires_and_logs_unattested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Given NO_PHALA + non-empty weights, When push_once, Then HTTP posts and CRITICAL log."""

    _enable_no_phala(monkeypatch)
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-np.sqlite3'}")
    await db.init()

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        captured["weights"] = dict(parsed.weights)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": "agent-challenge",
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-np-1",
                "payload_digest": parsed.payload_digest,
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = RawWeightPushClient(
        database=db,
        challenge_slug="agent-challenge",
        master_base_url="http://master.test",
        shared_token="test-token-secret",
        weights_fn=lambda: {WINNER_HOTKEY: 0.85},
        epoch_fn=lambda: 42,
        http_client=http,
    )

    with caplog.at_level(logging.CRITICAL, logger="agent_challenge.evaluation.raw_weight_push"):
        with activate_role(Role.CHALLENGE):
            result = await client.push_once()
    await http.aclose()

    assert result.status == "acknowledged"
    assert result.cursor_advanced is True
    assert captured["weights"] == {WINNER_HOTKEY: 0.85}
    assert any(
        "NO_PHALA raw weight push" in rec.message and "NOT TEE-attested" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_mode_off_weight_push_does_not_log_no_phala_banner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression: NO_PHALA off → push still works, no unattested CRITICAL."""

    monkeypatch.delenv(NO_PHALA_ENV, raising=False)
    monkeypatch.delenv("CHALLENGE_NO_PHALA", raising=False)
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-off.sqlite3'}")
    await db.init()

    def handler(request: httpx.Request) -> httpx.Response:
        parsed = RawWeightPushRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "protocol_version": "1.0",
                "challenge_slug": "agent-challenge",
                "epoch": parsed.epoch,
                "revision": parsed.revision,
                "snapshot_id": "snap-off",
                "payload_digest": parsed.payload_digest,
            },
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = RawWeightPushClient(
        database=db,
        challenge_slug="agent-challenge",
        master_base_url="http://master.test",
        shared_token="test-token-secret",
        weights_fn=lambda: {WINNER_HOTKEY: 1.0},
        epoch_fn=lambda: 7,
        http_client=http,
    )
    with caplog.at_level(logging.CRITICAL, logger="agent_challenge.evaluation.raw_weight_push"):
        with activate_role(Role.CHALLENGE):
            result = await client.push_once()
    await http.aclose()
    assert result.status == "acknowledged"
    assert not any("NO_PHALA raw weight push" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_weight_push_empty_skips_network(tmp_path: Path) -> None:
    """Edge: empty weights → skipped_empty, no HTTP."""

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'push-empty.sqlite3'}")
    await db.init()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = RawWeightPushClient(
        database=db,
        challenge_slug="agent-challenge",
        master_base_url="http://master.test",
        shared_token="test-token-secret",
        weights_fn=lambda: {},
        epoch_fn=lambda: 1,
        http_client=http,
    )
    with activate_role(Role.CHALLENGE):
        result = await client.push_once()
    await http.aclose()
    assert result.status == "skipped_empty"
    assert calls == 0
