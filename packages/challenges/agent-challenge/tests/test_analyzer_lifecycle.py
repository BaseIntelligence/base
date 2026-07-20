from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.analyzer.lifecycle import run_analysis_for_submission, run_next_analysis
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    KimiLlmReviewer,
    LlmProviderResponse,
    LlmProviderTimeout,
    LlmProviderUnavailable,
    LlmReviewOutcome,
    LlmToolCall,
    SubmitVerdictArgs,
    build_llm_verdict_row,
)
from agent_challenge.analyzer.schemas import EvidenceItem, ReviewerResult
from agent_challenge.app import app
from agent_challenge.evaluation.worker import run_worker_once
from agent_challenge.models import (
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    EvaluationJob,
    LlmVerdict,
    PythonAstFeature,
    SimilarityMatch,
    SubmissionEnvVar,
    SubmissionStatusEvent,
)
from agent_challenge.security import SignedRequestAuth
from agent_challenge.submissions.state_machine import ensure_submission_status
from agent_challenge.weights import get_weights

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


@pytest.fixture
def signed_submission_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="signed-miner-hotkey",
            signature="test-signature",
            nonce="test-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


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
    ):
        raise AssertionError("network-backed reviewer provider must not be called in tests")


class StaticReviewer:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    def review(self, *, analysis_run_id, manifest, read_session, similarity_evidence):
        self.calls += 1
        verdict = SubmitVerdictArgs(
            verdict=self.verdict,
            confidence=0.9,
            rationale=f"mock {self.verdict}",
            evidence_paths=["agent.py"],
            similarity_assessment=json.dumps(list(similarity_evidence), sort_keys=True),
            policy_flags=[f"mock_{self.verdict}"],
        )
        transcript = {
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


class ProviderUnavailableReviewer:
    def review(self, *, analysis_run_id, manifest, read_session, similarity_evidence):
        raise LlmProviderUnavailable(
            "LLM gateway token is not configured for /tmp/private/path with sk-test-secret"
        )


class FakeRulesReviewer:
    """Fake pipeline rules reviewer (``review(request) -> ReviewerResult``)."""

    def __init__(self, verdict, *, reason_codes=None, evidence=None) -> None:
        self.verdict = verdict
        self.reason_codes = reason_codes or [f"rules_{verdict}"]
        self.evidence = evidence or []
        self.calls = 0

    def review(self, request):
        self.calls += 1
        return ReviewerResult(
            verdict=self.verdict,
            reason_codes=self.reason_codes,
            evidence=self.evidence,
            notes="fake rules review",
        )


class ConnectionReleaseProbeReviewer(StaticReviewer):
    def __init__(self, verdict: str, session) -> None:
        super().__init__(verdict)
        self._session = session
        self.in_transaction_at_review: bool | None = None

    def review(self, **kwargs):
        self.in_transaction_at_review = self._session.in_transaction()
        return super().review(**kwargs)


class _TimeoutProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def complete(self, *, messages, tools, tool_choice, timeout_seconds):
        raise LlmProviderTimeout("LLM gateway request timed out")


class _ScriptedProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL

    def __init__(self, responses) -> None:
        self._responses = list(responses)

    def complete(self, *, messages, tools, tool_choice, timeout_seconds):
        return self._responses.pop(0)


def _disallowed_tool_reviewer() -> KimiLlmReviewer:
    return KimiLlmReviewer(
        provider=_ScriptedProvider(
            [
                LlmProviderResponse(
                    tool_calls=(
                        LlmToolCall(id="shell-1", name="run_shell", arguments={"command": "ls"}),
                    )
                )
            ]
        ),
        max_attempts=1,
    )


def configure_master(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "master")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)


def configure_normal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "normal")
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.validator_role", "normal")
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.validator_role", "normal")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )


async def test_master_submission_queues_analysis_without_evaluation_job(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    archive_bytes = build_zip({"agent.py": "def solve():\n    return 1\n"})
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/submissions",
        json={
            "name": "master-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    assert response.json()["latest_evaluation"] is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent.to_status).order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    assert submission is not None
    assert submission.zip_sha256 == zip_sha256
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert submission.latest_evaluation_job_id is None
    assert job_count == 0
    assert events == ["received", "upload_verified", "rate_limit_reserved", "analysis_queued"]


async def test_worker_allow_persists_analysis_and_waits_for_miner_env(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    reviewer = StaticReviewer("allow")
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_lifecycle_reviewer",
        lambda: reviewer,
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        submission.env_confirmed_empty = False
        submission.env_confirmed_empty_at = None
        await session.commit()

    iteration = await run_worker_once(worker_id="analysis-worker")

    assert iteration.analysis_summary is not None
    assert iteration.analysis_summary.verdict == "allow"
    assert iteration.summary is None
    assert reviewer.calls == 1
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        analysis_run_count = await session.scalar(select(func.count(AnalysisRun.id)))
        ast_count = await session.scalar(select(func.count(PythonAstFeature.id)))
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))

    assert submission is not None
    assert submission.raw_status == "waiting_miner_env"
    assert submission.effective_status == "Waiting environments"
    assert job_count == 0
    assert submission.latest_evaluation_job_id is None
    assert analysis_run_count == 1
    assert ast_count and ast_count > 0
    assert llm_count == 1


async def test_worker_allow_with_preexisting_env_enqueues_one_evaluation_and_locks_env(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    key_file = tmp_path / "env-key"
    key_file.write_text(Fernet.generate_key().decode("ascii"), encoding="utf-8")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.submission_env_encryption_key_file",
        str(key_file),
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        session.add(
            SubmissionEnvVar.encrypted(
                submission_id=submission.id,
                key="API_TOKEN",
                value="preexisting-sensitive-value",
                settings=routes.settings,
            )
        )
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"
    assert summary.evaluation_job_id is not None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        env_var = await session.scalar(select(SubmissionEnvVar))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_locked_at is not None
    assert submission.latest_evaluation_job_id is not None
    assert job_count == 1
    assert env_var is not None
    assert env_var.locked_at is not None


async def test_worker_allow_with_confirmed_empty_env_enqueues_one_evaluation_and_locks_env(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        submission.env_confirmed_empty = True
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"
    assert summary.evaluation_job_id is not None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_confirmed_empty is True
    assert submission.env_locked_at is not None
    assert submission.latest_evaluation_job_id is not None
    assert job_count == 1


async def test_create_submission_marks_env_confirmed_empty_and_auto_enqueues_on_allow(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        assert submission.env_confirmed_empty is True
        assert submission.env_confirmed_empty_at is not None
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"
    assert summary.evaluation_job_id is not None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert submission.env_locked_at is not None
    assert submission.latest_evaluation_job_id is not None
    assert job_count == 1


async def test_analysis_commits_before_llm_call_to_release_connection(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        probe = ConnectionReleaseProbeReviewer("allow", session)
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=probe,
        )
        await session.commit()

    assert probe.calls == 1
    # FIX-2: connection must be released (txn committed) before the slow LLM call
    # so an idle cross-node socket cannot be black-holed and held across it.
    assert probe.in_transaction_at_review is False
    assert summary is not None
    assert summary.verdict == "allow"


async def test_provider_unavailable_moves_submission_to_llm_standby_without_side_effects(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=ProviderUnavailableReviewer(),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "standby"
    assert summary.status == "llm_standby"
    assert summary.evaluation_job_id is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        latest_event = await session.scalar(
            select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence.desc())
        )
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))

    assert submission is not None
    assert submission.raw_status == "llm_standby"
    assert submission.effective_status == "LLM standby"
    assert latest_event is not None
    assert latest_event.to_status == "llm_standby"
    assert latest_event.reason == "llm_provider_unavailable"
    serialized_event = json.dumps(
        {"reason": latest_event.reason, "metadata": json.loads(latest_event.metadata_json)},
        sort_keys=True,
    )
    assert "sk-test-secret" not in serialized_event
    assert "/tmp/private/path" not in serialized_event
    assert "LLM gateway token" not in serialized_event
    assert llm_count == 0
    assert job_count == 0
    assert admin_count == 0


async def test_transient_timeout_routes_to_standby_not_escalate(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=KimiLlmReviewer(provider=_TimeoutProvider(), max_attempts=1),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "standby"
    assert summary.status == "llm_standby"
    assert summary.evaluation_job_id is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))
        analysis_run = await session.scalar(select(AnalysisRun))

    assert submission is not None
    assert submission.raw_status == "llm_standby"
    assert submission.effective_status == "LLM standby"
    assert admin_count == 0
    assert llm_count == 0
    assert analysis_run is not None
    assert json.loads(analysis_run.reason_codes_json) == ["provider_timeout"]


async def test_disallowed_tool_after_retries_routes_to_standby(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=_disallowed_tool_reviewer(),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "standby"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))

    assert submission is not None
    assert submission.raw_status == "llm_standby"
    assert admin_count == 0


async def test_retry_include_exclude_now_consumed(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    # Move disallowed_tool out of the retryable set: it must now escalate to
    # admin review instead of parking in standby, proving the policy is live.
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_reviewer_retry_exclude",
        ("unsafe_path", "submit_verdict_not_final", "disallowed_tool"),
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=_disallowed_tool_reviewer(),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "escalate"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))

    assert submission is not None
    assert submission.raw_status == "admin_paused"
    assert admin_count == 1
    assert llm_count == 1


async def test_standby_requeue_backoff_bounded_before_final_escalate(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_reviewer_max_standby_cycles", 1
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        first = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=KimiLlmReviewer(provider=_TimeoutProvider(), max_attempts=1),
        )
        await session.commit()

    assert first is not None
    assert first.verdict == "standby"

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        await ensure_submission_status(
            session, submission, "analysis_queued", actor="test", reason="requeue"
        )
        await session.commit()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        second = await run_analysis_for_submission(
            session,
            submission.id,
            actor="analysis-worker",
            reviewer=KimiLlmReviewer(provider=_TimeoutProvider(), max_attempts=1),
        )
        await session.commit()

    assert second.verdict == "escalate"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        standby_runs = await session.scalar(
            select(func.count(AnalysisRun.id)).where(AnalysisRun.status == "llm_standby")
        )

    assert submission is not None
    assert submission.raw_status == "admin_paused"
    assert admin_count == 1
    assert standby_runs == 1


async def test_missing_gateway_token_standby_does_not_tight_loop(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    # No gateway token/base URL configured: the configured reviewer cannot reach
    # the master gateway, so the submission parks in llm_standby.
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.llm_gateway_base_url", None)
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.llm_gateway_token", None)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    first_iteration = await run_worker_once(worker_id="analysis-worker")
    async with database_session() as session:
        first_event_count = await session.scalar(select(func.count(SubmissionStatusEvent.id)))
        first_analysis_count = await session.scalar(select(func.count(AnalysisRun.id)))

    second_iteration = await run_worker_once(worker_id="analysis-worker")

    assert first_iteration.analysis_summary is not None
    assert first_iteration.analysis_summary.verdict == "standby"
    assert first_iteration.analysis_summary.status == "llm_standby"
    assert second_iteration.analysis_summary is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        event_count = await session.scalar(select(func.count(SubmissionStatusEvent.id)))
        analysis_count = await session.scalar(select(func.count(AnalysisRun.id)))
        latest_event = await session.scalar(
            select(SubmissionStatusEvent).order_by(SubmissionStatusEvent.sequence.desc())
        )

    assert submission is not None
    assert submission.raw_status == "llm_standby"
    assert analysis_count == first_analysis_count == 1
    assert event_count == first_event_count
    assert latest_event is not None
    assert latest_event.reason == "missing_llm_gateway_token"


async def test_llm_standby_requeues_when_gateway_token_becomes_available(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.llm_gateway_base_url", None)
    monkeypatch.setattr("agent_challenge.analyzer.lifecycle.settings.llm_gateway_token", None)
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    first_iteration = await run_worker_once(worker_id="analysis-worker")
    assert first_iteration.analysis_summary is not None
    assert first_iteration.analysis_summary.verdict == "standby"

    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_gateway_base_url",
        "http://master:19080",
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_gateway_token",
        "scoped-token",
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_lifecycle_reviewer",
        lambda: StaticReviewer("allow"),
    )
    second_iteration = await run_worker_once(worker_id="analysis-worker")

    assert second_iteration.analysis_summary is not None
    assert second_iteration.analysis_summary.verdict == "allow"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        analysis_count = await session.scalar(select(func.count(AnalysisRun.id)))
        events = (
            (
                await session.execute(
                    select(SubmissionStatusEvent.to_status).order_by(SubmissionStatusEvent.sequence)
                )
            )
            .scalars()
            .all()
        )

    assert submission is not None
    assert submission.raw_status == "tb_completed"
    assert analysis_count == 2
    assert events[-9:] == [
        "llm_standby",
        "analysis_queued",
        "ast_running",
        "llm_running",
        "analysis_allowed",
        "waiting_miner_env",
        "tb_queued",
        "tb_running",
        "tb_completed",
    ]


@pytest.mark.parametrize(
    ("verdict", "raw_status", "effective_status"),
    [
        ("reject", "analysis_rejected", "invalid"),
        ("escalate", "admin_paused", "admin_paused"),
    ],
)
async def test_reject_and_escalate_do_not_queue_evaluation_or_weights(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
    verdict,
    raw_status,
    effective_status,
):
    configure_master(monkeypatch, tmp_path)
    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer(verdict),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == verdict
    assert summary.evaluation_job_id is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        admin_review_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))
        similarity_count = await session.scalar(select(func.count(SimilarityMatch.id)))

    assert submission is not None
    assert submission.raw_status == raw_status
    assert submission.effective_status == effective_status
    assert job_count == 0
    assert llm_count == 1
    assert similarity_count == 0
    assert admin_review_count == (1 if verdict == "escalate" else 0)
    assert await get_weights() == {}


async def test_legacy_normal_role_still_queues_analysis_centrally(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    """The legacy ``normal`` role is inert: central analysis is queued regardless."""
    configure_normal(monkeypatch, tmp_path)

    response = await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert job_count == 0


async def test_analysis_offloads_blocking_review_and_ast_off_event_loop(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    """Fix #1: the blocking LLM review and CPU-bound AST extraction must run off
    the shared event-loop thread (via asyncio.to_thread) so the combined worker
    cannot freeze HTTP/SSE handling."""

    import threading

    from agent_challenge.analyzer import lifecycle

    configure_master(monkeypatch, tmp_path)
    main_thread = threading.get_ident()
    review_threads: list[int] = []
    ast_threads: list[int] = []

    class ThreadRecordingReviewer:
        def review(self, *, analysis_run_id, manifest, read_session, similarity_evidence):
            review_threads.append(threading.get_ident())
            verdict = SubmitVerdictArgs(
                verdict="escalate",
                confidence=0.9,
                rationale="mock escalate",
                evidence_paths=["agent.py"],
                similarity_assessment="",
                policy_flags=["mock_escalate"],
            )
            transcript = {
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

    real_extract = lifecycle.extract_python_ast_features

    def _spy_extract(*args, **kwargs):
        ast_threads.append(threading.get_ident())
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "extract_python_ast_features", _spy_extract)

    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=ThreadRecordingReviewer(),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "escalate"
    assert review_threads and all(thread != main_thread for thread in review_threads)
    assert ast_threads and all(thread != main_thread for thread in ast_threads)


async def test_gate_blocks_submission_that_trips_rules_check(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_lifecycle_reviewer",
        lambda: StaticReviewer("allow"),
    )
    rules_reviewer = FakeRulesReviewer(
        "invalid",
        reason_codes=["reads_hidden_tests"],
        evidence=[
            EvidenceItem(
                path="agent.py",
                line_start=2,
                line_end=2,
                snippet="open('/app/tests/test_x.py')",
                reason_code="reads_hidden_tests",
                description="reads hidden benchmark tests",
            )
        ],
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_rules_reviewer",
        lambda: rules_reviewer,
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(session, lease_owner="analysis-worker")
        await session.commit()

    assert summary is not None
    assert summary.verdict == "reject"
    assert summary.evaluation_job_id is None
    assert rules_reviewer.calls == 1
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        analysis_run = await session.scalar(select(AnalysisRun))

    assert submission is not None
    assert submission.raw_status == "analysis_rejected"
    assert submission.effective_status == "invalid"
    assert job_count == 0
    assert admin_count == 0
    assert analysis_run is not None
    report = json.loads(analysis_run.report_json)
    assert report["rules_check"]["overall_verdict"] == "invalid"
    assert "reads_hidden_tests" in report["rules_check"]["reason_codes"]
    assert report["rules_check"]["evidence"][0]["reason_code"] == "reads_hidden_tests"
    assert report["ast"]["verdict"] == "clean"
    assert report["ast"]["verdict_reason"]
    assert "reads_hidden_tests" in json.loads(analysis_run.reason_codes_json)


async def test_gate_escalates_when_rules_check_uncertain(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_lifecycle_reviewer",
        lambda: StaticReviewer("allow"),
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_rules_reviewer",
        lambda: FakeRulesReviewer("suspicious", reason_codes=["needs_review"]),
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(session, lease_owner="analysis-worker")
        await session.commit()

    assert summary is not None
    assert summary.verdict == "escalate"
    assert summary.evaluation_job_id is None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        admin_count = await session.scalar(select(func.count(AdminReviewDecision.id)))
        analysis_run = await session.scalar(select(AnalysisRun))

    assert submission is not None
    assert submission.raw_status == "admin_paused"
    assert submission.effective_status == "admin_paused"
    assert job_count == 0
    assert admin_count == 1
    report = json.loads(analysis_run.report_json)
    assert report["rules_check"]["overall_verdict"] == "suspicious"


async def test_gate_clean_submission_allows_and_records_ast_and_rules(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_master(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_lifecycle_reviewer",
        lambda: StaticReviewer("allow"),
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.build_configured_rules_reviewer",
        lambda: FakeRulesReviewer("valid", reason_codes=["rules_passed"]),
    )
    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(session, lease_owner="analysis-worker")
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"
    assert summary.evaluation_job_id is not None
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        analysis_run = await session.scalar(select(AnalysisRun))

    assert submission is not None
    assert submission.raw_status == "tb_queued"
    assert job_count == 1
    report = json.loads(analysis_run.report_json)
    assert report["ast"]["verdict"] == "clean"
    assert report["ast"]["verdict_reason"]
    assert report["rules_check"]["overall_verdict"] == "valid"
    assert report["llm_verdict"]["verdict"] == "allow"


async def submit_agent(client, files: dict[str, str | bytes]):
    archive_bytes = build_zip(files)
    return await client.post(
        "/submissions",
        json={
            "name": "agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
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
