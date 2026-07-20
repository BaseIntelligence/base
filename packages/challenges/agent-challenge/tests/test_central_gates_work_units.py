"""Central gates feed pending work units (VAL-AC-004..009).

The AST + LLM gates run centrally before a submission's tasks become assignable.
An ``allow`` verdict expands the deterministic selected tasks into pending work
units for the coordination plane; ``reject`` and ``escalate`` produce none. The
gate's LLM review routes through the master gateway (mocked here).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.analyzer.lifecycle import gateway_llm_base_url, run_next_analysis
from agent_challenge.analyzer.llm_reviewer import (
    GATEWAY_PLACEHOLDER_MODEL,
    GatewayReviewProvider,
    LlmReviewOutcome,
    SubmitVerdictArgs,
    build_llm_verdict_row,
)
from agent_challenge.app import app
from agent_challenge.evaluation.benchmarks import BenchmarkTask, select_benchmark_tasks
from agent_challenge.evaluation.work_units import (
    list_pending_work_units,
    work_unit_id_for,
)
from agent_challenge.models import (
    AgentSubmission,
    EvaluationJob,
    LlmVerdict,
    TaskResult,
)
from agent_challenge.security import SignedRequestAuth
from agent_challenge.weights import get_weights

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"
NOW = datetime(2026, 5, 23, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def signed_submission_override() -> AsyncIterator[None]:
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


@pytest.fixture
def owner_auth_override() -> AsyncIterator[None]:
    calls = 0

    async def authenticate() -> SignedRequestAuth:
        nonlocal calls
        calls += 1
        return SignedRequestAuth(
            hotkey="owner-hotkey",
            signature=f"owner-signature-{calls}",
            nonce=f"owner-nonce-{calls}",
            timestamp=NOW.isoformat(),
            body_sha256=hashlib.sha256(f"admin-body-{calls}".encode()).hexdigest(),
            canonical_request=f"owner-request-{calls}",
        )

    app.dependency_overrides[routes.owner_signed_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.owner_signed_auth, None)


def configure_decentralized(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No master role gating; writable artifact root for stored zips."""

    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )


def use_benchmark_tasks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_count: int,
    selected_count: int,
) -> list[BenchmarkTask]:
    tasks = [
        BenchmarkTask(
            task_id=f"terminal-bench/task-{index}",
            docker_image=f"ghcr.io/baseintelligence/runner:{index}",
            prompt=f"task {index}",
            benchmark="terminal_bench",
        )
        for index in range(task_count)
    ]
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.load_benchmark_tasks",
        lambda: list(tasks),
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.settings.evaluation_task_count",
        selected_count,
    )
    return tasks


class StaticReviewer:
    """Deterministic gate reviewer that returns a fixed verdict (no network)."""

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
            similarity_assessment="",
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
            provider=_MockProvider(),
            verdict=verdict,
            transcript=transcript,
            manifest=manifest,
            similarity_evidence=list(similarity_evidence),
        )
        return LlmReviewOutcome(verdict=verdict, llm_verdict_row=row, transcript=transcript)


class _MockProvider:
    provider_name = "mock"
    model_name = GATEWAY_PLACEHOLDER_MODEL


class _FakeGatewayResponse:
    """Minimal httpx-like response simulating the master gateway."""

    def __init__(self, payload: Mapping[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Mapping[str, Any]:
        return self._payload


def _gateway_allow_payload() -> dict[str, Any]:
    return {
        "id": "gw-resp-1",
        "model": "anthropic/claude-opus-4.8",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "submit_verdict",
                                "arguments": json.dumps(
                                    {
                                        "verdict": "allow",
                                        "confidence": 0.95,
                                        "rationale": "looks fine via gateway",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _gateway_rules_valid_payload() -> dict[str, Any]:
    return {
        "id": "gw-rules-1",
        "model": "anthropic/claude-opus-4.8",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "rules-call-1",
                            "type": "function",
                            "function": {
                                "name": "submit_rules_review",
                                "arguments": json.dumps(
                                    {
                                        "verdict": "valid",
                                        "reason_codes": ["rules_passed"],
                                        "notes": "no policy violations via gateway",
                                    }
                                ),
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _fake_gateway_response_for(body: Mapping[str, Any]) -> _FakeGatewayResponse:
    """Return the primary submit_verdict allow, or the rules-review valid payload.

    The central gate makes two gateway calls per submission: the KimiLlmReviewer
    review (permits/forces ``submit_verdict``) and the rules-check (forces
    ``submit_rules_review``). The mock answers each with its matching tool payload.
    """

    tool_choice = body.get("tool_choice")
    if (
        isinstance(tool_choice, Mapping)
        and isinstance(tool_choice.get("function"), Mapping)
        and tool_choice["function"].get("name") == "submit_rules_review"
    ):
        return _FakeGatewayResponse(_gateway_rules_valid_payload())
    return _FakeGatewayResponse(_gateway_allow_payload())


async def submit_agent(client, files: dict[str, str | bytes]):
    archive_bytes = build_zip(files)
    return await client.post(
        "/submissions",
        json={
            "name": "agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )


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


# --------------------------------------------------------------------------- #
# VAL-AC-004: gates run before tasks become assignable
# --------------------------------------------------------------------------- #
async def test_no_work_units_before_verdict(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=8, selected_count=5)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        units = await list_pending_work_units(session)

    assert submission is not None
    assert submission.raw_status == "analysis_queued"
    assert job_count == 0
    assert units == []


# --------------------------------------------------------------------------- #
# VAL-AC-005: allow exposes the deterministic selected tasks as work units
# --------------------------------------------------------------------------- #
async def test_allow_exposes_selected_tasks_as_work_units(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    tasks = use_benchmark_tasks(monkeypatch, task_count=12, selected_count=5)

    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job = await session.scalar(select(EvaluationJob))
        units = await list_pending_work_units(session)

    assert submission is not None
    assert job is not None
    expected_tasks = select_benchmark_tasks(tasks, agent_hash=submission.agent_hash, count=5)
    assert job.total_tasks == len(expected_tasks) == 5
    assert len(units) == job.total_tasks
    assert {unit.task_id for unit in units} == {task.task_id for task in expected_tasks}
    assert {unit.work_unit_id for unit in units} == {
        work_unit_id_for(submission.id, task.task_id) for task in expected_tasks
    }
    assert all(unit.required_capability == "cpu" for unit in units)
    assert all(unit.submission_ref == submission.agent_hash for unit in units)


async def test_allow_work_units_exposed_via_internal_endpoint(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
    internal_headers,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=6, selected_count=4)

    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 2\n"})
    async with database_session() as session:
        await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    unauthorized = await client.get("/internal/v1/work_units")
    assert unauthorized.status_code in (401, 403)

    response = await client.get("/internal/v1/work_units", headers=internal_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["challenge_slug"] == "agent-challenge"
    assert len(body["work_units"]) == 4
    first = body["work_units"][0]
    assert set(first) == {
        "work_unit_id",
        "submission_id",
        "submission_ref",
        "miner_hotkey",
        "job_id",
        "task_id",
        "docker_image",
        "required_capability",
    }
    assert first["required_capability"] == "cpu"


# --------------------------------------------------------------------------- #
# VAL-AC-006: reject produces no work units
# --------------------------------------------------------------------------- #
async def test_reject_produces_no_work_units(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=8, selected_count=5)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("reject"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "reject"

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        units = await list_pending_work_units(session)

    assert submission is not None
    assert submission.raw_status == "analysis_rejected"
    assert job_count == 0
    assert units == []
    assert await get_weights() == {}


# --------------------------------------------------------------------------- #
# VAL-AC-007: escalate withholds work units pending admin review
# --------------------------------------------------------------------------- #
async def test_escalate_withholds_work_units(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=8, selected_count=5)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})

    async with database_session() as session:
        summary = await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("escalate"),
        )
        await session.commit()

    assert summary is not None
    assert summary.verdict == "escalate"

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))
        units = await list_pending_work_units(session)

    assert submission is not None
    assert submission.raw_status in {"analysis_escalated", "admin_paused"}
    assert job_count == 0
    assert units == []


async def test_admin_allow_after_escalate_creates_work_units(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    owner_auth_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=10, selected_count=6)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})
    async with database_session() as session:
        await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("escalate"),
        )
        await session.commit()

    # While paused for admin review there are no work units.
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        assert submission is not None
        submission_id = submission.id
        assert await list_pending_work_units(session) == []
        # Legacy confirmed-empty env so admin_allow enqueues directly.
        submission.env_confirmed_empty = True
        submission.env_confirmed_empty_at = NOW
        submission.env_locked_at = NOW
        submission.env_compatibility_reason = "pre_env_gate_analysis_allowed"
        await session.commit()

    response = await client.post(
        f"/owner/submissions/{submission_id}/admin-escalation",
        json={"decision": "admin_allow", "reason": "cleared"},
    )
    assert response.status_code == 200
    assert response.json()["job_id"] is not None

    async with database_session() as session:
        units = await list_pending_work_units(session)
    assert len(units) == 6


# --------------------------------------------------------------------------- #
# VAL-AC-008: gate LLM calls go through the master gateway (mocked)
# --------------------------------------------------------------------------- #
def test_gateway_provider_targets_gateway_url_without_provider_key(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _FakeGatewayResponse(_gateway_allow_payload())

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)

    provider = GatewayReviewProvider(
        gateway_token="scoped-token",
        base_url=gateway_llm_base_url("http://master:18080"),
    )
    result = provider.complete(messages=[], tools=[], tool_choice="auto", timeout_seconds=1.0)

    assert captured["url"] == "http://master:18080/llm/v1/chat/completions"
    assert captured["headers"]["X-Gateway-Token"] == "scoped-token"
    assert "Authorization" not in captured["headers"]
    # A placeholder model is sent; the gateway overwrites it from the token source.
    assert captured["body"]["model"] == GATEWAY_PLACEHOLDER_MODEL
    assert result.tool_calls and result.tool_calls[0].name == "submit_verdict"


async def test_gate_reaches_verdict_via_gateway_without_local_key(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=4, selected_count=2)

    # No local provider key; only the master gateway is wired.
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_gateway_base_url",
        "http://master:18080",
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.lifecycle.settings.llm_gateway_token",
        "scoped-token",
    )

    captured: dict[str, Any] = {}

    def fake_post(url, *, headers, json, timeout):  # noqa: A002 - mirror httpx.post
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _fake_gateway_response_for(json)

    monkeypatch.setattr("agent_challenge.analyzer.llm_reviewer.httpx.post", fake_post)

    await submit_agent(client, {"agent.py": "def solve(value):\n    return value + 1\n"})

    async with database_session() as session:
        # No injected reviewer: exercise the configured gateway-backed reviewer.
        summary = await run_next_analysis(session, lease_owner="analysis-worker")
        await session.commit()

    assert summary is not None
    assert summary.verdict == "allow"
    assert summary.status != "llm_standby"
    assert captured["url"] == "http://master:18080/llm/v1/chat/completions"
    assert captured["headers"]["X-Gateway-Token"] == "scoped-token"
    assert "Authorization" not in captured["headers"]
    # The configured central gate sends a placeholder model; the gateway injects
    # the real model from the token source.
    assert captured["body"]["model"] == GATEWAY_PLACEHOLDER_MODEL

    async with database_session() as session:
        llm_count = await session.scalar(select(func.count(LlmVerdict.id)))
        verdict_row = await session.scalar(select(LlmVerdict))
    assert llm_count == 1
    assert verdict_row is not None
    assert verdict_row.verdict == "allow"


# --------------------------------------------------------------------------- #
# VAL-AC-009: deterministic task selection by agent hash
# --------------------------------------------------------------------------- #
def test_select_benchmark_tasks_is_deterministic_by_hash():
    tasks = [BenchmarkTask(task_id=f"task-{index}", docker_image="img") for index in range(20)]

    first = select_benchmark_tasks(tasks, agent_hash="agent-hash-A", count=6)
    second = select_benchmark_tasks(tasks, agent_hash="agent-hash-A", count=6)
    other = select_benchmark_tasks(tasks, agent_hash="agent-hash-B", count=6)

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert [task.task_id for task in first] != [task.task_id for task in other]


async def test_work_unit_expansion_uses_same_deterministic_set(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    tasks = use_benchmark_tasks(monkeypatch, task_count=15, selected_count=7)

    await submit_agent(client, {"agent.py": "def solve():\n    return 3\n"})
    async with database_session() as session:
        await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        units = await list_pending_work_units(session)

    assert submission is not None
    expected = select_benchmark_tasks(tasks, agent_hash=submission.agent_hash, count=7)
    assert {unit.task_id for unit in units} == {task.task_id for task in expected}


# --------------------------------------------------------------------------- #
# Pending-unit derivation details
# --------------------------------------------------------------------------- #
async def test_completed_task_is_no_longer_a_pending_work_unit(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=8, selected_count=5)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})
    async with database_session() as session:
        await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob))
        assert job is not None
        before = await list_pending_work_units(session)
        finished_task_id = before[0].task_id
        session.add(
            TaskResult(
                job_id=job.id,
                task_id=finished_task_id,
                docker_image="img",
                status="completed",
                score=1.0,
            )
        )
        await session.commit()

    async with database_session() as session:
        after = await list_pending_work_units(session)

    assert len(after) == len(before) - 1
    assert finished_task_id not in {unit.task_id for unit in after}


async def test_list_pending_work_units_opens_its_own_session(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    configure_decentralized(monkeypatch, tmp_path)
    use_benchmark_tasks(monkeypatch, task_count=6, selected_count=3)

    await submit_agent(client, {"agent.py": "def solve():\n    return 1\n"})
    async with database_session() as session:
        await run_next_analysis(
            session,
            lease_owner="analysis-worker",
            reviewer=StaticReviewer("allow"),
        )
        await session.commit()

    units = await list_pending_work_units()
    assert len(units) == 3
