"""Cache-Control headers on public GET read endpoints.

Public reads advertise a ``public`` Cache-Control so the Cloudflare/Vercel edge
in front of the flaky origin tunnel (and browsers) can cache them, cutting
repeated round trips. Mutation and authenticated routes must never advertise a
public cache.
"""

from __future__ import annotations

import base64
import io
import zipfile

import pytest

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.core.config import settings
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import effective_evaluation_task_count
from agent_challenge.security import SignedRequestAuth

FULL_TASK_COUNT = effective_evaluation_task_count(settings.evaluation_task_count)

READ_CACHE = "public, max-age=5, s-maxage=5, stale-while-revalidate=30"
SHORT_CACHE = "public, max-age=2, s-maxage=2, stale-while-revalidate=15"
BENCHMARK_CACHE = "public, max-age=60, s-maxage=60, stale-while-revalidate=300"
SOURCE_CACHE = "public, max-age=300, s-maxage=300, stale-while-revalidate=86400"


def _make_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", "class Agent:\n    pass\n")
    return buffer.getvalue()


async def _seed_scored_submission(
    session,
    *,
    hotkey: str = "hk-cache",
    agent_hash: str = "hash-cache",
    score: float = 0.75,
) -> int:
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="tb_completed",
        raw_status="tb_completed",
        effective_status="valid",
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json="[]",
        score=score,
        passed_tasks=1,
        total_tasks=FULL_TASK_COUNT,
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    await session.commit()
    return submission.id


@pytest.fixture
def signed_submission_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="hotkey-a",
            signature="test-signature",
            nonce="test-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="signed-test-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


async def test_submissions_list_cache_control(client):
    response = await client.get("/submissions")

    assert response.status_code == 200
    assert response.headers["cache-control"] == READ_CACHE


async def test_submissions_count_cache_control(client):
    response = await client.get("/submissions/count")

    assert response.status_code == 200
    assert response.headers["cache-control"] == READ_CACHE


async def test_leaderboard_cache_control(client, database_session):
    async with database_session() as session:
        await _seed_scored_submission(session)

    response = await client.get("/leaderboard")

    assert response.status_code == 200
    assert response.headers["cache-control"] == READ_CACHE


async def test_submission_status_cache_control(client, database_session):
    async with database_session() as session:
        submission_id = await _seed_scored_submission(session)

    response = await client.get(f"/submissions/{submission_id}/status")

    assert response.status_code == 200
    assert response.headers["cache-control"] == SHORT_CACHE


async def test_v1_submission_status_cache_control(client, database_session):
    async with database_session() as session:
        submission_id = await _seed_scored_submission(session)

    response = await client.get(f"/v1/submissions/{submission_id}/status")

    assert response.status_code == 200
    assert response.headers["cache-control"] == SHORT_CACHE


async def test_agent_evaluation_cache_control(client, database_session):
    async with database_session() as session:
        await _seed_scored_submission(session, agent_hash="hash-eval")

    response = await client.get("/agents/hash-eval/evaluation")

    assert response.status_code == 200
    assert response.headers["cache-control"] == READ_CACHE


async def test_agent_source_cache_control(client, database_session):
    async with database_session() as session:
        await _seed_scored_submission(session, agent_hash="hash-src")

    response = await client.get("/agents/hash-src/source")

    assert response.status_code == 200
    assert response.headers["cache-control"] == SOURCE_CACHE


async def test_benchmarks_cache_control(client, monkeypatch):
    monkeypatch.setattr("agent_challenge.api.routes.settings.benchmark_backend", "terminal_bench")
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.terminal_bench_dataset",
        "terminal-bench/terminal-bench-2-1",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.benchmark_backend",
        "terminal_bench",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.terminal_bench_dataset",
        "terminal-bench/terminal-bench-2-1",
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.settings.terminal_bench_task_ids",
        (),
    )

    response = await client.get("/benchmarks")

    assert response.status_code == 200
    assert response.headers["cache-control"] == BENCHMARK_CACHE


async def test_mutation_route_has_no_public_cache_control(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_zip_base64": base64.b64encode(_make_zip()).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert "public" not in response.headers.get("cache-control", "")
