from __future__ import annotations

import base64
import hashlib
import io
import zipfile

import pytest
from _routing import public_route_paths

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.security import SignedRequestAuth
from agent_challenge.swe_forge import SweForgeTask

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


def make_zip(contents: str = "print('ok')\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", agent_source(contents))
    return buffer.getvalue()


async def test_public_submission_route(client, monkeypatch, signed_submission_override, tmp_path):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    archive_bytes = make_zip()
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["zip_sha256"] == zip_sha256
    assert payload["status"] == "queued"

    count_response = await client.get("/submissions/count")
    assert count_response.status_code == 200
    assert count_response.json() == {"count": 1}

    evaluation = await client.get(f"/agents/{payload['zip_sha256']}/evaluation")
    assert evaluation.status_code == 404

    submission = await client.get(f"/submissions/{payload['submission_id']}")
    assert submission.status_code == 200
    assert submission.json()["agent_hash"] == payload["zip_sha256"]


async def test_submission_rejects_missing_artifact(client, signed_submission_override):
    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_uri": "/tmp/agent-does-not-exist",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "artifact_uri_not_found"


async def test_submission_stages_base64_zip(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.benchmarks.load_swe_forge_tasks",
        lambda: [SweForgeTask(task_id="task-a", docker_image="baseintelligence/swe-forge:task-a")],
    )
    archive_bytes = make_zip()
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert (tmp_path / "agents" / zip_sha256 / "agent.zip").read_bytes() == archive_bytes
    assert not (tmp_path / "agents" / zip_sha256 / "agent.py").exists()


async def test_duplicate_agent_hash_returns_conflict(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = make_zip()
    payload = {
        "miner_hotkey": "hotkey-a",
        "name": "agent-a",
        "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        "agent_hash": "abc12345",
    }

    assert (await client.post("/submissions", json=payload)).status_code == 201
    assert (await client.post("/submissions", json=payload)).status_code == 409


async def test_submission_rejects_artifact_outside_root(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    agent_zip = tmp_path / "agent.zip"
    agent_zip.write_bytes(make_zip())
    monkeypatch.setattr("agent_challenge.api.routes.settings.artifact_root", str(tmp_path / "root"))

    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_uri": str(agent_zip),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "artifact_uri_outside_root"


async def test_submission_rejects_unsafe_agent_hash(client, signed_submission_override):
    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "hotkey-a",
            "name": "agent-a",
            "artifact_zip_base64": "not-a-zip",
            "agent_hash": "../bad-hash",
        },
    )

    assert response.status_code == 422


async def test_internal_get_weights_requires_auth(client):
    response = await client.get("/internal/v1/get_weights")

    assert response.status_code == 403


async def test_internal_get_weights(client, internal_headers):
    response = await client.get("/internal/v1/get_weights", headers=internal_headers)

    assert response.status_code == 200
    assert response.json()["challenge_slug"] == "agent-challenge"
    assert response.json()["weights"] == {}


async def test_benchmark_routes_use_terminal_bench_fallback_tasks(client, monkeypatch):
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
    assert response.json()["backend"] == "terminal_bench"
    assert response.json()["dataset"] == "terminal-bench/terminal-bench-2-1"
    assert response.json()["task_count"] == 30

    tasks = await client.get("/benchmarks/tasks")
    assert tasks.status_code == 200
    payload = tasks.json()
    assert len(payload) == 30
    assert payload[0]["task_id"] == "terminal-bench/adaptive-rejection-sampler"
    assert payload[29]["task_id"] == "terminal-bench/fix-git"
    assert payload[0]["benchmark"] == "terminal_bench"


def test_internal_launch_route_is_not_decorated_for_proxy_discovery():
    public_paths = public_route_paths(app)

    assert "/submissions" in public_paths
    assert "/submissions/count" in public_paths
    assert "/submissions/{submission_id}" in public_paths
    assert "/submissions/{submission_id}/status" in public_paths
    assert "/submissions/{submission_id}/events" in public_paths
    assert "/benchmarks" in public_paths
    assert "/benchmarks/tasks" in public_paths
    assert "/leaderboard" in public_paths
    assert "/agents/{agent_hash}/evaluation" in public_paths
    assert "/internal/v1/get_weights" not in public_paths
    assert "/internal/v1/submissions/{submission_id}/launch" not in public_paths
