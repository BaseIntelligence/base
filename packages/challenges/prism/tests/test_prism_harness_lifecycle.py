from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.gpu_scheduler import (
    BaseGpuTarget,
    GpuLeaseRequest,
    GpuLeaseScheduler,
)

# A real nn.Module so the static instantiation phase passes before the (faked) container eval.
REMOTE_ONLY_CODE = """
import torch

def build_model(ctx):
    return torch.nn.Linear(8, 8)

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""


def _settings(tmp_path: Path, name: str) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / name}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        # These doubles exercise runtime failure handling with single-process loops; the
        # multi-GPU static contract is covered in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )


def _submit(client: TestClient, nonce: str) -> str:
    payload = {"code": two_script_bundle(arch_code=REMOTE_ONLY_CODE), "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _active_leases(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM gpu_leases WHERE submission_id=? AND status=?",
            (submission_id, "active"),
        ).fetchone()
    finally:
        conn.close()
    return int(count)


def _score_rows(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return int(count)


def test_crashing_miner_loop_lands_failed_lease_released_worker_survives(tmp_path, monkeypatch):
    db_path = tmp_path / "crash.sqlite3"

    def crashing_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="Traceback (most recent call last):\nRuntimeError: miner train loop exploded",
            returncode=1,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", crashing_run)

    settings = _settings(tmp_path, "crash.sqlite3")
    with TestClient(create_app(settings)) as client:
        first = _submit(client, nonce="crash-1")
        _process(client)
        status = client.get(f"/v1/submissions/{first}").json()
        assert status["status"] == "failed"
        assert "miner train loop exploded" in (status["error"] or "")

        # Lease released, no score written for the crashed run.
        assert _active_leases(db_path, first) == 0
        assert _score_rows(db_path, first) == 0

        # The worker survives and keeps serving: /health 200 and the next submission processes.
        assert client.get("/health").status_code == 200
        second = _submit(client, nonce="crash-2")
        _process(client)
        next_status = client.get(f"/v1/submissions/{second}").json()
        assert next_status["status"] == "failed"
        assert _active_leases(db_path, second) == 0


def test_resource_oom_lands_failed_with_oom_reason_lease_released(tmp_path, monkeypatch):
    db_path = tmp_path / "oom.sqlite3"

    def oom_run(self, spec, timeout_seconds):
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="RuntimeError: CUDA out of memory. Tried to allocate 40.00 GiB",
            returncode=137,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", oom_run)

    settings = _settings(tmp_path, "oom.sqlite3")
    with TestClient(create_app(settings)) as client:
        submission = _submit(client, nonce="oom-1")
        _process(client)
        status = client.get(f"/v1/submissions/{submission}").json()
        assert status["status"] == "failed"
        assert "out of memory" in (status["error"] or "").lower()
        assert _active_leases(db_path, submission) == 0
        assert _score_rows(db_path, submission) == 0
        assert client.get("/health").status_code == 200


def test_terminal_eval_replicated_job_is_reaped(tmp_path, monkeypatch):
    reaped: list[str] = []

    def crashing_run(self, spec, timeout_seconds):
        return DockerRunResult("prism-eval", "", "RuntimeError: boom", 1)

    def record_cleanup(self, job_id):
        reaped.append(job_id)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", crashing_run)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.cleanup_job", record_cleanup
    )

    settings = _settings(tmp_path, "reap.sqlite3")
    with TestClient(create_app(settings)) as client:
        submission = _submit(client, nonce="reap-1")
        _process(client)
        status = client.get(f"/v1/submissions/{submission}").json()
        assert status["status"] == "failed"
    # The terminal eval's job/service is reaped exactly once for this submission id.
    assert submission in reaped


def _eval_request(submission_id: str) -> GpuLeaseRequest:
    return GpuLeaseRequest(
        submission_id=submission_id,
        job_id=f"job-{submission_id}",
        mode="gpu_proxy_eval",
        tier="proxy",
        score_eligible=True,
        min_gpu_count=1,
        max_gpu_count=1,
        requested_gpu_count=1,
        autosplit_allowed=False,
        official_fixed_profile=True,
    )


async def test_concurrent_eval_submissions_serialized_on_single_gpu(tmp_path) -> None:
    database = Database(tmp_path / "serialize.sqlite3")
    await database.init()
    scheduler = GpuLeaseScheduler(
        database, (BaseGpuTarget(id="gpu-a", server="server-a", gpu_count=1),)
    )

    first = await scheduler.enqueue_or_allocate(_eval_request("sub-A"))
    second = await scheduler.enqueue_or_allocate(_eval_request("sub-B"))

    # Single physical GPU: exactly one active eval lease at a time, the other waits.
    assert first.active
    assert second.status == "queued"
    active = [lease for lease in await scheduler.leases() if lease.active]
    assert len(active) == 1

    # The same submission re-requesting does not acquire a second concurrent lease.
    repeat = await scheduler.enqueue_or_allocate(_eval_request("sub-A"))
    assert repeat.active
    active_for_a = [
        lease
        for lease in await scheduler.leases()
        if lease.submission_id == "sub-A" and lease.active
    ]
    assert len(active_for_a) == 1

    # The second only runs after the first releases (no double-eval / GPU contention).
    await scheduler.release_for_submission("sub-A", "terminal")
    promoted = await scheduler.active_lease_for_submission("sub-B")
    assert promoted is not None
    assert promoted.gpu_count == 1
    active_after = [lease for lease in await scheduler.leases() if lease.active]
    assert len(active_after) == 1
