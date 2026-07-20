from __future__ import annotations

import json
import sqlite3
import threading

import pytest
from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.gpu_scheduler import (
    BaseGpuTarget,
    GpuLeaseRequest,
    GpuLeaseScheduler,
)
from prism_challenge.repository import PrismRepository

REMOTE_ONLY_CODE = """
import torch

def build_model(ctx):
    return torch.nn.Linear(8, 8)

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""


def _submit(client: TestClient, code: str, nonce: str) -> str:
    payload = {"code": two_script_bundle(arch_code=code), "filename": "project.zip"}
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


def _lease_request(submission_id: str) -> GpuLeaseRequest:
    return GpuLeaseRequest(
        submission_id=submission_id,
        job_id=None,
        mode="gpu_proxy_eval",
        tier="dev",
        score_eligible=False,
        min_gpu_count=1,
        max_gpu_count=1,
        requested_gpu_count=1,
        autosplit_allowed=True,
        official_fixed_profile=False,
    )


def test_wall_time_overrun_force_kills_and_releases_lease(tmp_path, monkeypatch):
    db_file = tmp_path / "wall-time.sqlite3"
    container_started = threading.Event()
    release_blocked_run = threading.Event()
    reaped: list[str] = []

    def blocking_run(self, spec, timeout_seconds):
        container_started.set()
        # Model an eval that never returns on its own; only the orchestration reap unblocks it,
        # exactly as a force-killed container would unwind the worker thread.
        release_blocked_run.wait(timeout=10)
        return DockerRunResult(container_name="prism-eval", stdout="", stderr="", returncode=0)

    def fake_reap(self, submission_id: str) -> None:
        reaped.append(submission_id)
        release_blocked_run.set()

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", blocking_run)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.PrismContainerEvaluator.reap_job", fake_reap
    )

    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_file}",
        shared_token="secret",
        allow_insecure_signatures=True,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_orchestration_timeout_seconds=0.2,
    )

    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="wall-time")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()

    assert container_started.is_set()
    assert status["status"] == "failed"
    assert "wall-time" in (status["error"] or "")
    assert reaped == [submission_id]

    connection = sqlite3.connect(db_file)
    try:
        rows = connection.execute(
            "SELECT status FROM gpu_leases WHERE submission_id=?", (submission_id,)
        ).fetchall()
    finally:
        connection.close()
    assert rows, "submission must have held a GPU lease"
    assert all(row[0] == "released" for row in rows)


async def test_capacity_snapshot_reports_backpressure(tmp_path):
    database = Database(tmp_path / "capacity.sqlite3")
    await database.init()
    PrismRepository(database, epoch_seconds=60)
    scheduler = GpuLeaseScheduler(
        database, (BaseGpuTarget(id="target-a", server="server-a", gpu_count=1),)
    )

    first = await scheduler.enqueue_or_allocate(_lease_request("submission-1"))
    second = await scheduler.enqueue_or_allocate(_lease_request("submission-2"))
    assert first.active
    assert second.status == "queued"

    busy = await scheduler.capacity_snapshot()
    assert busy.total_devices == 1
    assert busy.active_devices == 1
    assert busy.free_devices == 0
    assert busy.at_capacity is True
    assert busy.oversubscribed is False
    assert busy.active_leases == 1
    assert busy.queued_leases == 1

    await scheduler.release_for_submission("submission-1", "completed")
    promoted = await scheduler.capacity_snapshot()
    assert promoted.active_leases == 1
    assert promoted.queued_leases == 0
    assert promoted.at_capacity is True


@pytest.mark.parametrize("vram_mib,expected", [(0, False), (24576, True)])
def test_vram_cap_env_injection(vram_mib, expected):
    settings = PrismSettings(base_eval_gpu_vram_mib=vram_mib)
    evaluator = PrismContainerEvaluator(settings=settings, ctx=PrismContext(sequence_length=16))

    env = evaluator._env(
        submission_id="sub",
        code_hash="codehash",
        arch_hash="archhash",
        backend="base_gpu",
    )

    assert ("PRISM_GPU_VRAM_CAP_MIB" in env) is expected
    if expected:
        assert env["PRISM_GPU_VRAM_CAP_MIB"] == str(vram_mib)


def test_orchestration_timeout_derives_above_hard_timeout():
    derived = PrismSettings(
        base_eval_orchestration_timeout_seconds=0.0,
        base_eval_orchestration_grace_seconds=120,
    )
    assert derived.resolved_orchestration_timeout_seconds == float(
        derived.base_eval_hard_timeout_seconds + 120
    )

    explicit = PrismSettings(base_eval_orchestration_timeout_seconds=5.0)
    assert explicit.resolved_orchestration_timeout_seconds == 5.0
