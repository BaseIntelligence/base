"""RED tests: Prism live execution pool read + adjacent submission/curve regression.

Pins:

* ``GET /v1/execution-pool/live`` — in-flight jobs with latest event; terminal excluded
* S9 adjacent regression — existing submission status + curve endpoints unchanged
"""

from __future__ import annotations

import hmac
import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import anyio
import pytest
from base.challenge_sdk.executor import DockerRunResult
from conftest import VALID_CODE, signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.auth import canonical_submission_message
from prism_challenge.config import PrismSettings

_INTERNAL_TOKEN = "secret"
_HOTKEY = "hk-pool-live"


def _settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'pool-live.sqlite3'}",
        shared_token=_INTERNAL_TOKEN,
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    with TestClient(create_app(_settings(tmp_path))) as test_client:
        yield test_client


def _internal_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_INTERNAL_TOKEN}"}


def _sign_body(body: bytes, *, hotkey: str, nonce: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(_INTERNAL_TOKEN.encode(), message, sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
    }


def _seed_job(
    client: TestClient,
    *,
    job_id: str,
    submission_id: str,
    status: str,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    _HOTKEY,
                    1,
                    "project.zip",
                    "e30=",
                    f"hash-{submission_id}",
                    "{}",
                    "running" if status == "running" else status,
                    created_at,
                    created_at,
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id,
                    submission_id,
                    "l2",
                    status,
                    0,
                    "{}",
                    created_at,
                    created_at,
                ),
            )

    anyio.run(insert)


def _open_session(client: TestClient, *, eval_job_id: str, nonce: str) -> str:
    payload = {
        "eval_job_id": eval_job_id,
        "instance_id": "instance-pool-0",
        "hotkey_ss58": _HOTKEY,
        "nonce": nonce,
        "timestamp": str(int(time.time())),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        **_internal_headers(),
        **_sign_body(raw, hotkey=_HOTKEY, nonce=nonce),
    }
    resp = client.post("/v1/execution/telemetry-session", content=raw, headers=headers)
    assert resp.status_code == 200, resp.text
    return str(resp.json()["session_id"])


def _ingest(
    client: TestClient,
    *,
    session_id: str,
    eval_job_id: str,
    sequence: int,
    message: str,
) -> None:
    events = [
        {
            "eval_job_id": eval_job_id,
            "task_id": "train",
            "sequence": sequence,
            "event_type": "execution.progress",
            "message": message,
            "progress": min(1.0, sequence / 10.0),
        }
    ]
    body = json.dumps({"session_id": session_id, "events": events}, separators=(",", ":")).encode()
    headers = {
        "Content-Type": "application/json",
        **_internal_headers(),
        "X-Telemetry-Session": session_id,
    }
    resp = client.post("/v1/execution/events", content=body, headers=headers)
    assert resp.status_code == 200, resp.text
    assert "score" not in resp.json()
    assert "final_score" not in resp.json()


def test_s7_pool_live_shows_job(client: TestClient) -> None:
    """GET /v1/execution-pool/live returns in-flight job with latest event; terminal excluded."""

    live_job = "job-pool-live"
    done_job = "job-pool-done"
    failed_job = "job-pool-failed"
    _seed_job(client, job_id=live_job, submission_id="sub-pool-live", status="running")
    _seed_job(
        client,
        job_id=done_job,
        submission_id="sub-pool-done",
        status="completed",
        created_at="2026-01-01T00:01:00+00:00",
    )
    _seed_job(
        client,
        job_id=failed_job,
        submission_id="sub-pool-failed",
        status="failed",
        created_at="2026-01-01T00:02:00+00:00",
    )

    session_id = _open_session(client, eval_job_id=live_job, nonce="pool-n1")
    _ingest(
        client,
        session_id=session_id,
        eval_job_id=live_job,
        sequence=1,
        message="boot",
    )
    _ingest(
        client,
        session_id=session_id,
        eval_job_id=live_job,
        sequence=2,
        message="latest-progress-marker",
    )

    done_session = _open_session(client, eval_job_id=done_job, nonce="pool-n-done")
    _ingest(
        client,
        session_id=done_session,
        eval_job_id=done_job,
        sequence=1,
        message="finished-should-be-hidden",
    )

    resp = client.get("/v1/execution-pool/live")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    jobs: list[dict[str, Any]]
    if isinstance(body, list):
        jobs = body
    elif isinstance(body, dict) and isinstance(body.get("jobs"), list):
        jobs = body["jobs"]
    else:
        pytest.fail(f"unexpected live pool payload shape: {body!r}")

    job_ids = {
        str(item.get("eval_job_id") or item.get("id") or item.get("job_id")) for item in jobs
    }
    assert live_job in job_ids, f"in-flight job missing from live pool: {jobs!r}"
    assert done_job not in job_ids, "completed job must be excluded from live pool"
    assert failed_job not in job_ids, "failed job must be excluded from live pool"

    live_row = next(
        item
        for item in jobs
        if str(item.get("eval_job_id") or item.get("id") or item.get("job_id")) == live_job
    )
    blob = json.dumps(live_row)
    assert "latest-progress-marker" in blob, f"latest event not present: {live_row!r}"
    assert "final_score" not in live_row or live_row.get("final_score") is None
    for banned in ("mnemonic", "wallet_seed", "private_key"):
        assert banned not in blob


def test_execution_pool_live_empty_when_no_running_jobs(client: TestClient) -> None:
    """Empty pool is honest empty list / empty jobs array — not 404."""

    _seed_job(client, job_id="job-only-done", submission_id="sub-only-done", status="completed")
    resp = client.get("/v1/execution-pool/live")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if isinstance(body, list):
        assert body == []
    else:
        assert body.get("jobs") == []


def test_s9_existing_submission_api_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /v1/submissions/{id} and curve endpoint behave exactly as before telemetry."""

    def fake_run(self: Any, spec: Any, timeout_seconds: float) -> DockerRunResult:
        del self, timeout_seconds
        artifact_dir = next(mount.source for mount in spec.mounts if mount.target == "/artifacts")
        manifest = {
            "schema_version": "prism_run_manifest.v2",
            "metrics": {
                "covered_bytes": 4096,
                "sum_neg_log_likelihood_nats": 2200.0,
                "online_loss": [3.1, 2.9, 2.4],
                "predicted_tokens": 800,
                "tokens_seen": 800,
                "heldout_delta": 0.35,
                "val_bpb_trained": 1.10,
                "val_bpb_random_init": 1.45,
                "param_ladder_stage": "explore",
            },
        }
        (Path(artifact_dir) / "prism_run_manifest.v2.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        fake_run,
    )

    payload = {"code": two_script_bundle(arch_code=VALID_CODE), "filename": "project.zip"}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    submit = client.post(
        "/v1/submissions",
        content=raw,
        headers={
            **signed_headers(_INTERNAL_TOKEN, raw),
            "Content-Type": "application/json",
        },
    )
    assert submit.status_code == 200, submit.text
    submission_id = submit.json()["id"]

    process = client.post(
        "/internal/v1/worker/process-next",
        headers=_internal_headers(),
    )
    assert process.status_code == 200, process.text
    assert process.json()["submission_id"] == submission_id

    status = client.get(f"/v1/submissions/{submission_id}")
    assert status.status_code == 200, status.text
    status_body = status.json()
    assert status_body["id"] == submission_id
    assert status_body["status"] == "completed"
    assert "final_score" in status_body
    assert status_body["final_score"] is None or status_body["final_score"] >= 0

    curve = client.get(f"/v1/submissions/{submission_id}/curve")
    assert curve.status_code in {200, 404}, curve.text
    if curve.status_code == 200:
        curve_body = curve.json()
        assert curve_body["submission_id"] == submission_id
        assert "loss_curve" in curve_body
        assert "online_loss" in curve_body["loss_curve"]
        for banned in ("session_id", "telemetry", "execution_events", "mnemonic"):
            assert banned not in curve_body

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["slug"] == "prism"
