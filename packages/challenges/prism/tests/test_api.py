from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest
from base.challenge_sdk.executor import DockerRunResult
from conftest import VALID_CODE, signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings


@pytest.fixture
def small_cap_client(tmp_path) -> TestClient:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        max_code_bytes=2_000,
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _post_code(client: TestClient, code: str):
    payload = {"code": code, "filename": "model.py"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    return client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body), "Content-Type": "application/json"},
    )


def test_size_check_accepts_just_under_cap(small_cap_client):
    cap = small_cap_client.app.state.settings.max_code_bytes
    response = _post_code(small_cap_client, "A" * (cap - 1))
    assert response.status_code == 200, response.text


def test_size_check_rejects_just_over_cap(small_cap_client):
    cap = small_cap_client.app.state.settings.max_code_bytes
    response = _post_code(small_cap_client, "A" * (cap + 1))
    assert response.status_code == 413, response.text
    assert response.json()["detail"] == "submission too large"


def test_health_version_and_internal_auth(client):
    assert client.get("/health").json()["slug"] == "prism"
    capabilities = client.get("/version").json()["capabilities"]
    assert "nas" not in capabilities
    assert "challenge.scoring" in capabilities
    assert "challenge.ordinary_proof" in capabilities
    assert client.get("/internal/v1/get_weights").status_code == 401
    response = client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer secret", "X-Base-Challenge-Slug": "prism"},
    )
    assert response.status_code == 200
    assert response.json()["weights"] == {}


def test_submit_status_process_and_leaderboard(client, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        artifact_dir = next(mount.source for mount in spec.mounts if mount.target == "/artifacts")
        # Held-out primary must be present for emission crown / get_weights (VAL-RESLAB-006).
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
        (artifact_dir / "prism_run_manifest.v2.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    payload = {"code": two_script_bundle(arch_code=VALID_CODE), "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    submission_id = response.json()["id"]

    process = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert process.status_code == 200, process.text
    assert process.json()["submission_id"] == submission_id

    status = client.get(f"/v1/submissions/{submission_id}").json()
    assert status["status"] == "completed"
    assert status["final_score"] >= 0

    leaderboard = client.get("/v1/leaderboard").json()
    assert leaderboard["entries"][0]["submission_id"] == submission_id

    weights = client.get(
        "/internal/v1/get_weights",
        headers={"Authorization": "Bearer secret", "X-Base-Challenge-Slug": "prism"},
    ).json()["weights"]
    assert weights == {"hk": 1.0}


def test_rejects_bad_signature(client):
    response = client.post(
        "/v1/submissions",
        json={"code": VALID_CODE},
        headers={"X-Hotkey": "hk", "X-Signature": "bad", "X-Nonce": "x", "X-Timestamp": "1"},
    )
    assert response.status_code == 401


def _zip_submission_bytes() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("model.py", VALID_CODE)
    return stream.getvalue()


def _read_submission_row(client: TestClient, submission_id: str) -> dict:
    import anyio

    stored = client.app.state.repository

    async def read_code():
        async with stored.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT code, filename FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    return anyio.run(read_code)


def test_public_submission_accepts_raw_zip(client):
    raw = _zip_submission_bytes()
    response = client.post(
        "/v1/submissions",
        content=raw,
        headers={
            **signed_headers("secret", raw),
            "Content-Type": "application/zip",
            "X-Submission-Filename": "project.zip",
        },
    )
    assert response.status_code == 200, response.text
    submission_id = response.json()["id"]
    row = _read_submission_row(client, submission_id)
    assert row["filename"] == "project.zip"
    # Signature contract: bytes consumed by the handler must equal the bytes
    # authenticate_miner signed over (Starlette caches request.body()).
    assert base64.b64decode(row["code"]) == raw


def test_public_submission_raw_zip_signature_contract(client):
    raw = _zip_submission_bytes()
    good = client.post(
        "/v1/submissions",
        content=raw,
        headers={**signed_headers("secret", raw), "Content-Type": "application/zip"},
    )
    assert good.status_code == 200, good.text
    row = _read_submission_row(client, good.json()["id"])
    assert row["filename"] == "submission.zip"
    assert base64.b64decode(row["code"]) == raw

    bad = client.post(
        "/v1/submissions",
        content=raw,
        headers={
            "X-Hotkey": "hk",
            "X-Signature": "deadbeef",
            "X-Nonce": "n-bad",
            "X-Timestamp": signed_headers("secret", raw)["X-Timestamp"],
            "Content-Type": "application/zip",
        },
    )
    assert bad.status_code == 401, bad.text


def test_public_submission_accepts_raw_python_octet_stream(client):
    raw = VALID_CODE.encode()
    response = client.post(
        "/v1/submissions",
        content=raw,
        headers={
            **signed_headers("secret", raw),
            "Content-Type": "application/octet-stream",
            "X-Submission-Filename": "entry.py",
        },
    )
    assert response.status_code == 200, response.text
    row = _read_submission_row(client, response.json()["id"])
    assert row["filename"] == "entry.py"
    assert base64.b64decode(row["code"]) == raw


def test_public_submission_json_still_accepted(client):
    payload = {"code": VALID_CODE, "filename": "model.py"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["hotkey"] == "hk"


def test_public_submission_rejects_traversal_filename(client):
    raw = _zip_submission_bytes()
    response = client.post(
        "/v1/submissions",
        content=raw,
        headers={
            **signed_headers("secret", raw),
            "Content-Type": "application/zip",
            "X-Submission-Filename": "../escape.py",
        },
    )
    assert response.status_code == 422, response.text
    assert response.status_code != 500


def test_public_submission_malformed_json_no_500(client):
    raw = b"{not valid json"
    response = client.post(
        "/v1/submissions",
        content=raw,
        headers={**signed_headers("secret", raw), "Content-Type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert response.status_code != 500


def test_public_submission_oversized_raw_zip_413(small_cap_client):
    cap = small_cap_client.app.state.settings.max_code_bytes
    raw = b"P" * (cap + 1)
    response = small_cap_client.post(
        "/v1/submissions",
        content=raw,
        headers={**signed_headers("secret", raw), "Content-Type": "application/zip"},
    )
    assert response.status_code == 413, response.text
    assert response.json()["detail"] == "submission too large"


def test_internal_bridge_accepts_raw_zip_submission(client):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("model.py", VALID_CODE)
    raw = stream.getvalue()

    response = client.post(
        "/internal/v1/bridge/submissions",
        content=raw,
        headers={
            "Authorization": "Bearer secret",
            "X-Base-Verified-Hotkey": "hk-bridge",
            "X-Submission-Filename": "project.zip",
            "Content-Type": "application/zip",
        },
    )

    assert response.status_code == 200, response.text
    submission_id = response.json()["id"]
    status = client.get(f"/v1/submissions/{submission_id}").json()
    assert status["hotkey"] == "hk-bridge"
    stored = client.app.state.repository

    async def read_code():
        async with stored.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT code, filename FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    import anyio

    row = anyio.run(read_code)
    assert row["filename"] == "project.zip"
    assert base64.b64decode(row["code"]) == raw
