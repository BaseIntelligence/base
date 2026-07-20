"""Worker-plane regression tests: scoring + read surfaces are unchanged ON vs OFF.

VAL-PRISM-016: a fixture manifest finalized with ``worker_plane.enabled`` ON yields the exact same
``final_score`` as with the flag OFF (proof/plausibility/audit/admission never touch scoring).
VAL-PRISM-022: ``get_weights``/``leaderboard``/``epochs`` surfaces are byte-identical for an
identical database state whether the worker plane is ON or OFF (modulo the WeightsResponse epoch
timestamp, which is not compared here).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from base.challenge_sdk.executor import DockerRunResult
from conftest import VALID_CODE, signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig

_INTERNAL_HEADERS = {"Authorization": "Bearer secret", "X-Base-Challenge-Slug": "prism"}


def _fake_run(self: Any, spec: Any, timeout_seconds: Any) -> DockerRunResult:
    artifact_dir = next(mount.source for mount in spec.mounts if mount.target == "/artifacts")
    # Held-out primary required so crowning / get_weights stay non-empty under reslab invert.
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
    (artifact_dir / "prism_run_manifest.v2.json").write_text(json.dumps(manifest), encoding="utf-8")
    return DockerRunResult(container_name="prism-eval", stdout="", stderr="", returncode=0)


def _settings(db_url: str, *, enabled: bool) -> PrismSettings:
    return PrismSettings(
        database_url=db_url,
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
        worker_plane=WorkerPlaneConfig(enabled=enabled),
    )


def _submit_and_process(client: TestClient) -> str:
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
        "/internal/v1/worker/process-next", headers={"Authorization": "Bearer secret"}
    )
    assert process.status_code == 200, process.text
    assert process.json()["submission_id"] == submission_id
    return submission_id


def _read_surfaces(client: TestClient) -> dict[str, Any]:
    weights = client.get("/internal/v1/get_weights", headers=_INTERNAL_HEADERS).json()["weights"]
    leaderboard = client.get("/v1/leaderboard").json()
    epochs = client.get("/v1/epochs").json()
    current = client.get("/v1/epochs/current").json()
    return {
        "weights": weights,
        "leaderboard": leaderboard,
        "epochs": epochs,
        "current": current,
    }


def test_final_score_identical_worker_plane_on_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)

    on_url = f"sqlite+aiosqlite:///{tmp_path / 'on.sqlite3'}"
    with TestClient(create_app(_settings(on_url, enabled=True))) as client:
        sub_id = _submit_and_process(client)
        on_status = client.get(f"/v1/submissions/{sub_id}").json()

    off_url = f"sqlite+aiosqlite:///{tmp_path / 'off.sqlite3'}"
    with TestClient(create_app(_settings(off_url, enabled=False))) as client:
        sub_id = _submit_and_process(client)
        off_status = client.get(f"/v1/submissions/{sub_id}").json()

    assert on_status["status"] == "completed"
    assert off_status["status"] == "completed"
    assert on_status["final_score"] == off_status["final_score"]


def test_read_surfaces_identical_worker_plane_on_off(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", _fake_run)
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'shared.sqlite3'}"

    # Author a crowned-architecture state under the flag ON, then read every surface.
    with TestClient(create_app(_settings(db_url, enabled=True))) as client:
        _submit_and_process(client)
        on_surfaces = _read_surfaces(client)

    # Read the SAME database state under the flag OFF: the worker plane never touches these reads.
    with TestClient(create_app(_settings(db_url, enabled=False))) as client:
        off_surfaces = _read_surfaces(client)

    assert on_surfaces == off_surfaces
    assert on_surfaces["weights"] != {}  # a crowned architecture is paid
    assert on_surfaces["leaderboard"]["entries"]  # non-empty leaderboard


def test_read_surfaces_identical_worker_plane_on_off_burn(tmp_path: Path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'burn.sqlite3'}"

    with TestClient(create_app(_settings(db_url, enabled=True))) as client:
        on_surfaces = _read_surfaces(client)

    with TestClient(create_app(_settings(db_url, enabled=False))) as client:
        off_surfaces = _read_surfaces(client)

    # BURN state: no positive q_arch_best => empty weights, empty leaderboard, identical ON/OFF.
    assert on_surfaces == off_surfaces
    assert on_surfaces["weights"] == {}
    assert on_surfaces["leaderboard"]["entries"] == []
