"""Explicit CPU re-exec test-mode config (VAL-PRISM-013/015; mission harness seam).

Covers the opt-in ``worker_plane.cpu_reexec_test_mode`` wiring that installs the repo's own CPU
re-exec seam as configuration (no monkeypatch): the helper stages tiny locked train data, installs
``DockerExecutor.run`` -> CPU runner, authors a real ``prism_run_manifest.v2`` on CPU, normalizes
the volatile timing fields so two honest replicas of the SAME submission agree on one
``manifest_sha256``, and ``create_app`` drives it end-to-end to a recorded prequential-bpb score.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import container as _container
from prism_challenge.evaluator.cpu_test_mode import (
    TINY_ARCHITECTURE,
    TINY_TRAINING,
    VOLATILE_COMPUTE_FIELDS,
    configure_cpu_reexec_test_mode,
    evaluate_cpu_reexec,
    normalize_manifest_for_replication,
    stage_tiny_train_data,
)


@pytest.fixture(autouse=True)
def _restore_docker_run():
    """The seam is a process-wide class-attr swap; snapshot + restore so it never leaks."""

    original = _container.DockerExecutor.run
    try:
        yield
    finally:
        _container.DockerExecutor.run = original


def _settings(tmp_path: Path, *, data_dir: Path | None = None) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'cpu.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane={
            "cpu_reexec_test_mode": True,
            "cpu_reexec_train_data_dir": str(data_dir) if data_dir else None,
        },
    )


def test_configure_installs_seam_and_stages_data(tmp_path):
    settings = _settings(tmp_path)
    data_dir = configure_cpu_reexec_test_mode(settings)

    assert data_dir.is_dir()
    assert (data_dir / "train-00000.jsonl").is_file()
    # The seam is installed as an explicit config, not a test monkeypatch.
    assert _container.DockerExecutor.run.__name__ == "_run"


def test_evaluate_cpu_reexec_authors_real_manifest(tmp_path):
    settings = _settings(tmp_path)
    configure_cpu_reexec_test_mode(settings)

    outcome = evaluate_cpu_reexec(settings, submission_id="sub-1")

    assert outcome.manifest["schema_version"] == "prism_run_manifest.v2"
    assert outcome.manifest["run"]["device"] == "cpu"
    assert len(outcome.manifest_sha256) == 64
    assert outcome.artifact_dir.is_dir()


def test_normalize_drops_volatile_compute_fields():
    manifest = {"compute": {"wall_clock_seconds": 12.5, "peak_rss_bytes": 999, "world_size": 1}}
    normalized = normalize_manifest_for_replication(manifest)
    for field in VOLATILE_COMPUTE_FIELDS:
        if field in normalized["compute"]:
            assert normalized["compute"][field] == 0
    # Non-volatile fields are preserved and the input is not mutated.
    assert normalized["compute"]["world_size"] == 1
    assert manifest["compute"]["wall_clock_seconds"] == 12.5


def test_normalize_zeros_train_series_wall_and_realigns_digest():
    """Telemetry wall_s must not keep two honest replicas from sharing one hash."""
    series = {
        "schema_version": "prism_train_series.v1",
        "points": [
            {"i": 0, "wall_s": 0.0123, "train_ce_nats": 1.0, "tokens_seen": 8},
            {"i": 1, "wall_s": 0.0456, "train_ce_nats": 0.9, "tokens_seen": 16},
        ],
    }
    manifest = {
        "metrics": {
            "train_series": series,
            "train_series_sha256": "deadbeef",
        },
        "artifacts": {"train_series_sha256": "deadbeef"},
    }
    normalized = normalize_manifest_for_replication(manifest)
    points = normalized["metrics"]["train_series"]["points"]
    assert all(point["wall_s"] == 0.0 for point in points)
    assert points[0]["train_ce_nats"] == 1.0  # non-timing preserved
    digest = normalized["metrics"]["train_series_sha256"]
    assert digest != "deadbeef" and len(digest) == 64
    assert normalized["artifacts"]["train_series_sha256"] == digest
    # Input not mutated.
    assert series["points"][0]["wall_s"] == 0.0123


def test_two_replicas_of_same_submission_agree_on_hash(tmp_path):
    # Two independent worker hosts (distinct artifact roots) re-exec the SAME submission; the
    # normalized manifest hash MUST match so the base worker plane accepts (not disputes) them.
    data_dir = stage_tiny_train_data(tmp_path / "shared")
    settings_a = _settings(tmp_path / "a", data_dir=data_dir)
    settings_b = _settings(tmp_path / "b", data_dir=data_dir)
    configure_cpu_reexec_test_mode(settings_a)
    outcome_a = evaluate_cpu_reexec(settings_a, submission_id="agree-1")
    configure_cpu_reexec_test_mode(settings_b)
    outcome_b = evaluate_cpu_reexec(settings_b, submission_id="agree-1")

    assert outcome_a.manifest_sha256 == outcome_b.manifest_sha256


def test_create_app_test_mode_scores_without_monkeypatch(tmp_path):
    settings = _settings(tmp_path)
    db_path = tmp_path / "cpu.sqlite3"
    with TestClient(create_app(settings)) as client:
        payload = {
            "code": two_script_bundle(arch_code=TINY_ARCHITECTURE, train_code=TINY_TRAINING),
            "filename": "project.zip",
        }
        import json

        body = json.dumps(payload, separators=(",", ":")).encode()
        response = client.post(
            "/v1/submissions",
            content=body,
            headers={
                **signed_headers("secret", body, nonce="cpu-cfg"),
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200, response.text
        submission_id = str(response.json()["id"])
        process = client.post(
            "/internal/v1/worker/process-next", headers={"Authorization": "Bearer secret"}
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] > 0.0
