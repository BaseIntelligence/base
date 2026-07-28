"""S8: Prism public attestation challenge/answer roundtrip (proxy product path)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig

WORKER_KEY = "//WorkerAttestS8"
DIGEST = "sha256:" + ("11" * 32)
COMMIT = "a" * 40
TREE = "b" * 40


def _settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 's8.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=False,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY),
        # No constation_base_url → in-process SoT on prism app.state
    )


def test_s8_challenge_answer_roundtrip_on_prism(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        # Public surface exactly as published through BASE proxy under
        # /challenges/prism/v1/attestation/* (challenge app owns the path).
        ch = client.get(
            "/v1/attestation/challenge",
            params={
                "phase": "start",
                "work_unit_id": "wu-s8",
                "miner_hotkey": "hk-s8",
                "pod_id": "pod-s8",
            },
        )
        assert ch.status_code == 200, ch.text
        body = ch.json()
        assert body["nonce"]
        assert body["phase"] == "start"
        assert body["work_unit_id"] == "wu-s8"
        assert body["challenge_id"] == body["nonce"]

        ans = client.post(
            "/v1/attestation/answer",
            json={"nonce": body["nonce"], "phase": "start"},
        )
        assert ans.status_code == 200, ans.text
        assert ans.json()["status"] == "accepted"


def test_s8_inprocess_register_check_and_nonce_consume(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    headers = {"Authorization": "Bearer secret"}
    with TestClient(create_app(settings)) as client:
        reg = client.post(
            "/internal/v1/constation/register_digest",
            headers=headers,
            json={
                "commit_sha": COMMIT,
                "tree_sha": TREE,
                "variant": "cuda",
                "digest": DIGEST,
            },
        )
        assert reg.status_code == 200, reg.text

        hit = client.post(
            "/internal/v1/constation/check_allowlist",
            headers=headers,
            json={
                "digest": DIGEST,
                "commit_sha": COMMIT,
                "tree_sha": TREE,
                "variant": "cuda",
            },
        )
        assert hit.status_code == 200
        assert hit.json() == {"ok": True, "reason": "ok"}

        ch = client.get(
            "/v1/attestation/challenge",
            params={
                "phase": "interval",
                "work_unit_id": "wu-s8b",
                "miner_hotkey": "hk-s8b",
                "pod_id": "pod-s8b",
            },
        )
        nonce = ch.json()["nonce"]
        body = {
            "nonce": nonce,
            "work_unit_id": "wu-s8b",
            "miner_hotkey": "hk-s8b",
            "pod_id": "pod-s8b",
        }
        first = client.post(
            "/internal/v1/constation/check_nonce", headers=headers, json=body
        )
        second = client.post(
            "/internal/v1/constation/check_nonce", headers=headers, json=body
        )
        assert first.json() == {"ok": True, "reason": "ok"}
        assert second.json()["ok"] is False
        assert second.json()["reason"] == "already_consumed"


def test_s8_missing_binding_query_422(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        r = client.get("/v1/attestation/challenge", params={"phase": "start"})
        assert r.status_code == 422
