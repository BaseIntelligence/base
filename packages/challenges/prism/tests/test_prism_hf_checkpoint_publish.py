"""Validator checkpoint upload / HF-publish endpoint + cadence + push client.

Covers VAL-PRISM-021 (HF token resolution + redaction), VAL-PRISM-022 (published checkpoint_ref
recorded on the assignment) and VAL-PRISM-038 (the upload endpoint is hotkey-signed + validator-
permit gated; unsigned/forged/ineligible are rejected with no checkpoint_ref recorded). The HF
publisher is the in-memory mock (no real network).
"""

from __future__ import annotations

import base64
import hmac
import json
import sqlite3
import time
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.auth import canonical_checkpoint_message
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.checkpoint_publisher import MockCheckpointPublisher
from prism_challenge.evaluator.checkpoint_push import (
    CheckpointCadence,
    CheckpointPushClient,
    CheckpointPushError,
    DevHmacCheckpointSigner,
)

ELIGIBLE_HOTKEY = "val-a"


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "prism.sqlite3"


def _settings(
    tmp_path: Path,
    *,
    validator_hotkeys: tuple[str, ...] = (ELIGIBLE_HOTKEY,),
    allow_insecure: bool = True,
) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{_db_path(tmp_path)}",
        shared_token="secret",
        allow_insecure_signatures=allow_insecure,
        validator_hotkeys=validator_hotkeys,
        distributed_contract_policy="off",
    )


def _upload_body(
    *, submission_id: str = "sub-1", attempt: int = 1, files: dict[str, str] | None = None
) -> bytes:
    payload = {
        "submission_id": submission_id,
        "attempt": attempt,
        "files": files or {"model.pt": base64.b64encode(b"trained-weights").decode("ascii")},
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def _dev_headers(
    secret: str,
    body: bytes,
    *,
    hotkey: str = ELIGIBLE_HOTKEY,
    nonce: str = "ckpt-nonce-1",
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = canonical_checkpoint_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(secret.encode(), message, sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


def _recorded_checkpoint_ref(tmp_path: Path, submission_id: str) -> str | None:
    conn = sqlite3.connect(_db_path(tmp_path))
    try:
        row = conn.execute(
            "SELECT checkpoint_ref FROM evaluation_assignments WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


# --- VAL-PRISM-021: HF token resolves only from a secret file / env, never logged ----------------


def test_hf_token_from_prism_env(monkeypatch):
    monkeypatch.setenv("PRISM_HF_TOKEN", "tok-prism")
    assert PrismSettings().hf_token_value() == "tok-prism"


def test_hf_token_from_hf_token_env(monkeypatch):
    monkeypatch.delenv("PRISM_HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "tok-hf")
    assert PrismSettings().hf_token_value() == "tok-hf"


def test_hf_token_from_secret_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISM_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    secret = tmp_path / "hf_token"
    secret.write_text("tok-from-file\n", encoding="utf-8")
    assert PrismSettings(hf_token_file=secret).hf_token_value() == "tok-from-file"


def test_hf_token_absent_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.delenv("PRISM_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    settings = PrismSettings(hf_token=None, hf_token_file=tmp_path / "missing")
    assert settings.hf_token_value() is None


def test_hf_token_is_redacted_in_repr(monkeypatch):
    monkeypatch.setenv("PRISM_HF_TOKEN", "super-secret-token-xyz")
    settings = PrismSettings()
    assert "super-secret-token-xyz" not in repr(settings)


# --- VAL-PRISM-038 / 022: signed + permit-gated upload is published and the ref is recorded -------


def test_signed_eligible_upload_published_and_ref_recorded(tmp_path):
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-pub")
        response = client.post(
            "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["checkpoint_ref"].startswith(mock.repo_id + "@")
        assert "model.pt" in data["files"]

    # VAL-PRISM-038: the (mock) publisher was actually invoked, with no real network.
    assert mock.call_count == 1
    assert mock.uploads[0].files == ("model.pt",)
    # VAL-PRISM-022: the published ref is recorded on the submission's assignment for resume.
    assert _recorded_checkpoint_ref(tmp_path, "sub-pub") == data["checkpoint_ref"]


def test_unsigned_upload_rejected_and_nothing_recorded(tmp_path):
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-unsigned")
        headers = _dev_headers("secret", body)
        headers["X-Signature"] = "not-a-real-signature"
        response = client.post("/internal/v1/checkpoints", content=body, headers=headers)
        assert response.status_code == 401, response.text

    assert mock.call_count == 0
    assert _recorded_checkpoint_ref(tmp_path, "sub-unsigned") is None


def test_forged_signature_rejected_and_nothing_recorded(tmp_path):
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-forged")
        # A correctly-shaped HMAC computed with the WRONG secret: a forged signature.
        forged = _dev_headers("wrong-secret", body)
        response = client.post("/internal/v1/checkpoints", content=body, headers=forged)
        assert response.status_code == 401, response.text

    assert mock.call_count == 0
    assert _recorded_checkpoint_ref(tmp_path, "sub-forged") is None


def test_signed_but_ineligible_hotkey_rejected_403(tmp_path):
    mock = MockCheckpointPublisher()
    settings = _settings(tmp_path, validator_hotkeys=(ELIGIBLE_HOTKEY,))
    with TestClient(create_app(settings, checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-ineligible")
        headers = _dev_headers("secret", body, hotkey="intruder-hotkey")
        response = client.post("/internal/v1/checkpoints", content=body, headers=headers)
        assert response.status_code == 403, response.text

    assert mock.call_count == 0
    assert _recorded_checkpoint_ref(tmp_path, "sub-ineligible") is None


def test_replayed_nonce_rejected_409(tmp_path):
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-replay")
        headers = _dev_headers("secret", body, nonce="reused-nonce")
        first = client.post("/internal/v1/checkpoints", content=body, headers=headers)
        assert first.status_code == 200, first.text
        # Same nonce reused -> replay guard.
        second_body = _upload_body(submission_id="sub-replay", attempt=2)
        replay_headers = _dev_headers("secret", second_body, nonce="reused-nonce")
        replay_headers["X-Nonce"] = "reused-nonce"
        # Re-sign with the reused nonce over the new body.
        message = canonical_checkpoint_message(
            hotkey=ELIGIBLE_HOTKEY,
            nonce="reused-nonce",
            timestamp=replay_headers["X-Timestamp"],
            body=second_body,
        )
        replay_headers["X-Signature"] = hmac.new(b"secret", message, sha256).hexdigest()
        second = client.post(
            "/internal/v1/checkpoints", content=second_body, headers=replay_headers
        )
        assert second.status_code == 409, second.text


def test_malformed_payload_rejected_400(tmp_path):
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = b"{not json"
        response = client.post(
            "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
        )
        assert response.status_code == 400, response.text
    assert mock.call_count == 0


# --- VAL-PRISM-038: production sr25519 hotkey signatures (insecure dev path disabled) -------------


def test_real_keypair_eligible_upload_accepted(tmp_path):
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    keypair = keypair_from_uri("//Alice")
    hotkey = keypair.ss58_address
    mock = MockCheckpointPublisher()
    settings = _settings(tmp_path, validator_hotkeys=(hotkey,), allow_insecure=False)
    with TestClient(create_app(settings, checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="real-ok")
        nonce = "real-ok-nonce"
        timestamp = str(int(time.time()))
        message = canonical_checkpoint_message(
            hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
        )
        response = client.post(
            "/internal/v1/checkpoints",
            content=body,
            headers={
                "X-Hotkey": hotkey,
                "X-Signature": keypair.sign(message).hex(),
                "X-Nonce": nonce,
                "X-Timestamp": timestamp,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200, response.text
    assert mock.call_count == 1
    assert _recorded_checkpoint_ref(tmp_path, "real-ok") is not None


def test_real_keypair_forged_upload_rejected(tmp_path):
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    alice = keypair_from_uri("//Alice")
    bob = keypair_from_uri("//Bob")
    hotkey = alice.ss58_address
    mock = MockCheckpointPublisher()
    settings = _settings(tmp_path, validator_hotkeys=(hotkey,), allow_insecure=False)
    with TestClient(create_app(settings, checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="real-forge")
        nonce = "real-forge-nonce"
        timestamp = str(int(time.time()))
        message = canonical_checkpoint_message(
            hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
        )
        response = client.post(
            "/internal/v1/checkpoints",
            content=body,
            headers={
                "X-Hotkey": hotkey,
                "X-Signature": bob.sign(message).hex(),
                "X-Nonce": nonce,
                "X-Timestamp": timestamp,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 401, response.text
    assert mock.call_count == 0
    assert _recorded_checkpoint_ref(tmp_path, "real-forge") is None


def test_real_keypair_unpermitted_upload_rejected_403(tmp_path):
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    keypair = keypair_from_uri("//Alice")
    hotkey = keypair.ss58_address
    mock = MockCheckpointPublisher()
    # Correctly signed by Alice, but Alice holds no validator permit (empty permit set).
    settings = _settings(tmp_path, validator_hotkeys=(), allow_insecure=False)
    with TestClient(create_app(settings, checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="real-noperm")
        nonce = "real-noperm-nonce"
        timestamp = str(int(time.time()))
        message = canonical_checkpoint_message(
            hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
        )
        response = client.post(
            "/internal/v1/checkpoints",
            content=body,
            headers={
                "X-Hotkey": hotkey,
                "X-Signature": keypair.sign(message).hex(),
                "X-Nonce": nonce,
                "X-Timestamp": timestamp,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 403, response.text
    assert mock.call_count == 0
    assert _recorded_checkpoint_ref(tmp_path, "real-noperm") is None


# --- configurable cadence (hourly default) -------------------------------------------------------


def test_checkpoint_cadence_due_threshold():
    cadence = CheckpointCadence(interval_seconds=3600)
    assert cadence.due(elapsed_seconds=3600) is True
    assert cadence.due(elapsed_seconds=3599.999) is False
    assert cadence.due_at(last_checkpoint_at=1_000.0, now=1_000.0 + 3600) is True
    assert cadence.due_at(last_checkpoint_at=1_000.0, now=1_500.0) is False


def test_checkpoint_cadence_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        CheckpointCadence(interval_seconds=0)


def test_cadence_from_settings_defaults_hourly():
    assert (
        CheckpointCadence(
            interval_seconds=PrismSettings().checkpoint_cadence_seconds
        ).interval_seconds
        == 3600
    )


# --- validator-side push client round-trips through the master endpoint ---------------------------


async def test_push_client_round_trips_and_records_ref(tmp_path):
    mock = MockCheckpointPublisher()
    app = create_app(_settings(tmp_path), checkpoint_publisher=mock)
    await app.state.database.init()
    transport = httpx.ASGITransport(app=app)
    signer = DevHmacCheckpointSigner(hotkey=ELIGIBLE_HOTKEY, secret="secret")
    client = CheckpointPushClient("http://prism", signer, transport=transport)

    result = await client.push(
        submission_id="sub-push", attempt=2, files={"model.pt": b"weights-bytes"}
    )

    assert result["checkpoint_ref"].startswith(mock.repo_id + "@")
    assert mock.call_count == 1
    ref = await app.state.repository.latest_checkpoint_ref("sub-push")
    assert ref == result["checkpoint_ref"]


async def test_push_client_ineligible_hotkey_raises_403(tmp_path):
    mock = MockCheckpointPublisher()
    app = create_app(
        _settings(tmp_path, validator_hotkeys=(ELIGIBLE_HOTKEY,)), checkpoint_publisher=mock
    )
    await app.state.database.init()
    transport = httpx.ASGITransport(app=app)
    signer = DevHmacCheckpointSigner(hotkey="intruder", secret="secret")
    client = CheckpointPushClient("http://prism", signer, transport=transport)

    with pytest.raises(CheckpointPushError) as excinfo:
        await client.push(submission_id="sub-bad", attempt=1, files={"model.pt": b"x"})

    assert excinfo.value.status_code == 403
    assert mock.call_count == 0
    assert await app.state.repository.latest_checkpoint_ref("sub-bad") is None


# --- Publish observability + fail-closed success (HF configured requires checkpoint_ref) ------


class _FailingPublisher:
    """Publisher that always raises (simulates HF Hub failure; no network)."""

    repo_id = "BaseIntelligence/top-prism-architecture"

    def publish(self, upload):  # noqa: ANN001
        raise RuntimeError("simulated hub upload failure")

    def download(self, checkpoint_ref: str, dest_dir: Path) -> Path:
        raise NotImplementedError


def test_publish_failure_records_no_checkpoint_ref_and_exposes_last_error(tmp_path):
    """Failing publisher: no ref persisted; last_error is set on intake."""
    from prism_challenge.evaluator.checkpoint_intake import CheckpointIntakeService

    failing = _FailingPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=failing)) as client:  # type: ignore[arg-type]
        body = _upload_body(submission_id="sub-fail-pub")
        response = client.post(
            "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
        )
        assert response.status_code >= 400, response.text
        intake = client.app.state.checkpoint_intake
        assert isinstance(intake, CheckpointIntakeService)
        assert intake.last_error is not None
        assert "simulated hub upload failure" in intake.last_error

    assert _recorded_checkpoint_ref(tmp_path, "sub-fail-pub") is None


def test_publish_success_exposes_status_and_checkpoint_ref(tmp_path):
    """Mock publish success: status=success and checkpoint_ref observable."""
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-ok-obs")
        response = client.post(
            "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["checkpoint_ref"].startswith(mock.repo_id + "@")
        assert data.get("status") == "success"
        intake = client.app.state.checkpoint_intake
        assert intake.last_error is None
        assert intake.last_status == "success"
        assert intake.last_checkpoint_ref == data["checkpoint_ref"]

    assert _recorded_checkpoint_ref(tmp_path, "sub-ok-obs") == data["checkpoint_ref"]


def test_training_publish_complete_fail_closed_when_hf_configured():
    """When HF token is configured, completion without checkpoint_ref is incomplete."""
    from prism_challenge.evaluator.checkpoint_intake import is_training_publish_complete

    assert is_training_publish_complete(hf_token_configured=True, checkpoint_ref=None) is False
    assert (
        is_training_publish_complete(
            hf_token_configured=True,
            checkpoint_ref="BaseIntelligence/top-prism-architecture@rev1",
        )
        is True
    )


def test_training_publish_complete_allows_missing_ref_when_hf_disabled():
    """When HF is disabled/dev mock path, missing checkpoint_ref is not a hard incomplete."""
    from prism_challenge.evaluator.checkpoint_intake import is_training_publish_complete

    assert is_training_publish_complete(hf_token_configured=False, checkpoint_ref=None) is True


def test_mock_publisher_path_still_works_offline(tmp_path, monkeypatch):
    """Dev mock path remains offline-safe and still records a checkpoint_ref."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    mock = MockCheckpointPublisher()
    with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
        body = _upload_body(submission_id="sub-mock-offline")
        response = client.post(
            "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
        )
        assert response.status_code == 200, response.text
        assert response.json()["checkpoint_ref"]
    assert mock.call_count == 1
    assert _recorded_checkpoint_ref(tmp_path, "sub-mock-offline") is not None


def test_publish_success_and_failure_emit_structured_log_fields(tmp_path, caplog):
    """Structured logs on success/failure include repo_id + submission_id and never tokens."""
    import logging

    mock = MockCheckpointPublisher()
    with caplog.at_level(logging.INFO, logger="prism_challenge.evaluator.checkpoint_intake"):
        with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=mock)) as client:
            body = _upload_body(submission_id="sub-log-ok")
            ok = client.post(
                "/internal/v1/checkpoints", content=body, headers=_dev_headers("secret", body)
            )
            assert ok.status_code == 200, ok.text

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "sub-log-ok" in joined
    assert mock.repo_id in joined or "checkpoint publish" in joined.lower()
    assert "secret" not in joined  # shared_token must never appear

    failing = _FailingPublisher()
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="prism_challenge.evaluator.checkpoint_intake"):
        with TestClient(create_app(_settings(tmp_path), checkpoint_publisher=failing)) as client:  # type: ignore[arg-type]
            body = _upload_body(submission_id="sub-log-fail")
            bad = client.post(
                "/internal/v1/checkpoints",
                content=body,
                headers=_dev_headers("secret", body, nonce="ckpt-nonce-fail-log"),
            )
            assert bad.status_code >= 400, bad.text

    fail_joined = " ".join(r.getMessage() for r in caplog.records)
    assert "sub-log-fail" in fail_joined
    assert "secret" not in fail_joined
