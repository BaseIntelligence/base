"""T16 — observability of submit/review failure reasons (TDD RED->GREEN).

Three proven defects are covered here:

(a) Nonce-burn-before-flag ordering: with ``public_submissions_enabled`` OFF the
    submit route must return 404 *before* the nonce is consumed, so a legitimate
    retry once the flag is re-enabled succeeds (NOT 409 "nonce already used").
(b) Silent signature-failure collapse: a rejected signature must emit a structured
    log record AND return 401, WITHOUT leaking the signature/secret/shared_token.
(c) No structured logging at review held/rejected transitions: a rejected review
    transition must emit a log record (reasons are already persisted in the DB;
    we only assert added log visibility, not what is persisted).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings

TERMINAL_STATES = {"completed", "failed", "rejected", "held"}

# A distinctive signature value that must NEVER appear in any log output.
LEAK_SENTINEL = "LEAKCANARY_signature_value_must_not_be_logged_0xdeadbeef"

# Syntactically invalid Python -> static review rejects the submission.
INVALID_CODE = "def build_model(ctx):\n    return (((\n"


def _settings(tmp_path: Path) -> PrismSettings:
    db_path = tmp_path / "obs.sqlite3"
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="secret",
        allow_insecure_signatures=True,
        validator_hotkeys=("val-a", "val-b"),
        plagiarism_enabled=False,
        fineweb_sample_count=4,
    )


def test_flag_off_returns_404_without_consuming_nonce_then_retry_succeeds(tmp_path):
    """Defect (a): flag OFF -> 404 "submission route disabled" AND the nonce is
    NOT consumed; re-enabling the flag + retrying the SAME nonce succeeds (200),
    proving the flag check runs BEFORE the nonce is burned (no misleading 409)."""
    settings = _settings(tmp_path)
    payload = {"code": VALID_CODE, "filename": "model.py", "metadata": {}}
    body = json.dumps(payload).encode()
    nonce = "flag-retry-nonce"
    with TestClient(create_app(settings)) as client:
        # Flag OFF on the live app.
        client.app.state.settings.public_submissions_enabled = False
        headers = {
            **signed_headers("secret", body, hotkey="miner-1", nonce=nonce),
            "Content-Type": "application/json",
        }
        r1 = client.post("/v1/submissions", content=body, headers=headers)
        assert r1.status_code == 404, r1.text
        assert r1.json()["detail"] == "submission route disabled"

        # Re-enable the flag and retry with the SAME nonce -> must succeed, NOT 409.
        client.app.state.settings.public_submissions_enabled = True
        headers2 = {
            **signed_headers("secret", body, hotkey="miner-1", nonce=nonce),
            "Content-Type": "application/json",
        }
        r2 = client.post("/v1/submissions", content=body, headers=headers2)
        assert r2.status_code == 200, (
            f"retry with same nonce should succeed (nonce not burned), got "
            f"{r2.status_code}: {r2.text}"
        )


def test_signature_failure_logs_record_and_does_not_leak_secret(tmp_path, caplog):
    """Defect (b): a bad signature -> 401 AND a structured log record is emitted,
    but the signature/secret value must NOT appear in the logs."""
    settings = _settings(tmp_path)
    payload = {"code": VALID_CODE, "filename": "model.py", "metadata": {}}
    body = json.dumps(payload).encode()
    # Valid timestamp + non-validator hotkey so we reach the signature check, then
    # override the signature with a clearly-invalid sentinel value.
    headers = {
        **signed_headers("secret", body, hotkey="miner-1", nonce="badsig-n1"),
        "Content-Type": "application/json",
    }
    headers["X-Signature"] = LEAK_SENTINEL
    with TestClient(create_app(settings)) as client:
        with caplog.at_level(logging.DEBUG, logger="prism_challenge.auth"):
            resp = client.post("/v1/submissions", content=body, headers=headers)
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "invalid signature"
    # A log record was emitted on the rejection path.
    assert any("signature" in rec.getMessage().lower() for rec in caplog.records), (
        f"expected a signature-failure log record, got: {caplog.text!r}"
    )
    # And it must NOT leak the signature / secret material.
    assert LEAK_SENTINEL not in caplog.text, "signature value leaked into logs!"
    assert "secret" not in caplog.text, "shared_token value leaked into logs!"


def _drive_to_terminal(client: TestClient, submission_id: str) -> str:
    for _ in range(25):
        resp = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert resp.status_code == 200, resp.text
        status = client.get(f"/v1/submissions/{submission_id}").json()["status"]
        if status in TERMINAL_STATES:
            return str(status)
    raise AssertionError("submission never reached a terminal state")


def test_review_rejected_transition_emits_log_record(tmp_path, caplog):
    """Defect (c): a rejected review transition emits a structured log record
    (reason is already persisted in submissions.error; we assert log visibility)."""
    settings = _settings(tmp_path)
    payload = {
        "code": INVALID_CODE,
        "filename": "model.py",
        "metadata": {"execution_mode": "gpu_proxy_eval"},
    }
    body = json.dumps(payload).encode()
    headers = {
        **signed_headers("secret", body, hotkey="miner-1", nonce="reject-n1"),
        "Content-Type": "application/json",
    }
    with TestClient(create_app(settings)) as client:
        resp = client.post("/v1/submissions", content=body, headers=headers)
        assert resp.status_code == 200, resp.text
        submission_id = str(resp.json()["id"])
        with caplog.at_level(logging.WARNING, logger="prism_challenge.queue"):
            status = _drive_to_terminal(client, submission_id)
    assert status == "rejected", f"expected rejected, got {status}"
    assert any(
        "reject" in rec.getMessage().lower() and submission_id in rec.getMessage()
        for rec in caplog.records
    ), f"expected a rejected-transition log record for {submission_id}, got: {caplog.text!r}"
