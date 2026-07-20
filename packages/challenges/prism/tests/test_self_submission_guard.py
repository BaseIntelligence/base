from __future__ import annotations

import json

from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings


def _client(tmp_path, **overrides):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'self_submission.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        validator_hotkeys=("val-a", "val-b"),
        plagiarism_enabled=False,
        **overrides,
    )
    return TestClient(create_app(settings))


def _submit(client: TestClient, *, hotkey: str, nonce: str):
    body = json.dumps({"code": VALID_CODE, "filename": "model.py"}, separators=(",", ":")).encode()
    return client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey=hotkey, nonce=nonce),
            "Content-Type": "application/json",
        },
    )


def test_validator_hotkey_self_submission_is_rejected(tmp_path):
    """A submission signed with hotkey == a configured validator_hotkey must be
    rejected (403). This proves the anti-self-submission integrity guard."""
    with _client(tmp_path) as client:
        response = _submit(client, hotkey="val-a", nonce="validator-self")
        assert response.status_code == 403, response.text
        assert "validator" in response.text.lower()


def test_normal_miner_submission_still_accepted(tmp_path):
    """A normal miner hotkey (not a validator) must remain accepted unchanged."""
    with _client(tmp_path) as client:
        response = _submit(client, hotkey="miner-1", nonce="miner-ok")
        assert response.status_code == 200, response.text
        assert response.json()["id"]
