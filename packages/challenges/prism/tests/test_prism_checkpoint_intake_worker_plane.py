"""Checkpoint intake trust boundary + resume-ref flow survive the worker plane (VAL-PRISM-025).

(a) The checkpoint publish intake (`POST /internal/v1/checkpoints`) stays validator-permit gated
    even with the worker plane ON: a worker-bound / non-permitted hotkey is rejected 403 and records
    no checkpoint ref (worker-plane enablement does not widen the permit set).
(b) A submission that HAS a validator-published checkpoint still exposes `resume_checkpoint_ref` in
    its `GET /internal/v1/work_units` payload so a reassigned executor resumes; a fresh unit carries
    none.

Offline: dev (hmac) signatures via ``allow_insecure_signatures`` (no bittensor keys); no GPU.
"""

from __future__ import annotations

import base64
import hmac
import json
import sqlite3
import time
from hashlib import sha256
from pathlib import Path

import anyio
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.auth import canonical_checkpoint_message
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.coordination import RESUME_CHECKPOINT_PAYLOAD_KEY
from prism_challenge.models import SubmissionCreate

INTERNAL_TOKEN = "secret"
VALIDATOR_HOTKEY = "hk-validator"
WORKER_HOTKEY = "hk-worker"


def _settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'ckpt.sqlite3'}",
        shared_token=INTERNAL_TOKEN,
        allow_insecure_signatures=True,
        validator_hotkeys=[VALIDATOR_HOTKEY],
        # Worker plane ON: the trust boundary must NOT widen.
        worker_plane=WorkerPlaneConfig(enabled=True),
    )


def _bundle() -> str:
    code = "def build_model(ctx):\n    return None\n"
    return base64.b64encode(code.encode()).decode("ascii")


def _signed_checkpoint_headers(body: bytes, hotkey: str, nonce: str = "cn1") -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = canonical_checkpoint_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(INTERNAL_TOKEN.encode(), message, sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


def _checkpoint_body(submission_id: str) -> bytes:
    payload = {
        "submission_id": submission_id,
        "attempt": 1,
        "files": {"model.pt": base64.b64encode(b"weights").decode("ascii")},
    }
    return json.dumps(payload).encode("utf-8")


def _assignment_rows(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM evaluation_assignments WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


# --- VAL-PRISM-025(a): worker hotkeys cannot publish checkpoints ----------------------------------


def test_worker_hotkey_cannot_publish_checkpoint(tmp_path) -> None:
    settings = _settings(tmp_path)
    db_path = tmp_path / "ckpt.sqlite3"
    submission_id = "sub-ckpt"
    with TestClient(create_app(settings)) as client:
        body = _checkpoint_body(submission_id)
        # A correctly-signed upload from a non-validator (worker-bound) hotkey is rejected 403.
        resp = client.post(
            "/internal/v1/checkpoints",
            content=body,
            headers=_signed_checkpoint_headers(body, WORKER_HOTKEY),
        )
        assert resp.status_code == 403, resp.text
        assert "not an eligible validator" in resp.text
        # No checkpoint ref was recorded (the publish body was never reached).
        assert _assignment_rows(db_path, submission_id) == 0


def test_validator_hotkey_still_publishes_checkpoint(tmp_path) -> None:
    from prism_challenge.evaluator.checkpoint_publisher import MockCheckpointPublisher

    settings = _settings(tmp_path)
    app = create_app(settings, checkpoint_publisher=MockCheckpointPublisher())
    with TestClient(app) as client:
        body = _checkpoint_body("sub-ok")
        resp = client.post(
            "/internal/v1/checkpoints",
            content=body,
            headers=_signed_checkpoint_headers(body, VALIDATOR_HOTKEY),
        )
        # The permitted validator passes the gate (mock publisher publishes; ref returned).
        assert resp.status_code == 200, resp.text
        assert resp.json()["checkpoint_ref"]


# --- VAL-PRISM-025(b): resume_checkpoint_ref reaches reassigned units -----------------------------


def test_resume_checkpoint_ref_flows_to_reassigned_unit(tmp_path) -> None:
    settings = _settings(tmp_path)
    headers = {"Authorization": f"Bearer {INTERNAL_TOKEN}"}
    with TestClient(create_app(settings)) as client:
        app = client.app

        async def _seed() -> tuple[str, str]:
            repo = app.state.repository
            resumed = await repo.create_submission(
                "hk-a", SubmissionCreate(code=_bundle(), filename="a.py")
            )
            fresh = await repo.create_submission(
                "hk-b", SubmissionCreate(code=_bundle(), filename="b.py")
            )
            # A prior validator attempt published a checkpoint for the resumed submission.
            await repo.record_published_checkpoint(
                submission_id=resumed.id,
                attempt=1,
                validator_hotkey=VALIDATOR_HOTKEY,
                checkpoint_ref="hf://prism/resume@rev1",
            )
            return resumed.id, fresh.id

        resumed_id, fresh_id = anyio.run(_seed)

        listed = client.get("/internal/v1/work_units", headers=headers)
        assert listed.status_code == 200, listed.text
        by_submission = {
            unit["submission_id"]: unit
            for unit in listed.json()["work_units"]
            if not unit.get("audit")
        }
        # The reassigned unit carries the resume ref; the fresh first-attempt unit carries none.
        assert (
            by_submission[resumed_id]["payload"][RESUME_CHECKPOINT_PAYLOAD_KEY]
            == "hf://prism/resume@rev1"
        )
        assert RESUME_CHECKPOINT_PAYLOAD_KEY not in by_submission[fresh_id]["payload"]
