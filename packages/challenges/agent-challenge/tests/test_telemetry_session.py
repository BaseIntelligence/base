"""RED contract: telemetry-session handshake before progress ingest.

POST /evaluation/v1/runs/{eval_run_id}/telemetry-session
Auth: Authorization: Bearer <EVAL_RUN_TOKEN> vs EvalRun.token_sha256
Body: schema_version=1, eval_run_id, instance_id, hotkey_ss58, nonce, timestamp, signature
Signature over: ac-telemetry-session:v1|{eval_run_id}|{instance_id}|{nonce}|{timestamp}
Response: {session_id, expires_at}

Server receives hotkey_ss58 + signature only — never a mnemonic.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import AgentSubmission, EvalRun, TaskLogEvent
from agent_challenge.evaluation.authorization import load_eval_run_plan
from agent_challenge.evaluation.plan_scoring import canonical_eval_plan_json

TOKEN = "telemetry-session-token"
SESSION_PATH = "/evaluation/v1/runs/{eval_run_id}/telemetry-session"
PROGRESS_PATH = "/evaluation/v1/runs/{eval_run_id}/progress"
TELEMETRY_HEADER = "X-Telemetry-Session"
AGENT_HASH = "55" * 32
COMPOSE_HASH = "ab" * 32
PACKAGE_TREE_SHA = "bb" * 32
CANONICAL_PREFIX = "ac-telemetry-session:v1"
TEST_MNEMONIC_FRAGMENT = "abandon abandon abandon"


def _plan(*, eval_run_id: str) -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": eval_run_id,
            "submission_id": f"submission-{eval_run_id}",
            "submission_version": 1,
            "authorizing_review_digest": "66" * 32,
            "agent_hash": AGENT_HASH,
            "package_tree_sha": PACKAGE_TREE_SHA,
            "selected_tasks": [
                {
                    "task_id": "task-a",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": policy,
            "scoring_policy_digest": ew.scoring_policy_digest(policy),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": {
                    "mrtd": "11" * 48,
                    "rtmr0": "22" * 48,
                    "rtmr1": "33" * 48,
                    "rtmr2": "44" * 48,
                    "os_image_hash": "cc" * 32,
                    "key_provider": "validator-kms",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
            "key_release_nonce": f"key-release-{eval_run_id}",
            "score_nonce": f"score-{eval_run_id}",
            "run_token_sha256": hashlib.sha256(TOKEN.encode("utf-8")).hexdigest(),
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


async def _seed_run(
    database_session,
    plan: dict[str, Any],
    *,
    token: str = TOKEN,
    phase: str = "eval_running",
) -> EvalRun:
    now = datetime.now(UTC)
    async with database_session() as session:
        submission_agent_hash = hashlib.sha256(plan["eval_run_id"].encode("utf-8")).hexdigest()
        submission = AgentSubmission(
            miner_hotkey=f"telemetry-miner-{plan['eval_run_id']}",
            name=f"telemetry-agent-{plan['eval_run_id']}",
            agent_hash=submission_agent_hash,
            package_tree_sha=PACKAGE_TREE_SHA,
            artifact_uri=f"/tmp/telemetry-{plan['eval_run_id']}.zip",
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        plan = {
            **plan,
            "submission_id": str(submission.id),
            "run_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        }
        plan = ew.validate_eval_plan(plan)
        run = EvalRun(
            eval_run_id=plan["eval_run_id"],
            submission_id=submission.id,
            submission_version=1,
            authorizing_review_digest="66" * 32,
            plan_json=canonical_eval_plan_json(plan),
            plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode("utf-8")).hexdigest(),
            token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            phase=phase,
            retryable=False,
            score=None,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run


def _enable_attested(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.api import routes as routes_mod

    monkeypatch.setattr(routes_mod.settings, "attested_review_enabled", True)
    monkeypatch.setattr(routes_mod.settings, "phala_attestation_enabled", True)


def _keypair():
    import bittensor as bt

    return bt.Keypair.create_from_uri("//Alice")


def _encode_sig(signature: object) -> str:
    if isinstance(signature, bytes | bytearray):
        return "0x" + bytes(signature).hex()
    text = str(signature)
    return text if text.startswith("0x") else "0x" + text


def _session_body(
    *,
    eval_run_id: str,
    instance_id: str = "cvm-instance-1",
    nonce: str = "telemetry-nonce-1",
    timestamp: str | None = None,
    keypair=None,
    tamper_signature: bool = False,
) -> dict[str, Any]:
    kp = keypair or _keypair()
    ts = timestamp or datetime.now(UTC).replace(microsecond=0).isoformat()
    canonical = f"{CANONICAL_PREFIX}|{eval_run_id}|{instance_id}|{nonce}|{ts}"
    signature = _encode_sig(kp.sign(canonical))
    if tamper_signature:
        # Flip last nibble so verify fails.
        signature = signature[:-1] + ("0" if signature[-1] != "0" else "1")
    body = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "instance_id": instance_id,
        "hotkey_ss58": kp.ss58_address,
        "nonce": nonce,
        "timestamp": ts,
        "signature": signature,
    }
    # Contract: mnemonic never leaves the runner — only ss58 + signature on the wire.
    dumped = json.dumps(body)
    assert "mnemonic" not in body  # key must never be present on happy path
    assert TEST_MNEMONIC_FRAGMENT not in dumped.lower()
    return body


async def _post_session(client, eval_run_id: str, body: dict[str, Any], *, token: str = TOKEN):
    return await client.post(
        SESSION_PATH.format(eval_run_id=eval_run_id),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        content=json.dumps(body, separators=(",", ":")),
    )


async def test_telemetry_session_happy_open(
    client, database_session, monkeypatch: pytest.MonkeyPatch
):
    """Happy open: valid Bearer + hotkey signature → 200 {session_id, expires_at}."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-telemetry-happy")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    body = _session_body(eval_run_id=plan["eval_run_id"])

    response = await _post_session(client, plan["eval_run_id"], body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) >= {"session_id", "expires_at"}
    assert isinstance(payload["session_id"], str) and payload["session_id"]
    assert payload["expires_at"]
    # Response must not echo mnemonic or raw signing material beyond session id.
    assert "mnemonic" not in payload
    assert TEST_MNEMONIC_FRAGMENT not in json.dumps(payload).lower()
    assert run.token_sha256 == hashlib.sha256(TOKEN.encode("utf-8")).hexdigest()


async def test_telemetry_session_bad_signature(
    client, database_session, monkeypatch: pytest.MonkeyPatch
):
    """Bad signature → 401."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-telemetry-badsig")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    body = _session_body(eval_run_id=plan["eval_run_id"], tamper_signature=True)

    response = await _post_session(client, plan["eval_run_id"], body)

    assert response.status_code == 401
    detail = response.json()["detail"]
    code = detail["code"] if isinstance(detail, dict) else detail
    assert code in {
        "invalid_telemetry_signature",
        "invalid_telemetry_session",
        "telemetry_signature_invalid",
    }


async def test_telemetry_session_terminal_eval_run_phase(
    client, database_session, monkeypatch: pytest.MonkeyPatch
):
    """Terminal eval_run phase (eval_accepted) → 409."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-telemetry-terminal")
    run = await _seed_run(database_session, plan, phase="eval_accepted")
    plan = load_eval_run_plan(run)
    body = _session_body(eval_run_id=plan["eval_run_id"])

    response = await _post_session(client, plan["eval_run_id"], body)

    assert response.status_code == 409
    detail = response.json()["detail"]
    code = detail["code"] if isinstance(detail, dict) else detail
    assert code in {
        "eval_run_terminal",
        "telemetry_session_forbidden",
        "eval_run_not_open",
    }


async def test_telemetry_session_expired_then_progress_rejected(
    client, database_session, monkeypatch: pytest.MonkeyPatch
):
    """Expired/closed session then progress → 401, zero TaskLogEvent rows."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-telemetry-expired")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    body = _session_body(eval_run_id=plan["eval_run_id"])

    opened = await _post_session(client, plan["eval_run_id"], body)
    assert opened.status_code == 200, opened.text
    session_id = opened.json()["session_id"]

    # Force expiry/close via the session module contract (implementation under test).
    from agent_challenge.evaluation import telemetry_session as ts_mod

    if hasattr(ts_mod, "close_telemetry_session"):
        await ts_mod.close_telemetry_session(session_id)
    elif hasattr(ts_mod, "expire_telemetry_session"):
        await ts_mod.expire_telemetry_session(session_id)
    else:
        # Pin that a close/expire API exists for the handshake lifecycle.
        raise AssertionError(
            "telemetry_session module must expose close_telemetry_session or "
            "expire_telemetry_session"
        )

    progress_body = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "task_id": "task-a",
        "sequence": 1,
        "status": "running",
        "event_type": "task.status",
        "progress": 0.1,
        "message": "should be rejected",
    }
    progress = await client.post(
        PROGRESS_PATH.format(eval_run_id=plan["eval_run_id"]),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            TELEMETRY_HEADER: session_id,
        },
        content=json.dumps(progress_body, separators=(",", ":")),
    )

    assert progress.status_code == 401
    detail = progress.json()["detail"]
    code = detail["code"] if isinstance(detail, dict) else detail
    assert code in {
        "invalid_telemetry_session",
        "telemetry_session_expired",
        "telemetry_session_closed",
        "telemetry_session_unknown",
    }
    async with database_session() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(TaskLogEvent)
            .where(TaskLogEvent.submission_id == run.submission_id)
        )
    assert int(count or 0) == 0


async def test_telemetry_session_rejects_mnemonic_in_body(
    client, database_session, monkeypatch: pytest.MonkeyPatch
):
    """Server must reject bodies that attempt to send a mnemonic (hotkey_ss58 only)."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-telemetry-no-mne")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    body = _session_body(eval_run_id=plan["eval_run_id"])
    # Obvious test mnemonic — must never be accepted as auth material.
    body["mnemonic"] = (
        "abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon about"
    )
    assert "mnemonic" in body  # deliberate injection for this negative test

    response = await _post_session(client, plan["eval_run_id"], body)

    assert response.status_code != 404, "telemetry-session route missing"
    assert response.status_code in {400, 401, 422}
    # Even on error, response must not echo the mnemonic value.
    assert TEST_MNEMONIC_FRAGMENT not in response.text.lower()
    assert run.submission_id > 0
