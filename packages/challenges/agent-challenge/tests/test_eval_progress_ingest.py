"""RED contract: attested mid-run progress ingest + telemetry-session gate.

Phase A: Bearer EVAL_RUN_TOKEN (sha256 vs EvalRun.token_sha256).
Phase B: telemetry-session opened with hotkey-signed nonce.
Phase C: every progress POST carries Bearer + X-Telemetry-Session.

Progress is observability-only — never mutates scores. Final
POST /evaluation/v1/runs/{id}/result remains the only score path.
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
from agent_challenge.evaluation.progress import PROGRESS_SOURCE
from agent_challenge.evaluation.task_events import SAFE_TASK_PHASE_STATUSES

TOKEN = "progress-good-token"
TELEMETRY_HEADER = "X-Telemetry-Session"
PROGRESS_PATH = "/evaluation/v1/runs/{eval_run_id}/progress"
SESSION_PATH = "/evaluation/v1/runs/{eval_run_id}/telemetry-session"
RESULT_PATH = "/evaluation/v1/runs/{eval_run_id}/result"
AGENT_HASH = "55" * 32
COMPOSE_HASH = "ab" * 32
PACKAGE_TREE_SHA = "bb" * 32


def _assert_progress_wire_present() -> None:
    """Pin the known monorepo defect: progress.py calls these missing symbols."""
    assert hasattr(ew, "EVAL_PROGRESS_PHASES"), "eval_wire.EVAL_PROGRESS_PHASES missing"
    assert hasattr(ew, "validate_eval_progress_request"), (
        "eval_wire.validate_eval_progress_request missing"
    )
    assert hasattr(ew, "validate_eval_progress_receipt"), (
        "eval_wire.validate_eval_progress_receipt missing"
    )
    assert ew.EVAL_PROGRESS_PHASES == SAFE_TASK_PHASE_STATUSES


def _plan(*, eval_run_id: str = "eval-progress-1") -> dict[str, Any]:
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
            miner_hotkey=f"progress-miner-{plan['eval_run_id']}",
            name=f"progress-agent-{plan['eval_run_id']}",
            agent_hash=submission_agent_hash,
            package_tree_sha=PACKAGE_TREE_SHA,
            artifact_uri=f"/tmp/progress-{plan['eval_run_id']}.zip",
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


def _progress_body(
    plan: dict[str, Any],
    *,
    sequence: int = 1,
    status: str = "running",
    event_type: str = "task.status",
    progress: float | None = 0.25,
    message: str | None = "task running",
    task_id: str = "task-a",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "task_id": task_id,
        "sequence": sequence,
        "status": status,
        "event_type": event_type,
        "progress": progress,
        "message": message,
    }
    if extra:
        body.update(extra)
    return body


def _enable_attested(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_challenge.api import routes as routes_mod

    monkeypatch.setattr(routes_mod.settings, "attested_review_enabled", True)
    monkeypatch.setattr(routes_mod.settings, "phala_attestation_enabled", True)


def _sign_session_body(
    *,
    eval_run_id: str,
    instance_id: str = "instance-1",
    nonce: str = "nonce-1",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a hotkey-signed telemetry-session open body (mnemonic never included)."""
    import bittensor as bt

    keypair = bt.Keypair.create_from_uri("//Alice")
    ts = timestamp or datetime.now(UTC).isoformat()
    canonical = f"ac-telemetry-session:v1|{eval_run_id}|{instance_id}|{nonce}|{ts}"
    signature = keypair.sign(canonical)
    sig_hex = (
        "0x" + bytes(signature).hex()
        if isinstance(signature, bytes | bytearray)
        else (signature if str(signature).startswith("0x") else "0x" + str(signature))
    )
    return {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "instance_id": instance_id,
        "hotkey_ss58": keypair.ss58_address,
        "nonce": nonce,
        "timestamp": ts,
        "signature": sig_hex,
    }


async def _open_session(client, eval_run_id: str, *, token: str = TOKEN) -> str:
    body = _sign_session_body(eval_run_id=eval_run_id)
    # Server must never receive a mnemonic — only hotkey_ss58 + signature.
    assert "mnemonic" not in json.dumps(body).lower()
    response = await client.post(
        SESSION_PATH.format(eval_run_id=eval_run_id),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        content=json.dumps(body, separators=(",", ":")),
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "session_id" in payload
    assert "expires_at" in payload
    return str(payload["session_id"])


async def _post_progress(
    client,
    plan: dict[str, Any],
    body: dict[str, Any],
    *,
    token: str | None = TOKEN,
    session_id: str | None = None,
):
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if session_id is not None:
        headers[TELEMETRY_HEADER] = session_id
    return await client.post(
        PROGRESS_PATH.format(eval_run_id=plan["eval_run_id"]),
        headers=headers,
        content=json.dumps(body, separators=(",", ":")),
    )


async def _count_progress_events(database_session, submission_id: int) -> int:
    async with database_session() as session:
        count = await session.scalar(
            select(func.count())
            .select_from(TaskLogEvent)
            .where(TaskLogEvent.submission_id == submission_id)
        )
    return int(count or 0)


async def test_s1_happy_records_event(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S1: open session then POST progress sequence 1..3 → 202 + TaskLogEvent rows."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s1")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)

    session_id = await _open_session(client, plan["eval_run_id"])

    for seq in (1, 2, 3):
        status = ("assigned", "starting", "running")[seq - 1]
        response = await _post_progress(
            client,
            plan,
            _progress_body(plan, sequence=seq, status=status, progress=seq / 10),
            session_id=session_id,
        )
        assert response.status_code == 202, response.text
        receipt = response.json()
        assert receipt["created"] is True
        assert receipt["sequence"] == seq
        assert receipt["eval_run_id"] == plan["eval_run_id"]
        assert receipt["task_id"] == "task-a"

    async with database_session() as session:
        rows = list(
            await session.scalars(
                select(TaskLogEvent)
                .where(TaskLogEvent.submission_id == run.submission_id)
                .order_by(TaskLogEvent.sequence)
            )
        )
        refreshed = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == plan["eval_run_id"])
        )

    assert len(rows) == 3
    client_sequences: list[int] = []
    for row in rows:
        meta = json.loads(row.metadata_json)
        assert meta["source"] == PROGRESS_SOURCE == "eval_progress"
        assert meta["eval_run_id"] == plan["eval_run_id"]
        assert isinstance(meta["client_sequence"], int)
        client_sequences.append(meta["client_sequence"])
    assert client_sequences == [1, 2, 3]
    assert refreshed is not None
    assert refreshed.score is None


async def test_s2_auth_fail(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S2: wrong/absent Bearer → 401 and zero TaskLogEvent rows."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s2-auth")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    session_id = await _open_session(client, plan["eval_run_id"])
    body = _progress_body(plan)

    missing = await _post_progress(client, plan, body, token=None, session_id=session_id)
    wrong = await _post_progress(client, plan, body, token="not-the-token", session_id=session_id)

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json()["detail"] == {"code": "invalid_eval_token"}
    assert wrong.json()["detail"] == {"code": "invalid_eval_token"}
    assert await _count_progress_events(database_session, run.submission_id) == 0


async def test_s2_missing_session(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S2: valid Bearer but no/unknown X-Telemetry-Session → 401, zero rows."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s2-session")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    body = _progress_body(plan)

    no_header = await _post_progress(client, plan, body, session_id=None)
    unknown = await _post_progress(client, plan, body, session_id="sess-does-not-exist")

    assert no_header.status_code == 401
    assert unknown.status_code == 401
    for response in (no_header, unknown):
        detail = response.json()["detail"]
        code = detail["code"] if isinstance(detail, dict) else detail
        assert code in {
            "invalid_telemetry_session",
            "telemetry_session_required",
            "telemetry_session_unknown",
            "telemetry_session_expired",
        }
    assert await _count_progress_events(database_session, run.submission_id) == 0


async def test_s3_unknown_phase(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S3: status not in safe phases → 422."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s3")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    session_id = await _open_session(client, plan["eval_run_id"])

    response = await _post_progress(
        client,
        plan,
        _progress_body(plan, status="scoring"),
        session_id=session_id,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] in {
        "progress_invalid",
        "progress_phase_invalid",
    }
    assert "scoring" not in SAFE_TASK_PHASE_STATUSES
    assert "scoring" not in ew.EVAL_PROGRESS_PHASES
    assert await _count_progress_events(database_session, run.submission_id) == 0


async def test_s4_score_forbidden(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S4: body containing any score field → 422, zero rows, score stays None."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s4")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    session_id = await _open_session(client, plan["eval_run_id"])

    response = await _post_progress(
        client,
        plan,
        _progress_body(plan, extra={"score": 0.99, "score_record": {"score": 0.99}}),
        session_id=session_id,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] in {
        "progress_score_forbidden",
        "progress_invalid",
    }
    assert await _count_progress_events(database_session, run.submission_id) == 0
    async with database_session() as session:
        refreshed = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == plan["eval_run_id"])
        )
    assert refreshed is not None
    assert refreshed.score is None


async def test_s5_idempotent(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S5: replay same (eval_run_id, task_id, client_sequence) → 200, no duplicate row."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s5-idem")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    session_id = await _open_session(client, plan["eval_run_id"])
    body = _progress_body(plan, sequence=1, status="running", progress=0.4)

    first = await _post_progress(client, plan, body, session_id=session_id)
    second = await _post_progress(client, plan, body, session_id=session_id)

    assert first.status_code == 202, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()
    assert first_body["created"] is True
    assert second_body["created"] is False
    assert first_body["sequence"] == second_body["sequence"] == 1
    assert first_body["event_id"] == second_body["event_id"]
    assert await _count_progress_events(database_session, run.submission_id) == 1


async def test_s5_sequence_regression(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """S5: a lower sequence after a higher one → 422."""
    _assert_progress_wire_present()
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-s5-seq")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)
    session_id = await _open_session(client, plan["eval_run_id"])

    high = await _post_progress(
        client,
        plan,
        _progress_body(plan, sequence=5, status="running"),
        session_id=session_id,
    )
    low = await _post_progress(
        client,
        plan,
        _progress_body(plan, sequence=2, status="starting"),
        session_id=session_id,
    )

    assert high.status_code == 202, high.text
    assert low.status_code == 422
    assert low.json()["detail"]["code"] in {
        "progress_sequence_stale",
        "progress_invalid",
    }
    assert await _count_progress_events(database_session, run.submission_id) == 1


async def test_result_still_scores(client, database_session, monkeypatch: pytest.MonkeyPatch):
    """Adjacent: POST .../result still authenticates; only score-mutating path."""
    _enable_attested(monkeypatch)
    plan = _plan(eval_run_id="eval-progress-result-adj")
    run = await _seed_run(database_session, plan)
    plan = load_eval_run_plan(run)

    missing = await client.post(
        RESULT_PATH.format(eval_run_id=plan["eval_run_id"]),
        headers={"Content-Type": "application/json"},
        content=b"{}",
    )
    wrong = await client.post(
        RESULT_PATH.format(eval_run_id=plan["eval_run_id"]),
        headers={
            "Authorization": "Bearer wrong-token",
            "Content-Type": "application/json",
        },
        content=b"{}",
    )
    # Valid Bearer reaches body validation (not auth rejection).
    authed = await client.post(
        RESULT_PATH.format(eval_run_id=plan["eval_run_id"]),
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
        content=b"{}",
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json()["detail"] == {"code": "invalid_eval_token"}
    assert wrong.json()["detail"] == {"code": "invalid_eval_token"}
    # Auth passed: not 401/404. Body is invalid so 422 (or 413) — never score write.
    assert authed.status_code not in {401, 404}
    assert authed.status_code in {422, 400, 413}

    async with database_session() as session:
        refreshed = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == plan["eval_run_id"])
        )
    assert refreshed is not None
    assert refreshed.score is None
    assert refreshed.id == run.id
