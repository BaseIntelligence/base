"""RED contract: Agent Challenge live execution pool for master fan-out.

Master GET /v1/pools/executing calls each challenge's
GET /v1/execution-pool/live and extracts body["units"].

Source of truth: non-terminal EvalRun rows + latest TaskLogEvent
(observability only — never score fields).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import AgentSubmission, EvalRun, TaskLogEvent
from agent_challenge.evaluation.plan_scoring import canonical_eval_plan_json

POOL_PATH = "/v1/execution-pool/live"
AGENT_HASH = "55" * 32
PACKAGE_TREE_SHA = "bb" * 32
COMPOSE_HASH = "ab" * 32
TOKEN = "pool-live-token"

_SCORE_KEYS = frozenset(
    {
        "score",
        "scores",
        "final_score",
        "raw_score",
        "normalized_score",
        "weight",
        "weights",
        "emission",
        "emission_percent",
        "incentive",
        "passed_tasks",
        "total_tasks",
        "canonical_score_record_json",
        "canonical_score_record_sha256",
    }
)


def _assert_no_score_keys(payload: Any, *, path: str = "$") -> None:
    if isinstance(payload, dict):
        lowered = {str(k).lower() for k in payload}
        leaked = _SCORE_KEYS & lowered
        assert not leaked, f"score-like keys at {path}: {sorted(leaked)}"
        for key, value in payload.items():
            _assert_no_score_keys(value, path=f"{path}.{key}")
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _assert_no_score_keys(item, path=f"{path}[{index}]")


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
    *,
    eval_run_id: str,
    phase: str,
    score: float | None = None,
) -> EvalRun:
    plan = _plan(eval_run_id=eval_run_id)
    now = datetime.now(UTC)
    # token_sha256 is unique per EvalRun — derive from eval_run_id.
    token = f"{TOKEN}-{eval_run_id}"
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"pool-miner-{eval_run_id}",
            name=f"pool-agent-{eval_run_id}",
            agent_hash=hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest(),
            package_tree_sha=PACKAGE_TREE_SHA,
            artifact_uri=f"/tmp/pool-{eval_run_id}.zip",
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
            eval_run_id=eval_run_id,
            submission_id=submission.id,
            submission_version=1,
            authorizing_review_digest="66" * 32,
            plan_json=canonical_eval_plan_json(plan),
            plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode("utf-8")).hexdigest(),
            token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            phase=phase,
            retryable=False,
            score=score,
            finalized_at=now if phase in {"eval_accepted", "eval_rejected"} else None,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run


async def _seed_progress_event(
    database_session,
    *,
    submission_id: int,
    eval_run_id: str,
    sequence: int,
    message: str,
    progress: float = 0.4,
    status: str = "running",
) -> None:
    async with database_session() as session:
        event = TaskLogEvent(
            submission_id=submission_id,
            job_id=None,
            task_id="task-a",
            sequence=sequence,
            event_type="task.status",
            stream=None,
            message=message,
            message_bytes=len(message.encode("utf-8")),
            progress=progress,
            status=status,
            metadata_json=json.dumps(
                {
                    "source": "eval_progress",
                    "eval_run_id": eval_run_id,
                    "client_sequence": sequence,
                    "phase": status,
                },
                separators=(",", ":"),
            ),
        )
        session.add(event)
        await session.commit()


@pytest.mark.asyncio
async def test_execution_pool_live_empty_when_no_running_evals(client) -> None:
    """Empty pool is honest empty units list — not 404."""

    response = await client.get(POOL_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"units": []}
    _assert_no_score_keys(body)


@pytest.mark.asyncio
async def test_execution_pool_live_shows_running_eval_with_latest_event(
    client, database_session
) -> None:
    """In-flight eval_running appears with latest progress event; observability only."""

    live = await _seed_run(database_session, eval_run_id="eval-pool-live", phase="eval_running")
    await _seed_progress_event(
        database_session,
        submission_id=live.submission_id,
        eval_run_id=live.eval_run_id,
        sequence=1,
        message="boot",
        progress=0.1,
    )
    await _seed_progress_event(
        database_session,
        submission_id=live.submission_id,
        eval_run_id=live.eval_run_id,
        sequence=2,
        message="latest-progress-marker",
        progress=0.55,
        status="running",
    )

    response = await client.get(POOL_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    units = body.get("units")
    assert isinstance(units, list)
    assert len(units) == 1

    unit = units[0]
    unit_id = str(unit.get("unit_id") or unit.get("eval_run_id") or "")
    assert unit_id == live.eval_run_id
    assert unit.get("eval_run_id") == live.eval_run_id
    assert str(unit.get("status") or unit.get("phase") or "") in {
        "eval_running",
        "running",
        "executing",
    }

    blob = json.dumps(unit)
    assert "latest-progress-marker" in blob, f"latest event missing: {unit!r}"
    latest = unit.get("latest_event")
    assert isinstance(latest, dict)
    assert latest.get("message") == "latest-progress-marker"
    assert latest.get("task_id") == "task-a"
    assert latest.get("progress") == pytest.approx(0.55)
    _assert_no_score_keys(body)


@pytest.mark.asyncio
async def test_execution_pool_live_excludes_completed_eval(client, database_session) -> None:
    """Terminal/completed evals are excluded from the live pool."""

    live = await _seed_run(
        database_session, eval_run_id="eval-pool-still-live", phase="eval_running"
    )
    done = await _seed_run(
        database_session,
        eval_run_id="eval-pool-done",
        phase="eval_accepted",
        score=0.91,
    )
    await _seed_progress_event(
        database_session,
        submission_id=done.submission_id,
        eval_run_id=done.eval_run_id,
        sequence=1,
        message="finished-should-be-hidden",
    )
    await _seed_progress_event(
        database_session,
        submission_id=live.submission_id,
        eval_run_id=live.eval_run_id,
        sequence=1,
        message="still-going",
    )

    response = await client.get(POOL_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    units = body["units"]
    ids = {str(item.get("unit_id") or item.get("eval_run_id") or "") for item in units}
    assert live.eval_run_id in ids
    assert done.eval_run_id not in ids
    assert "finished-should-be-hidden" not in json.dumps(body)
    _assert_no_score_keys(body)


@pytest.mark.asyncio
async def test_execution_pool_live_never_exposes_score_fields(client, database_session) -> None:
    """Even when EvalRun.score is set on a non-terminal row, pool omits score keys."""

    run = await _seed_run(
        database_session,
        eval_run_id="eval-pool-score-guard",
        phase="eval_verifying",
        score=0.42,
    )
    # Defensive: score column may exist on the row; pool must not surface it.
    async with database_session() as session:
        refreshed = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        assert refreshed is not None
        assert refreshed.score == pytest.approx(0.42)

    response = await client.get(POOL_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert any(
        str(u.get("unit_id") or u.get("eval_run_id")) == run.eval_run_id for u in body["units"]
    )
    _assert_no_score_keys(body)
    blob = response.text.lower()
    for token in ('"score"', '"final_score"', '"weights"', '"emission"'):
        assert token not in blob, f"forbidden score token present: {token}"
