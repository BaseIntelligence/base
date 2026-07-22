"""Admission, capacity, and failure-mode hardening for direct Eval result ingestion.

These tests encode the scrutiny findings for VAL-VERIFY-001/013/014/028 and
VAL-CROSS-007/034: auth-first bounded transport reading, mandatory endpoint
signer configuration, database-backed global resource reservations, literal
zero-limit handling, cancellable verification, and exhaustive retryable mapping.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalResourceCounter,
    EvalRun,
)
from agent_challenge.evaluation.attestation import (
    AttestationGate,
    AttestationOutcome,
    ResultMeasurementAllowlist,
)
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationConflict,
    receipt_eval_result,
)
from agent_challenge.evaluation.direct_result import (
    process_direct_eval_result,
    require_endpoint_result_signer,
)
from agent_challenge.evaluation.plan_scoring import (
    build_score_record_from_eval_plan,
    canonical_eval_plan_json,
)
from agent_challenge.keyrelease.quote import (
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.sdk.config import ChallengeSettings

REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "ab" * 32
OS_IMAGE_HASH = os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])
AGENT_HASH = "55" * 32


def _plan(*, eval_run_id: str = "eval-admission-1") -> dict[str, Any]:
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
            "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
                    **REGS,
                    "os_image_hash": OS_IMAGE_HASH,
                    "key_provider": "validator-kms",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
            "key_release_nonce": f"key-release-{eval_run_id}",
            "score_nonce": f"score-{eval_run_id}",
            "run_token_sha256": hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest(),
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _settings(**overrides: Any) -> ChallengeSettings:
    values = {
        "attested_review_enabled": True,
        "phala_attestation_enabled": True,
        "eval_app_measurement_allowlist": (
            {
                "mrtd": REGS["mrtd"],
                "rtmr0": REGS["rtmr0"],
                "rtmr1": REGS["rtmr1"],
                "rtmr2": REGS["rtmr2"],
                "compose_hash": COMPOSE_HASH,
                "os_image_hash": OS_IMAGE_HASH,
            },
        ),
        "eval_result_max_bytes": 16 * 1024 * 1024,
        "eval_result_max_tasks": 4,
        "eval_result_max_event_log_entries": 8,
        "eval_result_max_event_log_bytes": 64 * 1024,
        "eval_result_max_vm_config_bytes": 4096,
        "eval_result_max_string_bytes": 4096,
        "eval_result_max_quote_bytes": 64 * 1024,
        "eval_result_max_submissions_per_run_per_minute": 10,
        "eval_result_max_outstanding": 2,
        "attestation_max_concurrent_verifications": 1,
        "eval_result_verifier_deadline_seconds": 0.25,
        "eval_result_signer_uri": "//Alice",
    }
    values.update(overrides)
    return ChallengeSettings(**values)


async def _seed_run(
    database_session,
    plan: dict[str, Any],
    *,
    token: str = "direct-token",
) -> EvalRun:
    now = datetime.now(UTC)
    async with database_session() as session:
        # agent_hash is unique across submissions; derive a stable per-run value
        # while keeping plan agent_hash for attestation fixtures.
        submission_agent_hash = hashlib.sha256(plan["eval_run_id"].encode("utf-8")).hexdigest()
        submission = AgentSubmission(
            miner_hotkey=f"admission-miner-{plan['eval_run_id']}",
            name=f"admission-agent-{plan['eval_run_id']}",
            agent_hash=submission_agent_hash,
            artifact_uri=f"/tmp/admission-{plan['eval_run_id']}.zip",
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        run = EvalRun(
            eval_run_id=plan["eval_run_id"],
            submission_id=submission.id,
            submission_version=1,
            authorizing_review_digest="66" * 32,
            plan_json=canonical_eval_plan_json(plan),
            plan_sha256=hashlib.sha256(canonical_eval_plan_json(plan).encode("utf-8")).hexdigest(),
            token_sha256=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            phase="eval_running",
            retryable=False,
            key_granted_at=now,
            issued_at=now,
            expires_at=now + timedelta(hours=1),
        )
        session.add(run)
        await session.flush()
        session.add(
            EvalNonce(
                eval_run_id=run.id,
                nonce=plan["score_nonce"],
                purpose="score",
                state="outstanding",
                expires_at=now + timedelta(hours=1),
            )
        )
        await session.commit()
        return run


def _request(plan: dict[str, Any]) -> dict[str, Any]:
    event_log, rtmr3 = build_rtmr3_event_log(
        [("compose-hash", bytes.fromhex(COMPOSE_HASH)), ("key-provider", b"validator-kms")]
    )
    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    binding = ew.build_score_binding(
        canonical_measurement={
            "mrtd": REGS["mrtd"],
            "rtmr0": REGS["rtmr0"],
            "rtmr1": REGS["rtmr1"],
            "rtmr2": REGS["rtmr2"],
            "compose_hash": COMPOSE_HASH,
            "os_image_hash": OS_IMAGE_HASH,
        },
        agent_hash=AGENT_HASH,
        eval_run_id=plan["eval_run_id"],
        score_nonce=plan["score_nonce"],
        scores_digest=scores_digest,
        task_ids=["task-a"],
    )
    report_data = ew.score_report_data_hex(binding)
    quote = build_tdx_quote(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    return {
        "schema_version": 1,
        "eval_run_id": plan["eval_run_id"],
        "submission_id": plan["submission_id"],
        "agent_hash": AGENT_HASH,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "cc" * 32,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": {"worker_pubkey": "", "sig": ""},
            "attestation": {
                "tdx_quote": quote,
                "event_log": event_log,
                "report_data": report_data,
                "measurement": {
                    **REGS,
                    "rtmr3": rtmr3,
                    "compose_hash": COMPOSE_HASH,
                    "os_image_hash": OS_IMAGE_HASH,
                },
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": OS_IMAGE_HASH,
                },
            },
        },
    }


def test_production_attested_mode_fails_closed_without_endpoint_signer() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_result_signer_uri=None,
        eval_result_signer_mnemonic=None,
    )
    with pytest.raises(ValueError, match="eval result signer"):
        settings.require_eval_result_signer_for_production()


def test_require_endpoint_result_signer_rejected_without_config() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=False,
        phala_attestation_enabled=False,
        eval_result_signer_uri=None,
        eval_result_signer_mnemonic=None,
    )
    with pytest.raises(ValueError, match="eval result signer"):
        require_endpoint_result_signer(settings)


def test_zero_submission_rate_is_accepted_by_settings() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=False,
        phala_attestation_enabled=False,
        eval_result_max_submissions_per_run_per_minute=0,
    )
    assert settings.eval_result_max_submissions_per_run_per_minute == 0


async def test_zero_submission_rate_admits_no_receipts(database_session) -> None:
    plan = _plan()
    run = await _seed_run(database_session, plan)
    raw = b'{"schema_version":1}'
    digest = hashlib.sha256(raw).hexdigest()
    async with database_session() as session:
        loaded = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert loaded is not None
        with pytest.raises(EvalAuthorizationConflict) as exc:
            await receipt_eval_result(
                session,
                eval_run_id=loaded.eval_run_id,
                body_sha256=digest,
                body=raw,
                max_submissions_per_minute=0,
                max_outstanding=10,
            )
        assert exc.value.code == "eval_result_rate_limited"
        await session.refresh(loaded)
        assert loaded.receipt_id is None


async def test_global_outstanding_cap_is_database_atomic(database_session) -> None:
    plan_a = _plan(eval_run_id="eval-admission-a")
    plan_b = _plan(eval_run_id="eval-admission-b")
    plan_c = _plan(eval_run_id="eval-admission-c")
    run_a = await _seed_run(database_session, plan_a, token="token-a")
    run_b = await _seed_run(database_session, plan_b, token="token-b")
    run_c = await _seed_run(database_session, plan_c, token="token-c")
    digest_a = "aa" * 32
    digest_b = "bb" * 32
    digest_c = "cc" * 32
    async with database_session() as session:
        for run, digest in ((run_a, digest_a), (run_b, digest_b)):
            loaded = await session.scalar(
                select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
            )
            assert loaded is not None
            await receipt_eval_result(
                session,
                eval_run_id=loaded.eval_run_id,
                body_sha256=digest,
                body=b"{}",
                max_submissions_per_minute=10,
                max_outstanding=2,
            )
            await session.commit()
        loaded_c = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run_c.eval_run_id)
        )
        assert loaded_c is not None
        with pytest.raises(EvalAuthorizationConflict) as exc:
            await receipt_eval_result(
                session,
                eval_run_id=loaded_c.eval_run_id,
                body_sha256=digest_c,
                body=b"{}",
                max_submissions_per_minute=10,
                max_outstanding=2,
            )
        assert exc.value.code == "eval_result_overloaded"
        from agent_challenge.evaluation.authorization import OUTSTANDING_RESULT_RESOURCE

        counter = await session.get(EvalResourceCounter, OUTSTANDING_RESULT_RESOURCE)
        assert counter is not None
        assert counter.value == 2


async def test_unexpected_verifier_exception_is_retryable_and_preserves_nonce(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from product_score_chain_fixtures import (
        bind_key_release_grant_on_run,
        build_fixture_review_envelope,
        rebind_plan_authorizing_digest,
        seed_authorizing_review_assignment,
    )

    envelope = build_fixture_review_envelope()
    plan = rebind_plan_authorizing_digest(_plan(), envelope)
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    settings = _settings()
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )

    class BrokenVerifier:
        def verify(self, _quote: str):
            raise OSError("dcap backend crashed")

    async with database_session() as session:
        run = await _seed_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        await seed_authorizing_review_assignment(
            session,
            submission_id=run.submission_id,
            envelope=envelope,
            authorizing_review_digest=plan["authorizing_review_digest"],
        )
        bind_key_release_grant_on_run(run, plan)
        await session.commit()
        parked, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=settings,
            quote_verifier=BrokenVerifier(),
        )
        assert created is True
        assert parked["phase"] == "verifier_unavailable"
        assert parked["retryable"] is True
        nonce = await session.scalar(
            select(EvalNonce).where(EvalNonce.eval_run_id == run.id, EvalNonce.purpose == "score")
        )
        assert nonce is not None
        assert nonce.state == "outstanding"


def test_direct_gate_maps_unexpected_backend_errors_to_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    request = _request(plan)
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )

    class BrokenVerifier:
        def verify(self, _quote: str):
            raise RuntimeError("unexpected dcap failure")

    gate = AttestationGate(
        quote_verifier=BrokenVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements(
            [
                {
                    "mrtd": REGS["mrtd"],
                    "rtmr0": REGS["rtmr0"],
                    "rtmr1": REGS["rtmr1"],
                    "rtmr2": REGS["rtmr2"],
                    "compose_hash": COMPOSE_HASH,
                    "os_image_hash": OS_IMAGE_HASH,
                }
            ]
        ),
    )
    decision = gate.decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
        endpoint_rebound=True,
        rebound_worker_signature={"worker_pubkey": "validator", "sig": "signature"},
    )
    assert decision.outcome is AttestationOutcome.VERIFIER_UNAVAILABLE


def _hang_until_killed(duration: float = 30.0) -> None:
    try:
        time.sleep(duration)
    except KeyboardInterrupt:
        return


async def test_verification_timeout_terminates_subprocess(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from product_score_chain_fixtures import (
        bind_key_release_grant_on_run,
        build_fixture_review_envelope,
        rebind_plan_authorizing_digest,
        seed_authorizing_review_assignment,
    )

    envelope = build_fixture_review_envelope()
    plan = rebind_plan_authorizing_digest(_plan(), envelope)
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    settings = _settings(eval_result_verifier_deadline_seconds=0.2)
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    processes: list[multiprocessing.Process] = []

    class SlowVerifier:
        def __init__(self) -> None:
            self._process: multiprocessing.Process | None = None

        def verify(self, _quote: str):
            proc = multiprocessing.Process(target=_hang_until_killed, args=(30.0,))
            proc.start()
            self._process = proc
            processes.append(proc)
            proc.join()
            return StaticQuoteVerifier().verify(_quote)

        def cancel(self) -> None:
            proc = self._process
            if proc is None or not proc.is_alive():
                return
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)

    async with database_session() as session:
        run = await _seed_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        await seed_authorizing_review_assignment(
            session,
            submission_id=run.submission_id,
            envelope=envelope,
            authorizing_review_digest=plan["authorizing_review_digest"],
        )
        bind_key_release_grant_on_run(run, plan)
        await session.commit()
        parked, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=settings,
            quote_verifier=SlowVerifier(),
        )
        assert created is True
        assert parked["phase"] == "verifier_unavailable"
    assert processes, "expected a verifier subprocess to start"
    for proc in processes:
        proc.join(timeout=1.0)
        assert not proc.is_alive(), "timeout must terminate the verifier process"


async def test_route_auth_fails_before_buffered_body(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unauthorized request must not force the route to allocate the oversized body."""

    from fastapi import HTTPException

    from agent_challenge.api import routes as routes_mod
    from agent_challenge.core.config import settings as app_settings

    plan = _plan(eval_run_id="eval-admission-route")
    await _seed_run(database_session, plan, token="good-token")
    monkeypatch.setattr(app_settings, "attested_review_enabled", True)
    monkeypatch.setattr(app_settings, "phala_attestation_enabled", True)
    monkeypatch.setattr(app_settings, "eval_result_max_bytes", 64)

    chunks_read = {"bytes": 0}

    async def hostile_stream():
        for _ in range(8):
            chunk = b"x" * 32
            chunks_read["bytes"] += len(chunk)
            yield chunk
            await asyncio.sleep(0)

    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "content-length": "16",
        }

        def stream(self):
            return hostile_stream()

        async def body(self) -> bytes:  # pragma: no cover - must not be used
            raise AssertionError("unbounded body() must not be used before auth")

    async with database_session() as session:
        with pytest.raises(HTTPException) as exc:
            await routes_mod.receive_direct_eval_result(
                plan["eval_run_id"],
                FakeRequest(),  # type: ignore[arg-type]
                session=session,
                authorization=None,
            )
    assert chunks_read["bytes"] == 0
    assert exc.value.status_code == 401
    assert exc.value.detail == {"code": "invalid_eval_token"}


async def test_route_reads_only_bounded_body_after_auth(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from agent_challenge.api import routes as routes_mod
    from agent_challenge.core.config import settings as app_settings

    plan = _plan(eval_run_id="eval-admission-bound")
    await _seed_run(database_session, plan, token="good-token")
    monkeypatch.setattr(app_settings, "attested_review_enabled", True)
    monkeypatch.setattr(app_settings, "phala_attestation_enabled", True)
    monkeypatch.setattr(app_settings, "eval_result_max_bytes", 48)

    bytes_streamed = {"n": 0}

    async def oversize_stream():
        for _ in range(4):
            chunk = b"y" * 20
            bytes_streamed["n"] += len(chunk)
            yield chunk
            await asyncio.sleep(0)

    class FakeRequest:
        headers = {"content-type": "application/json"}

        def stream(self):
            return oversize_stream()

        async def body(self) -> bytes:  # pragma: no cover
            raise AssertionError("must use stream() for bounded admission")

    async with database_session() as session:
        with pytest.raises(HTTPException) as exc:
            await routes_mod.receive_direct_eval_result(
                plan["eval_run_id"],
                FakeRequest(),  # type: ignore[arg-type]
                session=session,
                authorization="Bearer good-token",
            )
        assert exc.value.status_code == 413
        assert exc.value.detail == {"code": "result_too_large"}
        assert bytes_streamed["n"] <= app_settings.eval_result_max_bytes + 20


async def test_valid_result_rebinds_only_signature_placeholder(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from product_score_chain_fixtures import (
        bind_key_release_grant_on_run,
        build_fixture_review_envelope,
        rebind_plan_authorizing_digest,
        seed_authorizing_review_assignment,
    )

    envelope = build_fixture_review_envelope()
    plan = rebind_plan_authorizing_digest(_plan(), envelope)
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    settings = _settings()
    rebound = {"worker_pubkey": "validator-hotkey", "sig": "0xabc"}
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: rebound,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    original_quote = request["execution_proof"]["attestation"]["tdx_quote"]
    original_event_log = json.dumps(request["execution_proof"]["attestation"]["event_log"])

    async with database_session() as session:
        run = await _seed_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        await seed_authorizing_review_assignment(
            session,
            submission_id=run.submission_id,
            envelope=envelope,
            authorizing_review_digest=plan["authorizing_review_digest"],
        )
        bind_key_release_grant_on_run(run, plan)
        await session.commit()
        receipt, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=settings,
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert receipt["phase"] == "verified"
        assert request["execution_proof"]["worker_signature"] == {
            "worker_pubkey": "",
            "sig": "",
        }
        assert request["execution_proof"]["attestation"]["tdx_quote"] == original_quote
        assert (
            json.dumps(request["execution_proof"]["attestation"]["event_log"]) == original_event_log
        )


async def test_concurrent_verification_slots_are_database_backed(database_session) -> None:
    from agent_challenge.evaluation.authorization import (
        release_eval_resource,
        reserve_eval_resource,
    )

    async with database_session() as session:
        from agent_challenge.evaluation.authorization import VERIFYING_RESULT_RESOURCE

        await reserve_eval_resource(
            session,
            name=VERIFYING_RESULT_RESOURCE,
            limit=1,
            conflict_code="eval_result_overloaded",
        )
        await session.commit()
        with pytest.raises(EvalAuthorizationConflict) as exc:
            await reserve_eval_resource(
                session,
                name=VERIFYING_RESULT_RESOURCE,
                limit=1,
                conflict_code="eval_result_overloaded",
            )
        assert exc.value.code == "eval_result_overloaded"
        await release_eval_resource(session, name=VERIFYING_RESULT_RESOURCE)
        await session.commit()
        await reserve_eval_resource(
            session,
            name=VERIFYING_RESULT_RESOURCE,
            limit=1,
            conflict_code="eval_result_overloaded",
        )
        await session.commit()
        counter = await session.get(EvalResourceCounter, VERIFYING_RESULT_RESOURCE)
        assert counter is not None
        assert counter.value == 1
        await release_eval_resource(session, name=VERIFYING_RESULT_RESOURCE)
        await session.commit()
        counter = await session.get(EvalResourceCounter, VERIFYING_RESULT_RESOURCE)
        assert counter is not None
        assert counter.value == 0
