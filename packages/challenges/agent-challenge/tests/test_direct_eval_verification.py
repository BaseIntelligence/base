from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalRun,
    EvaluationJob,
    TaskAttestation,
    TaskResult,
)
from agent_challenge.evaluation.attestation import (
    AttestationGate,
    AttestationOutcome,
    ResultMeasurementAllowlist,
)
from agent_challenge.evaluation.direct_result import (
    DirectEvalResultError,
    _persist_verified_result,
    process_direct_eval_result,
    retry_receipted_eval_result,
    validate_result_bounds,
)
from agent_challenge.evaluation.plan_scoring import (
    build_score_record_from_eval_plan,
    canonical_eval_plan_json,
)
from agent_challenge.evaluation.validator_executor import job_attestation_verified
from agent_challenge.evaluation.weights import is_reward_eligible_job
from agent_challenge.keyrelease.quote import (
    QuoteVerifierUnavailable,
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


def _plan() -> dict:
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
            "eval_run_id": "eval-direct-1",
            "submission_id": "submission-direct-1",
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
            "result_endpoint": "/evaluation/v1/runs/eval-direct-1/result",
            "key_release_nonce": "key-release-direct-1",
            "score_nonce": "score-direct-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _request(plan: dict, *, score_nonce: str | None = None) -> dict:
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
        score_nonce=score_nonce or plan["score_nonce"],
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


def _gate(plan: dict, *, nonce_outstanding: bool = True) -> AttestationGate:
    return AttestationGate(
        quote_verifier=StaticQuoteVerifier(),
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


def _direct_settings() -> ChallengeSettings:
    plan = _plan()
    return ChallengeSettings(
        eval_app_measurement_allowlist=(
            {
                "mrtd": REGS["mrtd"],
                "rtmr0": REGS["rtmr0"],
                "rtmr1": REGS["rtmr1"],
                "rtmr2": REGS["rtmr2"],
                "compose_hash": COMPOSE_HASH,
                "os_image_hash": OS_IMAGE_HASH,
            },
        ),
        eval_result_max_bytes=16 * 1024 * 1024,
        eval_result_max_tasks=4,
        eval_result_max_event_log_entries=8,
        eval_result_max_event_log_bytes=64 * 1024,
        eval_result_max_vm_config_bytes=4096,
        eval_result_max_string_bytes=4096,
        eval_result_max_quote_bytes=64 * 1024,
        attestation_max_concurrent_verifications=2,
        eval_app_image_ref=plan["eval_app"]["image_ref"],
    )


async def _direct_run(database_session, plan: dict) -> EvalRun:
    now = datetime.now(UTC)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey="direct-miner",
            name="direct-agent",
            agent_hash=AGENT_HASH,
            artifact_uri="/tmp/direct.zip",
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
            token_sha256=hashlib.sha256(b"direct-token").hexdigest(),
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


async def test_direct_result_persists_one_attested_job_and_is_idempotent(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        receipt, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert receipt["phase"] == "verified"
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.id == run.result_job_id)
        )
        assert job is not None
        assert job.status == "completed"
        assert await session.scalar(select(TaskResult.job_id)) == job.id
        assert await session.scalar(select(TaskAttestation.job_id)) == job.id
        replay, replay_created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert replay_created is False
        assert replay == receipt


async def test_full_attested_result_persists_only_challenge_owned_eval_score(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production attested path never creates validator job/task rows."""

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
    direct_settings = _direct_settings()
    direct_settings.attested_review_enabled = True
    direct_settings.phala_attestation_enabled = True
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )

    async with database_session() as session:
        run = await _direct_run(database_session, plan)
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
            settings=direct_settings,
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert receipt["phase"] == "verified"
        assert run.result_job_id is None
        assert run.score is not None
        assert run.total_tasks == len(plan["selected_tasks"])
        assert await session.scalar(select(EvaluationJob.id)) is None
        assert await session.scalar(select(TaskResult.id)) is None
        assert await session.scalar(select(TaskAttestation.id)) is None


async def test_direct_result_outage_retries_exact_receipt_without_nonce_consume(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )

    class OutageVerifier:
        def verify(self, _quote: str):
            raise QuoteVerifierUnavailable("collateral unavailable")

    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        parked, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=OutageVerifier(),
        )
        assert created is True
        assert parked["phase"] == "verifier_unavailable"
        nonce = await session.scalar(
            select(EvalNonce).where(
                EvalNonce.eval_run_id == run.id,
                EvalNonce.purpose == "score",
            )
        )
        assert nonce is not None
        assert nonce.state == "outstanding"
        recovered, recovered_created = await retry_receipted_eval_result(
            session,
            run=run,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert recovered_created is True
        assert recovered["phase"] == "verified"
        assert recovered["body_sha256"] == parked["body_sha256"]
        await session.refresh(nonce)
        assert nonce.state == "consumed"


async def test_invalid_result_after_key_grant_is_terminal_non_verified(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad proof after golden-key release is still a terminal liveness result."""

    plan = _plan()
    request = _request(plan)
    request["execution_proof"]["attestation"]["report_data"] = "00" * 64
    raw_body = ew.canonical_json_v1(request)
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )

    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        receipt, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert receipt["phase"] == "rejected"
        assert receipt["verified"] is False
        assert receipt["retryable"] is False
        assert run.failure_origin == "attestation"
        assert run.reward_eligible is False
        nonce = await session.scalar(
            select(EvalNonce).where(
                EvalNonce.eval_run_id == run.id,
                EvalNonce.purpose == "score",
            )
        )
        assert nonce is not None
        assert nonce.state == "consumed"
        assert await session.scalar(select(TaskResult.id)) is None


async def test_key_granted_run_with_no_result_terminalizes_as_non_retryable(
    database_session,
) -> None:
    """A run that released golden material but never reports cannot be retried."""

    plan = _plan()
    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        run.key_release_receipt_sha256 = "aa" * 32
        run.expires_at = datetime(2024, 1, 1, tzinfo=UTC)
        page = await __import__(
            "agent_challenge.evaluation.authorization",
            fromlist=["eval_status_page"],
        ).eval_status_page(
            session,
            await session.get(AgentSubmission, run.submission_id),
        )
        assert page["items"][0]["phase"] == "eval_expired"
        assert run.failure_origin == "no_result"
        assert run.verified is False
        assert run.reward_eligible is False
        assert run.retryable is False


async def test_weight_evidence_requires_verified_review_allow(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete score attestations alone never earn weight without matching review."""

    plan = _plan()
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        receipt, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert receipt["phase"] == "verified"
        await session.refresh(run)
        job = await session.scalar(
            select(EvaluationJob).where(EvaluationJob.id == run.result_job_id)
        )
        assert job is not None
        assert await job_attestation_verified(session, job) is True
        assert (
            is_reward_eligible_job(job, 1, attestation_verified=True, review_verified=False)
            is False
        )


async def test_verified_result_persistence_rolls_back_all_rows_on_failure(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after task writes parks the receipt without partial scoring rows."""

    plan = _plan()
    request = _request(plan)
    raw_body = ew.canonical_json_v1(request)
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    original = _persist_verified_result

    async def fail_after_persist(*args, **kwargs):
        await original(*args, **kwargs)
        raise RuntimeError("injected persistence failure")

    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._persist_verified_result",
        fail_after_persist,
    )

    async with database_session() as session:
        run = await _direct_run(database_session, plan)
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id))
        assert run is not None
        parked, created = await process_direct_eval_result(
            session,
            run=run,
            raw_body=raw_body,
            result_request=request,
            settings=_direct_settings(),
            quote_verifier=StaticQuoteVerifier(),
        )
        assert created is True
        assert parked["phase"] == "verifier_unavailable"
        assert parked["reason_code"] == "persistence_unavailable"
        assert await session.scalar(select(TaskResult.id)) is None
        assert await session.scalar(select(TaskAttestation.id)) is None
        assert await session.scalar(select(EvaluationJob.id)) is None


def test_direct_gate_accepts_only_plan_bound_result_without_consuming_inline_nonce() -> None:
    plan = _plan()
    request = _request(plan)
    decision = _gate(plan).decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFIED


def test_direct_gate_fails_closed_for_empty_allowlist_and_missing_key_grant() -> None:
    plan = _plan()
    request = _request(plan)
    empty_gate = AttestationGate(quote_verifier=StaticQuoteVerifier())
    assert (
        empty_gate.decide_eval_result(
            request,
            eval_plan=plan,
            expected_agent_hash=AGENT_HASH,
            nonce_outstanding=True,
            key_granted=True,
        ).outcome
        is AttestationOutcome.VERIFICATION_FAILED
    )
    assert (
        _gate(plan)
        .decide_eval_result(
            request,
            eval_plan=plan,
            expected_agent_hash=AGENT_HASH,
            nonce_outstanding=True,
            key_granted=False,
        )
        .outcome
        is AttestationOutcome.VERIFICATION_FAILED
    )


def test_direct_gate_rejects_caller_signature_instead_of_rebinding_it() -> None:
    plan = _plan()
    request = _request(plan)
    request["execution_proof"]["worker_signature"] = {
        "worker_pubkey": "caller-selected",
        "sig": "caller-selected",
    }
    decision = _gate(plan).decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
        endpoint_rebound=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


def test_direct_gate_rebinds_only_validator_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    request = _request(plan)
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )
    decision = _gate(plan).decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=AGENT_HASH,
        nonce_outstanding=True,
        key_granted=True,
        endpoint_rebound=True,
        rebound_worker_signature={"worker_pubkey": "validator", "sig": "signature"},
    )
    assert decision.outcome is AttestationOutcome.VERIFIED


def test_direct_result_nested_bounds_fail_before_verification() -> None:
    plan = _plan()
    request = _request(plan)
    with pytest.raises(DirectEvalResultError, match="event-log"):
        validate_result_bounds(
            {
                **request,
                "execution_proof": {
                    **request["execution_proof"],
                    "attestation": {
                        **request["execution_proof"]["attestation"],
                        "event_log": request["execution_proof"]["attestation"]["event_log"] * 2,
                    },
                },
            },
            max_tasks=4,
            max_event_log_entries=1,
            max_quote_bytes=1024,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["execution_proof"]["attestation"].update(report_data="00" * 64),
        lambda value: value["execution_proof"]["attestation"]["measurement"].update(
            compose_hash="dd" * 32
        ),
        lambda value: value["score_record"]["tasks"][0].update(
            trial_scores_f64be=["0000000000000000"]
        ),
        lambda value: value["execution_proof"]["attestation"].update(
            tdx_quote=value["execution_proof"]["attestation"]["tdx_quote"][:-2]
        ),
    ],
)
def test_direct_gate_rejects_tampering_and_crossed_nonce(mutation) -> None:
    plan = _plan()
    request = _request(plan)
    mutation(copy.deepcopy(request))
    mutated = copy.deepcopy(request)
    mutation(mutated)
    assert (
        _gate(plan)
        .decide_eval_result(
            mutated,
            eval_plan=plan,
            expected_agent_hash=AGENT_HASH,
            nonce_outstanding=True,
            key_granted=True,
        )
        .outcome
        is AttestationOutcome.VERIFICATION_FAILED
    )

    crossed = _request(plan, score_nonce=plan["key_release_nonce"])
    assert (
        _gate(plan)
        .decide_eval_result(
            crossed,
            eval_plan=plan,
            expected_agent_hash=AGENT_HASH,
            nonce_outstanding=True,
            key_granted=True,
        )
        .outcome
        is AttestationOutcome.VERIFICATION_FAILED
    )
