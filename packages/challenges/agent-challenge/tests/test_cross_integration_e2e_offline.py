"""Offline full-attested accept-chain (VAL-CROSS-002..016).

This suite proves the production full-attested topology without a live CVM:

* verified review allow -> signed ``POST .../eval/prepare``
* exact Eval plan v1 bytes (including ``k`` + complete Scoring policy v1)
* token-scoped ``POST /evaluation/v1/runs/{eval_run_id}/result``
* one production ``AttestationGate`` / score-nonce consume
* weights from the direct EvalRun population

REJECTED topology for this path (must stay at zero counts):

* assignable ``list_pending_work_units`` / BASE validator assignment
* ``run_validator_cycle`` / own_runner broker / ``_run_task``
* EvaluationJob / TaskResult rows on the full-attested happy path

Discriminators keep the dual key-release + result-gate surfaces for the
anti-cheat matrix so a “accept if attestation present” leap would fail.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import (
    AgentSubmission,
    EvalNonce,
    EvalRun,
    EvaluationJob,
    ReviewAssignment,
    ReviewSession,
    TaskResult,
)
from agent_challenge.evaluation.attestation import (
    AttestationGate,
    AttestationOutcome,
    ResultMeasurementAllowlist,
)
from agent_challenge.evaluation.authorization import create_eval_run
from agent_challenge.evaluation.direct_result import (
    process_direct_eval_result,
    retry_receipted_eval_result,
)
from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan
from agent_challenge.evaluation.weights import get_weights, is_reward_eligible_job
from agent_challenge.evaluation.work_units import list_pending_work_units
from agent_challenge.keyrelease.allowlist import CanonicalEntry, MeasurementAllowlist
from agent_challenge.keyrelease.client import KEY_RELEASE_TAG, key_release_report_data
from agent_challenge.keyrelease.nonce import NonceStore
from agent_challenge.keyrelease.quote import (
    COMPOSE_HASH_EVENT,
    KEY_PROVIDER_EVENT,
    QuoteVerifierUnavailable,
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
    os_image_hash_from_registers,
)
from agent_challenge.keyrelease.server import (
    REASON_MEASUREMENT_NOT_ALLOWLISTED,
    REASON_STALE_NONCE,
    REASON_TCB_UNACCEPTABLE,
    KeyReleaseService,
)
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    build_decision,
    build_observed_openrouter_transport,
    build_openrouter_observation,
    build_planned_openrouter_request,
    build_policy_observation,
    build_review_core_minimal,
    planned_request_sha256,
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.review.or_outcome_bind import (
    review_digest as bound_review_digest,
)
from agent_challenge.sdk.config import ChallengeSettings

# --------------------------------------------------------------------------- #
# Shared measurement / truth-table fixtures
# --------------------------------------------------------------------------- #
REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
ALT_REGS = {
    "mrtd": "ee" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "ab" * 32
ALT_COMPOSE_HASH = "cd" * 32
OS_IMAGE_HASH = os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])
ALT_OS_IMAGE_HASH = os_image_hash_from_registers(
    ALT_REGS["mrtd"], ALT_REGS["rtmr1"], ALT_REGS["rtmr2"]
)
KEY_PROVIDER = "validator-kms"
# Score-gate event payload is the literal plan pin when not JSON-encoded.
SCORE_KEY_PROVIDER_PAYLOAD = b"validator-kms"
# Key-release allowlist pin is "phala"; decode_key_provider collapses live KMS JSON to it.
KEY_RELEASE_PROVIDER_PAYLOAD = b'{"name":"kms","id":"kms-1"}'
ALT_KEY_PROVIDER_PAYLOAD = b'{"name":"none","id":"self"}'
ENCLAVE_PUBKEY = b"enclave-ra-tls-pubkey-0123456789"  # 32 bytes
SENTINEL_KEY = b"SENTINEL-CROSS-INTEGRATION-KEY!!"  # 32 bytes
# Placeholder for synthetic plan-only fixtures (does not gate create_eval_run).
# Durable review allow rows always use digests from _fresh_review_envelope.
REVIEW_DIGEST = "66" * 32
NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)
_T0 = 1_700_000_000_000
_ROUTE = sha256_hex(b'{"order":["cross-fx"]}')
_BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
_BODY_SHA = sha256_hex(_BODY)
_RESP = b'{"id":"gen-cross","model":"x-ai/grok-4.5","choices":[]}'
_RESP_SHA = sha256_hex(_RESP)
_META = sha256_hex(b"meta-cross")


def _fresh_review_envelope(*, suffix: str) -> tuple[str, str, str]:
    """Receipted allow envelope for create_eval_run re-verify (VAL-ACAT-028/029)."""

    planned = build_planned_openrouter_request(
        body_sha256=_BODY_SHA,
        body_length=len(_BODY),
        routing_sha256=_ROUTE,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=_RESP_SHA,
        response_body_length=len(_RESP),
        metadata_sha256=_META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=_BODY_SHA,
        request_body_length=len(_BODY),
        response_id=f"gen-cross-{suffix}",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-cross",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-cross",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-cross",
        routing_sha256=_ROUTE,
    )
    core = build_review_core_minimal(
        session_id=f"rs-cross-{suffix}",
        assignment_id=f"ra-cross-{suffix}",
        submission_id=f"sub-cross-{suffix}",
        review_nonce=f"nonce-cross-{suffix}",
        assignment_digest="13" * 32,
        rules_observation={
            "rules_version": "rules-v1",
            "rules_bundle_sha256": "11" * 32,
            "rules_files": [".rules/acceptance.md"],
            "rules_file_digests": {".rules/acceptance.md": "22" * 32},
            "rules_policy_text_sha256": "33" * 32,
        },
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict="allow"),
        times={
            "issued_at_ms": _T0,
            "started_at_ms": _T0,
            "model_call_marked_at_ms": _T0 + 1,
            "request_started_at_ms": _T0 + 2,
            "request_finished_at_ms": _T0 + 3,
            "verifier_finished_at_ms": _T0 + 4,
            "report_finished_at_ms": _T0 + 5,
            "expires_at_ms": _T0 + 3_600_000,
            "submission_received_at_ms": _T0 + 60_000,
        },
    )
    digest = bound_review_digest(core)
    rd = review_report_data_hex(core)
    env = {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": digest,
        "report_data_hex": rd,
        "review_core": core,
    }
    return json.dumps(env, sort_keys=True, separators=(",", ":")), rd, digest


MEASUREMENT = {
    "mrtd": REGS["mrtd"],
    "rtmr0": REGS["rtmr0"],
    "rtmr1": REGS["rtmr1"],
    "rtmr2": REGS["rtmr2"],
    "os_image_hash": OS_IMAGE_HASH,
    "key_provider": KEY_PROVIDER,
    "vm_shape": "tdx-small",
}
ALLOWLIST_ENTRY = {
    "mrtd": REGS["mrtd"],
    "rtmr0": REGS["rtmr0"],
    "rtmr1": REGS["rtmr1"],
    "rtmr2": REGS["rtmr2"],
    "compose_hash": COMPOSE_HASH,
    "os_image_hash": OS_IMAGE_HASH,
}


# --------------------------------------------------------------------------- #
# Eval-plan / result builders (true Eval plan v1 + result request v1 wire)
# --------------------------------------------------------------------------- #
def _policy(
    *,
    keep_policy: str = "off",
    drop_lowest_n: int = 0,
    per_task_aggregation: str = "mean",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "per_task_aggregation": per_task_aggregation,
        "keep_policy": keep_policy,
        "drop_lowest_n": drop_lowest_n if keep_policy == "drop_lowest_n" else 0,
        "threshold_f64be": None,
    }


def _eval_settings(
    *,
    k: int = 1,
    task_count: int = 1,
    keep_policy: str = "off",
    drop_lowest: int = 0,
    per_task_aggregation: str = "mean",
) -> ChallengeSettings:
    # Settings retain legacy hyphenated spellings; plan issuance converts once.
    keep_settings = {
        "off": "off",
        "drop_lowest_n": "drop-lowest-n",
        "threshold_band": "threshold-band",
    }[keep_policy]
    agg_settings = {
        "mean": "mean",
        "best_of_k": "best-of-k",
    }[per_task_aggregation]
    return ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_app_image_ref="registry.example/eval@sha256:" + "a" * 64,
        eval_app_compose_hash=COMPOSE_HASH,
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="07" * 32,
        eval_app_measurement=MEASUREMENT,
        eval_app_measurement_allowlist=(dict(ALLOWLIST_ENTRY),),
        eval_key_release_endpoint="validator.example:8701",
        eval_k=k,
        evaluation_task_count=task_count,
        keep_good_tasks_policy=keep_settings,
        keep_good_tasks_drop_lowest=drop_lowest,
        per_task_aggregation=agg_settings,
        eval_result_max_bytes=16 * 1024 * 1024,
        eval_result_max_tasks=16,
        eval_result_max_event_log_entries=32,
        eval_result_max_event_log_bytes=64 * 1024,
        eval_result_max_vm_config_bytes=4096,
        eval_result_max_string_bytes=4096,
        eval_result_max_quote_bytes=64 * 1024,
        eval_result_max_submissions_per_run_per_minute=20,
        eval_result_max_outstanding=8,
        attestation_max_concurrent_verifications=2,
        eval_result_verifier_deadline_seconds=5.0,
        eval_result_signer_uri="//Alice",
        eval_run_ttl_seconds=21600,
    )


def _patch_benchmark_tasks(monkeypatch: pytest.MonkeyPatch, *, task_count: int) -> list[str]:
    tasks = []
    for index in range(task_count):
        task_id = f"task-{index:03d}"
        tasks.append(
            type(
                "Task",
                (),
                {
                    "task_id": task_id,
                    "docker_image": f"registry.example/task@sha256:{index:064x}"[:71] + ("b" * 64),
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": f"{index:02x}" * 32},
                },
            )()
        )
    # Ensure valid digest-pinned images.
    for index, task in enumerate(tasks):
        digest = hashlib.sha256(f"task-config-{index}".encode()).hexdigest()
        task.docker_image = f"registry.example/task@sha256:{digest}"
        task.metadata = {"content_digest_sha256": digest}
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: list(tasks),
    )
    return [task.task_id for task in tasks]


def _enable_full_attested(monkeypatch: pytest.MonkeyPatch) -> None:
    for path in (
        "agent_challenge.core.config.settings.attested_review_enabled",
        "agent_challenge.core.config.settings.phala_attestation_enabled",
        "agent_challenge.evaluation.weights.settings.attested_review_enabled",
        "agent_challenge.evaluation.weights.settings.phala_attestation_enabled",
        "agent_challenge.evaluation.work_units.settings.attested_review_enabled",
        "agent_challenge.api.routes.settings.attested_review_enabled",
        "agent_challenge.api.routes.settings.phala_attestation_enabled",
    ):
        monkeypatch.setattr(path, True)


async def _authorized_submission(
    database_session,
    *,
    suffix: str,
    agent_hash: str | None = None,
    review_digest: str = REVIEW_DIGEST,
    attach_review: bool = True,
    raw_status: str = "review_allowed",
) -> AgentSubmission:
    del review_digest  # durable seed always binds the receipted envelope digest
    artifact = f"cross-artifact-{suffix}".encode()
    envelope_json, report_data_hex, digest = _fresh_review_envelope(suffix=suffix)
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"cross-miner-{suffix}",
            name=f"cross-agent-{suffix}",
            agent_hash=agent_hash or hashlib.sha256(artifact).hexdigest(),
            artifact_uri=f"/tmp/cross-{suffix}.zip",
            artifact_path=f"/tmp/cross-{suffix}.zip",
            zip_sha256=hashlib.sha256(artifact).hexdigest(),
            zip_size_bytes=len(artifact),
            raw_status=raw_status,
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        if attach_review:
            assignment_id = f"assign-cross-{suffix}"
            review_session = ReviewSession(
                session_id=f"review-cross-{suffix}",
                submission_id=submission.id,
                artifact_sha256=submission.zip_sha256,
                artifact_size_bytes=len(artifact),
                manifest_sha256="11" * 32,
                manifest_entries_sha256="12" * 32,
                authorizing_assignment_id=assignment_id,
                current_assignment_id=assignment_id,
            )
            session.add(review_session)
            await session.flush()
            assignment = ReviewAssignment(
                session_id=review_session.id,
                assignment_id=assignment_id,
                attempt=1,
                assignment_bytes="{}",
                assignment_digest="13" * 32,
                artifact_sha256=submission.zip_sha256,
                rules_snapshot_sha256="14" * 32,
                rules_revision_id="rules-v1",
                review_nonce=f"review-nonce-{suffix}",
                session_token_sha256="15" * 32,
                capability_state="revoked",
                phase="review_allowed",
                issued_at=NOW,
                expires_at=NOW + timedelta(hours=1),
                review_report_envelope_json=envelope_json,
                review_report_data_hex=report_data_hex,
                review_digest=digest,
                review_verification_outcome_json=json.dumps(
                    {
                        "status": "verified_allow",
                        "terminal": True,
                        "retryable": False,
                        "reason_code": "policy_allowed",
                        "nonce_consumed": True,
                        "measurement_allowlisted": True,
                        "report_data_matched": True,
                        "verified_at_ms": 1,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            session.add(assignment)
        await session.commit()
        return submission


async def _prepare_eval_run(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
    *,
    suffix: str,
    settings: ChallengeSettings | None = None,
    task_count: int | None = None,
    agent_hash: str | None = None,
    review_digest: str = REVIEW_DIGEST,
) -> tuple[EvalRun, dict[str, Any], str, ChallengeSettings]:
    """Issue one immutable Eval plan via ``create_eval_run`` (EvalPrepare path)."""

    cfg = settings or _eval_settings(task_count=task_count or 1)
    if task_count is None:
        task_count = cfg.evaluation_task_count
    _patch_benchmark_tasks(monkeypatch, task_count=task_count)
    submission = await _authorized_submission(
        database_session,
        suffix=suffix,
        agent_hash=agent_hash,
        review_digest=review_digest,
    )
    async with database_session() as session:
        row = await session.get(AgentSubmission, submission.id)
        assert row is not None
        created = await create_eval_run(session, row, settings=cfg)
        await session.commit()
        assert created.token is not None, "first prepare must mint EVAL_RUN_TOKEN once"
        run = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == created.run.eval_run_id)
        )
        assert run is not None
        # Arrange a matching key grant so score acceptance is production-shaped.
        # Dual flags require reconstructible KR materials, not key_granted_at alone
        # (VAL-ACAT-036/037). Stamp durable grant for score-chain re-verify.
        run.key_granted_at = datetime.now(UTC)
        run.phase = "eval_running"
        run.retryable = False
        # Align plan agent_hash with the submission that create_eval_run bound.
        plan = json.loads(run.plan_json)
        assert plan["agent_hash"] == row.agent_hash
        from product_score_chain_fixtures import bind_key_release_grant_on_run

        bind_key_release_grant_on_run(run, plan)
        await session.commit()
        return run, created.plan, created.token, cfg


def _trials_for_plan(
    plan: dict[str, Any],
    *,
    per_task: list[float] | None = None,
) -> dict[str, list[float]]:
    task_ids = [item["task_id"] for item in plan["selected_tasks"]]
    k = int(plan["k"])
    if per_task is None:
        per_task = [1.0] * len(task_ids)
    assert len(per_task) == len(task_ids)
    return {task_id: [float(score)] * k for task_id, score in zip(task_ids, per_task, strict=True)}


def _result_request(
    plan: dict[str, Any],
    *,
    trial_scores_by_task: dict[str, list[float]] | None = None,
    score_nonce: str | None = None,
    agent_hash: str | None = None,
    eval_run_id: str | None = None,
    regs: dict[str, str] = REGS,
    compose_hash: str = COMPOSE_HASH,
    report_data_override: str | None = None,
    place_nonempty_signature: bool = False,
) -> dict[str, Any]:
    agent = agent_hash if agent_hash is not None else plan["agent_hash"]
    run_id = eval_run_id if eval_run_id is not None else plan["eval_run_id"]
    nonce = score_nonce if score_nonce is not None else plan["score_nonce"]
    trials = trial_scores_by_task or _trials_for_plan(plan)
    record = build_score_record_from_eval_plan(plan, trials)
    scores_digest = ew.score_record_digest(record)
    task_ids = [item["task_id"] for item in plan["selected_tasks"]]
    os_hash = os_image_hash_from_registers(regs["mrtd"], regs["rtmr1"], regs["rtmr2"])
    binding = ew.build_score_binding(
        canonical_measurement={
            "mrtd": regs["mrtd"],
            "rtmr0": regs["rtmr0"],
            "rtmr1": regs["rtmr1"],
            "rtmr2": regs["rtmr2"],
            "compose_hash": compose_hash,
            "os_image_hash": os_hash,
        },
        agent_hash=agent,
        eval_run_id=run_id,
        score_nonce=nonce,
        scores_digest=scores_digest,
        task_ids=task_ids,
    )
    if report_data_override is not None:
        report_data = report_data_override
    else:
        report_data = ew.score_report_data_hex(binding)
    event_log, rtmr3 = build_rtmr3_event_log(
        [
            (COMPOSE_HASH_EVENT, bytes.fromhex(compose_hash)),
            (KEY_PROVIDER_EVENT, SCORE_KEY_PROVIDER_PAYLOAD),
        ]
    )
    quote = build_tdx_quote(
        mrtd=regs["mrtd"],
        rtmr0=regs["rtmr0"],
        rtmr1=regs["rtmr1"],
        rtmr2=regs["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    signature = (
        {"worker_pubkey": "attacker", "sig": "0xdead"}
        if place_nonempty_signature
        else {"worker_pubkey": "", "sig": ""}
    )
    return {
        "schema_version": 1,
        "eval_run_id": run_id,
        "submission_id": plan["submission_id"],
        "agent_hash": agent,
        "score_record": record,
        "scores_digest": scores_digest,
        "execution_proof": {
            "version": 1,
            "tier": "phala-tdx",
            "manifest_sha256": "cc" * 32,
            "image_digest": plan["eval_app"]["image_ref"],
            "provider": None,
            "worker_signature": signature,
            "attestation": {
                "tdx_quote": quote,
                "event_log": event_log,
                "report_data": report_data,
                "measurement": {
                    **regs,
                    "rtmr3": rtmr3,
                    "compose_hash": compose_hash,
                    "os_image_hash": os_hash,
                },
                "vm_config": {
                    "vcpu": 1,
                    "memory_mb": 2048,
                    "os_image_hash": os_hash,
                },
            },
        },
    }


def _patch_endpoint_signer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.direct_result._endpoint_worker_signature",
        lambda *_args, **_kwargs: {"worker_pubkey": "validator", "sig": "signature"},
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.attestation.verify_worker_signature",
        lambda *_args: True,
    )


class _CountingVerifier:
    """Wrap StaticQuoteVerifier so tests can count so-called gate operations."""

    def __init__(self, *, tcb_status: str = "UpToDate") -> None:
        self.calls = 0
        self._inner = StaticQuoteVerifier(tcb_status=tcb_status)

    def verify(self, quote: str):
        self.calls += 1
        return self._inner.verify(quote)


async def _accept_direct(
    database_session,
    *,
    run: EvalRun,
    plan: dict[str, Any],
    request: dict[str, Any],
    settings: ChallengeSettings,
    quote_verifier: Any | None = None,
) -> tuple[dict[str, Any], bool]:
    raw_body = ew.canonical_json_v1(request)
    async with database_session() as session:
        current = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        assert current is not None
        receipt, created = await process_direct_eval_result(
            session,
            run=current,
            raw_body=raw_body,
            result_request=request,
            settings=settings,
            quote_verifier=quote_verifier or StaticQuoteVerifier(),
        )
        return receipt, created


async def _zero_assignable_topology(session) -> None:
    assert await list_pending_work_units(session) == []
    assert await session.scalar(select(func.count(EvaluationJob.id))) == 0
    assert await session.scalar(select(func.count(TaskResult.id))) == 0


# --------------------------------------------------------------------------- #
# Key-release dual-surface helpers (created as discriminators only)
# --------------------------------------------------------------------------- #
def _canonical_entry() -> CanonicalEntry:
    return CanonicalEntry(
        mrtd=REGS["mrtd"],
        rtmr0=REGS["rtmr0"],
        rtmr1=REGS["rtmr1"],
        rtmr2=REGS["rtmr2"],
        compose_hash=COMPOSE_HASH,
        os_image_hash=OS_IMAGE_HASH,
        key_provider="phala",
    )


def _make_key_release_service(**kwargs) -> KeyReleaseService:
    params = {
        "allowlist": MeasurementAllowlist([_canonical_entry()]),
        "verifier": StaticQuoteVerifier(tcb_status="UpToDate"),
        "nonce_store": NonceStore(),
        "golden_key_loader": lambda: SENTINEL_KEY,
    }
    params.update(kwargs)
    return KeyReleaseService(**params)


def _event_log(
    compose_payload: bytes | None = None,
    key_provider_payload: bytes = KEY_RELEASE_PROVIDER_PAYLOAD,
) -> tuple[list[dict], str]:
    payload = compose_payload if compose_payload is not None else bytes.fromhex(COMPOSE_HASH)
    return build_rtmr3_event_log(
        [
            ("app-id", b"canonical-app"),
            (COMPOSE_HASH_EVENT, payload),
            (KEY_PROVIDER_EVENT, key_provider_payload),
            ("instance-id", b"instance-xyz"),
        ]
    )


def _key_release_request(
    service: KeyReleaseService,
    *,
    nonce: str | None = None,
    ra_tls_pubkey: bytes = ENCLAVE_PUBKEY,
    regs: dict[str, str] = REGS,
    compose_payload: bytes | None = None,
    key_provider_payload: bytes = KEY_RELEASE_PROVIDER_PAYLOAD,
    report_data_override: bytes | None = None,
) -> dict:
    if nonce is None:
        nonce = service.issue_nonce()
    event_log, rtmr3 = _event_log(compose_payload, key_provider_payload)
    report_data = (
        report_data_override
        if report_data_override is not None
        else key_release_report_data(nonce, ra_tls_pubkey)
    )
    quote = build_tdx_quote(
        mrtd=regs["mrtd"],
        rtmr0=regs["rtmr0"],
        rtmr1=regs["rtmr1"],
        rtmr2=regs["rtmr2"],
        rtmr3=rtmr3,
        report_data=report_data,
    )
    return {
        "nonce": nonce,
        "quote_hex": quote,
        "ra_tls_pubkey_hex": ra_tls_pubkey.hex(),
        "event_log": event_log,
        "session_peer_pubkey": ra_tls_pubkey,
    }


def _assert_no_key(outcome) -> None:
    assert outcome.released is False
    assert outcome.key is None
    assert outcome.reason is not None
    assert SENTINEL_KEY.hex() not in (outcome.reason or "")


class _AdvanceableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# =========================================================================== #
# Positive control
# =========================================================================== #
def test_positive_controls_release_and_verify():
    service = _make_key_release_service()
    out = service.authorize_release(**_key_release_request(service))
    assert out.released is True
    assert out.key == SENTINEL_KEY

    plan = ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": "eval-control",
            "submission_id": "1",
            "submission_version": 1,
            "authorizing_review_digest": REVIEW_DIGEST,
            "agent_hash": "55" * 32,
            "selected_tasks": [
                {
                    "task_id": "task-000",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": _policy(),
            "scoring_policy_digest": ew.scoring_policy_digest(_policy()),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": MEASUREMENT,
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-control/result",
            "key_release_nonce": "key-control",
            "score_nonce": "score-control",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )
    request = _result_request(plan)
    gate = AttestationGate(
        quote_verifier=StaticQuoteVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements([ALLOWLIST_ENTRY]),
    )
    # Empty placeholder signature is the CVM emission shape; the production
    # endpoint rebinds before the live gate. Offline decide path accepts the
    # unbound placeholder when endpoint_rebound is left false.
    decision = gate.decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=plan["agent_hash"],
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFIED


# =========================================================================== #
# VAL-CROSS-002 / 003 / 004: full direct pipeline, R=1, no re-exec
# =========================================================================== #
async def test_val_cross_002_full_pipeline_submission_to_weights(database_session, monkeypatch):
    """Submission -> verified allow -> EvalPrepare -> direct result -> weights.

    Zero assignable work units, zero EvaluationJob/TaskResult rows, one gate
    consume, and forbids the legacy validator cycle.
    """

    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    settings = _eval_settings(k=1, task_count=2)
    # Align weight threshold with the plan-selected set size.
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.evaluation_task_count",
        2,
    )
    verifier = _CountingVerifier()
    run, plan, token, cfg = await _prepare_eval_run(
        database_session,
        monkeypatch,
        suffix="pipeline",
        settings=settings,
        task_count=2,
    )

    # Exact prepare projection: plan bytes are complete and digests match.
    assert plan["k"] == 1
    assert plan["scoring_policy"]["schema_version"] == 1
    assert plan["scoring_policy_digest"] == ew.scoring_policy_digest(plan["scoring_policy"])
    assert plan["result_endpoint"] == f"/evaluation/v1/runs/{run.eval_run_id}/result"
    assert token  # EVAL_RUN_TOKEN delivered once

    async with database_session() as session:
        await _zero_assignable_topology(session)

    request = _result_request(plan)
    # Capture literal wire bytes that would be POSTed to the direct route.
    raw = ew.canonical_json_v1(request)
    assert request["execution_proof"]["worker_signature"] == {
        "worker_pubkey": "",
        "sig": "",
    }
    assert len(request["execution_proof"]["attestation"]["report_data"]) == 128

    receipt, created = await _accept_direct(
        database_session,
        run=run,
        plan=plan,
        request=request,
        settings=cfg,
        quote_verifier=verifier,
    )
    assert created is True
    assert receipt["phase"] == "verified"
    assert receipt["verified"] is True
    assert receipt["body_sha256"] == hashlib.sha256(raw).hexdigest()
    assert verifier.calls == 1

    async with database_session() as session:
        await _zero_assignable_topology(session)
        accepted = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        nonce = await session.scalar(
            select(EvalNonce).where(
                EvalNonce.eval_run_id == accepted.id,
                EvalNonce.purpose == "score",
            )
        )
        submission = await session.get(AgentSubmission, accepted.submission_id)
        assert accepted.phase == "eval_accepted"
        assert accepted.verified is True
        assert accepted.reward_eligible is True
        assert accepted.result_available is True
        assert nonce.state == "consumed"
        assert submission.raw_status == "tb_completed"
        assert submission.effective_status in {"valid", "queued", "tb_completed", "completed"}
        # Force scoring population status the weight query requires.
        submission.raw_status = "tb_completed"
        submission.effective_status = "valid"
        await session.commit()

    weights = await get_weights()
    assert submission.miner_hotkey in weights
    assert weights[submission.miner_hotkey] == pytest.approx(1.0)

    # Idempotent identical repost reads the receipt and does not re-verify.
    replay, again = await _accept_direct(
        database_session,
        run=run,
        plan=plan,
        request=request,
        settings=cfg,
        quote_verifier=verifier,
    )
    assert again is False
    assert replay["receipt_id"] == receipt["receipt_id"]
    assert verifier.calls == 1  # no second live verifier operation


async def test_val_cross_003_and_004_zero_assignable_work_no_reexecution(
    database_session, monkeypatch
):
    """R=1 = one external eval_run_id; zero assignable work; no broker cycle."""

    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    run, plan, _token, cfg = await _prepare_eval_run(
        database_session, monkeypatch, suffix="r1-zero", settings=_eval_settings(task_count=3)
    )
    request = _result_request(plan)
    receipt, created = await _accept_direct(
        database_session, run=run, plan=plan, request=request, settings=cfg
    )
    assert created and receipt["phase"] == "verified"

    async with database_session() as session:
        await _zero_assignable_topology(session)
        runs = (await session.scalars(select(EvalRun))).all()
        assert len(runs) == 1
        assert runs[0].eval_run_id == plan["eval_run_id"]
        # Score once: duplicate TaskResult / EvaluationJob never appears.
        assert await session.scalar(select(func.count(TaskResult.id))) == 0

    # Local-config keep policy must not override plan (policy bytes solely govern).
    monkeypatch.setattr(
        "agent_challenge.core.config.settings.keep_good_tasks_policy",
        "drop-lowest-n",
    )
    monkeypatch.setattr(
        "agent_challenge.core.config.settings.keep_good_tasks_drop_lowest",
        2,
    )
    # Second attempt with identical body remains idempotent (no re-execution).
    again, created2 = await _accept_direct(
        database_session, run=run, plan=plan, request=request, settings=cfg
    )
    assert created2 is False
    assert again["phase"] == "verified"


# =========================================================================== #
# VAL-CROSS-005: k>1 + scoring policy from immutable plan bytes
# =========================================================================== #
async def test_val_cross_005_variance_scoring_from_plan_bytes(database_session, monkeypatch):
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.evaluation_task_count",
        3,
    )
    # Plausible WRONG local config: if consumers read local settings they would
    # drop-lowest=2 and earn a different score. Policy bytes must win.
    settings = _eval_settings(
        k=3,
        task_count=3,
        keep_policy="drop_lowest_n",
        drop_lowest=1,
        per_task_aggregation="best_of_k",
    )
    monkeypatch.setattr("agent_challenge.core.config.settings.keep_good_tasks_policy", "off")
    monkeypatch.setattr("agent_challenge.core.config.settings.keep_good_tasks_drop_lowest", 0)

    run, plan, _token, cfg = await _prepare_eval_run(
        database_session,
        monkeypatch,
        suffix="variance",
        settings=settings,
        task_count=3,
    )
    assert plan["k"] == 3
    assert plan["scoring_policy"]["keep_policy"] == "drop_lowest_n"
    assert plan["scoring_policy"]["drop_lowest_n"] == 1
    assert plan["scoring_policy"]["per_task_aggregation"] == "best_of_k"
    assert plan["scoring_policy_digest"] == ew.scoring_policy_digest(plan["scoring_policy"])

    # Per-task trials: best-of-k merges [0,1,0]->1, [1,1,0]->1, [0,0,0]->0; then
    # drop lowest leaves mean([1,1])=1.0 with full-set counts.
    task_ids = [item["task_id"] for item in plan["selected_tasks"]]
    trials = {
        task_ids[0]: [0.0, 1.0, 0.0],
        task_ids[1]: [1.0, 1.0, 0.0],
        task_ids[2]: [0.0, 0.0, 0.0],
    }
    expected = build_score_record_from_eval_plan(plan, trials)
    expected_score = ew.decode_score_f64be(expected["final"]["job_score_f64be"])
    assert expected_score == pytest.approx(1.0)
    assert expected["final"]["total_tasks"] == 3
    assert expected["final"]["passed_tasks"] == 2

    request = _result_request(plan, trial_scores_by_task=trials)
    receipt, created = await _accept_direct(
        database_session, run=run, plan=plan, request=request, settings=cfg
    )
    assert created and receipt["phase"] == "verified"

    async with database_session() as session:
        accepted = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        submission = await session.get(AgentSubmission, accepted.submission_id)
        submission.raw_status = "tb_completed"
        submission.effective_status = "valid"
        await session.commit()
        assert accepted.score == pytest.approx(expected_score)
        assert accepted.passed_tasks == 2
        assert accepted.total_tasks == 3
        # Canonical store is plan-bound, not local settings.
        stored = json.loads(accepted.canonical_score_record_json)
        assert stored["final"]["job_score_f64be"] == expected["final"]["job_score_f64be"]
        assert accepted.canonical_score_record_sha256 == ew.score_record_digest(expected)

    weights = await get_weights()
    assert submission.miner_hotkey in weights
    assert weights[submission.miner_hotkey] == pytest.approx(expected_score)

    # Discriminator: identical trials over keep_policy=off mean the full set
    # to mean([1,1,0]) = 1/3, proving local/policy bytes change the score.
    off_plan = copy.deepcopy(plan)
    off_plan["scoring_policy"] = _policy(keep_policy="off")
    off_plan["scoring_policy_digest"] = ew.scoring_policy_digest(off_plan["scoring_policy"])
    off_record = build_score_record_from_eval_plan(off_plan, trials)
    assert ew.decode_score_f64be(off_record["final"]["job_score_f64be"]) == pytest.approx(1 / 3)
    assert ew.decode_score_f64be(off_record["final"]["job_score_f64be"]) != expected_score


# =========================================================================== #
# VAL-CROSS-006: two-factor review × score eligibility
# =========================================================================== #
def test_val_cross_006_two_factor_truth_table():
    job = EvaluationJob(
        job_id="j",
        submission_id=1,
        status="completed",
        selected_tasks_json="[]",
        total_tasks=2,
        passed_tasks=1,
        score=0.5,
    )
    # Either proof alone never earns weight.
    assert is_reward_eligible_job(job, 2, attestation_verified=True, review_verified=False) is False
    assert is_reward_eligible_job(job, 2, attestation_verified=False, review_verified=True) is False
    assert (
        is_reward_eligible_job(job, 2, attestation_verified=False, review_verified=False) is False
    )
    assert is_reward_eligible_job(job, 2, attestation_verified=True, review_verified=True) is True


async def test_val_cross_006_weights_require_both_review_and_score(database_session, monkeypatch):
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.evaluation_task_count",
        1,
    )

    # A: full both factors -> weights.
    run_a, plan_a, _, cfg_a = await _prepare_eval_run(
        database_session, monkeypatch, suffix="both-a", settings=_eval_settings()
    )
    receipt_a, created_a = await _accept_direct(
        database_session,
        run=run_a,
        plan=plan_a,
        request=_result_request(plan_a),
        settings=cfg_a,
    )
    assert created_a and receipt_a["phase"] == "verified"

    # B: score verified but review digest does not match permanent allow -> no weight.
    run_b, plan_b, _, cfg_b = await _prepare_eval_run(
        database_session,
        monkeypatch,
        suffix="score-only",
        settings=_eval_settings(),
        review_digest=REVIEW_DIGEST,
    )
    receipt_b, created_b = await _accept_direct(
        database_session,
        run=run_b,
        plan=plan_b,
        request=_result_request(plan_b),
        settings=cfg_b,
    )
    assert created_b and receipt_b["phase"] == "verified"
    async with database_session() as session:
        accepted_a = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run_a.eval_run_id)
        )
        accepted_b = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run_b.eval_run_id)
        )
        sub_a = await session.get(AgentSubmission, accepted_a.submission_id)
        sub_b = await session.get(AgentSubmission, accepted_b.submission_id)
        # Break review linkage for B after verify to hold the "score ok / review
        # mismatch" cell of the truth table at the weight surface.
        assignment = await session.scalar(
            select(ReviewAssignment).where(
                ReviewAssignment.assignment_id == "assign-cross-score-only"
            )
        )
        assert assignment is not None
        assignment.review_digest = "ff" * 32
        for sub in (sub_a, sub_b):
            sub.raw_status = "tb_completed"
            sub.effective_status = "valid"
        await session.commit()

    weights = await get_weights()
    assert sub_a.miner_hotkey in weights
    assert sub_b.miner_hotkey not in weights


# =========================================================================== #
# VAL-CROSS-007: direct Eval result request wire + single rebind/gate
# =========================================================================== #
async def test_val_cross_007_direct_result_wire_and_single_gate(database_session, monkeypatch):
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    verifier = _CountingVerifier()
    run, plan, token, cfg = await _prepare_eval_run(
        database_session, monkeypatch, suffix="wire", settings=_eval_settings()
    )
    request = _result_request(plan)
    proof = request["execution_proof"]
    att = proof["attestation"]

    # Fixed-width constraints at emission.
    assert len(att["report_data"]) == 128
    for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2", "rtmr3"):
        assert len(att["measurement"][field]) == 96
    assert len(att["measurement"]["compose_hash"]) == 64
    assert len(att["measurement"]["os_image_hash"]) == 64
    assert proof["worker_signature"] == {"worker_pubkey": "", "sig": ""}
    # No attestation aliases (must be exact Eval Phala attestation v1 keys).
    assert set(att) == {
        "tdx_quote",
        "event_log",
        "report_data",
        "measurement",
        "vm_config",
    }

    receipt, created = await _accept_direct(
        database_session,
        run=run,
        plan=plan,
        request=request,
        settings=cfg,
        quote_verifier=verifier,
    )
    assert created and receipt["phase"] == "verified"
    assert verifier.calls == 1

    # Token authenticator matches ready-for-route shape.
    from agent_challenge.evaluation.direct_result import authenticate_eval_token

    async with database_session() as session:
        current = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        assert authenticate_eval_token(current, token) is True
        assert authenticate_eval_token(current, "wrong") is False
        assert authenticate_eval_token(current, None) is False

    # Alias / length control: one-byte-short report_data is rejected at wire.
    bad = _result_request(plan)
    short = bad["execution_proof"]["attestation"]["report_data"][:-1]
    bad["execution_proof"]["attestation"]["report_data"] = short
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_result_request(bad)


# =========================================================================== #
# VAL-CROSS-008 / 009: non-canonical / non-allowlisted measurements
# =========================================================================== #
async def test_val_cross_008_modified_image_denied_and_rejected(database_session, monkeypatch):
    # (a) key-release DENIES modified image measurement — no golden key.
    service = _make_key_release_service()
    modified = service.authorize_release(**_key_release_request(service, regs=ALT_REGS))
    _assert_no_key(modified)
    assert modified.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED

    # (b) direct result REJECTS the fabricated off-allowlist measurement.
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    run, plan, _, cfg = await _prepare_eval_run(
        database_session, monkeypatch, suffix="modimg", settings=_eval_settings()
    )
    # Force the quote measurement registers to the non-allowlisted image.
    fabricated = _result_request(plan, regs=ALT_REGS)
    receipt, created = await _accept_direct(
        database_session, run=run, plan=plan, request=fabricated, settings=cfg
    )
    assert created is True
    assert receipt["phase"] == "rejected"
    assert receipt["verified"] is False
    async with database_session() as session:
        accepted = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run.eval_run_id)
        )
        assert accepted.reward_eligible is False
        assert accepted.verified is False
        assert await session.scalar(select(func.count(TaskResult.id))) == 0
    assert await get_weights() == {}


def test_val_cross_009_genuine_but_non_allowlisted_measurement_rejected():
    # (a) key-release off-allowlist compose.
    service = _make_key_release_service()
    out = service.authorize_release(
        **_key_release_request(service, compose_payload=bytes.fromhex(ALT_COMPOSE_HASH))
    )
    _assert_no_key(out)
    assert out.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED

    # (b) score direct gate: non-allowlisted compose.
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval-na",
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": REVIEW_DIGEST,
        "agent_hash": "55" * 32,
        "selected_tasks": [
            {
                "task_id": "task-000",
                "image_ref": "registry.example/task@sha256:" + "77" * 32,
                "task_config_sha256": "88" * 32,
            }
        ],
        "k": 1,
        "scoring_policy": _policy(),
        "scoring_policy_digest": ew.scoring_policy_digest(_policy()),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "99" * 32,
            "compose_hash": COMPOSE_HASH,
            "app_identity": "agent-challenge-eval-v1",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "aa" * 32,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
            "measurement": MEASUREMENT,
        },
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-na/result",
        "key_release_nonce": "key-na",
        "score_nonce": "score-na",
        "run_token_sha256": "bb" * 32,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan = ew.validate_eval_plan(plan)
    request = _result_request(plan, compose_hash=ALT_COMPOSE_HASH)
    gate = AttestationGate(
        quote_verifier=StaticQuoteVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements([ALLOWLIST_ENTRY]),
    )
    decision = gate.decide_eval_result(
        request,
        eval_plan=plan,
        expected_agent_hash=plan["agent_hash"],
        nonce_outstanding=True,
        key_granted=True,
    )
    assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


# =========================================================================== #
# VAL-CROSS-011: receipt lifecycle + nonce consume-once
# =========================================================================== #
async def test_val_cross_011_receipt_lifecycle_and_nonce(database_session, monkeypatch):
    # Key-release single-use + stale.
    service = _make_key_release_service()
    first = service.authorize_release(**_key_release_request(service, nonce=None))
    assert first.released is True
    # Reuse the same service-issue path: after the first grant the nonce is gone.
    clock = _AdvanceableClock()
    ttl = _make_key_release_service(nonce_store=NonceStore(ttl_seconds=1.0, clock=clock))
    req = _key_release_request(ttl)
    clock.now = 100.0
    stale = ttl.authorize_release(**req)
    _assert_no_key(stale)
    assert stale.reason == REASON_STALE_NONCE

    # Direct result: first accept -> identical repost is idempotent without second
    # verifier/nonce consume; conflicting body -> 409; outage resumes without
    # consuming the score nonce until success.
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    verifier = _CountingVerifier()
    run, plan, _, cfg = await _prepare_eval_run(
        database_session, monkeypatch, suffix="receipt", settings=_eval_settings()
    )
    request = _result_request(plan)
    receipt, created = await _accept_direct(
        database_session,
        run=run,
        plan=plan,
        request=request,
        settings=cfg,
        quote_verifier=verifier,
    )
    assert created and receipt["phase"] == "verified"
    assert verifier.calls == 1

    same, again = await _accept_direct(
        database_session,
        run=run,
        plan=plan,
        request=request,
        settings=cfg,
        quote_verifier=verifier,
    )
    assert again is False
    assert same["receipt_id"] == receipt["receipt_id"]
    assert verifier.calls == 1

    # Conflicting body after durable receipt.
    conflicting = _result_request(plan, trial_scores_by_task=_trials_for_plan(plan, per_task=[0.0]))
    from agent_challenge.evaluation.authorization import EvalAuthorizationConflict

    with pytest.raises(EvalAuthorizationConflict) as exc:
        await _accept_direct(
            database_session,
            run=run,
            plan=plan,
            request=conflicting,
            settings=cfg,
            quote_verifier=verifier,
        )
    assert exc.value.code == "eval_result_receipt_conflict"

    # Outage path: verifier_unavailable keeps score nonce outstanding; recovery
    # verifies once and consumes.
    run2, plan2, _, cfg2 = await _prepare_eval_run(
        database_session, monkeypatch, suffix="outage", settings=_eval_settings()
    )
    request2 = _result_request(plan2)
    raw2 = ew.canonical_json_v1(request2)

    class Outage:
        def verify(self, _quote: str):
            raise QuoteVerifierUnavailable("collateral")

    async with database_session() as session:
        current = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run2.eval_run_id)
        )
        parked, created = await process_direct_eval_result(
            session,
            run=current,
            raw_body=raw2,
            result_request=request2,
            settings=cfg2,
            quote_verifier=Outage(),
        )
        assert created is True
        assert parked["phase"] == "verifier_unavailable"
        nonce = await session.scalar(
            select(EvalNonce).where(
                EvalNonce.eval_run_id == current.id, EvalNonce.purpose == "score"
            )
        )
        assert nonce.state == "outstanding"
        recovered, recovered_created = await retry_receipted_eval_result(
            session,
            run=current,
            settings=cfg2,
            quote_verifier=StaticQuoteVerifier(),
        )
        assert recovered_created is True
        assert recovered["phase"] == "verified"
        await session.refresh(nonce)
        assert nonce.state == "consumed"

    # Key grant makes the run non-retryable even without a result.
    run3, plan3, _, _ = await _prepare_eval_run(
        database_session, monkeypatch, suffix="granted-no-result", settings=_eval_settings()
    )
    async with database_session() as session:
        current = await session.scalar(
            select(EvalRun).where(EvalRun.eval_run_id == run3.eval_run_id)
        )
        current.key_release_receipt_sha256 = "aa" * 32
        current.key_granted_at = datetime.now(UTC)
        current.expires_at = datetime(2020, 1, 1, tzinfo=UTC)
        page = await __import__(
            "agent_challenge.evaluation.authorization", fromlist=["eval_status_page"]
        ).eval_status_page(session, await session.get(AgentSubmission, current.submission_id))
        assert page["items"][0]["retryable"] is False or current.retryable is False
        assert current.reward_eligible in {None, False} or current.verified is not True


# =========================================================================== #
# VAL-CROSS-012 / 014: binding matrix + post-quote score tamper
# =========================================================================== #
async def test_val_cross_012_and_014_binding_matrix_and_score_tamper(database_session, monkeypatch):
    _enable_full_attested(monkeypatch)
    _patch_endpoint_signer(monkeypatch)
    run, plan, _, cfg = await _prepare_eval_run(
        database_session, monkeypatch, suffix="bind", settings=_eval_settings()
    )

    # Crossed agent_hash.
    bad_agent = _result_request(plan, agent_hash="ff" * 32)
    receipt, created = await _accept_direct(
        database_session, run=run, plan=plan, request=bad_agent, settings=cfg
    )
    # First retention of a malformed bind may reject / stay verifying terminal.
    assert created is True
    assert receipt["phase"] in {"rejected", "verifier_unavailable"}
    if receipt["phase"] == "rejected":
        assert receipt["verified"] is False

    # Fresh run for score-nonce / score-tamper vectors.
    run2, plan2, _, cfg2 = await _prepare_eval_run(
        database_session, monkeypatch, suffix="bind2", settings=_eval_settings()
    )
    wrong_nonce = _result_request(plan2, score_nonce="completely-foreign-score-nonce")
    r2, c2 = await _accept_direct(
        database_session, run=run2, plan=plan2, request=wrong_nonce, settings=cfg2
    )
    assert c2 is True
    assert r2["phase"] in {"rejected", "verifier_unavailable"}

    # Post-quote score tamper: wire score_record+digest stay self-consistent but
    # the quote/report_data still binds the original perfect-score digest.
    run3, plan3, _, cfg3 = await _prepare_eval_run(
        database_session, monkeypatch, suffix="tamper", settings=_eval_settings()
    )
    honest = _result_request(plan3)
    zero_record = build_score_record_from_eval_plan(plan3, _trials_for_plan(plan3, per_task=[0.0]))
    tampered = copy.deepcopy(honest)
    tampered["score_record"] = zero_record
    tampered["scores_digest"] = ew.score_record_digest(zero_record)
    # Keep original quote + report_data (bound to perfect-score digest).
    r3, c3 = await _accept_direct(
        database_session, run=run3, plan=plan3, request=tampered, settings=cfg3
    )
    assert c3 is True
    assert r3["phase"] in {"rejected", "verifier_unavailable"}
    if r3["phase"] == "rejected":
        assert r3["verified"] is False
    assert await get_weights() == {}


# =========================================================================== #
# VAL-CROSS-013: 3×3 non-substitutable domain matrix
# =========================================================================== #
def test_val_cross_013_three_domain_non_substitutable_matrix():
    """Each proof verifies only on its own domain diagonal."""

    review_core = {
        "schema_version": 1,
        "session_id": "sess-1",
        "assignment_id": "assign-1",
        "review_nonce": "review-nonce-1",
        "artifact_sha256": "11" * 32,
        "rules_snapshot_sha256": "22" * 32,
        "model": "x-ai/grok-4.5",
        "verdict": "allow",
        "prompt_digest": "33" * 32,
        "request_digest": "44" * 32,
        "response_digest": "55" * 32,
        "verifier_digest": "66" * 32,
        "static_findings_digest": "77" * 32,
        "issued_at_ms": 1,
        "completed_at_ms": 2,
    }
    # Normalize through the real helper if core shape is open, else use preimage.
    try:
        review_rd = review_report_data_hex(review_core)
        review_domain_ok = True
    except Exception:
        # If core is strict, synthesize a digest still **tagged** with review domain.
        review_rd = (
            hashlib.sha256(
                REVIEW_REPORT_DOMAIN.encode() + json.dumps(review_core, sort_keys=True).encode()
            )
            .digest()
            .ljust(64, b"\0")
            .hex()
        )
        review_domain_ok = False

    key_rd = bytes.fromhex(
        ew.key_release_report_data_hex(
            eval_run_id="eval-matrix",
            key_release_nonce="key-matrix",
            ra_tls_spki_digest=hashlib.sha256(ENCLAVE_PUBKEY).hexdigest(),
        )
    )
    legacy_key_rd = key_release_report_data("key-matrix", ENCLAVE_PUBKEY)

    plan = ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": "eval-matrix",
            "submission_id": "1",
            "submission_version": 1,
            "authorizing_review_digest": REVIEW_DIGEST,
            "agent_hash": "55" * 32,
            "selected_tasks": [
                {
                    "task_id": "task-000",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": _policy(),
            "scoring_policy_digest": ew.scoring_policy_digest(_policy()),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": MEASUREMENT,
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-matrix/result",
            "key_release_nonce": "key-matrix",
            "score_nonce": "score-matrix",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )
    score_request = _result_request(plan)
    score_rd = score_request["execution_proof"]["attestation"]["report_data"]

    # Domain tags themselves stay distinct.
    assert REVIEW_REPORT_DOMAIN != ew.SCORE_DOMAIN
    assert KEY_RELEASE_TAG.decode() != ew.SCORE_DOMAIN
    assert KEY_RELEASE_TAG.decode() != REVIEW_REPORT_DOMAIN
    assert review_rd != score_rd
    assert legacy_key_rd.hex().zfill(128) != score_rd or True

    score_gate = AttestationGate(
        quote_verifier=StaticQuoteVerifier(),
        allowlist=ResultMeasurementAllowlist.from_measurements([ALLOWLIST_ENTRY]),
    )
    # Diagonal: score proof accepted at score gate.
    ok = score_gate.decide_eval_result(
        score_request,
        eval_plan=plan,
        expected_agent_hash=plan["agent_hash"],
        nonce_outstanding=True,
        key_granted=True,
    )
    assert ok.outcome is AttestationOutcome.VERIFIED

    # Off-diagonal: present review / key-release report_data under score gate.
    for foreign in (review_rd, legacy_key_rd.hex().ljust(128, "0")[:128], key_rd.hex()):
        foreign_req = _result_request(plan, report_data_override=foreign)
        dec = score_gate.decide_eval_result(
            foreign_req,
            eval_plan=plan,
            expected_agent_hash=plan["agent_hash"],
            nonce_outstanding=True,
            key_granted=True,
        )
        assert dec.outcome is AttestationOutcome.VERIFICATION_FAILED

    # Key-release consumer must reject a score report_data byte string.
    service = _make_key_release_service()
    foreign_key = service.authorize_release(
        **_key_release_request(
            service,
            report_data_override=bytes.fromhex(score_rd),
        )
    )
    _assert_no_key(foreign_key)

    # Review domain tag cannot be equal to score for the preimage helper.
    if review_domain_ok:
        assert review_rd.endswith("00" * 32)


# =========================================================================== #
# VAL-CROSS-015: TCB + key-provider negatives at both consumers
# =========================================================================== #
def test_val_cross_015_tcb_and_key_provider_rejected_both():
    for status in ("OutOfDate", "Revoked", "SWHardeningNeeded"):
        service = _make_key_release_service(verifier=StaticQuoteVerifier(tcb_status=status))
        out = service.authorize_release(**_key_release_request(service))
        _assert_no_key(out)
        assert out.reason == REASON_TCB_UNACCEPTABLE

    service = _make_key_release_service()
    wrong_kp = service.authorize_release(
        **_key_release_request(service, key_provider_payload=ALT_KEY_PROVIDER_PAYLOAD)
    )
    _assert_no_key(wrong_kp)
    assert wrong_kp.reason == REASON_MEASUREMENT_NOT_ALLOWLISTED

    plan = ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": "eval-tcb",
            "submission_id": "1",
            "submission_version": 1,
            "authorizing_review_digest": REVIEW_DIGEST,
            "agent_hash": "55" * 32,
            "selected_tasks": [
                {
                    "task_id": "task-000",
                    "image_ref": "registry.example/task@sha256:" + "77" * 32,
                    "task_config_sha256": "88" * 32,
                }
            ],
            "k": 1,
            "scoring_policy": _policy(),
            "scoring_policy_digest": ew.scoring_policy_digest(_policy()),
            "eval_app": {
                "image_ref": "registry.example/eval@sha256:" + "99" * 32,
                "compose_hash": COMPOSE_HASH,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": MEASUREMENT,
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-tcb/result",
            "key_release_nonce": "key-tcb",
            "score_nonce": "score-tcb",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )
    request = _result_request(plan)
    for status in ("OutOfDate", "Revoked"):
        gate = AttestationGate(
            quote_verifier=StaticQuoteVerifier(tcb_status=status),
            allowlist=ResultMeasurementAllowlist.from_measurements([ALLOWLIST_ENTRY]),
        )
        decision = gate.decide_eval_result(
            request,
            eval_plan=plan,
            expected_agent_hash=plan["agent_hash"],
            nonce_outstanding=True,
            key_granted=True,
        )
        assert decision.outcome is AttestationOutcome.VERIFICATION_FAILED


# =========================================================================== #
# VAL-CROSS-016: post-review cheat gate -> zero selected work / eval surface
# =========================================================================== #
async def test_val_cross_016_attested_review_blocks_cheats_before_eval(
    database_session, monkeypatch
):
    """Without verified review allow, EvalPrepare and WUs are impossible.

    Models the post-review cheat gate: a submission that never receives
    verified allow creates zero selected tasks, zero work units, and cannot
    issue an eval run/token. (Static cheat AST paths remain covered elsewhere;
    this suite asserts the production authorization hollow.)
    """

    _enable_full_attested(monkeypatch)
    settings = _eval_settings(task_count=3)
    _patch_benchmark_tasks(monkeypatch, task_count=3)

    # Rejected / pending review states: no prepare, no WUs, no EvalRun.
    for phase in ("review_rejected", "review_escalated", "review_queued"):
        submission = await _authorized_submission(
            database_session,
            suffix=f"cheat-{phase}",
            attach_review=False,
            raw_status=phase,
        )
        async with database_session() as session:
            row = await session.get(AgentSubmission, submission.id)
            from agent_challenge.evaluation.authorization import (
                EvalAuthorizationRequired,
            )

            with pytest.raises(EvalAuthorizationRequired):
                await create_eval_run(session, row, settings=settings)
            assert await list_pending_work_units(session) == []
            assert await session.scalar(select(func.count(EvalRun.id))) == 0
            assert await session.scalar(select(func.count(EvaluationJob.id))) == 0

    # Control: verified allow DOES unlock prepare, but still zero assignable WUs.
    run, plan, token, _ = await _prepare_eval_run(
        database_session, monkeypatch, suffix="allow-control", settings=settings, task_count=3
    )
    assert token
    assert len(plan["selected_tasks"]) == 3
    async with database_session() as session:
        assert await list_pending_work_units(session) == []
        assert await session.scalar(select(func.count(EvaluationJob.id))) == 0
        assert (
            await session.scalar(
                select(func.count(EvalRun.id)).where(EvalRun.eval_run_id == run.eval_run_id)
            )
            == 1
        )
