"""Durable RA-TLS KR grant materials → score admission (VAL-ACAT-036/037/040).

Scrutiny residual on acat-eval-gate: production KR must stamp reconstructible
grant JSON on EvalRun; score admission must load that durable column (not only
the process-local registry). Multi-worker restart simulation: clear the
in-process map and still admit from durable JSON; missing grant refuse even
when key_granted_at alone is set.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import (
    AgentSubmission,
    EvalRun,
    ReviewAssignment,
    ReviewSession,
)
from agent_challenge.evaluation.authorization import (
    build_key_release_grant_materials,
    create_eval_run,
    load_eval_run_plan,
    mark_eval_key_granted,
    persist_key_release_grant_materials,
    receipt_eval_key_release,
)
from agent_challenge.evaluation.direct_result import _key_release_grant_from_result
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    REFUSE_MISSING_KEY_RELEASE,
    admit_production_score_for_eval_result,
    admit_production_score_from_chain,
    build_score_binding_from_plan_and_digest,
    clear_key_release_grant_for_score,
    load_durable_key_release_grant,
    lookup_key_release_grant_for_score,
    recompute_key_release_report_data_hex,
    register_key_release_grant_for_score,
    verify_key_release_grant,
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
    review_digest,
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.sdk.config import ChallengeSettings

_T0 = 1_700_000_000_000
SPKI = "aa" * 32
COMPOSE_HASH = "ab" * 32
MRTD = "11" * 48
RTMR0 = "22" * 48
RTMR1 = "33" * 48
RTMR2 = "44" * 48
MEASUREMENT = {
    "mrtd": MRTD,
    "rtmr0": RTMR0,
    "rtmr1": RTMR1,
    "rtmr2": RTMR2,
    "os_image_hash": "66" * 32,
    "key_provider": "validator-kms",
    "vm_shape": "tdx-small",
}
_ROUTING = sha256_hex(b'{"order":["durable-kr"]}')
_BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
_BODY_SHA = sha256_hex(_BODY)
_RESP = b'{"id":"gen-durable-kr","model":"x-ai/grok-4.5","choices":[]}'
_RESP_SHA = sha256_hex(_RESP)
_META = sha256_hex(b"meta-durable-kr")

_SUBMISSION_SEQ = 0


def _settings() -> ChallengeSettings:
    return ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        eval_app_image_ref="registry.example/eval@sha256:" + "a" * 64,
        eval_app_compose_hash=COMPOSE_HASH,
        eval_app_identity="agent-challenge-eval-v1",
        eval_app_kms_public_key_hex="07" * 32,
        eval_app_measurement=MEASUREMENT,
        eval_app_measurement_allowlist=(
            {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "compose_hash": COMPOSE_HASH,
                "os_image_hash": MEASUREMENT["os_image_hash"],
            },
        ),
        eval_key_release_endpoint="validator.example:8701",
        eval_k=1,
        evaluation_task_count=1,
    )


def _patch_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_challenge.evaluation.authorization.load_benchmark_tasks",
        lambda: [
            type(
                "Task",
                (),
                {
                    "task_id": "task-a",
                    "docker_image": "registry.example/task@sha256:" + "b" * 64,
                    "prompt": "",
                    "benchmark": "terminal_bench",
                    "metadata": {"content_digest_sha256": "aa" * 32},
                },
            )()
        ],
    )


def _fresh_review_envelope(
    *,
    session_id: str = "rs-durable-kr",
    assignment_id: str = "ra-durable-kr",
    submission_id: str = "sub-durable-kr",
    review_nonce: str = "nonce-durable-kr",
) -> tuple[str, str, str, dict[str, Any]]:
    """Full receipted envelope for create_eval_run re-verify (not cache-only)."""

    planned = build_planned_openrouter_request(
        body_sha256=_BODY_SHA,
        body_length=len(_BODY),
        routing_sha256=_ROUTING,
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
        response_id="gen-durable-kr",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-durable-kr",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-durable-kr",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-durable-kr",
        routing_sha256=_ROUTING,
    )
    core = build_review_core_minimal(
        session_id=session_id,
        assignment_id=assignment_id,
        submission_id=submission_id,
        review_nonce=review_nonce,
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
    digest = review_digest(core)
    rd = review_report_data_hex(core)
    env: dict[str, Any] = {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": digest,
        "report_data_hex": rd,
        "review_core": core,
    }
    # AGATE: create_eval_run / score dual-flag path requires residual + tree sha.
    from agent_challenge.evaluation.llm_rules_residual import (
        MEASURED_RESIDUAL_KIND,
        bind_package_residual_into_review_materials,
        build_package_residual_materials,
    )

    materials = build_package_residual_materials(
        residual_verdict="allow",
        rules_bundle_sha256="11" * 32,
        rules_version="rules-v1",
        rules_file_digests={".rules/acceptance.md": "22" * 32},
        package_tree_sha="bb" * 32,
        residual_kind=MEASURED_RESIDUAL_KIND,
        rules_policy_text_sha256="33" * 32,
        harness_kind="measured_review_cvm_script_zip",
    )
    env = bind_package_residual_into_review_materials(
        envelope=env,
        materials=materials,
    )["envelope"]
    return json.dumps(env, sort_keys=True, separators=(",", ":")), rd, digest, env


async def _authorized_submission(database_session) -> tuple[int, dict[str, Any]]:
    global _SUBMISSION_SEQ
    _SUBMISSION_SEQ += 1
    salt = f"kr-score-{_SUBMISSION_SEQ}".encode()
    sid = f"rs-durable-kr-{_SUBMISSION_SEQ}"
    aid = f"ra-durable-kr-{_SUBMISSION_SEQ}"
    rid = f"nonce-durable-kr-{_SUBMISSION_SEQ}"
    sub_label = f"sub-durable-kr-{_SUBMISSION_SEQ}"
    envelope_json, report_data_hex, digest, env = _fresh_review_envelope(
        session_id=sid,
        assignment_id=aid,
        submission_id=sub_label,
        review_nonce=rid,
    )
    async with database_session() as session:
        submission = AgentSubmission(
            miner_hotkey=f"kr-score-miner-{_SUBMISSION_SEQ}",
            name=f"kr-score-agent-{_SUBMISSION_SEQ}",
            agent_hash=hashlib.sha256(b"agent-" + salt).hexdigest(),
            artifact_uri=f"/tmp/agent-kr-{_SUBMISSION_SEQ}.zip",
            artifact_path=f"/tmp/agent-kr-{_SUBMISSION_SEQ}.zip",
            zip_sha256=hashlib.sha256(b"zip-" + salt).hexdigest(),
            package_tree_sha="bb" * 32,
            zip_size_bytes=3,
            raw_status="review_allowed",
            status="queued",
            effective_status="queued",
            version_number=1,
        )
        session.add(submission)
        await session.flush()
        review_session = ReviewSession(
            session_id=sid,
            submission_id=submission.id,
            artifact_sha256=submission.zip_sha256,
            artifact_size_bytes=3,
            manifest_sha256="11" * 32,
            manifest_entries_sha256="12" * 32,
            authorizing_assignment_id=aid,
            current_assignment_id=aid,
        )
        session.add(review_session)
        await session.flush()
        assignment = ReviewAssignment(
            session_id=review_session.id,
            assignment_id=aid,
            attempt=1,
            assignment_bytes="{}",
            assignment_digest="13" * 32,
            artifact_sha256=submission.zip_sha256,
            rules_snapshot_sha256="14" * 32,
            rules_revision_id="rules-1",
            review_nonce=rid,
            session_token_sha256="15" * 32,
            capability_state="revoked",
            phase="review_allowed",
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            review_report_envelope_json=envelope_json,
            review_report_data_hex=report_data_hex,
            review_digest=digest,
            review_verification_outcome_json=json.dumps(
                {
                    "status": "verified_allow",
                    "terminal": True,
                    "retryable": False,
                    "nonce_consumed": True,
                    "package_residual": env.get("package_residual"),
                },
                separators=(",", ":"),
            ),
        )
        session.add(assignment)
        await session.commit()
        return submission.id, env


async def _create_run(database_session, monkeypatch: pytest.MonkeyPatch):
    submission_id, env = await _authorized_submission(database_session)
    _patch_tasks(monkeypatch)
    async with database_session() as session:
        submission = await session.get(AgentSubmission, submission_id)
        assert submission is not None
        created = await create_eval_run(session, submission, settings=_settings())
        await session.commit()
        return created.run.eval_run_id, created.plan, submission_id, env


def test_build_key_release_grant_materials_closed_shape() -> None:
    grant = build_key_release_grant_materials(
        eval_run_id="eval-1",
        key_release_nonce="kr-nonce-1",
        ra_tls_spki_digest=SPKI,
        agent_hash="55" * 32,
    )
    assert grant["domain"] == KEY_RELEASE_DOMAIN
    assert grant["schema_version"] == 2
    assert grant["eval_run_id"] == "eval-1"
    assert grant["key_release_nonce"] == "kr-nonce-1"
    assert grant["ra_tls_spki_digest"] == SPKI
    assert grant["agent_hash"] == "55" * 32
    expected = recompute_key_release_report_data_hex(
        eval_run_id="eval-1",
        key_release_nonce="kr-nonce-1",
        ra_tls_spki_digest=SPKI,
    )
    assert grant["report_data_hex"] == expected
    err, rd = verify_key_release_grant(
        grant=grant,
        eval_plan={
            "eval_run_id": "eval-1",
            "key_release_nonce": "kr-nonce-1",
            "score_nonce": "score-nonce-1",
            "agent_hash": "55" * 32,
            "package_tree_sha": "bb" * 32,
        },
        key_granted_flag=True,
    )
    assert err is None
    assert rd == expected


@pytest.mark.asyncio
async def test_mark_eval_key_granted_persists_durable_grant_json(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live KR success stamping: durable grant_json + process registry."""

    clear_key_release_grant_for_score()
    eval_run_id, plan, _, _env = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"kr-grant-frame").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        granted = await mark_eval_key_granted(
            session,
            eval_run_id=eval_run_id,
            ra_tls_spki_digest=SPKI,
        )
        assert granted.key_granted_at is not None
        assert granted.key_release_state == "granted"
        assert isinstance(granted.key_release_grant_json, str)
        assert granted.key_release_grant_json
        durable = json.loads(granted.key_release_grant_json)
        assert durable["domain"] == KEY_RELEASE_DOMAIN
        assert durable["eval_run_id"] == eval_run_id
        assert durable["key_release_nonce"] == plan["key_release_nonce"]
        assert durable["ra_tls_spki_digest"] == SPKI
        assert durable["agent_hash"] == plan["agent_hash"]
        assert durable["report_data_hex"] == recompute_key_release_report_data_hex(
            eval_run_id=eval_run_id,
            key_release_nonce=str(plan["key_release_nonce"]),
            ra_tls_spki_digest=SPKI,
        )
        cached = lookup_key_release_grant_for_score(eval_run_id)
        assert cached is not None
        assert cached["ra_tls_spki_digest"] == SPKI
        await session.commit()

    # Restart simulation: wipe process-local map; reload from durable column.
    clear_key_release_grant_for_score(eval_run_id)
    assert lookup_key_release_grant_for_score(eval_run_id) is None
    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        reloaded = load_durable_key_release_grant(run)
        assert reloaded is not None
        assert reloaded["ra_tls_spki_digest"] == SPKI
        assert lookup_key_release_grant_for_score(eval_run_id) is not None


@pytest.mark.asyncio
async def test_score_admission_from_durable_grant_after_process_restart(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KR→score: after clearing process registry, durable JSON still admits."""

    clear_key_release_grant_for_score()
    eval_run_id, plan, _, env = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"kr-score-admit").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        await mark_eval_key_granted(
            session,
            eval_run_id=eval_run_id,
            ra_tls_spki_digest=SPKI,
        )
        await session.commit()

    # Simulate multi-worker / process bounce: only DB remains.
    clear_key_release_grant_for_score()
    assert lookup_key_release_grant_for_score(eval_run_id) is None

    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        plan_reloaded = load_eval_run_plan(run)
        grant = _key_release_grant_from_result(
            plan=plan_reloaded,
            validated={},
            run=run,
            raw_request=None,
        )
        assert grant is not None, "must load durable grant, not process dict alone"
        assert grant["domain"] == KEY_RELEASE_DOMAIN
        plan_for_score = dict(plan_reloaded)
        plan_for_score["authorizing_review_digest"] = env["review_digest"]
        # Synthetic scores_digest (hex) — gate re-checks binding equality only.
        scores_digest = "cd" * 32
        binding = build_score_binding_from_plan_and_digest(
            eval_plan=plan_for_score,
            scores_digest=scores_digest,
        )
        decision = admit_production_score_for_eval_result(
            settings_dual_flags_on=True,
            eval_plan=plan_for_score,
            review_envelope=env,
            key_release_grant=grant,
            key_granted_flag=run.key_granted_at is not None,
            score_binding=binding,
            score_report_data_hex=ew.score_report_data_hex(binding),
            scores_digest=scores_digest,
            score_nonce_outstanding=True,
        )
        assert decision.admitted is True, decision.reason_code
        assert decision.partial_score is False
        assert decision.production_emit is True
        assert KEY_RELEASE_DOMAIN in decision.domains_checked


@pytest.mark.asyncio
async def test_missing_durable_grant_refuses_even_with_key_granted_at(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-ACAT-037: key_granted_at alone is never enough under dual flags."""

    clear_key_release_grant_for_score()
    eval_run_id, plan, _, env = await _create_run(database_session, monkeypatch)
    digest = hashlib.sha256(b"kr-legacy-no-grant").hexdigest()
    async with database_session() as session:
        await receipt_eval_key_release(session, eval_run_id=eval_run_id, body_sha256=digest)
        # Legacy path: grant without SPKI materials.
        granted = await mark_eval_key_granted(session, eval_run_id=eval_run_id)
        assert granted.key_granted_at is not None
        assert granted.key_release_grant_json is None
        await session.commit()

    clear_key_release_grant_for_score()
    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        plan_reloaded = load_eval_run_plan(run)
        grant = _key_release_grant_from_result(
            plan=plan_reloaded,
            validated={},
            run=run,
            raw_request=None,
        )
        assert grant is None
        plan_for_score = dict(plan_reloaded)
        plan_for_score["authorizing_review_digest"] = env["review_digest"]
        scores_digest = "cd" * 32
        binding = build_score_binding_from_plan_and_digest(
            eval_plan=plan_for_score,
            scores_digest=scores_digest,
        )
        decision = admit_production_score_from_chain(
            dual_flags_on=True,
            review_envelope=env,
            key_release_grant=grant,
            key_granted_flag=True,
            eval_plan=plan_for_score,
            score_binding=binding,
            score_report_data_hex=ew.score_report_data_hex(binding),
            scores_digest=scores_digest,
            score_nonce_state="outstanding",
        )
        assert decision.admitted is False
        assert decision.reason_code == REFUSE_MISSING_KEY_RELEASE
        assert decision.score is None
        assert decision.partial_score is False
        _ = plan


@pytest.mark.asyncio
async def test_process_local_dict_alone_insufficient_after_restart(
    database_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If only the process dict held materials and was cleared, load fails closed."""

    clear_key_release_grant_for_score()
    eval_run_id, plan, _, _env = await _create_run(database_session, monkeypatch)
    # Process-only stamp (simulates pre-fix listener that never wrote DB).
    register_key_release_grant_for_score(
        eval_run_id,
        build_key_release_grant_materials(
            eval_run_id=eval_run_id,
            key_release_nonce=str(plan["key_release_nonce"]),
            ra_tls_spki_digest=SPKI,
            agent_hash=str(plan["agent_hash"]),
        ),
    )
    assert lookup_key_release_grant_for_score(eval_run_id) is not None
    clear_key_release_grant_for_score(eval_run_id)
    assert lookup_key_release_grant_for_score(eval_run_id) is None

    async with database_session() as session:
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == eval_run_id))
        assert run is not None
        assert run.key_release_grant_json is None
        grant = _key_release_grant_from_result(
            plan=load_eval_run_plan(run),
            validated={},
            run=run,
            raw_request=None,
        )
        assert grant is None
        materials = build_key_release_grant_materials(
            eval_run_id=eval_run_id,
            key_release_nonce=str(plan["key_release_nonce"]),
            ra_tls_spki_digest=SPKI,
            agent_hash=str(plan["agent_hash"]),
        )
        persist_key_release_grant_materials(run, materials)
        assert run.key_release_grant_json is not None
        clear_key_release_grant_for_score(eval_run_id)
        reloaded = load_durable_key_release_grant(run)
        assert reloaded is not None
        assert reloaded["ra_tls_spki_digest"] == SPKI
        await session.commit()
