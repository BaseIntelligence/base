"""VAL-ACAT-011/012/017/018/030–037/039/040: full score-chain fail-closed admission.

Production score emission under dual flags ON requires conjunction:
  review (bound times + ≤24h + OR digests + allow)
  + key-release RA-TLS grant re-checked at admission
  + score-domain report_data / nonce / domain separation
Any single ablation refuses with zero partial score. AST-only never emits.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.llm_rules_residual import (
    MEASURED_RESIDUAL_KIND,
    bind_package_residual_into_review_materials,
    build_package_residual_materials,
)
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    REFUSE_AST_ONLY,
    REFUSE_DOMAIN_CONFUSION,
    REFUSE_FLAGS_OFF,
    REFUSE_INCOMPLETE_CHAIN,
    REFUSE_KEY_RELEASE_MISMATCH,
    REFUSE_MISSING_KEY_RELEASE,
    REFUSE_NONCE_REPLAY,
    REFUSE_NONCE_STALE,
    REFUSE_REVIEW,
    REFUSE_REVIEW_STALE,
    REFUSE_SCORE_DOMAIN,
    REFUSE_STICKY,
    REFUSE_TAMPERED,
    REVIEW_DOMAIN,
    SCORE_DOMAIN,
    ScoreChainAdmissionError,
    admit_production_score_for_eval_result,
    admit_production_score_from_chain,
    build_score_binding_from_plan_and_digest,
    recompute_key_release_report_data_hex,
    require_production_score_from_chain,
    verify_key_release_grant,
)
from agent_challenge.review.attested_times import FRESHNESS_WINDOW_MS
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

PACKAGE_TREE_SHA = "bb" * 32

T0 = 1_700_000_000_000
MS_23H = FRESHNESS_WINDOW_MS - 60_000
MS_24H = FRESHNESS_WINDOW_MS
MS_24H_PLUS = FRESHNESS_WINDOW_MS + 1

REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "ab" * 32
AGENT_HASH = "55" * 32
OS_IMAGE_HASH = "66" * 32  # synthetic; plan validation may only check hex form
ROUTING = sha256_hex(b'{"order":["score-chain"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-score","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-score-chain")
SPKI = "aa" * 32


def _times(*, issued: int = T0, received: int = T0 + MS_23H) -> dict[str, int]:
    base = min(issued, received)
    return {
        "issued_at_ms": issued,
        "started_at_ms": base,
        "model_call_marked_at_ms": base + 1,
        "request_started_at_ms": base + 2,
        "request_finished_at_ms": base + 3,
        "verifier_finished_at_ms": base + 4,
        "report_finished_at_ms": base + 5,
        "expires_at_ms": max(issued, received) + 3_600_000,
        "submission_received_at_ms": received,
    }


def _rules() -> dict:
    return {
        "rules_version": "rules-v1",
        "rules_bundle_sha256": "11" * 32,
        "rules_files": [".rules/acceptance.md"],
        "rules_file_digests": {".rules/acceptance.md": "22" * 32},
        "rules_policy_text_sha256": "33" * 32,
    }


def _review_core(
    *,
    verdict: str = "allow",
    issued: int = T0,
    received: int = T0 + MS_23H,
) -> dict[str, Any]:
    planned = build_planned_openrouter_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_observed_openrouter_transport(
        planned_request_sha256_=p_digest,
        response_body_sha256=RESP_SHA,
        response_body_length=len(RESP),
        metadata_sha256=META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=BODY_SHA,
        request_body_length=len(BODY),
        response_id="gen-score-chain",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-score-chain",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-score-chain",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-score-chain",
        routing_sha256=ROUTING,
    )
    return build_review_core_minimal(
        session_id="rs-score-chain",
        assignment_id="ra-score-chain",
        submission_id="sub-score-chain",
        review_nonce="nonce-score-chain-review",
        assignment_digest="aa" * 32,
        rules_observation=_rules(),
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(issued=issued, received=received),
    )


def _review_envelope(
    *,
    verdict: str = "allow",
    issued: int = T0,
    received: int = T0 + MS_23H,
    mutilate: bool = False,
    domain: str = REVIEW_REPORT_DOMAIN,
    include_package_residual: bool = True,
    package_tree_sha: str = PACKAGE_TREE_SHA,
) -> dict[str, Any]:
    core = _review_core(verdict=verdict, issued=issued, received=received)
    rd = review_report_data_hex(core)
    if mutilate:
        rd = ("ff" * 32) + ("00" * 32)
    env: dict[str, Any] = {
        "schema_version": 1,
        "domain": domain,
        "review_digest": review_digest(core),
        "report_data_hex": rd,
        "review_core": core,
    }
    if include_package_residual and verdict == "allow" and not mutilate:
        materials = build_package_residual_materials(
            residual_verdict="allow",
            rules_bundle_sha256="11" * 32,
            rules_version="rules-v1",
            rules_file_digests={".rules/acceptance.md": "22" * 32},
            package_tree_sha=package_tree_sha,
            residual_kind=MEASURED_RESIDUAL_KIND,
            rules_policy_text_sha256="33" * 32,
            harness_kind="measured_review_cvm_script_zip",
        )
        env = bind_package_residual_into_review_materials(
            envelope=env,
            materials=materials,
        )["envelope"]
    return env


def _os_image_hash() -> str:
    from agent_challenge.keyrelease.quote import os_image_hash_from_registers

    return os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])


def _plan(*, authorizing_review_digest: str | None = None) -> dict[str, Any]:
    os_hash = _os_image_hash()
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    core = _review_core()
    review_d = authorizing_review_digest or review_digest(core)
    return ew.validate_eval_plan(
        {
            "schema_version": 1,
            "eval_run_id": "eval-score-chain-1",
            "submission_id": "submission-score-chain-1",
            "submission_version": 1,
            "authorizing_review_digest": review_d,
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
                    **REGS,
                    "os_image_hash": os_hash,
                    "key_provider": "phala",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-score-chain-1/result",
            "key_release_nonce": "key-release-score-chain-1",
            "score_nonce": "score-nonce-score-chain-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _grant(
    plan: dict[str, Any],
    *,
    eval_run_id: str | None = None,
    domain: str = KEY_RELEASE_DOMAIN,
    spki: str = SPKI,
    mutilate_rd: bool = False,
    agent_hash: str | None = None,
) -> dict[str, Any]:
    er = eval_run_id if eval_run_id is not None else plan["eval_run_id"]
    nonce = plan["key_release_nonce"]
    rd = recompute_key_release_report_data_hex(
        eval_run_id=er,
        key_release_nonce=nonce,
        ra_tls_spki_digest=spki,
    )
    if mutilate_rd:
        rd = ("00" * 32) + ("ff" * 32)
    out: dict[str, Any] = {
        "domain": domain,
        "schema_version": 2,
        "eval_run_id": er,
        "key_release_nonce": nonce,
        "ra_tls_spki_digest": spki,
        "report_data_hex": rd,
    }
    if agent_hash is not None:
        out["agent_hash"] = agent_hash
    return out


def _scores_digest(plan: dict[str, Any]) -> str:
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    return ew.score_record_digest(record)


def _score_binding(plan: dict[str, Any], scores_digest: str) -> dict[str, Any]:
    return build_score_binding_from_plan_and_digest(
        eval_plan=plan,
        scores_digest=scores_digest,
    )


def _full_ok(**overrides: Any) -> dict[str, Any]:
    plan = _plan()
    env = _review_envelope()
    # Force authorizing_review_digest to match envelope.
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    grant = _grant(plan)
    kwargs: dict[str, Any] = {
        "dual_flags_on": True,
        "review_envelope": env,
        "key_release_grant": grant,
        "key_granted_flag": True,
        "eval_plan": plan,
        "score_binding": binding,
        "score_report_data_hex": ew.score_report_data_hex(binding),
        "scores_digest": sd,
        "score_nonce_state": "outstanding",
        "offline_ast_pass": False,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# VAL-ACAT-030 / 040: only all-on control admits
# ---------------------------------------------------------------------------


def test_full_chain_control_admits_production() -> None:
    decision = admit_production_score_from_chain(**_full_ok())
    assert decision.admitted is True
    assert decision.production_emit is True
    assert decision.partial_score is False
    assert decision.score is None  # gate is binary; score set elsewhere
    assert decision.reverify_exercised is True
    assert REVIEW_DOMAIN in decision.domains_checked
    assert KEY_RELEASE_DOMAIN in decision.domains_checked
    assert SCORE_DOMAIN in decision.domains_checked
    assert decision.reason_code == "score_chain_verified"


def test_require_raises_on_refuse() -> None:
    with pytest.raises(ScoreChainAdmissionError) as exc:
        require_production_score_from_chain(
            **_full_ok(key_release_grant=None, key_granted_flag=False)
        )
    assert exc.value.code == REFUSE_MISSING_KEY_RELEASE


# ---------------------------------------------------------------------------
# VAL-ACAT-031: ablation of each required artifact → refuse, no partial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        ({"review_envelope": None, "cached_review_allow": True}, REFUSE_INCOMPLETE_CHAIN),
        ({"key_release_grant": None, "key_granted_flag": False}, REFUSE_MISSING_KEY_RELEASE),
        ({"score_binding": None}, REFUSE_INCOMPLETE_CHAIN),
        ({"score_report_data_hex": None, "score_binding": None}, REFUSE_INCOMPLETE_CHAIN),
    ],
)
def test_missing_chain_material_refuses_zero_partial(
    mutation: dict[str, Any], expected_code: str
) -> None:
    decision = admit_production_score_from_chain(**_full_ok(**mutation))
    assert decision.admitted is False
    assert decision.production_emit is False
    assert decision.partial_score is False
    assert decision.score is None
    assert decision.reason_code == expected_code


def test_db_key_granted_flag_alone_without_grant_materials_refuses() -> None:
    """VAL-ACAT-037: historic env-inject / DB key_granted alone is insufficient."""

    decision = admit_production_score_from_chain(
        **_full_ok(key_release_grant=None, key_granted_flag=True)
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_MISSING_KEY_RELEASE
    assert decision.partial_score is False


# ---------------------------------------------------------------------------
# VAL-ACAT-032: tamper / domain confusion fail closed
# ---------------------------------------------------------------------------


def test_tampered_review_report_data_refuses() -> None:
    decision = admit_production_score_from_chain(
        **_full_ok(review_envelope=_review_envelope(mutilate=True))
    )
    assert decision.admitted is False
    assert decision.partial_score is False
    assert decision.score is None
    assert decision.reverify_exercised is True
    # re-verify fail class (review)
    assert decision.reason_code in {
        REFUSE_REVIEW,
        REFUSE_TAMPERED,
        "review_attestation_reverify_failed",
    }


def test_tampered_key_release_report_data_refuses() -> None:
    plan = _plan()
    env = _review_envelope()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=_grant(plan, mutilate_rd=True),
        key_granted_flag=True,
        eval_plan=plan,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_TAMPERED
    assert decision.score is None


def test_domain_confusion_review_as_score_refuses() -> None:
    plan = _plan()
    env = _review_envelope()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    # Use review domain on score binding.
    bad_binding = _score_binding(plan, sd)
    bad_binding = dict(bad_binding)
    bad_binding["domain"] = REVIEW_DOMAIN
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=_grant(plan),
        key_granted_flag=True,
        eval_plan=plan,
        score_binding=bad_binding,
        score_report_data_hex=ew.score_report_data_hex(
            _score_binding(plan, sd)
        ),  # mismatch + domain
        scores_digest=sd,
    )
    assert decision.admitted is False
    assert decision.reason_code in {REFUSE_DOMAIN_CONFUSION, REFUSE_SCORE_DOMAIN, REFUSE_TAMPERED}
    assert decision.partial_score is False


def test_domain_confusion_score_materials_as_key_release_refuses() -> None:
    grant = {
        "domain": SCORE_DOMAIN,
        "eval_run_id": "eval-score-chain-1",
        "key_release_nonce": "key-release-score-chain-1",
        "ra_tls_spki_digest": SPKI,
        "report_data_hex": "00" * 64,
    }
    decision = admit_production_score_from_chain(**_full_ok(key_release_grant=grant))
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_DOMAIN_CONFUSION


def test_cross_session_key_release_grant_swap_refuses() -> None:
    """VAL-ACAT-037: grant for another eval_run_id does not authorize."""

    plan = _plan()
    env = _review_envelope()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    foreign = _grant(plan, eval_run_id="eval-FOREIGN-session")
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=foreign,
        key_granted_flag=True,
        eval_plan=plan,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_KEY_RELEASE_MISMATCH


def test_agent_hash_mismatch_on_grant_refuses() -> None:
    plan = _plan()
    env = _review_envelope()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    grant = _grant(plan, agent_hash="ff" * 32)
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        eval_plan=plan,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_KEY_RELEASE_MISMATCH


# ---------------------------------------------------------------------------
# VAL-ACAT-033 / 017: nonce replay / stale
# ---------------------------------------------------------------------------


def test_replayed_score_nonce_refuses() -> None:
    decision = admit_production_score_from_chain(**_full_ok(score_nonce_state="consumed"))
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_NONCE_REPLAY
    assert decision.score is None


def test_expired_score_nonce_refuses() -> None:
    decision = admit_production_score_from_chain(**_full_ok(score_nonce_state="expired"))
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_NONCE_STALE


def test_stale_review_over_24h_refuses_score() -> None:
    env = _review_envelope(received=T0 + MS_24H_PLUS)
    decision = admit_production_score_from_chain(**_full_ok(review_envelope=env))
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_REVIEW_STALE
    assert decision.partial_score is False


def test_exactly_24h_review_still_allows_score_chain() -> None:
    env = _review_envelope(received=T0 + MS_24H)
    plan = _plan()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=_grant(plan),
        key_granted_flag=True,
        eval_plan=plan,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_state="outstanding",
    )
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# VAL-ACAT-034 / 035: offline AST / tutte cannot score alone
# ---------------------------------------------------------------------------


def test_ast_only_green_without_attestation_refuses() -> None:
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        eval_plan=_plan(),
        review_envelope=None,
        key_release_grant=None,
        score_binding=None,
        offline_ast_pass=True,
        cached_review_allow=True,
        master_status_green=True,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_AST_ONLY
    assert decision.production_emit is False
    assert decision.score is None
    assert decision.partial_score is False


def test_ast_pass_does_not_upgrade_missing_key_release() -> None:
    env = _review_envelope()
    plan = _plan()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=None,
        key_granted_flag=False,
        eval_plan=plan,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        offline_ast_pass=True,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_AST_ONLY
    assert decision.score is None


# ---------------------------------------------------------------------------
# VAL-ACAT-018: flag-off cannot emit production scores
# ---------------------------------------------------------------------------


def test_flags_off_refuses_production_even_with_perfect_chain() -> None:
    decision = admit_production_score_from_chain(**_full_ok(dual_flags_on=False))
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_FLAGS_OFF
    assert decision.production_emit is False
    assert decision.score is None


# ---------------------------------------------------------------------------
# VAL-ACAT-012: master green + bad materials refuses
# ---------------------------------------------------------------------------


def test_master_green_with_bad_quote_materials_refuses() -> None:
    decision = admit_production_score_from_chain(
        **_full_ok(
            review_envelope=_review_envelope(mutilate=True),
            master_status_green=True,
            cached_score_ok=True,
        )
    )
    assert decision.admitted is False
    assert decision.production_emit is False


# ---------------------------------------------------------------------------
# VAL-ACAT-039: sticky refuse, no silent downgrade
# ---------------------------------------------------------------------------


def test_prior_reverify_failed_sticky_refuses() -> None:
    decision = admit_production_score_from_chain(
        **_full_ok(prior_reverify_failed=True, offline_ast_pass=True)
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_STICKY
    assert decision.sticky is True
    assert decision.score is None


# ---------------------------------------------------------------------------
# VAL-ACAT-017: domain-separated nonces - score_nonce ≠ key_release_nonce
# ---------------------------------------------------------------------------


def test_key_release_and_score_nonce_must_differ_in_plan_constructor() -> None:
    """eval_wire plan validation already enforces nonce separation."""

    with pytest.raises(ew.EvalWireError):
        policy = {
            "schema_version": 1,
            "per_task_aggregation": "mean",
            "keep_policy": "off",
            "drop_lowest_n": 0,
            "threshold_f64be": None,
        }
        os_hash = _os_image_hash()
        ew.validate_eval_plan(
            {
                "schema_version": 1,
                "eval_run_id": "eval-collide",
                "submission_id": "sub-collide",
                "submission_version": 1,
                "authorizing_review_digest": "66" * 32,
                "agent_hash": AGENT_HASH,
                "package_tree_sha": "bb" * 32,
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
                        "os_image_hash": os_hash,
                        "key_provider": "phala",
                        "vm_shape": "tdx-small",
                    },
                },
                "key_release_endpoint": "validator.example:8701",
                "result_endpoint": "/evaluation/v1/runs/eval-collide/result",
                "key_release_nonce": "SAME-NONCE",
                "score_nonce": "SAME-NONCE",
                "run_token_sha256": "bb" * 32,
                "issued_at_ms": 1,
                "expires_at_ms": 2,
            }
        )


def test_verify_key_release_grant_rejects_colliding_score_nonce() -> None:
    plan = {
        "eval_run_id": "eval-x",
        "key_release_nonce": "nonce-shared",
        "score_nonce": "nonce-shared",
        "agent_hash": AGENT_HASH,
        "package_tree_sha": PACKAGE_TREE_SHA,
    }
    grant = {
        "domain": KEY_RELEASE_DOMAIN,
        "eval_run_id": "eval-x",
        "key_release_nonce": "nonce-shared",
        "ra_tls_spki_digest": SPKI,
        "report_data_hex": recompute_key_release_report_data_hex(
            eval_run_id="eval-x",
            key_release_nonce="nonce-shared",
            ra_tls_spki_digest=SPKI,
        ),
    }
    err, _ = verify_key_release_grant(grant=grant, eval_plan=plan, key_granted_flag=True)
    assert err == REFUSE_DOMAIN_CONFUSION


# ---------------------------------------------------------------------------
# VAL-ACAT-040: single-factor ablation matrix (conjunction all-or-nothing)
# ---------------------------------------------------------------------------


def test_conjunction_ablation_matrix_only_all_on_admits() -> None:
    control = admit_production_score_from_chain(**_full_ok())
    assert control.admitted is True

    ablations: list[tuple[str, dict[str, Any]]] = [
        ("flags_off", {"dual_flags_on": False}),
        ("no_review", {"review_envelope": None}),
        ("stale_review", {"review_envelope": _review_envelope(received=T0 + MS_24H_PLUS)}),
        ("reject_review", {"review_envelope": _review_envelope(verdict="reject")}),
        ("no_key_release", {"key_release_grant": None, "key_granted_flag": False}),
        ("foreign_grant", {}),  # filled below
        ("no_score_binding", {"score_binding": None, "score_report_data_hex": None}),
        ("nonce_replay", {"score_nonce_state": "consumed"}),
        ("nonce_expired", {"score_nonce_state": "expired"}),
        ("sticky", {"prior_reverify_failed": True}),
        ("ast_only", {"review_envelope": None, "offline_ast_pass": True}),
    ]
    # Foreign grant ablation needs plan.
    base = _full_ok()
    plan = base["eval_plan"]
    ablations[5] = (
        "foreign_grant",
        {"key_release_grant": _grant(plan, eval_run_id="EVAL-OTHER")},
    )

    for name, mut in ablations:
        decision = admit_production_score_from_chain(**_full_ok(**mut))
        assert decision.admitted is False, f"ablation {name} unexpectedly admitted"
        assert decision.score is None, f"ablation {name} leaked score"
        assert decision.partial_score is False, f"ablation {name} partial"
        assert decision.production_emit is False, f"ablation {name} production_emit"


# ---------------------------------------------------------------------------
# Convenience wrapper (direct-result path surface)
# ---------------------------------------------------------------------------


def test_admit_production_score_for_eval_result_wrapper() -> None:
    plan = _plan()
    env = _review_envelope()
    plan = dict(plan)
    plan["authorizing_review_digest"] = env["review_digest"]
    sd = _scores_digest(plan)
    binding = _score_binding(plan, sd)
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=_grant(plan),
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
    )
    assert decision.admitted is True
    decision_fail = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=_grant(plan),
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=False,
    )
    assert decision_fail.admitted is False
    assert decision_fail.reason_code == REFUSE_NONCE_REPLAY


def test_refuse_codes_are_stable_nonsecret_strings() -> None:
    for code in (
        REFUSE_INCOMPLETE_CHAIN,
        REFUSE_AST_ONLY,
        REFUSE_FLAGS_OFF,
        REFUSE_MISSING_KEY_RELEASE,
        REFUSE_KEY_RELEASE_MISMATCH,
        REFUSE_NONCE_REPLAY,
        REFUSE_NONCE_STALE,
        REFUSE_DOMAIN_CONFUSION,
        REFUSE_TAMPERED,
        REFUSE_STICKY,
        REFUSE_REVIEW_STALE,
    ):
        assert isinstance(code, str)
        assert code
        assert " " not in code
        assert "secret" not in code.lower()
        assert "key=" not in code.lower()
