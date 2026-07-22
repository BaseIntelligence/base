"""VAL-ACAT-016 / 050–054: eval-agent OpenRouter only inside measured eval CVM.

- Guest-only measured OpenRouter with planned+observed digests bound into score materials
- Base /llm/v1 and BASE_GATEWAY_TOKEN refuse (GatewayConfigError never production success)
- Tools-only mode fails closed on model egress
- Dual-flag matrix: only ON/ON may emit production scores
- create_review_session retains harness identity
- live helpers require_real_or_digests / admit_production_from_bound_outcome
"""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.direct_result import extract_agent_llm_materials_for_score
from agent_challenge.evaluation.eval_agent_llm import (
    BASE_GATEWAY_ENV_NAMES,
    BASE_MASTER_KIND,
    MEASURED_EVAL_CVM_KIND,
    MINER_LAPTOP_KIND,
    MODE_MEASURED_OPENROUTER,
    MODE_TOOLS_ONLY,
    REFUSE_BASE_GATEWAY,
    REFUSE_BASE_GATEWAY_URL,
    REFUSE_DIGEST_MISMATCH,
    REFUSE_DIGEST_UNBOUND,
    REFUSE_FLAGS_OFF,
    REFUSE_MEASUREMENT,
    REFUSE_TOOLS_ONLY_EGRESS,
    REFUSE_UNMEASURED_OR,
    SIDECAR_PROXY_KIND,
    UNMEASURED_HOST_KIND,
    EvalAgentLlmError,
    admit_eval_agent_llm_for_score,
    assert_no_base_gateway_agent_env,
    assert_no_base_gateway_url,
    bind_eval_agent_or_digests_into_score_materials,
    build_eval_agent_observed_transport,
    build_eval_agent_planned_request,
    flag_matrix_production_emit,
    refuse_base_gateway_assignment_payload,
    require_eval_agent_llm_for_score,
    require_eval_agent_or_digests,
)
from agent_challenge.evaluation.gateway import (
    GATEWAY_TOKEN_ENV,
    LLM_GATEWAY_PATH,
    GatewayConfigError,
    GatewayExecutionConfig,
    agent_gateway_config_from_settings,
)
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    admit_production_score_for_eval_result,
    admit_production_score_from_chain,
    build_score_binding_from_plan_and_digest,
    recompute_key_release_report_data_hex,
)
from agent_challenge.evaluation.score_chain_gate import (
    REFUSE_FLAGS_OFF as SCORE_REFUSE_FLAGS_OFF,
)
from agent_challenge.review.attested_times import FRESHNESS_WINDOW_MS
from agent_challenge.review.canonical import canonical_json_v1
from agent_challenge.review.or_outcome_bind import (
    REVIEW_REPORT_DOMAIN,
    build_decision,
    build_openrouter_observation,
    build_planned_openrouter_request,
    build_policy_observation,
    build_review_core_minimal,
    planned_request_sha256,
    require_real_or_digests,
    review_digest,
    review_report_data_hex,
    sha256_hex,
    transport_observation_sha256,
)
from agent_challenge.review.or_outcome_bind import (
    build_observed_openrouter_transport as build_review_observed,
)
from agent_challenge.review.sessions import CreatedReviewSession, create_review_session
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.selfdeploy.eval import EVAL_REQUIRED_SECRET_ENVS


def _h(data: bytes) -> str:
    return sha256(data).hexdigest()


ROUTING = _h(b'{"order":["eval-agent"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[{"role":"user","content":"x"}]}'
BODY_SHA = _h(BODY)
RESP = b'{"id":"gen-eval-agent","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = _h(RESP)
META = _h(b"eval-agent-or-meta")

MEASUREMENT = {
    "compose_hash": "aa" * 32,
    "os_image_hash": "bb" * 32,
    "mrtd": "cc" * 48,
    "key_provider": "phala-kms",
    "vm_shape": "2c-4g",
}
ALLOWLIST = [dict(MEASUREMENT)]


def _materials() -> dict:
    planned = build_eval_agent_planned_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING,
        model="x-ai/grok-4.5",
    )
    observed = build_eval_agent_observed_transport(
        planned=planned,
        response_body_sha256=RESP_SHA,
        response_body_length=len(RESP),
        metadata_sha256=META,
    )
    return bind_eval_agent_or_digests_into_score_materials(planned=planned, observed=observed)


# ---------------------------------------------------------------------------
# VAL-ACAT-050 — OpenRouter only inside measured eval CVM
# ---------------------------------------------------------------------------


def test_measured_eval_cvm_openrouter_admits_with_digests() -> None:
    mats = _materials()
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=True,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        measurement=MEASUREMENT,
        allowlist=ALLOWLIST,
        claims_model_call=True,
        agent_or_materials=mats,
    )
    assert decision.admitted is True
    assert decision.production_emit_eligible is True
    assert decision.digests_bound is True
    assert decision.planned_request_sha256 == mats["planned_request_sha256"]
    assert decision.transport_observation_sha256 == mats["transport_observation_sha256"]
    assert decision.base_gateway_used is False


def test_gateway_module_unused_for_production_success() -> None:
    """VAL-ACAT-050: Base gateway config is never the production success path."""

    assert agent_gateway_config_from_settings(ChallengeSettings()) is None
    # Residual from_assignment_payload may raise GatewayConfigError — not success.
    with pytest.raises(GatewayConfigError):
        GatewayExecutionConfig.from_assignment_payload({})
    # Production path refuses gateway payload keys instead of succeeding.
    with pytest.raises(EvalAgentLlmError) as exc:
        refuse_base_gateway_assignment_payload(
            {"gateway_token": "t", "gateway_url": "https://master.example"}
        )
    assert exc.value.code == REFUSE_BASE_GATEWAY


def test_unmeasured_runtimes_refuse_openrouter_score_credit() -> None:
    mats = _materials()
    for kind in (
        UNMEASURED_HOST_KIND,
        BASE_MASTER_KIND,
        MINER_LAPTOP_KIND,
        SIDECAR_PROXY_KIND,
        "host_python",
        "",
    ):
        decision = admit_eval_agent_llm_for_score(
            mode=MODE_MEASURED_OPENROUTER,
            dual_flags_on=True,
            runtime_kind=kind or None,
            measurement=MEASUREMENT,
            allowlist=ALLOWLIST,
            claims_model_call=True,
            agent_or_materials=mats,
        )
        assert decision.admitted is False
        assert decision.reason_code == REFUSE_UNMEASURED_OR
        assert decision.production_emit_eligible is False


def test_eval_required_secrets_exclude_base_gateway_pair() -> None:
    """VAL-ACAT-050 evidence: EVAL_REQUIRED_SECRET_ENVS has no Base gateway pair."""

    assert "BASE_GATEWAY_TOKEN" not in EVAL_REQUIRED_SECRET_ENVS
    assert "BASE_LLM_GATEWAY_URL" not in EVAL_REQUIRED_SECRET_ENVS
    for name in BASE_GATEWAY_ENV_NAMES:
        assert name not in EVAL_REQUIRED_SECRET_ENVS


# ---------------------------------------------------------------------------
# VAL-ACAT-051 / 052 — planned + observed digests bind into eval/score materials
# ---------------------------------------------------------------------------


def test_planned_and_observed_digests_bound_into_score_materials() -> None:
    mats = _materials()
    assert len(mats["planned_request_sha256"]) == 64
    assert len(mats["transport_observation_sha256"]) == 64
    digests = require_eval_agent_or_digests(mats)
    assert digests["planned_request_sha256"] == mats["planned_request_sha256"]
    assert digests["transport_observation_sha256"] == mats["transport_observation_sha256"]
    # Live helper used on review path is also exercised (feature wire note).
    require_real_or_digests(
        planned=mats["planned"],
        observed=mats["observed"],
        openrouter_observation=mats["openrouter_observation"],
    )


def test_forged_planned_digest_refuses() -> None:
    mats = _materials()
    forged = dict(mats)
    planned = dict(mats["planned"])
    planned["body_sha256"] = "ff" * 32  # mutates planned digest vs observed link
    forged["planned"] = planned
    with pytest.raises(EvalAgentLlmError) as exc:
        require_eval_agent_or_digests(forged)
    assert exc.value.code in {REFUSE_DIGEST_MISMATCH, REFUSE_DIGEST_UNBOUND}


def test_missing_observed_materials_refuse_score() -> None:
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=True,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        measurement=MEASUREMENT,
        allowlist=ALLOWLIST,
        claims_model_call=True,
        agent_or_materials=None,
    )
    assert decision.admitted is False
    assert decision.production_emit_eligible is False
    assert decision.reason_code in {
        REFUSE_DIGEST_UNBOUND,
        "eval_agent_llm_claim_missing_digests",
    }


def test_mismatched_observed_planned_link_refuses() -> None:
    planned = build_eval_agent_planned_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING,
        model="x-ai/grok-4.5",
    )
    observed = build_eval_agent_observed_transport(
        planned=planned,
        response_body_sha256=RESP_SHA,
        response_body_length=len(RESP),
        metadata_sha256=META,
    )
    bad_observed = dict(observed)
    bad_observed["planned_request_sha256"] = "00" * 32
    with pytest.raises(EvalAgentLlmError) as exc:
        bind_eval_agent_or_digests_into_score_materials(planned=planned, observed=bad_observed)
    assert exc.value.code == REFUSE_DIGEST_MISMATCH


def test_score_empty_on_digest_reject_via_require() -> None:
    with pytest.raises(EvalAgentLlmError):
        require_eval_agent_llm_for_score(
            mode=MODE_MEASURED_OPENROUTER,
            dual_flags_on=True,
            runtime_kind=MEASURED_EVAL_CVM_KIND,
            measurement=MEASUREMENT,
            allowlist=ALLOWLIST,
            claims_model_call=True,
            agent_or_materials=None,
        )


# ---------------------------------------------------------------------------
# VAL-ACAT-053 — forbid Base gateway URLs and tokens
# ---------------------------------------------------------------------------


def test_base_llm_v1_url_refuses() -> None:
    with pytest.raises(EvalAgentLlmError) as exc:
        assert_no_base_gateway_url(f"https://master.example{LLM_GATEWAY_PATH}")
    assert exc.value.code == REFUSE_BASE_GATEWAY_URL

    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=True,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        measurement=MEASUREMENT,
        allowlist=ALLOWLIST,
        claims_model_call=True,
        agent_or_materials=_materials(),
        gateway_url=f"https://master.example{LLM_GATEWAY_PATH}",
        gateway_token_present=True,
        used_base_llm_v1=True,
    )
    assert decision.admitted is False
    assert decision.reason_code in {REFUSE_BASE_GATEWAY, REFUSE_BASE_GATEWAY_URL}
    assert decision.base_gateway_used is True
    assert decision.production_emit_eligible is False


def test_base_gateway_token_env_refuses() -> None:
    with pytest.raises(EvalAgentLlmError) as exc:
        assert_no_base_gateway_agent_env({GATEWAY_TOKEN_ENV: "scoped-token"})
    assert exc.value.code == REFUSE_BASE_GATEWAY


def test_llm_gateway_path_constant_is_forbidden_surface() -> None:
    assert LLM_GATEWAY_PATH == "/llm/v1"
    with pytest.raises(EvalAgentLlmError):
        assert_no_base_gateway_url(f"http://127.0.0.1:8000{LLM_GATEWAY_PATH}/chat/completions")


# ---------------------------------------------------------------------------
# VAL-ACAT-054 — flag-off residual cannot emit production scores
# ---------------------------------------------------------------------------


def test_flag_matrix_only_dual_on_emits() -> None:
    matrix = flag_matrix_production_emit(
        phala_attestation_enabled=True,
        attested_review_enabled=True,
    )
    assert matrix["dual_flags_on"] is True
    assert matrix["production_emit"] is True
    assert matrix["refuse_code"] is None
    by = {
        (r["phala_attestation_enabled"], r["attested_review_enabled"]): r for r in matrix["matrix"]
    }
    assert by[(True, True)]["production_emit"] is True
    assert by[(True, False)]["production_emit"] is False
    assert by[(False, True)]["production_emit"] is False
    assert by[(False, False)]["production_emit"] is False
    assert by[(False, False)]["refuse_code"] == REFUSE_FLAGS_OFF


def test_flag_off_refuses_agent_llm_and_score_chain() -> None:
    mats = _materials()
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=False,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        measurement=MEASUREMENT,
        allowlist=ALLOWLIST,
        claims_model_call=True,
        agent_or_materials=mats,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_FLAGS_OFF
    assert decision.production_emit_eligible is False

    chain = admit_production_score_from_chain(
        dual_flags_on=False,
        cached_review_allow=True,
        key_granted_flag=True,
        offline_ast_pass=True,
        master_status_green=True,
        cached_score_ok=True,
        eval_plan={"eval_run_id": "x"},
    )
    assert chain.admitted is False
    assert chain.production_emit is False
    assert chain.score is None
    assert chain.reason_code == SCORE_REFUSE_FLAGS_OFF


def test_flag_matrix_any_off_has_no_weight_material() -> None:
    for phala, review in ((False, False), (True, False), (False, True)):
        m = flag_matrix_production_emit(
            phala_attestation_enabled=phala,
            attested_review_enabled=review,
        )
        assert m["production_emit"] is False
        assert m["refuse_code"] == REFUSE_FLAGS_OFF


# ---------------------------------------------------------------------------
# VAL-ACAT-016 — tools-only vs measured mode
# ---------------------------------------------------------------------------


def test_tools_only_admits_without_model_call() -> None:
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_TOOLS_ONLY,
        dual_flags_on=True,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        claims_model_call=False,
    )
    assert decision.admitted is True
    assert decision.reason_code == "eval_agent_tools_only"
    assert decision.digests_bound is False


def test_tools_only_fails_closed_on_model_egress() -> None:
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_TOOLS_ONLY,
        dual_flags_on=True,
        claims_model_call=True,
        agent_or_materials=_materials(),
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_TOOLS_ONLY_EGRESS


def test_measurement_allowlist_miss_refuses() -> None:
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=True,
        runtime_kind=MEASURED_EVAL_CVM_KIND,
        measurement={**MEASUREMENT, "compose_hash": "dd" * 32},
        allowlist=ALLOWLIST,
        claims_model_call=True,
        agent_or_materials=_materials(),
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_MEASUREMENT


# ---------------------------------------------------------------------------
# Live call-site wire checks (feature text)
# ---------------------------------------------------------------------------


def test_create_review_session_retains_harness_identity_contract() -> None:
    assert callable(create_review_session)
    names = {f.name for f in fields(CreatedReviewSession)}
    assert "harness_identity" in names


def test_digest_builders_roundtrip_sha() -> None:
    planned = build_eval_agent_planned_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING,
        model="x-ai/grok-4.5",
    )
    observed = build_eval_agent_observed_transport(
        planned=planned,
        response_body_sha256=RESP_SHA,
        response_body_length=len(RESP),
        metadata_sha256=META,
    )
    assert planned_request_sha256(planned) == sha256_hex(canonical_json_v1(planned))
    assert transport_observation_sha256(observed) == sha256_hex(canonical_json_v1(observed))
    assert observed["planned_request_sha256"] == planned_request_sha256(planned)


# ---------------------------------------------------------------------------
# Emission-wrapper wire (VAL-ACAT-016/050–052 fix): for_eval_result + extract
# ---------------------------------------------------------------------------

_T0 = 1_700_000_000_000
_MS_23H = FRESHNESS_WINDOW_MS - 60_000
_REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
_AGENT_HASH = "55" * 32
_SPKI = "aa" * 32
_REVIEW_ROUTING = sha256_hex(b'{"order":["score-chain-agent-or"]}')
_REVIEW_BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
_REVIEW_BODY_SHA = sha256_hex(_REVIEW_BODY)
_REVIEW_RESP = b'{"id":"gen-score-agent","model":"x-ai/grok-4.5","choices":[]}'
_REVIEW_RESP_SHA = sha256_hex(_REVIEW_RESP)
_REVIEW_META = sha256_hex(b"meta-score-agent-or")


def _os_image_hash() -> str:
    from agent_challenge.keyrelease.quote import os_image_hash_from_registers

    return os_image_hash_from_registers(_REGS["mrtd"], _REGS["rtmr1"], _REGS["rtmr2"])


def _review_times(*, issued: int = _T0, received: int = _T0 + _MS_23H) -> dict[str, int]:
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


def _emission_review_envelope() -> dict[str, Any]:
    planned = build_planned_openrouter_request(
        body_sha256=_REVIEW_BODY_SHA,
        body_length=len(_REVIEW_BODY),
        routing_sha256=_REVIEW_ROUTING,
    )
    p_digest = planned_request_sha256(planned)
    observed = build_review_observed(
        planned_request_sha256_=p_digest,
        response_body_sha256=_REVIEW_RESP_SHA,
        response_body_length=len(_REVIEW_RESP),
        metadata_sha256=_REVIEW_META,
    )
    or_obs = build_openrouter_observation(
        planned=planned,
        observed=observed,
        request_body_sha256=_REVIEW_BODY_SHA,
        request_body_length=len(_REVIEW_BODY),
        response_id="gen-score-agent",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-agent-or",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-agent-or",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-agent-or",
        routing_sha256=_REVIEW_ROUTING,
    )
    rules = {
        "rules_version": "rules-v1",
        "rules_bundle_sha256": "11" * 32,
        "rules_files": [".rules/acceptance.md"],
        "rules_file_digests": {".rules/acceptance.md": "22" * 32},
        "rules_policy_text_sha256": "33" * 32,
    }
    core = build_review_core_minimal(
        session_id="rs-agent-or-emit",
        assignment_id="ra-agent-or-emit",
        submission_id="sub-agent-or-emit",
        review_nonce="nonce-agent-or-emit",
        assignment_digest="aa" * 32,
        rules_observation=rules,
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict="allow"),
        times=_review_times(),
    )
    return {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": review_digest(core),
        "report_data_hex": review_report_data_hex(core),
        "review_core": core,
    }


def _emission_plan(*, authorizing_review_digest: str) -> dict[str, Any]:
    os_hash = _os_image_hash()
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
            "eval_run_id": "eval-agent-or-emit-1",
            "submission_id": "submission-agent-or-emit-1",
            "submission_version": 1,
            "authorizing_review_digest": authorizing_review_digest,
            "agent_hash": _AGENT_HASH,
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
                "compose_hash": "ab" * 32,
                "app_identity": "agent-challenge-eval-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "aa" * 32,
                "kms_public_key_sha256": sha256(bytes.fromhex("aa" * 32)).hexdigest(),
                "measurement": {
                    **_REGS,
                    "os_image_hash": os_hash,
                    "key_provider": "phala",
                    "vm_shape": "tdx-small",
                },
            },
            "key_release_endpoint": "validator.example:8701",
            "result_endpoint": "/evaluation/v1/runs/eval-agent-or-emit-1/result",
            "key_release_nonce": "key-release-agent-or-emit-1",
            "score_nonce": "score-nonce-agent-or-emit-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _emission_grant(plan: dict[str, Any]) -> dict[str, Any]:
    rd = recompute_key_release_report_data_hex(
        eval_run_id=str(plan["eval_run_id"]),
        key_release_nonce=str(plan["key_release_nonce"]),
        ra_tls_spki_digest=_SPKI,
    )
    return {
        "domain": KEY_RELEASE_DOMAIN,
        "schema_version": 2,
        "eval_run_id": plan["eval_run_id"],
        "key_release_nonce": plan["key_release_nonce"],
        "ra_tls_spki_digest": _SPKI,
        "report_data_hex": rd,
        "agent_hash": plan.get("agent_hash"),
    }


def _emission_chain_kit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str, dict]:
    env = _emission_review_envelope()
    plan = _emission_plan(authorizing_review_digest=env["review_digest"])
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan

    record = build_score_record_from_eval_plan(plan, {"task-a": [1.0]})
    scores_digest = ew.score_record_digest(record)
    binding = build_score_binding_from_plan_and_digest(
        eval_plan=plan,
        scores_digest=scores_digest,
    )
    return plan, env, _emission_grant(plan), scores_digest, binding


def test_for_eval_result_admits_measured_agent_openrouter() -> None:
    """Emission wrapper succeeds when agent OR digests + measured runtime are wired."""

    plan, env, grant, sd, binding = _emission_chain_kit()
    mats = _materials()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        agent_or_materials=mats,
        agent_llm_runtime_kind=MEASURED_EVAL_CVM_KIND,
        agent_llm_measurement=MEASUREMENT,
        agent_llm_allowlist=ALLOWLIST,
        claims_agent_model_call=True,
    )
    assert decision.admitted is True, decision.reason_code
    assert decision.production_emit is True
    assert decision.partial_score is False
    assert decision.score is None


def test_for_eval_result_refuses_claim_without_digests() -> None:
    """Live dual-flag emission refuses model claim when planned/observed missing."""

    plan, env, grant, sd, binding = _emission_chain_kit()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        claims_agent_model_call=True,
        agent_or_materials=None,
        agent_llm_runtime_kind=MEASURED_EVAL_CVM_KIND,
        agent_llm_measurement=MEASUREMENT,
        agent_llm_allowlist=ALLOWLIST,
    )
    assert decision.admitted is False
    assert decision.production_emit is False
    assert decision.partial_score is False
    assert decision.score is None
    assert decision.reason_code in {
        REFUSE_DIGEST_UNBOUND,
        "eval_agent_llm_claim_missing_digests",
    }


def test_for_eval_result_refuses_unmeasured_runtime() -> None:
    plan, env, grant, sd, binding = _emission_chain_kit()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        claims_agent_model_call=True,
        agent_or_materials=_materials(),
        agent_llm_runtime_kind=UNMEASURED_HOST_KIND,
        agent_llm_measurement=MEASUREMENT,
        agent_llm_allowlist=ALLOWLIST,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_UNMEASURED_OR
    assert decision.production_emit is False


def test_for_eval_result_refuses_base_gateway_residue() -> None:
    plan, env, grant, sd, binding = _emission_chain_kit()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        claims_agent_model_call=True,
        agent_or_materials=_materials(),
        agent_llm_runtime_kind=MEASURED_EVAL_CVM_KIND,
        agent_llm_measurement=MEASUREMENT,
        agent_llm_allowlist=ALLOWLIST,
        agent_gateway_url=f"https://master.example{LLM_GATEWAY_PATH}",
        agent_gateway_token_present=True,
        agent_used_base_llm_v1=True,
    )
    assert decision.admitted is False
    assert decision.reason_code in {REFUSE_BASE_GATEWAY, REFUSE_BASE_GATEWAY_URL}
    assert decision.production_emit is False
    assert decision.score is None


def test_for_eval_result_tools_only_default_without_claim() -> None:
    """No claim / materials / residue → tools-only default still admits chain."""

    plan, env, grant, sd, binding = _emission_chain_kit()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        score_nonce_outstanding=True,
        # Explicit no claim; tools-only path does not require digests.
        claims_agent_model_call=False,
        agent_or_materials=None,
    )
    assert decision.admitted is True, decision.reason_code
    assert decision.production_emit is True


def test_extract_agent_llm_materials_forwards_claim_and_digests() -> None:
    """process_direct_eval_result extractor surfaces agent_or bags into kwargs."""

    plan, env, grant, sd, binding = _emission_chain_kit()
    _ = env, grant, sd, binding
    mats = _materials()
    raw = {
        "agent_llm": {
            "mode": MODE_MEASURED_OPENROUTER,
            "claims_model_call": True,
            "runtime_kind": MEASURED_EVAL_CVM_KIND,
            "materials": mats,
            "measurement": MEASUREMENT,
            "allowlist": ALLOWLIST,
        }
    }
    extracted = extract_agent_llm_materials_for_score(
        plan=plan,
        validated={"scores_digest": "00" * 32},
        raw_request=raw,
        settings=None,
    )
    assert extracted["claims_agent_model_call"] is True
    assert extracted["agent_llm_mode"] == MODE_MEASURED_OPENROUTER
    assert extracted["agent_llm_runtime_kind"] == MEASURED_EVAL_CVM_KIND
    assert extracted["agent_or_materials"] is not None
    assert (
        extracted["agent_or_materials"]["planned_request_sha256"] == mats["planned_request_sha256"]
    )

    # Emission path driven by extractor: must re-check digests via for_eval_result.
    plan2, env2, grant2, sd2, binding2 = _emission_chain_kit()
    decision = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        eval_plan=plan2,
        review_envelope=env2,
        key_release_grant=grant2,
        key_granted_flag=True,
        score_binding=binding2,
        score_report_data_hex=ew.score_report_data_hex(binding2),
        scores_digest=sd2,
        score_nonce_outstanding=True,
        **extracted,
    )
    assert decision.admitted is True, decision.reason_code


def test_extract_agent_llm_materials_surfaces_gateway_residue() -> None:
    extracted = extract_agent_llm_materials_for_score(
        plan={},
        validated={},
        raw_request={
            "agent_env": {"BASE_GATEWAY_TOKEN": "scoped"},
            "gateway_url": f"https://master.example{LLM_GATEWAY_PATH}/chat/completions",
            "used_base_llm_v1": True,
        },
    )
    assert extracted["agent_gateway_token_present"] is True
    assert extracted["agent_used_base_llm_v1"] is True
    assert extracted["agent_gateway_url"] is not None
    assert "/llm/v1" in str(extracted["agent_gateway_url"])


def test_from_chain_matches_for_eval_result_on_agent_refuse() -> None:
    """from_chain and for_eval_result must agree (no orphan helper-only path)."""

    plan, env, grant, sd, binding = _emission_chain_kit()
    common = dict(
        eval_plan=plan,
        review_envelope=env,
        key_release_grant=grant,
        key_granted_flag=True,
        score_binding=binding,
        score_report_data_hex=ew.score_report_data_hex(binding),
        scores_digest=sd,
        claims_agent_model_call=True,
        agent_or_materials=None,
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        agent_llm_runtime_kind=MEASURED_EVAL_CVM_KIND,
        agent_llm_measurement=MEASUREMENT,
        agent_llm_allowlist=ALLOWLIST,
    )
    via_wrapper = admit_production_score_for_eval_result(
        settings_dual_flags_on=True,
        score_nonce_outstanding=True,
        **common,
    )
    via_chain = admit_production_score_from_chain(
        dual_flags_on=True,
        score_nonce_state="outstanding",
        **common,
    )
    assert via_wrapper.admitted is False
    assert via_chain.admitted is False
    assert via_wrapper.reason_code == via_chain.reason_code
