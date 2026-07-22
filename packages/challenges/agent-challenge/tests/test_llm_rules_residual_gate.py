"""AGATE measured LLM rules residual + agent model rules (VAL-AGATE-004..007, 012, 013).

TDD contract:
- measured residual required before eval authorizable
- residual fail stops pipeline
- rules digests + residual verdict bound into review materials
- host analyzer alone insufficient for TEE auth
- no closed agent model catalog; ban personal finetunes
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_challenge.evaluation.eval_agent_llm import (
    MODE_MEASURED_OPENROUTER,
    admit_eval_agent_llm_for_score,
    build_eval_agent_planned_request,
)
from agent_challenge.evaluation.fresh_review_gate import (
    admit_eval_cvm_fresh_review,
    admit_eval_cvm_launch_from_assignment,
)
from agent_challenge.evaluation.llm_rules_residual import (
    HOST_ANALYZER_KIND,
    MEASURED_RESIDUAL_KIND,
    REFUSE_FINETUNE,
    REFUSE_HOST_ONLY,
    REFUSE_PACKAGE_TREE_MISSING,
    REFUSE_RESIDUAL_FAIL,
    REFUSE_RESIDUAL_MISSING,
    REFUSE_RESIDUAL_UNBOUND,
    PackageResidualError,
    admit_package_residual_for_eval,
    agent_model_requires_closed_catalog,
    assert_no_closed_agent_model_catalog,
    bind_package_residual_into_review_materials,
    build_package_residual_materials,
    filter_agent_model_or_refuse,
    host_analyzer_alone_insufficient,
    inventory_residual_gate,
    is_personal_finetune_model,
    refuse_personal_finetune_model,
    require_package_residual_for_eval,
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

T0 = 1_700_000_000_000
ROUTING = sha256_hex(b'{"order":["x"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-2","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-residual-gate")
TREE = "ab" * 32
BUNDLE = "11" * 32
RULES_V = "rules-v-agate-1"
FILE_DIG = {".rules/acceptance.md": "22" * 32, ".rules/security.md": "33" * 32}


def _honest_residual(
    *,
    verdict: str = "allow",
    kind: str = MEASURED_RESIDUAL_KIND,
    tree: str | None = TREE,
) -> dict:
    materials = build_package_residual_materials(
        residual_verdict=verdict,
        rules_bundle_sha256=BUNDLE,
        rules_version=RULES_V,
        rules_file_digests=FILE_DIG,
        package_tree_sha=tree,
        residual_kind=kind,
        rules_policy_text_sha256="44" * 32,
        harness_kind="measured_review_cvm_script_zip",
    )
    return materials.as_dict()


def _times(*, issued: int = T0, received: int = T0 + 3_600_000) -> dict[str, int]:
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


def _rules_obs() -> dict:
    # Closed report schema still uses snapshot_sha256/revision_id.
    return {
        "snapshot_sha256": "55" * 32,
        "revision_id": "rules-rev-agate-1",
    }


def _core(*, verdict: str = "allow") -> dict:
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
        response_id="gen-residual",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-residual",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-residual",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-residual",
        routing_sha256=ROUTING,
    )
    return build_review_core_minimal(
        session_id="rs-residual",
        assignment_id="ra-residual",
        submission_id="sub-residual",
        review_nonce="nonce-residual",
        assignment_digest="aa" * 32,
        rules_observation=_rules_obs(),
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(),
    )


def _envelope_with_residual(
    *,
    residual: dict | None = None,
    verdict: str = "allow",
    include_residual: bool = True,
) -> dict:
    core = _core(verdict=verdict)
    env = {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": review_digest(core),
        "report_data_hex": review_report_data_hex(core),
        "review_core": core,
        "attestation": {
            "tdx_quote_hex": "00" * 16,
            "event_log": [],
            "measurement": {},
        },
    }
    if include_residual:
        bag = residual if residual is not None else _honest_residual()
        bound = bind_package_residual_into_review_materials(
            envelope=env,
            materials=build_package_residual_materials(
                residual_verdict=str(bag["residual_verdict"]),
                rules_bundle_sha256=str(bag["rules_bundle_sha256"]),
                rules_version=str(bag["rules_version"]),
                rules_file_digests=dict(bag["rules_file_digests"]),
                package_tree_sha=bag.get("package_tree_sha"),
                residual_kind=str(bag["residual_kind"]),
                rules_policy_text_sha256=bag.get("rules_policy_text_sha256"),
                harness_kind=bag.get("harness_kind"),
            ),
        )
        env = bound["envelope"]
    return env


# ---------------------------------------------------------------------------
# VAL-AGATE-004 / 005 — residual required; fail stops
# --------------------------------------------------------------------------- #


def test_measured_residual_allow_admits_eval() -> None:
    materials = _honest_residual(verdict="allow")
    decision = admit_package_residual_for_eval(
        residual=materials,
        dual_flags_on=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.admitted is True
    assert decision.reason_code == "package_residual_allow"
    assert decision.rules_bundle_sha256 == BUNDLE
    assert decision.package_tree_sha == TREE


def test_missing_residual_refuses_eval_authorizable() -> None:
    decision = admit_package_residual_for_eval(dual_flags_on=True)
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_MISSING

    with pytest.raises(PackageResidualError) as exc:
        require_package_residual_for_eval(dual_flags_on=True)
    assert exc.value.code == REFUSE_RESIDUAL_MISSING


def test_residual_reject_stops_pipeline() -> None:
    materials = _honest_residual(verdict="reject")
    decision = admit_package_residual_for_eval(
        residual=materials,
        dual_flags_on=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_FAIL

    with pytest.raises(PackageResidualError) as exc:
        require_package_residual_for_eval(
            residual=materials,
            dual_flags_on=True,
            expected_package_tree_sha=TREE,
        )
    assert exc.value.code == REFUSE_RESIDUAL_FAIL


def test_residual_fail_verdict_also_stops() -> None:
    materials = _honest_residual(verdict="fail")
    decision = admit_package_residual_for_eval(residual=materials, dual_flags_on=True)
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_FAIL


def test_missing_package_tree_on_residual_refuses_when_required() -> None:
    materials = _honest_residual(tree=None)
    decision = admit_package_residual_for_eval(
        residual=materials,
        dual_flags_on=True,
        require_package_tree_sha=True,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_PACKAGE_TREE_MISSING


def test_package_tree_mismatch_refuses() -> None:
    materials = _honest_residual(tree=TREE)
    decision = admit_package_residual_for_eval(
        residual=materials,
        dual_flags_on=True,
        expected_package_tree_sha="cd" * 32,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_UNBOUND


# ---------------------------------------------------------------------------
# VAL-AGATE-006 — bind residual + rules digests into review materials
# --------------------------------------------------------------------------- #


def test_residual_bind_into_envelope_and_outcome() -> None:
    materials = build_package_residual_materials(
        residual_verdict="allow",
        rules_bundle_sha256=BUNDLE,
        rules_version=RULES_V,
        rules_file_digests=FILE_DIG,
        package_tree_sha=TREE,
    )
    env0 = {"schema_version": 1, "domain": REVIEW_REPORT_DOMAIN}
    out0 = {"status": "verified_allow", "terminal": True}
    bound = bind_package_residual_into_review_materials(
        envelope=env0,
        outcome=out0,
        materials=materials,
    )
    env = bound["envelope"]
    outcome = bound["outcome"]
    assert env["package_residual"]["residual_verdict"] == "allow"
    assert env["package_residual"]["rules_bundle_sha256"] == BUNDLE
    assert env["package_residual"]["rules_file_digests"] == FILE_DIG
    assert env["package_residual"]["package_tree_sha"] == TREE
    assert env["package_residual"]["residual_digest"] == materials.residual_digest
    assert outcome["package_residual"]["residual_digest"] == materials.residual_digest


def test_tampered_residual_digest_refuses() -> None:
    bag = _honest_residual()
    bag["residual_digest"] = "ff" * 32
    decision = admit_package_residual_for_eval(residual=bag, dual_flags_on=True)
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_UNBOUND


# ---------------------------------------------------------------------------
# VAL-AGATE-007 — host analyzer alone insufficient
# --------------------------------------------------------------------------- #


def test_host_analyzer_allow_alone_insufficient() -> None:
    assert host_analyzer_alone_insufficient(host_analyzer_allow=True, residual=None) is True
    decision = admit_package_residual_for_eval(
        dual_flags_on=True,
        host_analyzer_allow=True,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_HOST_ONLY


def test_host_only_residual_kind_refuses_tee_auth() -> None:
    bag = _honest_residual(kind=HOST_ANALYZER_KIND)
    decision = admit_package_residual_for_eval(
        residual=bag,
        dual_flags_on=True,
        host_analyzer_allow=True,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_HOST_ONLY


def test_host_allow_plus_measured_residual_admits() -> None:
    bag = _honest_residual()
    assert (
        host_analyzer_alone_insufficient(
            host_analyzer_allow=True,
            residual=bag,
        )
        is False
    )
    decision = admit_package_residual_for_eval(
        residual=bag,
        dual_flags_on=True,
        host_analyzer_allow=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# Fresh review gate integration: residual required under dual flags
# --------------------------------------------------------------------------- #


def test_fresh_review_gate_refuses_without_residual() -> None:
    env = _envelope_with_residual(include_residual=False)
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
    )
    assert decision.may_launch is False
    assert decision.reason_code in {
        REFUSE_RESIDUAL_MISSING,
        "package_residual_missing",
        "eval_cvm_refused_no_fresh_review",
    }


def test_fresh_review_gate_allows_with_bound_residual() -> None:
    env = _envelope_with_residual(include_residual=True)
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.may_launch is True
    assert decision.verdict == "allow"


def test_fresh_review_gate_refuses_residual_reject() -> None:
    env = _envelope_with_residual(residual=_honest_residual(verdict="reject"))
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.may_launch is False
    assert decision.reason_code in {REFUSE_RESIDUAL_FAIL, "package_residual_reject"}


def test_assignment_launch_requires_residual_on_envelope() -> None:
    env = _envelope_with_residual(include_residual=False)
    import json

    assignment = SimpleNamespace(
        phase="review_allowed",
        review_verification_outcome_json=json.dumps(
            {
                "status": "verified_allow",
                "terminal": True,
                "retryable": False,
                "nonce_consumed": True,
            }
        ),
        review_report_envelope_json=json.dumps(env),
        review_report_data_hex=env["report_data_hex"],
        review_digest=env["review_digest"],
    )
    decision = admit_eval_cvm_launch_from_assignment(
        assignment,
        dual_flags_on=True,
        require_package_residual=True,
    )
    assert decision.may_launch is False


# ---------------------------------------------------------------------------
# VAL-AGATE-012 / 013 — no catalog; finetune ban
# --------------------------------------------------------------------------- #


def test_no_closed_agent_model_catalog_required() -> None:
    assert agent_model_requires_closed_catalog() is False
    assert_no_closed_agent_model_catalog(None)
    assert_no_closed_agent_model_catalog([])
    inv = inventory_residual_gate()
    assert inv["agent_model_closed_catalog"] is False


def test_personal_finetune_models_detected() -> None:
    positives = [
        "ft:gpt-4o-mini:org:custom:abc123",
        "openai/gpt-4o:ft-acme-2024",
        "openrouter/customer/my-model:ft-xyz",
        "personal/my-finetune/gpt",
        "meta-llama/fine_tune-private-1",
        "anthropic/claude-3-personal-finetune",
    ]
    for mid in positives:
        assert is_personal_finetune_model(mid), mid
        with pytest.raises(PackageResidualError) as exc:
            refuse_personal_finetune_model(mid)
        assert exc.value.code == REFUSE_FINETUNE


def test_public_base_models_not_banned_by_catalog() -> None:
    # No closed catalog: common public models pass finetune filter.
    ok = [
        "x-ai/grok-4.5",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4",
        "google/gemini-2.5-pro",
        "deepseek/deepseek-chat",
        None,
    ]
    for mid in ok:
        assert is_personal_finetune_model(mid) is False
        assert filter_agent_model_or_refuse(mid) == (None if mid is None else mid)


def test_eval_agent_planned_request_refuses_finetune() -> None:
    with pytest.raises(Exception) as exc:
        build_eval_agent_planned_request(
            body_sha256=BODY_SHA,
            body_length=len(BODY),
            routing_sha256=ROUTING,
            model="ft:gpt-4o-mini:personal:xyz",
        )
    # EvalAgentLlmError or PackageResidualError with finetune code
    code = getattr(exc.value, "code", str(exc.value))
    assert REFUSE_FINETUNE in str(code) or "finetune" in str(exc.value).lower()


def test_eval_agent_score_admit_refuses_finetune_materials() -> None:
    materials = {
        "planned_request_sha256": "aa" * 32,
        "transport_observation_sha256": "bb" * 32,
        "model": "ft:gpt-4o:org:mine:1",
        "planned": {"model": "ft:gpt-4o:org:mine:1"},
    }
    decision = admit_eval_agent_llm_for_score(
        mode=MODE_MEASURED_OPENROUTER,
        dual_flags_on=True,
        runtime_kind="measured_eval_cvm",
        measurement={
            "compose_hash": "aa" * 32,
            "os_image_hash": "bb" * 32,
            "mrtd": "cc" * 48,
        },
        allowlist=[
            {
                "compose_hash": "aa" * 32,
                "os_image_hash": "bb" * 32,
                "mrtd": "cc" * 48,
            }
        ],
        claims_model_call=True,
        agent_or_materials=materials,
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_FINETUNE
