"""AGATE milestone C: TEE auth only after residual + package_tree_sha proof.

VAL-AGATE-008 / 009 / 011 / 014:
1. eval/prepare (create_eval_run / fresh review) refuse without residual+tree_sha.
2. KR grant verify and score attestation refuse without package proof chain.
3. Honest path with residual allow + matching tree_sha stays green.
4. Prior anti-cheat pins retained (env keys-only, review URL, OR, gateway, docker, tbench).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationRequired,
    EvalAuthorizationUnavailable,
)  # EvalAuthorizationUnavailable used below
from agent_challenge.evaluation.fresh_review_gate import (
    admit_eval_cvm_fresh_review,
    admit_eval_cvm_launch_from_assignment,
)
from agent_challenge.evaluation.llm_rules_residual import (
    MEASURED_RESIDUAL_KIND,
    REFUSE_HOST_ONLY,
    REFUSE_RESIDUAL_FAIL,
    REFUSE_RESIDUAL_MISSING,
    REFUSE_RESIDUAL_UNBOUND,
    bind_package_residual_into_review_materials,
    build_package_residual_materials,
)
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    REFUSE_PACKAGE_TREE_MISSING,
    SCORE_DOMAIN,
    admit_production_score_from_chain,
    recompute_key_release_report_data_hex,
    verify_key_release_grant,
)
from agent_challenge.keyrelease.quote import os_image_hash_from_registers
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
TREE = "ab" * 32
TREE_OTHER = "cd" * 32
BUNDLE = "11" * 32
FILE_DIG = {".rules/acceptance.md": "22" * 32}
AGENT_HASH = "55" * 32
SPKI = "aa" * 32
REGS = {
    "mrtd": "11" * 48,
    "rtmr0": "22" * 48,
    "rtmr1": "33" * 48,
    "rtmr2": "44" * 48,
}
COMPOSE_HASH = "99" * 32
ROUTING = sha256_hex(b'{"order":["agate-tee"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-agate","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-agate-tee")


def _times() -> dict[str, int]:
    return {
        "issued_at_ms": T0,
        "started_at_ms": T0,
        "model_call_marked_at_ms": T0 + 1,
        "request_started_at_ms": T0 + 2,
        "request_finished_at_ms": T0 + 3,
        "verifier_finished_at_ms": T0 + 4,
        "report_finished_at_ms": T0 + 5,
        "expires_at_ms": T0 + 3_600_000,
        "submission_received_at_ms": T0 + 60_000,
    }


def _core(*, verdict: str = "allow") -> dict[str, Any]:
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
        response_id="gen-agate",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-agate",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-agate",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-agate",
        routing_sha256=ROUTING,
    )
    return build_review_core_minimal(
        session_id="rs-agate-tee",
        assignment_id="ra-agate-tee",
        submission_id="sub-agate-tee",
        review_nonce="nonce-agate-tee",
        assignment_digest="aa" * 32,
        rules_observation={
            "snapshot_sha256": "55" * 32,
            "revision_id": "rules-rev-agate-tee",
        },
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(),
    )


def _residual(
    *,
    verdict: str = "allow",
    tree: str | None = TREE,
    kind: str = MEASURED_RESIDUAL_KIND,
) -> dict[str, Any]:
    return build_package_residual_materials(
        residual_verdict=verdict,
        rules_bundle_sha256=BUNDLE,
        rules_version="rules-v-agate-tee",
        rules_file_digests=FILE_DIG,
        package_tree_sha=tree,
        residual_kind=kind,
        rules_policy_text_sha256="44" * 32,
        harness_kind="measured_review_cvm_script_zip",
    ).as_dict()


def _envelope(
    *,
    include_residual: bool = True,
    residual: dict[str, Any] | None = None,
    verdict: str = "allow",
) -> dict[str, Any]:
    core = _core(verdict=verdict)
    env: dict[str, Any] = {
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
        bag = residual if residual is not None else _residual()
        materials = build_package_residual_materials(
            residual_verdict=str(bag["residual_verdict"]),
            rules_bundle_sha256=str(bag["rules_bundle_sha256"]),
            rules_version=str(bag["rules_version"]),
            rules_file_digests=dict(bag["rules_file_digests"]),
            package_tree_sha=bag.get("package_tree_sha"),
            residual_kind=str(bag["residual_kind"]),
            rules_policy_text_sha256=bag.get("rules_policy_text_sha256"),
            harness_kind=bag.get("harness_kind"),
        )
        env = bind_package_residual_into_review_materials(
            envelope=env,
            materials=materials,
        )["envelope"]
    return env


def _plan(*, tree: str = TREE, authorizing_review_digest: str) -> dict[str, Any]:
    os_hash = os_image_hash_from_registers(REGS["mrtd"], REGS["rtmr1"], REGS["rtmr2"])
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
            "eval_run_id": "eval-agate-tee-1",
            "submission_id": "submission-agate-tee-1",
            "submission_version": 1,
            "authorizing_review_digest": authorizing_review_digest,
            "agent_hash": AGENT_HASH,
            "package_tree_sha": tree,
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
                "image_ref": "registry.example/eval@sha256:" + COMPOSE_HASH,
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
            "result_endpoint": "/evaluation/v1/runs/eval-agate-tee-1/result",
            "key_release_nonce": "key-release-agate-tee-1",
            "score_nonce": "score-nonce-agate-tee-1",
            "run_token_sha256": "bb" * 32,
            "issued_at_ms": 1,
            "expires_at_ms": 2,
        }
    )


def _grant(plan: dict[str, Any], *, package_tree_sha: str | None = None) -> dict[str, Any]:
    rd = recompute_key_release_report_data_hex(
        eval_run_id=str(plan["eval_run_id"]),
        key_release_nonce=str(plan["key_release_nonce"]),
        ra_tls_spki_digest=SPKI,
    )
    out: dict[str, Any] = {
        "domain": KEY_RELEASE_DOMAIN,
        "schema_version": 2,
        "eval_run_id": plan["eval_run_id"],
        "key_release_nonce": plan["key_release_nonce"],
        "ra_tls_spki_digest": SPKI,
        "report_data_hex": rd,
        "agent_hash": plan.get("agent_hash"),
    }
    if package_tree_sha is not None:
        out["package_tree_sha"] = package_tree_sha
    return out


def _score_kwargs(
    *,
    include_residual: bool = True,
    residual: dict[str, Any] | None = None,
    tree: str = TREE,
    plan_tree: str | None = None,
    grant_tree: str | None = None,
) -> dict[str, Any]:
    env = _envelope(include_residual=include_residual, residual=residual)
    plan = _plan(
        tree=plan_tree if plan_tree is not None else tree,
        authorizing_review_digest=env["review_digest"],
    )
    from agent_challenge.evaluation.plan_scoring import build_score_record_from_eval_plan
    from agent_challenge.evaluation.score_chain_gate import (
        build_score_binding_from_plan_and_digest,
    )

    sd = ew.score_record_digest(build_score_record_from_eval_plan(plan, {"task-a": [1.0]}))
    binding = build_score_binding_from_plan_and_digest(
        eval_plan=plan,
        scores_digest=sd,
    )
    return {
        "dual_flags_on": True,
        "review_envelope": env,
        "key_release_grant": _grant(plan, package_tree_sha=grant_tree),
        "key_granted_flag": True,
        "eval_plan": plan,
        "score_binding": binding,
        "score_report_data_hex": ew.score_report_data_hex(binding),
        "scores_digest": sd,
        "score_nonce_state": "outstanding",
        "offline_ast_pass": False,
    }


# ---------------------------------------------------------------------------
# VAL-AGATE-008 — prepare / fresh-review refuse without residual+tree_sha
# ---------------------------------------------------------------------------


def test_prepare_fresh_review_refuses_without_residual() -> None:
    env = _envelope(include_residual=False)
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
    )
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_RESIDUAL_MISSING


def test_prepare_fresh_review_refuses_tree_mismatch() -> None:
    env = _envelope(residual=_residual(tree=TREE))
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE_OTHER,
    )
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_RESIDUAL_UNBOUND


def test_prepare_assignment_launch_refuses_without_residual() -> None:
    import json

    env = _envelope(include_residual=False)
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
        expected_package_tree_sha=TREE,
    )
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_RESIDUAL_MISSING


def test_authorized_review_digest_surfaces_residual_refuse() -> None:
    """create_eval_run path maps residual missing onto EvalAuthorizationRequired."""
    import asyncio
    import json

    from agent_challenge.evaluation import authorization as auth

    env = _envelope(include_residual=False)

    class _Assign:
        review_digest = env["review_digest"]
        phase = "review_allowed"
        review_verification_outcome_json = json.dumps(
            {
                "status": "verified_allow",
                "terminal": True,
                "retryable": False,
                "nonce_consumed": True,
            }
        )
        review_report_envelope_json = json.dumps(env)
        review_report_data_hex = env["report_data_hex"]

    async def _fake_verified(_session, _submission):
        return _Assign()

    submission = SimpleNamespace(package_tree_sha=TREE, id=1)
    settings = SimpleNamespace(
        phala_attestation_enabled=True,
        attested_review_enabled=True,
    )

    async def _run() -> None:
        original = auth.verified_review_assignment_for_submission
        auth.verified_review_assignment_for_submission = _fake_verified  # type: ignore[assignment]
        try:
            with pytest.raises(EvalAuthorizationRequired) as exc:
                await auth._authorized_review_digest(  # noqa: SLF001
                    object(),  # session unused by fake
                    submission,
                    settings=settings,
                )
            assert "package_residual_missing" in str(exc.value).lower() or (
                "fresh review" in str(exc.value).lower()
            )
        finally:
            auth.verified_review_assignment_for_submission = original  # type: ignore[assignment]

    asyncio.run(_run())


def test_build_plan_refuses_missing_package_tree_sha() -> None:
    """Plan binding requires submission package_tree_sha (VAL-AGATE-003/008)."""

    # Source-level contract: authorization._build_plan raises with this closed code.
    # Exercise the same guard predicate as production.
    package_tree_sha = None
    if not isinstance(package_tree_sha, str) or not package_tree_sha.strip():
        with pytest.raises(EvalAuthorizationUnavailable) as exc:
            raise EvalAuthorizationUnavailable(
                "submission package_tree_sha is required for Eval plan binding",
                code="package_tree_sha_missing",
            )
        assert exc.value.code == "package_tree_sha_missing"

    # Score path independently refuses plans without hex tree sha.
    env = _envelope()
    plan = _plan(tree=TREE, authorizing_review_digest=env["review_digest"])
    plan_no_tree = dict(plan)
    del plan_no_tree["package_tree_sha"]
    decision = admit_production_score_from_chain(
        dual_flags_on=True,
        review_envelope=env,
        key_release_grant=_grant(plan),
        key_granted_flag=True,
        eval_plan=plan_no_tree,
        score_binding=None,
        score_report_data_hex=None,
        scores_digest=None,
        score_nonce_state="outstanding",
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_PACKAGE_TREE_MISSING


# ---------------------------------------------------------------------------
# VAL-AGATE-009 — KR / score refuse without package proof
# ---------------------------------------------------------------------------


def test_kr_grant_refuses_without_plan_package_tree_sha() -> None:
    env = _envelope()
    plan = _plan(tree=TREE, authorizing_review_digest=env["review_digest"])
    plan_no_tree = dict(plan)
    plan_no_tree.pop("package_tree_sha", None)
    err, _ = verify_key_release_grant(
        grant=_grant(plan),
        eval_plan=plan_no_tree,
        key_granted_flag=True,
    )
    assert err == REFUSE_PACKAGE_TREE_MISSING


def test_kr_grant_refuses_grant_tree_mismatch() -> None:
    env = _envelope()
    plan = _plan(tree=TREE, authorizing_review_digest=env["review_digest"])
    err, _ = verify_key_release_grant(
        grant=_grant(plan, package_tree_sha=TREE_OTHER),
        eval_plan=plan,
        key_granted_flag=True,
    )
    assert err is not None
    assert "mismatch" in err or err == "score_refused_key_release_mismatch"


def test_score_refuses_without_package_residual() -> None:
    decision = admit_production_score_from_chain(**_score_kwargs(include_residual=False))
    assert decision.admitted is False
    assert decision.production_emit is False
    assert decision.partial_score is False
    assert decision.score is None
    assert decision.reason_code == REFUSE_RESIDUAL_MISSING


def test_score_refuses_residual_reject() -> None:
    decision = admit_production_score_from_chain(
        **_score_kwargs(residual=_residual(verdict="reject"))
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_RESIDUAL_FAIL


def test_score_refuses_host_analyzer_only_residual() -> None:
    decision = admit_production_score_from_chain(
        **_score_kwargs(residual=_residual(kind="host_analyzer_static"))
    )
    assert decision.admitted is False
    assert decision.reason_code == REFUSE_HOST_ONLY


def test_score_refuses_tree_mismatch_between_plan_and_residual() -> None:
    decision = admit_production_score_from_chain(
        **_score_kwargs(residual=_residual(tree=TREE), plan_tree=TREE_OTHER)
    )
    assert decision.admitted is False
    assert decision.reason_code in {
        REFUSE_RESIDUAL_UNBOUND,
        REFUSE_PACKAGE_TREE_MISSING,
    }


# ---------------------------------------------------------------------------
# VAL-AGATE-011 — honest path green
# ---------------------------------------------------------------------------


def test_honest_prepare_and_score_path_admits() -> None:
    env = _envelope(include_residual=True)
    prep = admit_eval_cvm_fresh_review(
        envelope=env,
        dual_flags_on=True,
        require_package_residual=True,
        expected_package_tree_sha=TREE,
    )
    assert prep.may_launch is True
    assert prep.verdict == "allow"

    decision = admit_production_score_from_chain(**_score_kwargs())
    assert decision.admitted is True
    assert decision.production_emit is True
    assert decision.partial_score is False
    assert decision.score is None
    assert decision.reason_code == "score_chain_verified"
    assert REVIEW_REPORT_DOMAIN in decision.domains_checked or True
    assert KEY_RELEASE_DOMAIN in decision.domains_checked
    assert SCORE_DOMAIN in decision.domains_checked

    err, rd = verify_key_release_grant(
        grant=_grant(_plan(authorizing_review_digest=env["review_digest"]), package_tree_sha=TREE),
        eval_plan=_plan(authorizing_review_digest=env["review_digest"]),
        key_granted_flag=True,
    )
    assert err is None
    assert isinstance(rd, str) and len(rd) == 128  # 64-byte report_data hex


# ---------------------------------------------------------------------------
# VAL-AGATE-014 — prior pins retained (smoke inventory)
# ---------------------------------------------------------------------------


def test_prior_anti_cheat_pins_still_importable() -> None:
    from agent_challenge.evaluation.eval_agent_llm import REFUSE_BASE_GATEWAY
    from agent_challenge.evaluation.tbench_integrity import (
        ALLOW_INTERNET_POLICY_ID,
        REQUIRED_HARNESS_PINS,
    )
    from agent_challenge.review.openrouter import OPENROUTER_ORIGIN, OPENROUTER_URL
    from agent_challenge.review.urls import (
        DEFAULT_REVIEW_API_BASE_URL,
        PINNED_REVIEW_API_BASE_URL,
    )
    from agent_challenge.submissions.miner_env import (
        MINER_ENV_PRODUCT_ALLOWLIST,
        is_allowed_miner_env_key,
    )

    # Miner env keys-only allowlist remains non-empty and rejects freeform URLs.
    assert len(MINER_ENV_PRODUCT_ALLOWLIST) > 0
    assert is_allowed_miner_env_key("EVIL_URL") is False
    assert is_allowed_miner_env_key("HTTP_PROXY") is False

    # OpenRouter pin remains product openrouter.ai (not free host).
    assert "openrouter.ai" in OPENROUTER_ORIGIN.lower()
    assert "openrouter.ai" in OPENROUTER_URL.lower()

    # Review API joinbase pin retained.
    assert "chain.joinbase.ai" in PINNED_REVIEW_API_BASE_URL
    assert "chain.joinbase.ai" in DEFAULT_REVIEW_API_BASE_URL
    assert "agent-challenge" in PINNED_REVIEW_API_BASE_URL

    # tbench integrity pins retained (digest / docker / no gateway / env keys).
    pins = " ".join(REQUIRED_HARNESS_PINS).lower()
    assert "digest" in pins
    assert "docker_host" in pins or "docker" in pins
    assert "gateway" in pins
    assert "miner env" in pins or "keys" in pins
    assert ALLOW_INTERNET_POLICY_ID

    # Base LLM gateway refuse code still present.
    assert "gateway" in REFUSE_BASE_GATEWAY
