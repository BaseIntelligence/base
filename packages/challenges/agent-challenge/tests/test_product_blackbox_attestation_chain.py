"""VAL-ACAT-020 + VAL-ACAT-X001..X012: black-box e2e attestation chain.

Product Mode B: scripted re-verifiable package covering
miner ZIP+script → rules → OR digests → ≤24h → eval RA-TLS → score → raw weights,
with independent re-verify fail-closed on mutilation and zero Base gateway
critical path. Weights path is challenge-push / master aggregate only.
"""

from __future__ import annotations

import copy

import pytest

from agent_challenge.evaluation.blackbox_attestation_chain import (
    FRESHNESS_WINDOW_MS,
    HOP_ASSERTIONS,
    HOP_EVAL_AGENT_OR,
    HOP_EVAL_CONJUNCTION,
    HOP_FRESHNESS_24H,
    HOP_GATEWAY_FREE,
    HOP_HONESTY,
    HOP_KEY_RELEASE,
    HOP_MINER_SUBMIT,
    HOP_OR_OBSERVED,
    HOP_OR_PLANNED,
    HOP_ORDER,
    HOP_RAW_WEIGHTS,
    HOP_RULES_LOAD,
    HOP_SCORE_DOMAIN,
    MS_23H,
    REFUSE_BURN_TRACKED,
    REFUSE_CHAIN_MUTILATED,
    REFUSE_GATEWAY_RESIDUAL,
    REFUSE_PACKAGE_TAMPER,
    REFUSE_SET_WEIGHTS,
    T0_DEFAULT,
    BlackboxChainError,
    build_complete_package,
    chronological_evidence_index,
    mutilate_package,
    package_digest,
    require_attestation_chain,
    reverify_package,
    run_attestation_chain,
    scan_gateway_residuals,
)
from agent_challenge.evaluation.eval_agent_llm import MODE_MEASURED_OPENROUTER


def _hop(admission, hop_id: str):
    for hop in admission.hops:
        if hop.hop_id == hop_id:
            return hop
    raise AssertionError(f"missing hop {hop_id}")


# ---------------------------------------------------------------------------
# VAL-ACAT-020 control path
# ---------------------------------------------------------------------------


def test_full_e2e_chain_admits_and_maps_all_hops() -> None:
    decision = run_attestation_chain()
    assert decision.admitted is True
    assert decision.production_emit is True
    assert decision.reverify_ok is True
    assert decision.gateway_free is True
    assert decision.reason_code == "e2e_chain_verified"
    assert decision.side_effects["set_weights"] == 0
    assert decision.side_effects["burn_weights_24h"] == 0
    assert list(h.hop_id for h in decision.hops) == list(HOP_ORDER)
    assert all(h.status == "pass" for h in decision.hops)
    for hop_id, assertion in HOP_ASSERTIONS.items():
        hop = _hop(decision, hop_id)
        assert hop.assertion_id == assertion
        assert hop.gateway_dependency is False
        assert hop.set_weights_invoked is False
    assert decision.raw_weights
    assert "payload_digest" in decision.package["raw_weights_payload"]
    index = chronological_evidence_index(decision)
    assert index["assertion_table"]["VAL-ACAT-020"]["status"] == "PASS"
    for assertion in HOP_ASSERTIONS.values():
        assert index["assertion_table"][assertion]["status"] == "PASS"
    assert index["no_set_weights"] is True
    assert index["gateway_free"] is True
    assert index["weights_path"].startswith("challenge-raw-weights")


def test_reverify_is_idempotent_on_same_package() -> None:
    package = build_complete_package()
    a = reverify_package(package)
    b = reverify_package(package)
    assert a.admitted is True
    assert b.admitted is True
    assert a.package_digest == b.package_digest == package["package_digest"]


def test_require_raises_on_refuse() -> None:
    with pytest.raises(BlackboxChainError) as exc:
        require_attestation_chain(dual_flags_on=False)
    # dual flags off fails score / agent path
    assert exc.value.code


# ---------------------------------------------------------------------------
# VAL-ACAT-X001 miner submit gateway free
# ---------------------------------------------------------------------------


def test_x001_gateway_env_on_master_refuses() -> None:
    decision = run_attestation_chain(
        master_env={"BASE_GATEWAY_TOKEN": "should-never-exist"},
    )
    assert decision.admitted is False
    assert _hop(decision, HOP_MINER_SUBMIT).status == "refuse"
    assert _hop(decision, HOP_MINER_SUBMIT).gateway_dependency is True
    assert decision.reason_code == REFUSE_GATEWAY_RESIDUAL


def test_x001_llm_v1_route_refuses_gateway_free_hop() -> None:
    decision = run_attestation_chain(
        master_routes=("/health", "/llm/v1/chat/completions"),
    )
    assert decision.admitted is False
    # residual is on gateway inventory; miner submit or gateway-free hop refuses
    assert any(
        h.status == "refuse"
        for h in decision.hops
        if h.hop_id
        in {
            HOP_MINER_SUBMIT,
            HOP_GATEWAY_FREE,
        }
    )


# ---------------------------------------------------------------------------
# VAL-ACAT-X002 rules
# ---------------------------------------------------------------------------


def test_x002_rules_digest_bound_into_review() -> None:
    decision = run_attestation_chain()
    hop = _hop(decision, HOP_RULES_LOAD)
    assert hop.status == "pass"
    assert hop.digests["rules_version"]
    core = decision.package["review_envelope"]["review_core"]
    assert core["rules_observation"]["rules_version"] == hop.digests["rules_version"]


def test_x002_empty_rules_refuses_entry() -> None:
    decision = run_attestation_chain(rules_files={})
    assert decision.admitted is False
    assert decision.package.get("identity_error") is not None


# ---------------------------------------------------------------------------
# VAL-ACAT-X003 / X004 OpenRouter digests
# ---------------------------------------------------------------------------


def test_x003_x004_planned_observed_digests_present() -> None:
    decision = run_attestation_chain()
    planned = _hop(decision, HOP_OR_PLANNED)
    observed = _hop(decision, HOP_OR_OBSERVED)
    assert planned.status == "pass"
    assert observed.status == "pass"
    assert len(planned.digests["planned_request_sha256"]) == 64
    assert len(observed.digests["transport_observation_sha256"]) == 64
    assert decision.package["gateway_inventory"]["openrouter_destination"].startswith(
        "https://openrouter.ai"
    )


def test_x003_mutilated_planned_digest_fail_closed() -> None:
    package = build_complete_package()
    core = package["review_envelope"]["review_core"]
    core = copy.deepcopy(core)
    core["openrouter_observation"]["planned_request_sha256"] = "00" * 32
    package = mutilate_package(
        package,
        path=("review_envelope", "review_core"),
        value=core,
    )
    # also fix report_data so we exercise OR check not package digest alone
    # (package_digest mismatch is also refuse; both acceptable fail-closed)
    decision = reverify_package(package)
    assert decision.admitted is False
    assert decision.reason_code in {
        REFUSE_PACKAGE_TAMPER,
        "review_or_planned_digest_missing",
        "score_refused_review_chain",
        "review_attestation_reverify_failed",
        "eval_cvm_refused_no_fresh_review",
        "score_refused_tampered",
    }


# ---------------------------------------------------------------------------
# VAL-ACAT-X005 24h window
# ---------------------------------------------------------------------------


def test_x005_stale_over_24h_refuses() -> None:
    decision = run_attestation_chain(
        issued_at_ms=T0_DEFAULT,
        received_at_ms=T0_DEFAULT + FRESHNESS_WINDOW_MS + 1,
    )
    assert decision.admitted is False
    hop = _hop(decision, HOP_FRESHNESS_24H)
    assert hop.status == "refuse"
    assert "24h" in hop.reason_code or hop.reason_code == "attestation_stale_over_24h"


def test_x005_exactly_24h_boundary_admits() -> None:
    decision = run_attestation_chain(
        issued_at_ms=T0_DEFAULT,
        received_at_ms=T0_DEFAULT + FRESHNESS_WINDOW_MS,
    )
    assert decision.admitted is True
    assert _hop(decision, HOP_FRESHNESS_24H).status == "pass"


def test_x005_23h_admits() -> None:
    decision = run_attestation_chain(
        issued_at_ms=T0_DEFAULT,
        received_at_ms=T0_DEFAULT + MS_23H,
    )
    assert decision.admitted is True


# ---------------------------------------------------------------------------
# VAL-ACAT-X006 conjunction ablation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip_review": True},
        {"verdict": "reject"},
        {
            "issued_at_ms": T0_DEFAULT,
            "received_at_ms": T0_DEFAULT + FRESHNESS_WINDOW_MS + 1,
        },
    ],
)
def test_x006_missing_conjunct_blocks_eval(kwargs: dict) -> None:
    decision = run_attestation_chain(**kwargs)
    assert decision.admitted is False
    # eval conjunction or earlier hop refused
    eval_hop = _hop(decision, HOP_EVAL_CONJUNCTION)
    fresh_hop = _hop(decision, HOP_FRESHNESS_24H)
    or_hop = _hop(decision, HOP_OR_PLANNED)
    assert eval_hop.status == "refuse" or fresh_hop.status == "refuse" or or_hop.status == "refuse"


# ---------------------------------------------------------------------------
# VAL-ACAT-X007 RA-TLS key release (no Base gateway mint)
# ---------------------------------------------------------------------------


def test_x007_key_release_present_and_no_gateway_mint() -> None:
    decision = run_attestation_chain()
    hop = _hop(decision, HOP_KEY_RELEASE)
    assert hop.status == "pass"
    grant = decision.package["key_release_grant"]
    assert grant["domain"] == "base-agent-challenge-keyrelease-v1"
    assert grant["base_gateway_token_minted"] is False
    assert grant["ra_tls_spki_digest"]


def test_x007_missing_key_release_refuses() -> None:
    decision = run_attestation_chain(skip_key_release=True)
    assert decision.admitted is False
    assert _hop(decision, HOP_KEY_RELEASE).status == "refuse"


# ---------------------------------------------------------------------------
# VAL-ACAT-X008 eval-agent OR measured or tools-only
# ---------------------------------------------------------------------------


def test_x008_tools_only_default_admits() -> None:
    decision = run_attestation_chain()
    hop = _hop(decision, HOP_EVAL_AGENT_OR)
    assert hop.status == "pass"
    assert hop.digests["mode"] == "tools_only"


def test_x008_measured_agent_or_admits() -> None:
    decision = run_attestation_chain(
        agent_llm_mode=MODE_MEASURED_OPENROUTER,
        include_measured_agent_or=True,
    )
    assert decision.admitted is True
    hop = _hop(decision, HOP_EVAL_AGENT_OR)
    assert hop.status == "pass"
    assert hop.digests["mode"] == MODE_MEASURED_OPENROUTER
    assert hop.digests.get("planned_request_sha256")


# ---------------------------------------------------------------------------
# VAL-ACAT-X009 score domain
# ---------------------------------------------------------------------------


def test_x009_score_admits_after_full_chain() -> None:
    decision = run_attestation_chain()
    hop = _hop(decision, HOP_SCORE_DOMAIN)
    assert hop.status == "pass"
    assert "base-agent-challenge-review-v1" in hop.digests["domains"]
    assert "base-agent-challenge-keyrelease-v1" in hop.digests["domains"]
    assert "base-agent-challenge-v1" in hop.digests["domains"]


def test_x009_skip_score_materials_refuses() -> None:
    decision = run_attestation_chain(skip_score=True)
    assert decision.admitted is False
    assert _hop(decision, HOP_SCORE_DOMAIN).status == "refuse"


# ---------------------------------------------------------------------------
# VAL-ACAT-X010 raw weights / no set_weights
# ---------------------------------------------------------------------------


def test_x010_raw_weights_path_only() -> None:
    decision = run_attestation_chain()
    hop = _hop(decision, HOP_RAW_WEIGHTS)
    assert hop.status == "pass"
    assert "raw-weights" in hop.digests["path"]
    assert decision.side_effects["set_weights"] == 0
    assert decision.package["raw_weights_payload"]["master_set_weights"] is False


def test_x010_set_weights_refuses() -> None:
    decision = run_attestation_chain(set_weights_count=1)
    assert decision.admitted is False
    hop = _hop(decision, HOP_RAW_WEIGHTS)
    assert hop.status == "refuse"
    assert hop.reason_code == REFUSE_SET_WEIGHTS
    assert hop.set_weights_invoked is True


def test_x010_burn_invoked_refuses() -> None:
    decision = run_attestation_chain(burn_invoked=True)
    assert decision.admitted is False
    assert _hop(decision, HOP_RAW_WEIGHTS).reason_code == REFUSE_BURN_TRACKED


# ---------------------------------------------------------------------------
# VAL-ACAT-X011 zero Base gateway
# ---------------------------------------------------------------------------


def test_x011_gateway_free_pass() -> None:
    decision = run_attestation_chain()
    assert decision.gateway_free is True
    assert _hop(decision, HOP_GATEWAY_FREE).status == "pass"
    assert scan_gateway_residuals(env={}) == []


def test_x011_gateway_token_scrapes() -> None:
    hits = scan_gateway_residuals(env={"BASE_LLM_GATEWAY_URL": "http://x/llm/v1"})
    assert hits
    hits2 = scan_gateway_residuals(urls=["https://master.example/llm/v1/chat"])
    assert hits2


# ---------------------------------------------------------------------------
# VAL-ACAT-X012 honesty
# ---------------------------------------------------------------------------


def test_x012_real_provider_blocked_on_pass() -> None:
    decision = run_attestation_chain()
    assert decision.tee_labels["REAL-PROVIDER"] == "BLOCKED"
    assert _hop(decision, HOP_HONESTY).status == "pass"


def test_x012_invented_real_provider_pass_refuses() -> None:
    decision = run_attestation_chain(real_provider_pass_claimed=True)
    assert decision.admitted is False
    hop = _hop(decision, HOP_HONESTY)
    assert hop.status == "refuse"
    assert "real_provider" in hop.reason_code


# ---------------------------------------------------------------------------
# Independent re-verify fail-closed on mutilation samples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,value",
    [
        (("review_envelope", "report_data_hex"), ("ff" * 32) + ("00" * 32)),
        (
            ("key_release_grant", "report_data_hex"),
            ("00" * 32) + ("ff" * 32),
        ),
        (("score_report_data_hex",), "aa" * 64),
        (("issued_at_ms",), T0_DEFAULT - 10_000_000),
        (("raw_weights_payload", "payload_digest"), "00" * 32),
    ],
)
def test_mutilation_samples_fail_closed(path: tuple[str, ...], value: object) -> None:
    package = build_complete_package()
    control = reverify_package(package)
    assert control.admitted is True
    mutilated = mutilate_package(package, path=path, value=value)
    decision = reverify_package(mutilated)
    assert decision.admitted is False
    assert decision.production_emit is False
    assert decision.raw_weights == {} or decision.reason_code != "e2e_chain_verified"
    assert decision.reason_code in {
        REFUSE_PACKAGE_TAMPER,
        REFUSE_CHAIN_MUTILATED,
        "score_refused_review_chain",
        "score_refused_tampered",
        "score_refused_key_release_mismatch",
        "score_refused_score_domain",
        "score_refused_incomplete_chain",
        "attestation_stale_over_24h",
        "review_attestation_reverify_failed",
        "eval_cvm_refused_no_fresh_review",
        "review_report_data_mismatch",
        "score_refused_domain_confusion",
    }


def test_package_digest_detects_any_material_change() -> None:
    package = build_complete_package()
    d1 = package_digest(package)
    package2 = copy.deepcopy(package)
    package2["issued_at_ms"] = package["issued_at_ms"] + 1
    d2 = package_digest(package2)
    assert d1 != d2


def test_dual_flags_off_cannot_emit() -> None:
    decision = run_attestation_chain(dual_flags_on=False)
    assert decision.admitted is False
    assert decision.production_emit is False


def test_index_lists_all_val_acat_x_assertions() -> None:
    decision = run_attestation_chain()
    index = chronological_evidence_index(decision)
    for i in range(1, 13):
        key = f"VAL-ACAT-X{i:03d}"
        assert key in index["assertion_table"]
    assert index["assertion_table"]["VAL-ACAT-020"]["status"] == "PASS"
