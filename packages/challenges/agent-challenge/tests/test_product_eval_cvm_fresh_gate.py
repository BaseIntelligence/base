"""VAL-ACAT-010 / 028 / 029: Eval CVM launches only with fresh re-verified review.

Expected behavior:
- Fresh allow re-verify → may_launch
- Stale / missing / reject / wrong domain → no eval lifecycle start
- Cache allow bit alone is insufficient (no re-verify → refuse)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_challenge.evaluation.fresh_review_gate import (
    REFUSE_ATTESTATION_MISSING,
    REFUSE_CACHED_ALLOW_ONLY,
    REFUSE_EVAL_CVM,
    REFUSE_REVERIFY_FAILED,
    REFUSE_STALE,
    REFUSE_WRONG_DOMAIN,
    EvalCvmFreshReviewError,
    admit_eval_cvm_fresh_review,
    admit_eval_cvm_launch_from_assignment,
    require_eval_cvm_fresh_review,
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

T0 = 1_700_000_000_000
ROUTING = sha256_hex(b'{"order":["x"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-2","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-eval-gate")


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


def _rules() -> dict:
    return {
        "rules_version": "rules-v1",
        "rules_bundle_sha256": "11" * 32,
        "rules_files": [".rules/acceptance.md"],
        "rules_file_digests": {".rules/acceptance.md": "22" * 32},
        "rules_policy_text_sha256": "33" * 32,
    }


def _core(*, verdict: str = "allow", issued: int = T0, received: int = T0 + 3_600_000) -> dict:
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
        response_id="gen-eval-gate",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-eval-gate",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-eval-gate",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-eval-gate",
        routing_sha256=ROUTING,
    )
    return build_review_core_minimal(
        session_id="rs-eval-gate",
        assignment_id="ra-eval-gate",
        submission_id="sub-eval-gate",
        review_nonce="nonce-eval-gate",
        assignment_digest="aa" * 32,
        rules_observation=_rules(),
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict=verdict),
        times=_times(issued=issued, received=received),
    )


def _envelope(
    *,
    verdict: str = "allow",
    issued: int = T0,
    received: int = T0 + 3_600_000,
    domain: str = REVIEW_REPORT_DOMAIN,
    mutilate_report_data: bool = False,
) -> dict:
    core = _core(verdict=verdict, issued=issued, received=received)
    rd = review_report_data_hex(core)
    if mutilate_report_data:
        # Force report_data mismatch while keeping bytes well-formed.
        rd = ("ff" * 32) + ("00" * 32)
    return {
        "schema_version": 1,
        "domain": domain,
        "review_digest": review_digest(core),
        "report_data_hex": rd,
        "review_core": core,
        "attestation": {
            "tdx_quote_hex": "00" * 16,
            "event_log": [],
            "measurement": {},
        },
    }


# ---------------------------------------------------------------------------
# VAL-ACAT-010 / 028: fresh allow → may launch; refuse matrix blocks lifecycle
# ---------------------------------------------------------------------------


def test_fresh_allow_reverify_may_launch() -> None:
    env = _envelope(verdict="allow", issued=T0, received=T0 + MS_23H)
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is True
    assert decision.reverify_exercised is True
    assert decision.reason_code == "review_verified"
    assert decision.verdict == "allow"
    assert decision.review_digest == env["review_digest"]
    assert decision.bound_issued_at_ms == T0
    assert decision.bound_received_at_ms == T0 + MS_23H


MS_23H = FRESHNESS_WINDOW_MS - 60_000
MS_24H = FRESHNESS_WINDOW_MS
MS_24H_PLUS = FRESHNESS_WINDOW_MS + 1


def test_exactly_24h_allow_may_launch() -> None:
    env = _envelope(verdict="allow", issued=T0, received=T0 + MS_24H)
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is True
    assert decision.reverify_exercised is True


def test_stale_over_24h_refuses_eval_launch() -> None:
    env = _envelope(verdict="allow", issued=T0, received=T0 + MS_24H_PLUS)
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is False
    assert decision.reverify_exercised is True
    assert decision.reason_code == REFUSE_STALE


def test_reject_verdict_refuses_eval_launch() -> None:
    env = _envelope(verdict="reject")
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        cached_outcome_status="verified_allow",  # cache tries to lie
        cached_phase="review_allowed",
    )
    assert decision.may_launch is False
    assert decision.reverify_exercised is True
    assert decision.reason_code == REFUSE_EVAL_CVM
    assert decision.verdict == "reject"


def test_reject_with_plain_status_allow_lie_refuses() -> None:
    """Caller claim 'allow' shears bound reject → still no eval spend."""

    env = _envelope(verdict="reject")
    decision = admit_eval_cvm_fresh_review(
        envelope=env,
        plain_status_allow=True,
    )
    assert decision.may_launch is False
    assert decision.reverify_exercised is True
    assert decision.reason_code == REFUSE_EVAL_CVM


def test_escalate_verdict_refuses_eval_spend_start() -> None:
    env = _envelope(verdict="escalate")
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_EVAL_CVM
    assert decision.verdict == "escalate"


def test_missing_attestation_refuses_even_without_cache() -> None:
    decision = admit_eval_cvm_fresh_review(envelope=None, review_core=None, report_data_hex=None)
    assert decision.may_launch is False
    assert decision.reverify_exercised is False
    assert decision.reason_code == REFUSE_ATTESTATION_MISSING


def test_wrong_domain_refuses() -> None:
    env = _envelope(domain="base-agent-challenge-v1")  # score domain, not review
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_WRONG_DOMAIN
    assert decision.reverify_exercised is True


def test_report_data_mismatch_refuses() -> None:
    env = _envelope(mutilate_report_data=True)
    decision = admit_eval_cvm_fresh_review(envelope=env)
    assert decision.may_launch is False
    assert decision.reverify_exercised is True
    assert decision.reason_code == REFUSE_REVERIFY_FAILED


# ---------------------------------------------------------------------------
# VAL-ACAT-029: cached allow alone is insufficient
# ---------------------------------------------------------------------------


def test_cached_allow_alone_without_envelope_refuses() -> None:
    decision = admit_eval_cvm_fresh_review(
        envelope=None,
        cached_phase="review_allowed",
        cached_outcome_status="verified_allow",
        cached_review_digest="ab" * 32,
        plain_status_allow=True,
    )
    assert decision.may_launch is False
    assert decision.reverify_exercised is False
    assert decision.reason_code == REFUSE_CACHED_ALLOW_ONLY


def test_cached_allow_with_only_digest_column_refuses() -> None:
    """Flip allow-bit storage without replaying quote materials → refuse."""

    assignment = SimpleNamespace(
        phase="review_allowed",
        review_verification_outcome_json=(
            '{"status":"verified_allow","terminal":true,"retryable":false,"nonce_consumed":true}'
        ),
        review_report_envelope_json='{"schema_version":1}',  # no core / report_data
        review_report_data_hex=None,
        review_digest="cd" * 32,
    )
    decision = admit_eval_cvm_launch_from_assignment(assignment)
    assert decision.may_launch is False
    assert decision.reverify_exercised is False
    assert decision.reason_code in {REFUSE_CACHED_ALLOW_ONLY, REFUSE_ATTESTATION_MISSING}


def test_assignment_with_fresh_envelope_admits() -> None:
    env = _envelope(verdict="allow")
    import json

    assignment = SimpleNamespace(
        phase="review_allowed",
        review_verification_outcome_json=(
            '{"status":"verified_allow","terminal":true,"retryable":false,"nonce_consumed":true}'
        ),
        review_report_envelope_json=json.dumps(env, sort_keys=True, separators=(",", ":")),
        review_report_data_hex=env["report_data_hex"],
        review_digest=env["review_digest"],
    )
    decision = admit_eval_cvm_launch_from_assignment(assignment)
    assert decision.may_launch is True
    assert decision.reverify_exercised is True
    assert decision.review_digest == env["review_digest"]


def test_require_raises_on_cached_only() -> None:
    with pytest.raises(EvalCvmFreshReviewError) as exc:
        require_eval_cvm_fresh_review(
            cached_outcome_status="verified_allow",
            cached_phase="review_allowed",
        )
    assert exc.value.code == REFUSE_CACHED_ALLOW_ONLY


def test_quote_reverify_failure_refuses() -> None:
    env = _envelope(verdict="allow")

    def _bad_quote(_materials: dict) -> bool:
        return False

    decision = admit_eval_cvm_fresh_review(envelope=env, quote_reverify=_bad_quote)
    assert decision.may_launch is False
    assert decision.reason_code == REFUSE_REVERIFY_FAILED
    assert decision.reverify_exercised is True


def test_quote_reverify_success_and_fresh_allow_launches() -> None:
    env = _envelope(verdict="allow")
    seen: list[bool] = []

    def _ok_quote(_materials: dict) -> bool:
        seen.append(True)
        return True

    decision = admit_eval_cvm_fresh_review(envelope=env, quote_reverify=_ok_quote)
    assert seen == [True]
    assert decision.may_launch is True
    assert decision.reverify_exercised is True


def test_require_raises_on_stale() -> None:
    env = _envelope(verdict="allow", issued=T0, received=T0 + MS_24H_PLUS)
    with pytest.raises(EvalCvmFreshReviewError) as exc:
        require_eval_cvm_fresh_review(envelope=env)
    assert exc.value.code == REFUSE_STALE
