"""Residual class report_timeline_invalid (sub25 / tdx.small measured path).

Live artifact: guest/history durable reason_code=report_timeline_invalid after
service tip ≥37fed2e8 + envelope diag; soft evaluate prepare 403; verified_allow
false. Evidence: eval-1task-service-tip-speed SUMMARY.

Root cause product bind failure (Mode B, this module):
  miner submit-first stamps challenge-domain submission_received_at_ms (ZIP
  admit) *before* assignment issued_at_ms. check_freshness incorrectly refused
  that normal order as attestation_time_order_invalid, which guest maps to
  report_timeline_invalid via /report detail code tokens.

Fix: ≤24h freshness allows either chronological order of the two bound fields
when absolute separation is within the window (still fail closed over 24h).
Does not invent times; both values stay opinionated challenge-domain stamps.
"""

from __future__ import annotations

import pytest

from agent_challenge.review.attested_times import (
    FRESHNESS_WINDOW_MS,
    REFUSE_STALE,
    REFUSE_TIME_ORDER,
    check_freshness,
    enforce_bound_freshness,
    production_freshness_from_reverified_materials,
)
from agent_challenge.review.or_outcome_bind import (
    ReviewOrOutcomeError,
    admit_production_from_bound_outcome,
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

T0 = 1_700_000_000_000
# Typical residual: ZIP admitted ~minutes earlier than assignment issue.
ZIP_BEFORE_ISSUE_MS = 120_000
MS_24H = FRESHNESS_WINDOW_MS
MS_24H_PLUS_1 = FRESHNESS_WINDOW_MS + 1

ROUTING_SHA = sha256_hex(b'{"order":["xai"]}')
BODY = b'{"model":"x-ai/grok-4.5"}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"timeline-residual"}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-timeline")


def _times(*, issued: int, received: int) -> dict[str, int]:
    """Internal report leaf chain stays valid relative to bound pair.

    The middle leaves (started…report_finished) sit near max(issue, receive)
    so schema ordering stays green either when ZIP pretangles issue or when
    issue pretangles ZIP.
    """

    base = max(issued, received)
    return {
        "issued_at_ms": issued,
        "started_at_ms": base,
        "model_call_marked_at_ms": base + 1,
        "request_started_at_ms": base + 2,
        "request_finished_at_ms": base + 3,
        "verifier_finished_at_ms": base + 4,
        "report_finished_at_ms": base + 5,
        "expires_at_ms": base + 3_600_000,
        "submission_received_at_ms": received,
    }


def _core(*, issued: int, received: int) -> dict:
    planned = build_planned_openrouter_request(
        body_sha256=BODY_SHA,
        body_length=len(BODY),
        routing_sha256=ROUTING_SHA,
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
        response_id="gen-timeline-residual",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier",
        routing_sha256=ROUTING_SHA,
    )
    decision = build_decision(verdict="allow")
    return build_review_core_minimal(
        session_id="rs-timeline-residual",
        assignment_id="ra-timeline-residual",
        submission_id="25",
        review_nonce="rn-timeline-residual",
        assignment_digest="aa" * 32,
        rules_observation={"snapshot_sha256": "bb" * 32, "revision_id": "rules-v1"},
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=decision,
        times=_times(issued=issued, received=received),
        artifact_observation={
            "agent_hash": "11" * 32,
            "zip_sha256": "22" * 32,
            "zip_size_bytes": 4,
            "manifest_sha256": "33" * 32,
            "manifest_entries_sha256": "44" * 32,
        },
    )


# ---------------------------------------------------------------------------
# RED residual class (submit-first: ZIP receive before assignment issue)
# ---------------------------------------------------------------------------


def test_submit_first_zip_before_issue_within_24h_admits() -> None:
    """Measured residual chronology: submission_received < issued, |Δ| << 24h.

    This is the durable product order for miner ZIP → later selfdeploy review
    deploy assignment. Must NOT map to report_timeline_invalid.
    """

    received = T0  # ZIP admit (challenge-domain)
    issued = T0 + ZIP_BEFORE_ISSUE_MS  # assignment issue after ZIP
    assert issued > received
    assert (issued - received) < FRESHNESS_WINDOW_MS

    assert check_freshness(issued_at_ms=issued, received_at_ms=received) is None
    core = _core(issued=issued, received=received)
    rd = review_report_data_hex(core)
    bound = production_freshness_from_reverified_materials(
        review_core=core,
        report_data_hex=rd,
    )
    assert bound == {"issued_at_ms": issued, "received_at_ms": received}
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True
    assert admission.status == "verified_allow"
    assert admission.reason_code == "review_verified"


def test_issue_before_zip_within_24h_still_admits() -> None:
    """Self-deploy-then-admit variant (issue ≤ ZIP) stays green."""

    issued = T0
    received = T0 + ZIP_BEFORE_ISSUE_MS
    assert check_freshness(issued_at_ms=issued, received_at_ms=received) is None
    core = _core(issued=issued, received=received)
    rd = review_report_data_hex(core)
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True


def test_absolute_age_over_24h_either_order_refuses_stale() -> None:
    """Stale refuse is absolute separation, not directional order."""

    # ZIP long before issue
    assert check_freshness(issued_at_ms=T0 + MS_24H_PLUS_1, received_at_ms=T0) == REFUSE_STALE
    # Issue long before ZIP
    assert check_freshness(issued_at_ms=T0, received_at_ms=T0 + MS_24H_PLUS_1) == REFUSE_STALE

    core = _core(issued=T0 + MS_24H_PLUS_1, received=T0)
    rd = review_report_data_hex(core)
    with pytest.raises(ReviewOrOutcomeError) as exc:
        admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex=rd,
            require_or_digests=True,
        )
    assert exc.value.code == REFUSE_STALE
    assert exc.value.code != REFUSE_TIME_ORDER


def test_enforce_bound_freshness_submit_first_no_time_order_error() -> None:
    bound = enforce_bound_freshness(
        issued_at_ms=T0 + ZIP_BEFORE_ISSUE_MS,
        received_at_ms=T0,
    )
    assert bound["issued_at_ms"] == T0 + ZIP_BEFORE_ISSUE_MS
    assert bound["received_at_ms"] == T0


def test_exact_24h_absolute_boundary_passes_both_orders() -> None:
    assert check_freshness(issued_at_ms=T0 + MS_24H, received_at_ms=T0) is None
    assert check_freshness(issued_at_ms=T0, received_at_ms=T0 + MS_24H) is None
