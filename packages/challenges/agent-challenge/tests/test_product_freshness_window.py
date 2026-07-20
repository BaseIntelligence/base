"""Product VAL-ACAT-009/025/026/027: ≤24h freshness over attestation-bound times.

Enforces absolute age using only cryptographically bound
``issued_at_ms`` / ``received_at_ms`` (report_data preimage v2).

Product boundary (exact, documented here and in library/ac-attestation.md):

- ``abs(received_at_ms - issued_at_ms) ≤ 86_400_000`` admits (exactly 24h00m00s **passes**)
- ``86_400_000 + 1`` ms absolute age refuses with ``attestation_stale_over_24h``
- Either chronological order of the two bound stamps admits inside the window
  (submit-first ZIP before issue is the measured residual path)
- Missing / invalid times refuse with stable codes (fail closed)
- HTTP Date / client header / auth skew alone never satisfy the window

Live admission (``admit_production_from_bound_outcome``) re-runs re-verify + freshness.
"""

from __future__ import annotations

import pytest

from agent_challenge.review.attested_times import (
    FRESHNESS_WINDOW_MS,
    REFUSE_STALE,
    REFUSE_TIME_ORDER,
    REFUSE_TIMES_INVALID,
    REFUSE_TIMES_MISSING,
    AttestedTimeError,
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
MS_23H59 = FRESHNESS_WINDOW_MS - 60_000  # 23h59m exactly relative to window constant
MS_24H = FRESHNESS_WINDOW_MS  # exactly 24h00m00s
MS_24H_PLUS_1 = FRESHNESS_WINDOW_MS + 1
MS_MULTI_DAY = FRESHNESS_WINDOW_MS * 3

ROUTING_SHA = sha256_hex(b'{"order":["a"]}')
BODY = b'{"model":"x-ai/grok-4.5"}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"x"}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"m")


def _times(*, issued: int = T0, received: int = T0 + 60_000) -> dict[str, int]:
    # Timeline fields stay nearly concurrent; only issued/received drive the window.
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


def _core(*, issued: int = T0, received: int = T0 + 60_000) -> dict:
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
        response_id="gen-fresh",
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
        session_id="rs-fresh",
        assignment_id="ra-fresh",
        submission_id="42",
        review_nonce="rn-fresh",
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
# Age matrix (exact boundary) — VAL-ACAT-009 / VAL-ACAT-025
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_ms", "expect_ok"),
    [
        (0, True),  # simultaneous evidence + submit
        (1, True),
        (MS_23H59, True),  # 23h59m
        (MS_24H - 1, True),  # 23h59m59s.999
        (MS_24H, True),  # exact 24h00m00s boundary PASSES (product ≤ rule)
        (MS_24H_PLUS_1, False),  # 24h + 1ms REFUSES
        (MS_MULTI_DAY, False),  # multi-day stale
    ],
)
def test_age_matrix_boundary_exact_24h(delta_ms: int, expect_ok: bool) -> None:
    """Documented product boundary: Δ ≤ 86400000 passes; Δ = 86400001 refuses."""

    issued = T0
    received = T0 + delta_ms
    code = check_freshness(issued_at_ms=issued, received_at_ms=received)
    if expect_ok:
        assert code is None, f"expected admit for delta={delta_ms}, got {code}"
    else:
        assert code == REFUSE_STALE, f"expected stale for delta={delta_ms}, got {code}"

    # Same matrix via enforce (exception path used by admission).
    if expect_ok:
        bound = enforce_bound_freshness(issued_at_ms=issued, received_at_ms=received)
        assert bound == {"issued_at_ms": issued, "received_at_ms": received}
    else:
        with pytest.raises(AttestedTimeError) as exc:
            enforce_bound_freshness(issued_at_ms=issued, received_at_ms=received)
        assert exc.value.code == REFUSE_STALE


def test_admit_production_accepts_exactly_24h() -> None:
    core = _core(issued=T0, received=T0 + MS_24H)
    rd = review_report_data_hex(core)
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True
    assert admission.reason_code == "review_verified"
    assert admission.verdict == "allow"


def test_admit_production_refuses_stale_over_24h_plus_1ms() -> None:
    core = _core(issued=T0, received=T0 + MS_24H_PLUS_1)
    rd = review_report_data_hex(core)
    with pytest.raises(ReviewOrOutcomeError) as exc:
        admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex=rd,
            require_or_digests=True,
        )
    assert exc.value.code == REFUSE_STALE


def test_admit_production_refuses_multi_day_stale() -> None:
    core = _core(issued=T0, received=T0 + MS_MULTI_DAY)
    rd = review_report_data_hex(core)
    with pytest.raises(ReviewOrOutcomeError) as exc:
        admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex=rd,
            require_or_digests=True,
        )
    assert exc.value.code == REFUSE_STALE


# ---------------------------------------------------------------------------
# Directional order — residual report_timeline_invalid (submit-first)
# ---------------------------------------------------------------------------


def test_submit_first_issued_after_received_admits_under_24h() -> None:
    """ZIP admit before assignment issue is normal; absolute age, not interval direction.

    Prior (buggy) product refused ``issued > received`` as
    ``attestation_time_order_invalid`` → guest ``report_timeline_invalid``.
    Absolute lag of 5s under 24h must admit for the measured dual-flag path.
    """

    issued = T0 + 5_000
    received = T0
    assert abs(issued - received) < FRESHNESS_WINDOW_MS
    assert check_freshness(issued_at_ms=issued, received_at_ms=received) is None

    core = _core(issued=issued, received=received)
    rd = review_report_data_hex(core)
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True
    assert admission.reason_code == "review_verified"


def test_order_equal_times_pass() -> None:
    assert check_freshness(issued_at_ms=T0, received_at_ms=T0) is None
    core = _core(issued=T0, received=T0)
    rd = review_report_data_hex(core)
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True


# ---------------------------------------------------------------------------
# Header-only / client skew cannot satisfy window — VAL-ACAT-027
# ---------------------------------------------------------------------------


def test_header_only_timestamps_cannot_satisfy_window() -> None:
    """HTTP Date 'looking fresh' does not rescue stale bound attestation times."""

    core = _core(issued=T0, received=T0 + MS_24H_PLUS_1)
    rd = review_report_data_hex(core)
    # Fresh-looking HTTP Date and client header times near "now" relative to issue.
    fake_fresh_http_date = T0 + 60_000
    with pytest.raises(ReviewOrOutcomeError) as exc:
        admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex=rd,
            require_or_digests=True,
            http_date_ms=fake_fresh_http_date,
            client_header_ms=fake_fresh_http_date,
            client_skew_ms=300_000,  # signed-request ~300s skew budget elsewhere
        )
    assert exc.value.code == REFUSE_STALE

    # Positive control: same headers, bound times within 24h → admit.
    core_ok = _core(issued=T0, received=T0 + 3_600_000)
    rd_ok = review_report_data_hex(core_ok)
    ok = admit_production_from_bound_outcome(
        review_core=core_ok,
        reported_report_data_hex=rd_ok,
        require_or_digests=True,
        http_date_ms=fake_fresh_http_date,
        client_header_ms=T0 - 1,  # adversarial header that would invert order
        client_skew_ms=-99_999_999,
    )
    assert ok.admitted is True


def test_check_freshness_never_uses_header_args() -> None:
    """API surface of enforce_bound_freshness discards non-bound clocks."""

    with pytest.raises(AttestedTimeError) as exc:
        enforce_bound_freshness(
            issued_at_ms=T0,
            received_at_ms=T0 + MS_24H_PLUS_1,
            http_date_ms=T0 + 1,  # pretend British fresh header
            client_header_ms=T0 + 1,
            client_skew_ms=0,
        )
    assert exc.value.code == REFUSE_STALE


def test_reverify_plus_freshness_helper_refuses_stale() -> None:
    core = _core(issued=T0, received=T0 + MS_24H_PLUS_1)
    rd = review_report_data_hex(core)
    with pytest.raises(AttestedTimeError) as exc:
        production_freshness_from_reverified_materials(
            review_core=core,
            report_data_hex=rd,
            http_date_ms=T0 + 10,
        )
    assert exc.value.code == REFUSE_STALE


# ---------------------------------------------------------------------------
# Missing / invalid times — fail closed — VAL-ACAT-009
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("issued", "received", "expect"),
    [
        (None, T0, REFUSE_TIMES_MISSING),
        (T0, None, REFUSE_TIMES_MISSING),
        (True, T0, REFUSE_TIMES_INVALID),  # bool is not int
        (T0, "1700000000000", REFUSE_TIMES_INVALID),
        (-1, T0, REFUSE_TIMES_INVALID),
        (T0, 2**63, REFUSE_TIMES_INVALID),
    ],
)
def test_missing_or_invalid_times_fail_closed(
    issued: object, received: object, expect: str
) -> None:
    assert check_freshness(issued_at_ms=issued, received_at_ms=received) == expect


# ---------------------------------------------------------------------------
# Stable reason codes registry (wire freeze)
# ---------------------------------------------------------------------------


def test_stable_stale_and_order_reason_codes() -> None:
    assert REFUSE_STALE == "attestation_stale_over_24h"
    assert REFUSE_TIME_ORDER == "attestation_time_order_invalid"
    assert REFUSE_TIMES_MISSING == "attestation_times_missing"
    assert REFUSE_TIMES_INVALID == "attestation_times_invalid"
    assert FRESHNESS_WINDOW_MS == 86_400_000


def test_fresh_under_24h_admission_via_reverify() -> None:
    """23h59 delta from verified report_data admits."""

    core = _core(issued=T0, received=T0 + MS_23H59)
    rd = review_report_data_hex(core)
    bound = production_freshness_from_reverified_materials(
        review_core=core,
        report_data_hex=rd,
    )
    assert bound["issued_at_ms"] == T0
    assert bound["received_at_ms"] == T0 + MS_23H59
    admission = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert admission.admitted is True
