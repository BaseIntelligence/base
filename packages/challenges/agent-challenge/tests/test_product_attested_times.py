"""Product VAL-ACAT-007/008/021-024/038: issued_at/received_at crypto bind.

Bound times inhabit review-domain report_data preimage v2. Guest clock alone,
unattested DB columns alone, and client-smuggled public bag times cannot
authorize production age decisions. Re-verify extracts only quote-bound times.
"""

from __future__ import annotations

import pytest

from agent_challenge.review.attested_times import (
    FRESHNESS_WINDOW_MS,
    REFUSE_DB_ONLY_TIMES,
    REFUSE_GUEST_CLOCK_ALONE,
    REFUSE_REPORT_DATA_MISMATCH,
    REFUSE_TIMES_MISSING,
    REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_V2,
    AttestedTimeError,
    check_freshness,
    client_smuggled_time_keys,
    ignore_client_smuggled_times,
    production_times_from_reverified_materials,
    refuse_db_only_time_authorization,
    refuse_guest_clock_alone_authorization,
    reverify_extract_bound_times,
    review_report_data_hex_v2,
    review_report_data_preimage_v2,
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
    review_digest,
    review_report_data_hex,
    sha256_hex,
)
from agent_challenge.review.report import (
    REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION,
    ReviewReportError,
    review_report_data_preimage,
)

T0 = 1_700_000_000_000
ROUTING_SHA = sha256_hex(b'{"order":["a"]}')
BODY = b'{"model":"x-ai/grok-4.5"}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"x"}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"m")


def _times(*, issued: int = T0, received: int = T0 + 60_000) -> dict[str, int]:
    return {
        "issued_at_ms": issued,
        "started_at_ms": issued,
        "model_call_marked_at_ms": issued + 1,
        "request_started_at_ms": issued + 2,
        "request_finished_at_ms": issued + 3,
        "verifier_finished_at_ms": issued + 4,
        "report_finished_at_ms": issued + 5,
        "expires_at_ms": issued + 3_600_000,
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
        response_id="gen-1",
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
        session_id="rs-times",
        assignment_id="ra-times",
        submission_id="42",
        review_nonce="rn-times",
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


def test_preimage_v2_schema_and_bound_times() -> None:
    core = _core()
    pre = review_report_data_preimage(core)
    assert pre["schema_version"] == REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_VERSION == 2
    assert pre["schema_version"] == REVIEW_REPORT_DATA_PREIMAGE_SCHEMA_V2
    assert pre["issued_at_ms"] == T0
    assert pre["received_at_ms"] == T0 + 60_000
    assert set(pre.keys()) == {
        "domain",
        "schema_version",
        "review_digest",
        "session_id",
        "review_nonce",
        "issued_at_ms",
        "received_at_ms",
    }


def test_report_data_mutates_when_issued_at_flipped() -> None:
    a = review_report_data_hex(_core(issued=T0, received=T0 + 100))
    b = review_report_data_hex(_core(issued=T0 + 1, received=T0 + 100))
    assert a != b
    assert len(a) == 128 and a.endswith("00" * 32)


def test_report_data_mutates_when_received_at_flipped() -> None:
    a = review_report_data_hex(_core(issued=T0, received=T0 + 100))
    b = review_report_data_hex(_core(issued=T0, received=T0 + 101))
    assert a != b


def test_reverify_extracts_bound_times() -> None:
    core = _core()
    rd = review_report_data_hex(core)
    bound = reverify_extract_bound_times(review_core=core, report_data_hex=rd)
    assert bound == {"issued_at_ms": T0, "received_at_ms": T0 + 60_000}


def test_reverify_refuses_mutated_report_data() -> None:
    core = _core()
    rd = review_report_data_hex(core)
    mutated = ("0" if rd[0] != "0" else "1") + rd[1:]
    with pytest.raises(AttestedTimeError) as exc:
        reverify_extract_bound_times(review_core=core, report_data_hex=mutated)
    assert exc.value.code == REFUSE_REPORT_DATA_MISMATCH


def test_db_only_times_cannot_authorize() -> None:
    # Without report_data re-verify, DB columns alone refuse.
    code = refuse_db_only_time_authorization(
        db_issued_at_ms=T0,
        db_received_at_ms=T0 + 1,
        report_data_reverified=False,
        bound_issued_at_ms=None,
        bound_received_at_ms=None,
    )
    assert code == REFUSE_DB_ONLY_TIMES

    core = _core()
    rd = review_report_data_hex(core)
    # With shear DB vs bound → refuse.
    with pytest.raises(AttestedTimeError) as exc:
        production_times_from_reverified_materials(
            review_core=core,
            report_data_hex=rd,
            db_issued_at_ms=T0 + 999_999,  # sheared cache
            db_received_at_ms=T0 + 60_000,
        )
    assert exc.value.code == REFUSE_DB_ONLY_TIMES


def test_guest_alone_clock_refuses() -> None:
    code = refuse_guest_clock_alone_authorization(
        guest_issued_at_ms=T0,
        guest_received_at_ms=T0 + 10,
        challenge_bound_issued_at_ms=None,
        challenge_bound_received_at_ms=None,
        report_data_matched=False,
    )
    assert code == REFUSE_GUEST_CLOCK_ALONE


def test_client_smuggled_times_ignored_for_security() -> None:
    bag = {
        "envelope": {"x": 1},
        "evidence": {},
        "issued_at_ms": 1,
        "received_at_ms": 2,
        "already_fresh": True,
        "created": 123,
    }
    assert client_smuggled_time_keys(bag) == frozenset(
        {"issued_at_ms", "received_at_ms", "already_fresh", "created"}
    )
    cleaned = ignore_client_smuggled_times(bag)
    assert "issued_at_ms" not in cleaned
    assert "already_fresh" not in cleaned
    assert cleaned["envelope"] == {"x": 1}

    core = _core()
    rd = review_report_data_hex(core)
    bound = production_times_from_reverified_materials(
        review_core=core,
        report_data_hex=rd,
        client_bag=bag,
    )
    # Smuggled times do not override bound values used for age.
    assert bound["issued_at_ms"] == T0
    assert bound["received_at_ms"] == T0 + 60_000
    assert bound["issued_at_ms"] != bag["issued_at_ms"]


def test_admit_production_requires_bound_times_in_report_data() -> None:
    core = _core()
    rd = review_report_data_hex(core)
    ok = admit_production_from_bound_outcome(
        review_core=core,
        reported_report_data_hex=rd,
        require_or_digests=True,
    )
    assert ok.admitted is True
    assert ok.verdict == "allow"

    # Wrong report_data refuses even if plain status would allow.
    with pytest.raises(ReviewOrOutcomeError):
        admit_production_from_bound_outcome(
            review_core=core,
            reported_report_data_hex="00" * 64,
            plain_status_allow=True,
            require_or_digests=True,
        )


def test_v2_preimage_independent_builder_matches_product() -> None:
    core = _core()
    digest = review_digest(core)
    product = review_report_data_preimage(core)
    direct = review_report_data_preimage_v2(
        review_digest=digest,
        session_id=core["session_id"],
        review_nonce=core["review_nonce"],
        issued_at_ms=core["times"]["issued_at_ms"],
        received_at_ms=core["times"]["submission_received_at_ms"],
    )
    assert product == direct
    assert review_report_data_hex_v2(
        review_digest=digest,
        session_id=core["session_id"],
        review_nonce=core["review_nonce"],
        issued_at_ms=core["times"]["issued_at_ms"],
        received_at_ms=core["times"]["submission_received_at_ms"],
    ) == review_report_data_hex(core)


def test_freshness_boundary_scaffold_aligned() -> None:
    assert check_freshness(issued_at_ms=T0, received_at_ms=T0 + FRESHNESS_WINDOW_MS) is None
    assert check_freshness(issued_at_ms=T0, received_at_ms=T0 + FRESHNESS_WINDOW_MS + 1) is not None
    # Submit-first ZIP before issue: absolute age under 24h admits (not REFUSE_TIME_ORDER).
    assert check_freshness(issued_at_ms=T0 + 5, received_at_ms=T0) is None
    assert check_freshness(issued_at_ms=T0 + FRESHNESS_WINDOW_MS + 1, received_at_ms=T0) is not None
    assert check_freshness(issued_at_ms=None, received_at_ms=T0) == REFUSE_TIMES_MISSING


def test_missing_submission_received_fails_closed() -> None:
    core = _core()
    del core["times"]["submission_received_at_ms"]
    with pytest.raises((ReviewReportError, AttestedTimeError, ValueError)):
        review_report_data_hex(core)
