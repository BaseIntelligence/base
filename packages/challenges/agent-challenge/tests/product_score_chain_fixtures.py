"""Shared dual-flag full-chain fixtures for score admission tests.

Provides:
* valid review envelope bound to a computable review_digest
* key-release grant reconstructible against an eval plan
* durable ReviewSession + ReviewAssignment seeder so
  ``process_direct_eval_result`` can re-load materials
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import ReviewAssignment, ReviewSession
from agent_challenge.evaluation.score_chain_gate import (
    KEY_RELEASE_DOMAIN,
    recompute_key_release_report_data_hex,
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
MS_23H = FRESHNESS_WINDOW_MS - 60_000
ROUTING = sha256_hex(b'{"order":["fixture-chain"]}')
BODY = b'{"model":"x-ai/grok-4.5","messages":[]}'
BODY_SHA = sha256_hex(BODY)
RESP = b'{"id":"gen-fx","model":"x-ai/grok-4.5","choices":[]}'
RESP_SHA = sha256_hex(RESP)
META = sha256_hex(b"meta-fx-chain")
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


def build_fixture_review_envelope(
    *,
    session_id: str = "rs-fx-chain",
    assignment_id: str = "ra-fx-chain",
    submission_id: str = "sub-fx-chain",
    review_nonce: str = "nonce-fx-review",
) -> dict[str, Any]:
    """Closed valid allow envelope with bound times ≤24h and OR digests."""

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
        response_id="gen-fx",
    )
    policy = build_policy_observation(
        prompt_version="review-policy-prompt-v1",
        prompt_bytes=b"prompt-fx",
        tool_schema_version="review-policy-tool-v1",
        tool_schema_bytes=b"tools-fx",
        verifier_version="review-policy-verifier-v1",
        verifier_bytes=b"verifier-fx",
        routing_sha256=ROUTING,
    )
    rules = {
        "rules_version": "rules-v1",
        "rules_bundle_sha256": "11" * 32,
        "rules_files": [".rules/acceptance.md"],
        "rules_file_digests": {".rules/acceptance.md": "22" * 32},
        "rules_policy_text_sha256": "33" * 32,
    }
    core = build_review_core_minimal(
        session_id=session_id,
        assignment_id=assignment_id,
        submission_id=submission_id,
        review_nonce=review_nonce,
        assignment_digest="aa" * 32,
        rules_observation=rules,
        policy_observation=policy,
        openrouter_observation=or_obs,
        decision=build_decision(verdict="allow"),
        times=_times(),
    )
    return {
        "schema_version": 1,
        "domain": REVIEW_REPORT_DOMAIN,
        "review_digest": review_digest(core),
        "report_data_hex": review_report_data_hex(core),
        "review_core": core,
    }


def build_key_release_grant_for_plan(
    plan: dict[str, Any],
    *,
    spki: str = SPKI,
) -> dict[str, Any]:
    rd = recompute_key_release_report_data_hex(
        eval_run_id=str(plan["eval_run_id"]),
        key_release_nonce=str(plan["key_release_nonce"]),
        ra_tls_spki_digest=spki,
    )
    return {
        "domain": KEY_RELEASE_DOMAIN,
        "schema_version": 2,
        "eval_run_id": plan["eval_run_id"],
        "key_release_nonce": plan["key_release_nonce"],
        "ra_tls_spki_digest": spki,
        "report_data_hex": rd,
        "agent_hash": plan.get("agent_hash"),
    }


def attach_key_release_grant(
    request: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Return the request unchanged (grant is not on the closed wire body).

    Prefer :func:`bind_key_release_grant_on_run` to inject grant materials into
    the durable EvalRun object that score admission re-checks.
    """

    _ = plan
    return request


def bind_key_release_grant_on_run(run: Any, plan: dict[str, Any]) -> dict[str, Any]:
    """Stamp KR grant materials for admission-time re-verify (durable + cache).

    Production path sometimes uses ``mark_eval_key_granted(..., ra_tls_spki_digest=)``
    which writes ``EvalRun.key_release_grant_json``. Fixtures mirror that durable
    column so multi-worker / restart simulations exercise the real load path.
    """

    import json

    from agent_challenge.evaluation.score_chain_gate import (
        register_key_release_grant_for_score,
    )

    grant = build_key_release_grant_for_plan(plan)
    run._score_chain_key_release_grant = grant
    run.key_release_grant_json = json.dumps(grant, sort_keys=True, separators=(",", ":"))
    register_key_release_grant_for_score(str(plan["eval_run_id"]), grant)
    return grant


async def seed_authorizing_review_assignment(
    session: Any,
    *,
    submission_id: int,
    envelope: dict[str, Any],
    authorizing_review_digest: str | None = None,
) -> ReviewAssignment:
    """Persist a verified_allow ReviewSession+Assignment for a submission."""

    digest = authorizing_review_digest or str(envelope["review_digest"])
    now = datetime.now(UTC)
    rs = ReviewSession(
        session_id=f"rs-seed-{submission_id}",
        submission_id=submission_id,
        artifact_sha256="ab" * 32,
        artifact_size_bytes=12,
        manifest_sha256="cd" * 32,
        manifest_entries_sha256="ef" * 32,
        current_assignment_id=f"ra-seed-{submission_id}",
        authorizing_assignment_id=f"ra-seed-{submission_id}",
        submission_received_at_ms=T0 + MS_23H,
    )
    session.add(rs)
    await session.flush()
    outcome = {
        "status": "verified_allow",
        "reason_code": "review_verified",
        "terminal": True,
        "retryable": False,
        "nonce_consumed": True,
    }
    ra = ReviewAssignment(
        session_id=rs.id,
        assignment_id=f"ra-seed-{submission_id}",
        attempt=1,
        assignment_bytes="{}",
        assignment_digest="aa" * 32,
        artifact_sha256="ab" * 32,
        rules_snapshot_sha256="11" * 32,
        rules_revision_id="rules-v1",
        review_nonce=f"nonce-seed-{submission_id}",
        session_token_sha256="bb" * 32,
        capability_state="consumed",
        phase="review_allowed",
        review_report_envelope_json=json.dumps(envelope, separators=(",", ":")),
        review_digest=digest,
        review_report_data_hex=str(envelope.get("report_data_hex")),
        review_verification_outcome_json=json.dumps(outcome, separators=(",", ":")),
        issued_at=now - timedelta(hours=1),
        expires_at=now + timedelta(hours=12),
    )
    session.add(ra)
    await session.flush()
    return ra


def rebind_plan_authorizing_digest(
    plan: dict[str, Any],
    envelope: dict[str, Any],
) -> dict[str, Any]:
    """Return a plan copy with authorizing_review_digest matching envelope."""

    updated = dict(plan)
    updated["authorizing_review_digest"] = envelope["review_digest"]
    return ew.validate_eval_plan(updated)


__all__ = [
    "MS_23H",
    "SPKI",
    "T0",
    "attach_key_release_grant",
    "bind_key_release_grant_on_run",
    "build_fixture_review_envelope",
    "build_key_release_grant_for_plan",
    "rebind_plan_authorizing_digest",
    "seed_authorizing_review_assignment",
]
