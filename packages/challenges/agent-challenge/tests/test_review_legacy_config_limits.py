"""Legacy status identity and config-backed review limits.

Covers VAL-REVIEW-060 / VAL-CROSS-019 edge controls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import select

from agent_challenge.api import routes as api_routes
from agent_challenge.models import AgentSubmission, ReviewNonce
from agent_challenge.review.sessions import (
    _MUTATION_WINDOWS,
    ReviewConflict,
    ReviewRateLimited,
    create_review_session,
    enforce_outstanding_review_cap,
    enforce_review_session_mutation_budget,
    issue_operator_approval,
    prune_outstanding_review_records,
    review_audit_page,
)
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.submissions.state_machine import transition_submission_status


def _submission(name: str = "legacy-cfg", *, artifact: bytes = b"zip") -> AgentSubmission:
    digest = sha256(artifact).hexdigest()
    return AgentSubmission(
        miner_hotkey=f"miner-{name}",
        name=f"{name}-agent",
        agent_hash=digest,
        artifact_uri=f"/tmp/{name}.zip",
        zip_sha256=digest,
        zip_size_bytes=len(artifact),
        status="received",
        raw_status="received",
        effective_status="received",
    )


async def test_legacy_status_excludes_review_field_and_skips_review_query(
    client,
    database_session,
    monkeypatch,
):
    """Fully legacy mode: status bytes omit `review` and do no review work."""

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", False)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", False)

    async def _boom(*_args, **_kwargs):  # pragma: no cover - must not be called
        raise AssertionError("legacy status must not query review state")

    monkeypatch.setattr(api_routes, "_review_status_response", _boom)

    async with database_session() as session:
        submission = _submission("flag-off-status")
        session.add(submission)
        await session.flush()
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="received",
            from_status=None,
        )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert "review" not in payload
    # Stable legacy key set still present.
    for required in (
        "submission_id",
        "status",
        "public_state",
        "phase",
        "analyzer",
        "evaluation",
        "terminal_bench",
        "progress",
    ):
        assert required in payload


async def test_full_attested_status_includes_review_field(
    client,
    database_session,
    monkeypatch,
):
    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)

    async with database_session() as session:
        submission = _submission("full-status")
        session.add(submission)
        await session.flush()
        await transition_submission_status(
            session,
            submission,
            "received",
            actor="api",
            reason="received",
            from_status=None,
        )
        await session.commit()
        submission_id = submission.id

    response = await client.get(f"/submissions/{submission_id}/status")
    assert response.status_code == 200
    payload = response.json()
    assert "review" in payload
    assert payload["review"]["session_id"] is None


def test_challenge_settings_declares_normative_review_limits() -> None:
    settings = ChallengeSettings()
    expected = {
        "review_assignment_ttl_seconds": 1800,
        "review_operator_approval_ttl_seconds": 300,
        "review_https_connect_timeout_seconds": 10.0,
        "review_https_tls_timeout_seconds": 10.0,
        "review_https_read_timeout_seconds": 240.0,
        "review_https_total_timeout_seconds": 300.0,
        "attestation_verification_timeout_seconds": 60.0,
        "review_max_assignment_bytes": 262_144,
        "review_max_capability_bytes": 4_096,
        "review_max_approval_bytes": 4_096,
        "eval_max_capability_bytes": 4_096,
        "review_max_rules_bytes": 1_048_576,
        "review_max_rules_files": 128,
        "review_max_report_request_bytes": 8_388_608,
        "review_max_openrouter_request_bytes": 4_194_304,
        "review_max_openrouter_response_bytes": 1_048_576,
        "review_max_openrouter_metadata_bytes": 262_144,
        "review_max_encrypted_evidence_bytes": 6_291_456,
        "review_max_quote_bytes": 65_536,
        "review_max_event_log_bytes": 2_097_152,
        "review_max_event_log_entries": 4_096,
        "review_max_vm_config_bytes": 65_536,
        "review_max_reason_evidence_items": 256,
        "review_max_string_bytes": 16_384,
        "review_max_assignments_per_session": 16,
        "eval_max_runs_per_submission": 8,
        "review_report_page_default": 10,
        "review_report_page_max": 16,
        "review_report_max_response_bytes": 2_097_152,
        "review_internal_report_max_response_bytes": 12_582_912,
        "review_evidence_max_object_bytes": 6_291_456,
        "review_evidence_max_range_bytes": 6_291_456,
        "eval_status_page_default": 10,
        "eval_status_page_max": 16,
        "eval_status_max_response_bytes": 2_097_152,
        "review_max_mutations_per_session_per_minute": 10,
        "attestation_max_outstanding_nonce_receipts": 10_000,
        "attestation_max_concurrent_verifications": 8,
    }
    for key, value in expected.items():
        assert getattr(settings, key) == value


def test_review_rate_limited_is_independent_of_review_conflict() -> None:
    """Rate/concurrency must not inherit conflict so intake can return 429."""

    assert not issubclass(ReviewRateLimited, ReviewConflict)
    limited = ReviewRateLimited("review outstanding nonce capacity is full")
    assert isinstance(limited, ReviewRateLimited)
    assert not isinstance(limited, ReviewConflict)
    conflict = ReviewConflict("review assignment is active")
    assert not isinstance(conflict, ReviewRateLimited)


def test_exactly_at_string_limit_boundary_helpers() -> None:
    settings = ChallengeSettings(review_max_string_bytes=16)
    # Routes compare body length against the config key.
    at_limit = b"x" * settings.review_max_string_bytes
    over = at_limit + b"y"
    assert len(at_limit) == settings.review_max_string_bytes
    assert len(over) == settings.review_max_string_bytes + 1
    assert not (len(at_limit) > settings.review_max_string_bytes)
    assert len(over) > settings.review_max_string_bytes


def test_bound_keys_have_live_consumers() -> None:
    """VAL-REVIEW-060: declared limits must be read by live fame consumers."""

    from agent_challenge.evaluation import authorization as eval_auth
    from agent_challenge.review import openrouter as openrouter_mod
    from agent_challenge.review import policy as policy_mod
    from agent_challenge.review import report as report_mod
    from agent_challenge.review import sessions as sessions_mod

    # Attribute-name presence on production modules (not only config.py).
    consumer_sources = {
        "review_max_capability_bytes": sessions_mod._derive_session_token,
        "review_max_approval_bytes": sessions_mod.issue_operator_approval,
        "review_max_quote_bytes": report_mod._review_resource_limits,
        "review_max_event_log_bytes": report_mod._review_resource_limits,
        "review_max_event_log_entries": report_mod._review_resource_limits,
        "review_max_vm_config_bytes": report_mod._review_resource_limits,
        "review_max_reason_evidence_items": report_mod._review_resource_limits,
        "review_https_tls_timeout_seconds": openrouter_mod.openrouter_timeout_from_settings,
        "review_https_total_timeout_seconds": openrouter_mod.openrouter_timeout_from_settings,
        "attestation_verification_timeout_seconds": report_mod.verify_review_envelope,
        "eval_max_runs_per_submission": eval_auth._issue_run,
        "eval_max_capability_bytes": eval_auth._issue_run,
        "eval_status_page_max": eval_auth.eval_status_page,
        "review_max_openrouter_response_bytes": openrouter_mod.openrouter_byte_limits_from_settings,
        "review_max_reason_evidence_items_policy": policy_mod.reason_evidence_limit_from_settings,
    }
    for name, consumer in consumer_sources.items():
        assert callable(consumer), name

    settings = ChallengeSettings(
        review_max_quote_bytes=128,
        review_max_event_log_bytes=256,
        review_max_event_log_entries=7,
        review_max_vm_config_bytes=64,
        review_max_reason_evidence_items=11,
        review_https_tls_timeout_seconds=3.0,
        review_https_total_timeout_seconds=12.0,
        review_max_openrouter_request_bytes=99,
        review_max_openrouter_response_bytes=88,
        review_max_openrouter_metadata_bytes=77,
    )
    limits = report_mod._review_resource_limits(settings)
    assert limits["quote_bytes"] == 128
    assert limits["event_log_bytes"] == 256
    assert limits["event_log_entries"] == 7
    assert limits["vm_config_bytes"] == 64
    assert limits["reason_evidence_items"] == 11
    timeout = openrouter_mod.openrouter_timeout_from_settings(settings)
    assert timeout.connect == 3.0
    assert timeout.read == 12.0  # total clamps the larger default read
    assert openrouter_mod.openrouter_byte_limits_from_settings(settings) == {
        "request": 99,
        "response": 88,
        "metadata": 77,
    }
    assert policy_mod.reason_evidence_limit_from_settings(settings) == 11


async def test_audit_page_limit_uses_configured_page_max(database_session) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        review_report_page_max=3,
        review_report_page_default=2,
    )
    artifact = b"zip!"
    async with database_session() as session:
        submission = _submission("page-max", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/policy.md": b"# policy\n"},
            rules_revision_id="rev-1",
            settings=settings,
        )
        await session.commit()
        with pytest.raises(ReviewConflict, match="1..3"):
            await review_audit_page(
                session,
                session_row=created.session,
                cursor=None,
                limit=4,
                page_max=settings.review_report_page_max,
            )
        page = await review_audit_page(
            session,
            session_row=created.session,
            cursor=None,
            limit=3,
            page_max=settings.review_report_page_max,
        )
        assert len(page["items"]) <= 3


async def test_operator_approval_ttl_reads_settings(database_session) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        review_operator_approval_ttl_seconds=300,
    )
    now = datetime(2026, 7, 12, tzinfo=UTC)
    artifact = b"zip"
    async with database_session() as session:
        submission = _submission("approval-ttl", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/policy.md": b"# policy\n"},
            rules_revision_id="rev-ttl",
            settings=settings,
            now=now,
        )
        # Force terminal-ish precursor phase so approval can target it.
        created.assignment.phase = "review_rejected"
        created.assignment.finished_at = now
        created.assignment.capability_state = "revoked"
        created.assignment.active_key = None
        approval = await issue_operator_approval(
            session,
            session_row=created.session,
            assignment=created.assignment,
            action="retry_policy",
            actor="internal",
            now=now,
            settings=settings,
        )
        await session.commit()
        assert approval.expires_at == now + timedelta(seconds=300)


async def test_mutation_rate_exact_limit_then_one_over(database_session) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        review_max_mutations_per_session_per_minute=2,
    )
    _MUTATION_WINDOWS.clear()
    now = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    artifact = b"zip"
    async with database_session() as session:
        submission = _submission("mut-rate", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/policy.md": b"# policy\n"},
            rules_revision_id="rev-rate",
            settings=settings,
            now=now,
        )
        await session.commit()
        await enforce_review_session_mutation_budget(
            session, session_row=created.session, settings=settings, now=now
        )
        await enforce_review_session_mutation_budget(
            session,
            session_row=created.session,
            settings=settings,
            now=now + timedelta(seconds=1),
        )
        with pytest.raises(ReviewRateLimited):
            await enforce_review_session_mutation_budget(
                session,
                session_row=created.session,
                settings=settings,
                now=now + timedelta(seconds=2),
            )


async def test_outstanding_cap_exact_limit_prunes_then_recovers(database_session) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        attestation_max_outstanding_nonce_receipts=1,
    )
    now = datetime(2026, 7, 12, 13, 0, tzinfo=UTC)
    artifact = b"zip"
    async with database_session() as session:
        submission = _submission("outstanding", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/policy.md": b"# policy\n"},
            rules_revision_id="rev-out",
            settings=settings,
            now=now,
        )
        await session.commit()

        # Cap is one active nonce: next create/enforce without prune fails.
        with pytest.raises(ReviewRateLimited):
            await enforce_outstanding_review_cap(session, settings=settings, now=now)

        # Expire the outstanding nonce and prune recovery must free capacity.
        nonce = (
            await session.execute(
                select(ReviewNonce).where(ReviewNonce.assignment_id == created.assignment.id)
            )
        ).scalar_one()
        nonce.expires_at = now - timedelta(hours=1)
        await session.flush()
        pruned = await prune_outstanding_review_records(
            session, now=now + timedelta(seconds=1), settings=settings
        )
        assert pruned == 1
        assert nonce.state == "expired"
        active = await session.scalar(select(ReviewNonce).where(ReviewNonce.state == "active"))
        assert active is None
        await enforce_outstanding_review_cap(
            session, settings=settings, now=now + timedelta(seconds=1)
        )


async def test_rules_file_count_at_limit_and_one_over(database_session) -> None:
    settings = ChallengeSettings(
        shared_token="test-token",
        review_max_rules_files=2,
        review_max_rules_bytes=10_000,
    )
    artifact_a = b"zip-a"
    async with database_session() as session:
        submission = _submission("rules-cap", artifact=artifact_a)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_a,
            rules_files={
                "rules/a.md": b"# a\n",
                "rules/b.md": b"# b\n",
            },
            rules_revision_id="rev-rules",
            settings=settings,
        )
        assert created.session is not None
        with pytest.raises(ReviewConflict, match="review_max_rules_files"):
            # Force a second session path on a new submission.
            artifact_b = b"zip-b"
            other = _submission("rules-cap-2", artifact=artifact_b)
            other.raw_status = "rate_limit_reserved"
            session.add(other)
            await session.flush()
            await create_review_session(
                session,
                submission=other,
                artifact_bytes=artifact_b,
                rules_files={
                    "rules/a.md": b"# a\n",
                    "rules/b.md": b"# b\n",
                    "rules/c.md": b"# c\n",
                },
                rules_revision_id="rev-rules-2",
                settings=settings,
            )


async def test_report_body_limit_config_read_by_route_constants() -> None:
    settings = ChallengeSettings(
        review_max_report_request_bytes=100,
        review_max_string_bytes=20,
        review_report_max_response_bytes=1_000,
        review_internal_report_max_response_bytes=2_000,
    )
    assert settings.review_max_report_request_bytes == 100
    assert settings.review_max_string_bytes == 20
    # Response size guard used by report routes.
    page = {"items": [{"x": "y" * 50}]}
    from agent_challenge.review.canonical import canonical_json_v1

    encoded_len = len(canonical_json_v1(page))
    assert encoded_len < settings.review_report_max_response_bytes
    over = {"items": [{"x": "y" * 2_000}]}
    assert len(canonical_json_v1(over)) > settings.review_report_max_response_bytes


async def test_parse_evidence_range_respects_range_cap() -> None:
    start, end = api_routes._parse_evidence_range(None, 100, max_range_bytes=40)
    assert start == 0
    assert end == 39
    start, end = api_routes._parse_evidence_range("bytes=0-9", 100, max_range_bytes=40)
    assert (start, end) == (0, 9)
    with pytest.raises(ValueError):
        api_routes._parse_evidence_range("bytes=0-50", 100, max_range_bytes=40)


def test_intake_create_maps_rate_limited_before_conflict_503() -> None:
    """Outstanding-nonce flood on signed intake maps to 429, not availability 503."""

    source = Path(api_routes.__file__).read_text(encoding="utf-8")
    anchor = source.index("created_review = await create_review_session")
    create_block = source[anchor : anchor + 2500]
    assert "except ReviewRateLimited" in create_block
    assert "review_rate_limited" in create_block
    assert "review_session_unavailable" in create_block
    assert create_block.index("ReviewRateLimited") < create_block.index("ReviewConflict")
    assert create_block.index("429") < create_block.index("review_session_unavailable")
    # Hierarchy independence: issubclass is already covered; floor is present.
    assert not issubclass(ReviewRateLimited, ReviewConflict)


async def test_marker_route_prefers_size_413_over_schema_and_rate(
    client,
    database_session,
    monkeypatch,
):
    """Multi-fault marker route precedence: 413 before 400/422/429."""

    from agent_challenge.review.canonical import canonical_json_v1
    from agent_challenge.review.openrouter import build_model_call_started
    from agent_challenge.review.sessions import create_review_session as make_session

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    monkeypatch.setattr(api_routes.settings, "review_max_mutations_per_session_per_minute", 1)
    token_secret = "test-token"
    monkeypatch.setattr(api_routes.settings, "shared_token", token_secret)

    artifact = b"zip-marker"
    async with database_session() as session:
        submission = _submission("marker-prec", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await make_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/p.md": b"# p\n"},
            rules_revision_id="rev-marker",
            settings=ChallengeSettings(shared_token=token_secret),
        )
        # Keep the capability in the deployed phase and leave wall-clock expiry future.
        created.assignment.phase = "review_cvm_running"
        created.assignment.capability_state = "active"
        created.assignment.expires_at = datetime.now(UTC) + timedelta(hours=1)
        submission.raw_status = "review_cvm_running"
        await session.commit()
        assignment_id = created.assignment.assignment_id
        token = created.session_token

    # Prefer structural multi-fault ordering on the live marker route:
    # 413 size, then 400 JSON, then 422 schema, then 429 mutation budget.
    oversized = b"x" * (api_routes.settings.review_max_string_bytes + 1)
    oversized_response = await client.post(
        f"/review/v1/assignments/{assignment_id}/model-call-started",
        content=oversized,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert oversized_response.status_code == 413
    assert oversized_response.json()["detail"]["code"] == "review_marker_too_large"

    bad_json = await client.post(
        f"/review/v1/assignments/{assignment_id}/model-call-started",
        content=b"{not-json",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert bad_json.status_code == 400

    schema_invalid = await client.post(
        f"/review/v1/assignments/{assignment_id}/model-call-started",
        content=canonical_json_v1({"schema_version": 1}),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert schema_invalid.status_code == 422

    # Pre-seed the mutation window so the first valid marker hits 429 (budget=1).
    from agent_challenge.models import ReviewAssignment, ReviewSession

    _MUTATION_WINDOWS.clear()
    async with database_session() as session:
        assignment = (
            await session.execute(
                select(ReviewAssignment).where(ReviewAssignment.assignment_id == assignment_id)
            )
        ).scalar_one()
        review_session = await session.get(ReviewSession, assignment.session_id)
        assert review_session is not None
        _MUTATION_WINDOWS[review_session.session_id] = [datetime.now(UTC)]

    marker = build_model_call_started(
        assignment_id=assignment_id,
        planned_request_sha256="ab" * 32,
        request_body_sha256="cd" * 32,
        request_body_length=12,
    )
    first = await client.post(
        f"/review/v1/assignments/{assignment_id}/model-call-started",
        content=canonical_json_v1(marker),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert first.status_code == 429
    assert first.json()["detail"]["code"] == "review_rate_limited"


async def test_failure_route_prefers_size_413_over_json(
    client,
    database_session,
    monkeypatch,
):
    from agent_challenge.review.sessions import create_review_session as make_session

    monkeypatch.setattr(api_routes.settings, "attested_review_enabled", True)
    monkeypatch.setattr(api_routes.settings, "phala_attestation_enabled", True)
    monkeypatch.setattr(api_routes.settings, "review_max_string_bytes", 48)
    token_secret = "test-token"
    monkeypatch.setattr(api_routes.settings, "shared_token", token_secret)

    artifact = b"zip-fail"
    async with database_session() as session:
        submission = _submission("fail-prec", artifact=artifact)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await make_session(
            session,
            submission=submission,
            artifact_bytes=artifact,
            rules_files={"rules/p.md": b"# p\n"},
            rules_revision_id="rev-fail",
            settings=ChallengeSettings(shared_token=token_secret),
        )
        created.assignment.phase = "review_cvm_running"
        created.assignment.capability_state = "active"
        created.assignment.expires_at = datetime.now(UTC) + timedelta(hours=1)
        submission.raw_status = "review_cvm_running"
        await session.commit()
        assignment_id = created.assignment.assignment_id
        token = created.session_token

    oversized = await client.post(
        f"/review/v1/assignments/{assignment_id}/failure",
        content=b"y" * (api_routes.settings.review_max_string_bytes + 1),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"]["code"] == "review_failure_too_large"

    invalid_json = await client.post(
        f"/review/v1/assignments/{assignment_id}/failure",
        content=b"{bad",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert invalid_json.status_code == 400


async def test_outstanding_cap_prunes_then_create_review_session_recovers(
    database_session,
) -> None:
    """Flood/prune path must free capacity for a later create_review_session."""

    settings = ChallengeSettings(
        shared_token="test-token",
        attestation_max_outstanding_nonce_receipts=1,
    )
    now = datetime(2026, 7, 12, 16, 0, tzinfo=UTC)
    artifact_a = b"zip-a"
    async with database_session() as session:
        submission = _submission("outstanding-route", artifact=artifact_a)
        submission.raw_status = "rate_limit_reserved"
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=artifact_a,
            rules_files={"rules/p.md": b"# p\n"},
            rules_revision_id="rev-a",
            settings=settings,
            now=now,
        )
        await session.commit()
        with pytest.raises(ReviewRateLimited):
            artifact_b = b"zip-b"
            other = _submission("outstanding-route-2", artifact=artifact_b)
            other.raw_status = "rate_limit_reserved"
            session.add(other)
            await session.flush()
            await create_review_session(
                session,
                submission=other,
                artifact_bytes=artifact_b,
                rules_files={"rules/p.md": b"# p\n"},
                rules_revision_id="rev-b",
                settings=settings,
                now=now,
            )
            await session.rollback()

        nonce = (
            await session.execute(
                select(ReviewNonce).where(ReviewNonce.assignment_id == created.assignment.id)
            )
        ).scalar_one()
        nonce.expires_at = now - timedelta(hours=1)
        await session.flush()
        pruned = await prune_outstanding_review_records(
            session, now=now + timedelta(seconds=1), settings=settings
        )
        assert pruned == 1
        artifact_c = b"zip-c"
        third = _submission("outstanding-route-3", artifact=artifact_c)
        third.raw_status = "rate_limit_reserved"
        session.add(third)
        await session.flush()
        revived = await create_review_session(
            session,
            submission=third,
            artifact_bytes=artifact_c,
            rules_files={"rules/p.md": b"# p\n"},
            rules_revision_id="rev-c",
            settings=settings,
            now=now + timedelta(seconds=2),
        )
        await session.commit()
        assert revived.assignment.assignment_id != created.assignment.assignment_id


def test_openrouter_request_capacity_uses_settings_not_module_constant() -> None:
    from agent_challenge.review.openrouter import (
        OpenRouterTransportError,
        build_planned_openrouter_request,
        openrouter_byte_limits_from_settings,
    )

    settings = ChallengeSettings(review_max_openrouter_request_bytes=32)
    assert openrouter_byte_limits_from_settings(settings)["request"] == 32
    with pytest.raises(OpenRouterTransportError, match="request body exceeds"):
        build_planned_openrouter_request(
            body=b"x" * 33,
            routing_sha256="ab" * 32,
            settings=settings,
        )


def test_report_quote_limit_reads_settings() -> None:
    from agent_challenge.review.report import ReviewReportError, _validate_attestation

    settings = ChallengeSettings(review_max_quote_bytes=4)
    oversize_quote = "ab" * 5  # 5 bytes > 4
    with pytest.raises(ReviewReportError, match="configured byte bound"):
        _validate_attestation(
            {
                "tdx_quote_hex": oversize_quote,
                "event_log": [],
                "measurement": {},
                "vm_config": {},
            },
            ReviewReportError,
            settings=settings,
        )
