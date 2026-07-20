from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import (
    AgentSubmission,
    EvaluationJob,
    ReviewAssignment,
    ReviewNonce,
    ReviewOperatorApproval,
    ReviewRulesSnapshot,
    ReviewSession,
)
from agent_challenge.review.canonical import CanonicalJsonError, canonical_json_v1
from agent_challenge.review.compose import generate_review_app_compose, review_app_compose_hash
from agent_challenge.review.rules import RulesSnapshotCaptureError, capture_rules_bundle
from agent_challenge.review.schemas import (
    MAX_RULES_BYTES,
    MAX_RULES_FILES,
    REVIEW_POLICY_PROMPT_BYTES,
    REVIEW_POLICY_PROMPT_VERSION,
    REVIEW_POLICY_TOOL_SCHEMA_BYTES,
    REVIEW_POLICY_TOOL_SCHEMA_VERSION,
    REVIEW_POLICY_VERIFIER_BYTES,
    REVIEW_POLICY_VERIFIER_VERSION,
    RulesSchemaError,
    build_rules_bundle,
    validate_review_assignment,
    validate_rules_bundle,
)
from agent_challenge.review.sessions import (
    ReviewCapabilityError,
    ReviewConflict,
    authenticate_assignment_capability,
    cancel_review_assignment,
    create_review_session,
    deliver_prepare_token,
    issue_operator_approval,
    mark_review_deployed,
    retry_review_assignment,
    review_audit_page,
)
from agent_challenge.sdk.auth import load_internal_token
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.security import SignedRequestAuth


def _submission() -> AgentSubmission:
    zip_bytes = b"review-zip-bytes"
    digest = hashlib.sha256(zip_bytes).hexdigest()
    return AgentSubmission(
        miner_hotkey="review-miner",
        name="review-agent",
        agent_hash=digest,
        artifact_uri="/tmp/review-agent.zip",
        artifact_path="/tmp/review-agent.zip",
        zip_sha256=digest,
        zip_size_bytes=len(zip_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )


def _zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", "class Agent:\n    pass\n")
    return buffer.getvalue()


@pytest.fixture
def signed_review_auth_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="review-miner",
            signature="signature",
            nonce="review-nonce",
            timestamp="2026-07-10T00:00:00+00:00",
            body_sha256="a" * 64,
            canonical_request="POST\n/review\n0\nnonce\nhash",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


def test_canonical_json_v1_normalizes_multilingual_text_and_rejects_malformed_values() -> None:
    value = {
        "z": "Cafe\u0301",
        "a": ["\u041f\u0440\u0438\u0432\u0435\u0442", "\u0645\u0631\u062d\u0628\u0627", "🚀"],
        "control": "\n\t",
    }

    assert canonical_json_v1(value) == (
        b'{"a":["\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82",'
        b'"\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7","\xf0\x9f\x9a\x80"],'
        b'"control":"\\n\\t","z":"Caf\xc3\xa9"}'
    )

    with pytest.raises(CanonicalJsonError):
        canonical_json_v1({"nan": float("nan")})
    with pytest.raises(CanonicalJsonError):
        canonical_json_v1({"surrogate": "\ud800"})
    with pytest.raises(CanonicalJsonError):
        canonical_json_v1({"Cafe\u0301": 1, "Caf\u00e9": 2})


def test_rules_snapshot_is_schema_closed_canonical_and_content_bound() -> None:
    bundle = build_rules_bundle(
        revision_id="rules-v1",
        files={
            ".rules/\u0645\u0631\u062d\u0628\u0627.md": "\u0645\u0631\u062d\u0628\u0627\n".encode(),
            ".rules/cafe\u0301.md": b"accent\n",
        },
    )

    assert [item["path"] for item in bundle["files"]] == [
        ".rules/caf\u00e9.md",
        ".rules/\u0645\u0631\u062d\u0628\u0627.md",
    ]
    assert validate_rules_bundle(bundle) == canonical_json_v1(bundle)

    for invalid in (
        {**bundle, "unknown": True},
        {**bundle, "files": list(reversed(bundle["files"]))},
        {
            **bundle,
            "files": [{**bundle["files"][0], "path": "../escape.md"}, bundle["files"][1]],
        },
        {
            **bundle,
            "files": [{**bundle["files"][0], "content_b64": "not base64"}, bundle["files"][1]],
        },
    ):
        with pytest.raises(RulesSchemaError):
            validate_rules_bundle(invalid)


async def test_review_session_pins_immutable_assignment_rules_and_nonce(database_session) -> None:
    submission = _submission()
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"allow only safe agents\n"},
            rules_revision_id="rules-v1",
            settings=ChallengeSettings(shared_token="review-token"),
            now=datetime(2026, 7, 10, tzinfo=UTC),
        )
        await session.commit()

        assignment = created.assignment
        assignment_object = json.loads(assignment.assignment_bytes)
        validate_review_assignment(assignment_object)
        assert assignment.assignment_digest == assignment_object["assignment_digest"]
        assert assignment.assignment_id != created.session.session_id
        assert assignment.attempt == 1
        assert assignment.phase == "review_queued"
        assert assignment.active_key == created.session.session_id
        assert (
            assignment.session_token_sha256
            == hashlib.sha256(created.session_token.encode("utf-8")).hexdigest()
        )
        assert assignment.review_nonce != created.session_token
        assert len(assignment.review_nonce.encode("ascii")) >= 22
        policy = assignment_object["assignment_core"]["policy"]
        assert policy["prompt_version"] == REVIEW_POLICY_PROMPT_VERSION
        assert policy["prompt_sha256"] == hashlib.sha256(REVIEW_POLICY_PROMPT_BYTES).hexdigest()
        assert policy["tool_schema_version"] == REVIEW_POLICY_TOOL_SCHEMA_VERSION
        assert (
            policy["tool_schema_sha256"]
            == hashlib.sha256(REVIEW_POLICY_TOOL_SCHEMA_BYTES).hexdigest()
        )
        assert policy["verifier_version"] == REVIEW_POLICY_VERIFIER_VERSION
        assert policy["verifier_sha256"] == hashlib.sha256(REVIEW_POLICY_VERIFIER_BYTES).hexdigest()

        session_row = await session.scalar(select(ReviewSession))
        snapshot = await session.scalar(select(ReviewRulesSnapshot))
        nonce = await session.scalar(select(ReviewNonce))

    assert session_row is not None
    assert snapshot is not None
    assert nonce is not None
    assert nonce.session_id == session_row.id
    assert nonce.assignment_id == assignment.id
    assert nonce.state == "active"
    assert snapshot.snapshot_sha256 in assignment.assignment_bytes
    assert "review-token" not in assignment.assignment_bytes


async def test_retry_preserves_snapshot_by_default_and_refresh_consumes_approval(
    database_session,
) -> None:
    submission = _submission()
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"first\n"},
            rules_revision_id="first",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_cancelled"
        created.assignment.active_key = None
        created.assignment.capability_state = "revoked"
        created.assignment.finished_at = now
        await session.commit()

        default_retry = await retry_review_assignment(
            session,
            session_row=created.session,
            expected_assignment_id=created.assignment.assignment_id,
            settings=settings,
            now=now + timedelta(seconds=1),
        )
        default_retry.assignment.phase = "review_cancelled"
        default_retry.assignment.active_key = None
        default_retry.assignment.capability_state = "revoked"
        default_retry.assignment.finished_at = now + timedelta(seconds=1)
        await session.flush()
        approval = await issue_operator_approval(
            session,
            session_row=created.session,
            assignment=default_retry.assignment,
            action="refresh_rules",
            rules_revision_id="second",
            actor="operator-a",
            now=now + timedelta(seconds=1),
        )

        refreshed = await retry_review_assignment(
            session,
            session_row=created.session,
            expected_assignment_id=default_retry.assignment.assignment_id,
            settings=settings,
            approval_id=approval.approval_id,
            refresh_rules_files={".rules/policy.md": b"second\n"},
            now=now + timedelta(seconds=2),
        )
        await session.commit()

        rows = (
            await session.scalars(
                select(ReviewAssignment)
                .where(ReviewAssignment.session_id == created.session.id)
                .order_by(ReviewAssignment.attempt)
            )
        ).all()
        approval_row = await session.scalar(select(ReviewOperatorApproval))

    assert [row.attempt for row in rows] == [1, 2, 3]
    assert rows[0].rules_snapshot_sha256 == rows[1].rules_snapshot_sha256
    assert rows[2].rules_snapshot_sha256 != rows[1].rules_snapshot_sha256
    assert len({row.assignment_id for row in rows}) == 3
    assert len({row.review_nonce for row in rows}) == 3
    assert refreshed.session_token != default_retry.session_token
    assert approval_row is not None and approval_row.used_at is not None


async def test_active_retry_conflicts_and_history_is_stable_paginated(database_session) -> None:
    submission = _submission()
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"first\n"},
            rules_revision_id="first",
            settings=settings,
            now=now,
        )
        await session.commit()

        with pytest.raises(ReviewConflict, match="active"):
            await retry_review_assignment(
                session,
                session_row=created.session,
                expected_assignment_id=created.assignment.assignment_id,
                settings=settings,
                now=now,
            )

        # Freeze clock within assignment TTL so history stays active (lazy expire
        # would otherwise terminalize past-TTL rows when now=None uses wall clock).
        page = await review_audit_page(
            session,
            session_row=created.session,
            cursor=None,
            limit=10,
            now=now,
        )
        nonce_count = await session.scalar(select(func.count(ReviewNonce.id)))

    assert page["total_count"] == 1
    assert page["items"][0]["assignment_id"] == created.assignment.assignment_id
    assert page["items"][0]["phase"] == "review_queued"
    assert nonce_count == 1


async def test_deliver_prepare_token_redelivers_before_deploy_not_after(
    database_session,
) -> None:
    """Active undepoyed assignments redeliver capability; post-deploy is sticky-null.

    Residual (review-v8): dry-run prepare spent the one-time token, so live
    deploy cancelled the still-usable assignment (cli_internal_prepare_cancel_retry)
    and burned attempts before the CVM ever started. Product must redeliver for
    active undepoyed rows and leave cancelled/sticky only for true terminal or
    already-deployed paths.
    """

    submission = _submission()
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"first\n"},
            rules_revision_id="first",
            settings=settings,
            now=now,
        )
        await session.commit()

        first_assignment, first_token = await deliver_prepare_token(
            session,
            session_row=created.session,
            settings=settings,
            now=now + timedelta(seconds=1),
        )
        await session.commit()
        assert first_token is not None
        assert first_assignment.token_delivered_at is not None
        assert first_assignment.phase == "review_queued"
        assert first_assignment.deployed_at is None

        # Dry-import/prepare path: second call must redeliver same capability.
        second_assignment, second_token = await deliver_prepare_token(
            session,
            session_row=created.session,
            settings=settings,
            now=now + timedelta(seconds=2),
        )
        await session.commit()
        assert second_assignment.assignment_id == first_assignment.assignment_id
        assert second_token == first_token
        assert second_token is not None

        # After deployment receipt, redelivery must stop (token lives in CVM only).
        assignment_body = json.loads(second_assignment.assignment_bytes)
        review_app = assignment_body["assignment_core"]["review_app"]
        receipt = {
            "schema_version": 1,
            "assignment_id": second_assignment.assignment_id,
            "cvm_id": "cvm-redeliver-1",
            "phala_create_receipt": {
                "request_id": "req-redeliver-1",
                "app_id": review_app["app_identity"],
                "cvm_id": "cvm-redeliver-1",
                "receipt_sha256": "a" * 64,
                "created_at_ms": 1_000,
            },
            "compose_identity": {
                "image_ref": review_app["image_ref"],
                "compose_hash": review_app["compose_hash"],
                "app_kms_public_key_sha256": review_app["kms_public_key_sha256"],
            },
        }
        deployed = await mark_review_deployed(
            session,
            session_row=created.session,
            expected_assignment_id=second_assignment.assignment_id,
            deployed_receipt=receipt,
            now=now + timedelta(seconds=3),
            settings=settings,
        )
        await session.commit()
        assert deployed.phase == "review_cvm_running"
        assert deployed.deployed_at is not None

        third_assignment, third_token = await deliver_prepare_token(
            session,
            session_row=created.session,
            settings=settings,
            now=now + timedelta(seconds=4),
        )
        await session.commit()
        assert third_assignment.assignment_id == first_assignment.assignment_id
        assert third_token is None


async def test_assignment_expiry_revokes_capability_and_preserves_terminal_history(
    database_session,
) -> None:
    submission = _submission()
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"first\n"},
            rules_revision_id="first",
            settings=settings,
            now=now,
        )
        await session.commit()

        assignment, token = await deliver_prepare_token(
            session,
            session_row=created.session,
            settings=settings,
            now=now + timedelta(seconds=1800),
        )
        await session.commit()
        nonce = await session.scalar(select(ReviewNonce))

        with pytest.raises(ReviewConflict, match="terminal"):
            await cancel_review_assignment(
                session,
                session_row=created.session,
                expected_assignment_id=assignment.assignment_id,
                now=now + timedelta(seconds=1801),
            )

    assert token is None
    assert assignment.phase == "review_expired"
    assert assignment.capability_state == "revoked"
    assert nonce is not None
    assert nonce.state == "expired"


async def test_review_audit_page_lazy_expires_active_past_ttl(database_session) -> None:
    """History reads must terminalize sticky active assignments past expires_at."""

    submission = _submission()
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"first\n"},
            rules_revision_id="first",
            settings=settings,
            now=now,
        )
        await session.commit()
        # Still active before TTL ends when history is queried normally.
        page_active = await review_audit_page(
            session,
            session_row=created.session,
            cursor=None,
            limit=10,
            now=now + timedelta(seconds=60),
        )
        assert page_active["items"][0]["phase"] == "review_queued"
        # Past expires_at (default TTL 1800s): lazy-expire on history read.
        page_expired = await review_audit_page(
            session,
            session_row=created.session,
            cursor=None,
            limit=10,
            now=now + timedelta(seconds=1801),
        )
        await session.commit()
        assignment = await session.scalar(
            select(ReviewAssignment).where(
                ReviewAssignment.assignment_id == created.assignment.assignment_id
            )
        )

    assert page_expired["items"][0]["phase"] == "review_expired"
    assert page_expired["items"][0]["reason_code"] == "expired"
    assert assignment is not None
    assert assignment.phase == "review_expired"
    assert assignment.capability_state == "revoked"
    assert assignment.reason_code == "expired"


async def test_file_backed_shared_token_derives_same_capability_as_internal_loader(
    database_session, tmp_path: Path
) -> None:
    secret_path = tmp_path / "challenge_token"
    secret_path.write_text("file-backed-review-secret\n", encoding="utf-8")
    settings = ChallengeSettings(shared_token=None, shared_token_file=str(secret_path))
    assert load_internal_token(settings) == "file-backed-review-secret"
    submission = _submission()
    now = datetime(2026, 7, 10, tzinfo=UTC)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"allow\n"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        await session.commit()
        mac = hmac.new(
            b"file-backed-review-secret",
            b"agent-challenge:review-session:v1:"
            + created.assignment.assignment_id.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        expected = f"{created.assignment.assignment_id}.{mac}"
        assert created.session_token == expected
        assert (
            created.assignment.session_token_sha256
            == hashlib.sha256(expected.encode("utf-8")).hexdigest()
        )
        authed = await authenticate_assignment_capability(
            session,
            assignment_id=created.assignment.assignment_id,
            token=created.session_token,
            now=now + timedelta(seconds=1),
        )
        assert authed.id == created.assignment.id


async def test_capability_auth_checks_expiry_revocation_and_session_binding(
    database_session,
) -> None:
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    submission = _submission()
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"allow\n"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        await session.commit()
        token = created.session_token
        assignment_id = created.assignment.assignment_id

        active = await authenticate_assignment_capability(
            session,
            assignment_id=assignment_id,
            token=token,
            now=now + timedelta(seconds=1),
        )
        assert active.capability_state == "active"
        assert active.assignment_id == assignment_id
        review_session = await session.get(ReviewSession, active.session_id)
        assert review_session is not None
        assert review_session.submission_id == submission.id

        with pytest.raises(ReviewCapabilityError, match="expired|revoked"):
            await authenticate_assignment_capability(
                session,
                assignment_id=assignment_id,
                token=token,
                now=now + timedelta(seconds=1800),
            )
        assignment = await session.scalar(
            select(ReviewAssignment).where(ReviewAssignment.assignment_id == assignment_id)
        )
        assert assignment is not None
        assert assignment.phase == "review_expired"
        assert assignment.capability_state == "revoked"

        with pytest.raises(ReviewCapabilityError):
            await authenticate_assignment_capability(
                session,
                assignment_id=assignment_id,
                token=token,
                now=now + timedelta(seconds=1801),
            )

        with pytest.raises(ReviewCapabilityError, match="invalid"):
            await authenticate_assignment_capability(
                session,
                assignment_id=assignment_id,
                token="wrong-token",
                now=now,
            )


async def test_receipted_assignment_capability_may_only_resume_report_after_expiry(
    database_session,
) -> None:
    settings = ChallengeSettings(shared_token="review-token")
    now = datetime(2026, 7, 10, tzinfo=UTC)
    submission = _submission()
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=b"review-zip-bytes",
            rules_files={".rules/policy.md": b"allow\n"},
            rules_revision_id="rules-v1",
            settings=settings,
            now=now,
        )
        created.assignment.phase = "review_verifying"
        created.assignment.review_report_envelope_json = '{"schema_version":1}'
        created.assignment.review_report_sha256 = "aa" * 32
        created.assignment.review_report_received_at = now + timedelta(seconds=10)
        await session.commit()

        # Unreceipted paths (artifact/rules style, no report-replay flag) stay denied.
        with pytest.raises(ReviewCapabilityError, match="expired|revoked"):
            await authenticate_assignment_capability(
                session,
                assignment_id=created.assignment.assignment_id,
                token=created.session_token,
                now=now + timedelta(seconds=1800),
            )

        # Already-receipted identical report soft-auth may proceed after wall expiry.
        resumed = await authenticate_assignment_capability(
            session,
            assignment_id=created.assignment.assignment_id,
            token=created.session_token,
            now=now + timedelta(seconds=1800),
            allow_report_replay=True,
        )
        assert resumed.review_report_envelope_json is not None
        assert resumed.assignment_id == created.assignment.assignment_id


def test_rules_bundle_enforces_file_count_and_aggregate_byte_bounds() -> None:
    ok_files = {f".rules/{index:03d}.md": b"x" for index in range(MAX_RULES_FILES)}
    assert len(ok_files) == MAX_RULES_FILES
    validate_rules_bundle(build_rules_bundle(revision_id="bound-ok", files=ok_files))

    too_many = {f".rules/{index:03d}.md": b"x" for index in range(MAX_RULES_FILES + 1)}
    with pytest.raises(RulesSchemaError, match="128"):
        build_rules_bundle(revision_id="too-many", files=too_many)

    oversize = {".rules/big.md": b"a" * (MAX_RULES_BYTES + 1)}
    with pytest.raises(RulesSchemaError, match="aggregate|1 MiB|1048576|bytes"):
        build_rules_bundle(revision_id="too-big", files=oversize)

    exact = {".rules/exact.md": b"b" * MAX_RULES_BYTES}
    validate_rules_bundle(build_rules_bundle(revision_id="exact", files=exact))


def test_capture_rules_bundle_is_atomic_old_or_new_and_bounds_aggregate(tmp_path: Path) -> None:
    root = tmp_path
    rules_dir = root / ".rules"
    rules_dir.mkdir()
    (rules_dir / "a.md").write_bytes(b"first-revision-aaaa\n")
    (rules_dir / "b.md").write_bytes(b"first-revision-bbbb\n")

    first = capture_rules_bundle(root)
    second = capture_rules_bundle(root)
    assert second == first
    assert validate_rules_bundle(first)

    # Same-size content rewrite must not yield a mixed or partial bundle: capture
    # either finishes on the complete old revision or retries onto the complete new.
    (rules_dir / "a.md").write_bytes(b"second-revision-aaa\n")
    (rules_dir / "b.md").write_bytes(b"second-revision-bbb\n")
    captured_after = capture_rules_bundle(root)
    assert captured_after != first
    paths = {
        item["path"]: base64.b64decode(item["content_b64"]) for item in captured_after["files"]
    }
    assert paths[".rules/a.md"] == b"second-revision-aaa\n"
    assert paths[".rules/b.md"] == b"second-revision-bbb\n"

    oversize_dir = root / "oversize"
    (oversize_dir / ".rules").mkdir(parents=True)
    (oversize_dir / ".rules" / "huge.md").write_bytes(b"z" * (MAX_RULES_BYTES + 1))
    with pytest.raises(RulesSnapshotCaptureError, match="aggregate|bound|byte"):
        capture_rules_bundle(oversize_dir)

    many_dir = root / "many"
    (many_dir / ".rules").mkdir(parents=True)
    for index in range(MAX_RULES_FILES + 1):
        (many_dir / ".rules" / f"{index:03d}.md").write_bytes(b"x")
    with pytest.raises(RulesSnapshotCaptureError, match="128|file"):
        capture_rules_bundle(many_dir)


async def test_signed_intake_creates_no_spend_review_session_and_one_time_capability(
    client,
    database_session,
    monkeypatch,
    signed_review_auth_override,
    tmp_path,
) -> None:
    monkeypatch.setattr("agent_challenge.api.routes.settings.attested_review_enabled", True)
    review_image = "docker.io/example/agent-challenge-review@sha256:" + ("a" * 64)
    review_compose = generate_review_app_compose(
        review_image=review_image,
        app_identity="agent-challenge-review-v1",
    )
    review_measurement = {
        "mrtd": "01" * 48,
        "rtmr0": "02" * 48,
        "rtmr1": "03" * 48,
        "rtmr2": "04" * 48,
        "os_image_hash": "05" * 32,
        "key_provider": "phala",
        "vm_shape": "tdx.small",
    }
    review_allowlist = {
        "mrtd": review_measurement["mrtd"],
        "rtmr0": review_measurement["rtmr0"],
        "rtmr1": review_measurement["rtmr1"],
        "rtmr2": review_measurement["rtmr2"],
        "compose_hash": review_app_compose_hash(review_compose),
        "os_image_hash": review_measurement["os_image_hash"],
    }
    monkeypatch.setattr("agent_challenge.api.routes.settings.review_app_image_ref", review_image)
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.review_app_compose_hash",
        review_allowlist["compose_hash"],
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.review_app_kms_public_key_hex",
        "f" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.review_app_measurement",
        review_measurement,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.review_app_measurement_allowlist",
        (review_allowlist,),
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.eval_app_identity",
        "agent-challenge-eval-v1",
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.eval_app_image_ref",
        "docker.io/example/agent-challenge-canonical@sha256:" + ("b" * 64),
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.eval_app_compose_hash",
        "c" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.eval_app_kms_public_key_hex",
        "e" * 64,
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.eval_app_measurement_allowlist",
        (
            {
                "mrtd": "06" * 48,
                "rtmr0": "07" * 48,
                "rtmr1": "08" * 48,
                "rtmr2": "09" * 48,
                "compose_hash": "0a" * 32,
                "os_image_hash": "0b" * 32,
            },
        ),
    )
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.review_rules_root", str(tmp_path))
    rules_dir = tmp_path / ".rules"
    rules_dir.mkdir()
    (rules_dir / "policy.md").write_text("do not spend\n", encoding="utf-8")
    payload = {
        "name": "review-agent",
        "artifact_zip_base64": base64.b64encode(_zip_bytes()).decode("ascii"),
    }

    submitted = await client.post("/submissions", json=payload)

    assert submitted.status_code == 201
    submission_id = submitted.json()["submission_id"]
    prepared = await client.post(f"/submissions/{submission_id}/review/prepare")
    repeated = await client.post(f"/submissions/{submission_id}/review/prepare")

    assert prepared.status_code == 200
    assert repeated.status_code == 200
    first = prepared.json()
    second = repeated.json()
    assert first["session_id"]
    assert first["assignment_id"]
    assert first["review_session_token"]
    assert second["session_id"] == first["session_id"]
    assert second["assignment"] == first["assignment"]
    # Pre-deploy redelivery: dry-run / second prepare keeps the same capability
    # so live deploy does not need cancel+retry burn of active attempts.
    assert second["review_session_token"] == first["review_session_token"]
    validate_review_assignment(first["assignment"])

    review_app = first["assignment"]["assignment_core"]["review_app"]
    assert review_app["measurement_allowlist_sha256"]
    assert review_app["measurement_allowlist"]
    deployed = await client.post(
        f"/submissions/{submission_id}/review/deployed",
        json={
            "schema_version": 1,
            "assignment_id": first["assignment_id"],
            "cvm_id": "cvm-review-1",
            "phala_create_receipt": {
                "request_id": "req-review-1",
                "app_id": review_app["app_identity"],
                "cvm_id": "cvm-review-1",
                "receipt_sha256": "7" * 64,
                "created_at_ms": 1_000,
            },
            "compose_identity": {
                "image_ref": review_app["image_ref"],
                "compose_hash": review_app["compose_hash"],
                "app_kms_public_key_sha256": review_app["kms_public_key_sha256"],
            },
        },
    )
    assert deployed.status_code == 200, deployed.text
    assert deployed.json()["phase"] == "review_cvm_running"

    # After deployment acknowledgement, prepare goes sticky-null (capability in CVM).
    post_deploy = await client.post(f"/submissions/{submission_id}/review/prepare")
    assert post_deploy.status_code == 200
    assert post_deploy.json()["review_session_token"] is None
    assert post_deploy.json()["assignment_id"] == first["assignment_id"]

    headers = {"Authorization": f"Bearer {first['review_session_token']}"}
    artifact = await client.get(
        f"/review/v1/assignments/{first['assignment_id']}/artifact",
        headers=headers,
    )
    rules = await client.get(
        f"/review/v1/assignments/{first['assignment_id']}/rules",
        headers=headers,
    )
    assert artifact.status_code == 200
    assert (
        hashlib.sha256(artifact.content).hexdigest()
        == first["assignment"]["assignment_core"]["artifact"]["zip_sha256"]
    )
    assert rules.status_code == 200
    assert json.loads(rules.content)["revision_id"]

    artifact_sha256 = first["assignment"]["assignment_core"]["artifact"]["zip_sha256"]
    stored_zip = tmp_path / "agents" / artifact_sha256
    artifact_path = stored_zip / "agent.zip"
    original_bytes = artifact_path.read_bytes()
    artifact_path.write_bytes(original_bytes + b"mutated")
    changed = await client.get(
        f"/review/v1/assignments/{first['assignment_id']}/artifact",
        headers=headers,
    )
    assert changed.status_code == 409
    artifact_path.write_bytes(original_bytes)

    cancelled = await client.post(
        f"/submissions/{submission_id}/review/cancel",
        json={"expected_assignment_id": first["assignment_id"]},
    )
    assert cancelled.status_code == 200
    denied = await client.get(
        f"/review/v1/assignments/{first['assignment_id']}/artifact",
        headers=headers,
    )
    assert denied.status_code == 401

    async with database_session() as session:
        review_count = await session.scalar(select(func.count(ReviewSession.id)))
        assignment_count = await session.scalar(select(func.count(ReviewAssignment.id)))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert review_count == 1
    assert assignment_count == 1
    assert job_count == 0
