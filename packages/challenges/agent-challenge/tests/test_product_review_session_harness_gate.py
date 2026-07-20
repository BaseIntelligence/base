"""create_review_session enforces ZIP+script harness and refuses parity path."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from agent_challenge.models import AgentSubmission
from agent_challenge.review.harness_entry import (
    REFUSE_PARITY_HARNESS,
    REFUSE_UNMEASURED_HOST,
)
from agent_challenge.review.sessions import ReviewConflict, create_review_session
from agent_challenge.sdk.config import ChallengeSettings


def _settings() -> ChallengeSettings:
    return ChallengeSettings(shared_token="review-token")


def _zip() -> bytes:
    return b"PK\x03\x04product-review-session-zip-v1"


def _rules() -> dict[str, bytes]:
    return {
        ".rules/acceptance.md": b"# acceptance\nok\n",
        ".rules/anti-cheat.md": b"# anti-cheat\nok\n",
    }


def _submission(zip_bytes: bytes) -> AgentSubmission:
    digest = hashlib.sha256(zip_bytes).hexdigest()
    return AgentSubmission(
        miner_hotkey="review-harness-miner",
        name="review-agent",
        agent_hash=digest,
        artifact_uri="/tmp/review-agent.zip",
        artifact_path="/tmp/review-agent.zip",
        zip_sha256=digest,
        zip_size_bytes=len(zip_bytes),
        raw_status="review_queued",
        effective_status="queued",
    )


async def test_parity_harness_refused_at_create_review_session(database_session) -> None:
    zip_bytes = _zip()
    submission = _submission(zip_bytes)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        with pytest.raises(ReviewConflict) as exc:
            await create_review_session(
                session,
                submission=submission,
                artifact_bytes=zip_bytes,
                rules_files=_rules(),
                rules_revision_id="rev-parity-refuse",
                settings=_settings(),
                now=datetime(2026, 7, 10, tzinfo=UTC),
                entry_script="tools/agent_parity_harness.py",
            )
        assert REFUSE_PARITY_HARNESS in str(exc.value)


async def test_unmeasured_offline_ast_refused(database_session) -> None:
    zip_bytes = _zip()
    submission = _submission(zip_bytes)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        with pytest.raises(ReviewConflict) as exc:
            await create_review_session(
                session,
                submission=submission,
                artifact_bytes=zip_bytes,
                rules_files=_rules(),
                rules_revision_id="rev-offline-refuse",
                settings=_settings(),
                now=datetime(2026, 7, 10, tzinfo=UTC),
                harness_kind="offline_ast",
            )
        assert REFUSE_UNMEASURED_HOST in str(exc.value)


async def test_product_selfdeploy_entry_admits(database_session) -> None:
    zip_bytes = _zip()
    submission = _submission(zip_bytes)
    async with database_session() as session:
        session.add(submission)
        await session.flush()
        created = await create_review_session(
            session,
            submission=submission,
            artifact_bytes=zip_bytes,
            rules_files=_rules(),
            rules_revision_id="rev-product-ok",
            settings=_settings(),
            now=datetime(2026, 7, 10, tzinfo=UTC),
            entry_script_identity="python -m agent_challenge.selfdeploy",
        )
        await session.commit()
        assert created.session is not None
        assert created.assignment is not None
        assert created.session_token
        assert created.session.artifact_sha256 == hashlib.sha256(zip_bytes).hexdigest()
        # create_review_session retains harness identity materials (not just gates).
        assert created.harness_identity is not None
        assert created.harness_identity.get("harness_kind") == "measured_review_cvm_script_zip"
        assert created.harness_identity.get("zip_sha256") == hashlib.sha256(zip_bytes).hexdigest()
        assert created.session.harness_identity_json
        assert created.session.harness_identity_sha256
        assert created.session.submission_received_at_ms is not None
