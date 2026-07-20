from __future__ import annotations

import pytest
from sqlalchemy import func, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent_challenge.db import Base
from agent_challenge.models import AgentSubmission, SubmissionFamily
from agent_challenge.submissions.versioning import normalize_submission_name, version_label


@pytest.fixture
async def model_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()


def _submission(
    *,
    family_id: int | None = None,
    version_number: int | None = None,
    agent_hash: str = "agent-hash",
    canonical_artifact_hash: str | None = None,
    is_latest_version: bool = False,
) -> AgentSubmission:
    return AgentSubmission(
        miner_hotkey="miner-hotkey",
        name="Example Agent",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        submission_family_id=family_id,
        version_number=version_number,
        version_label=version_label(version_number) if version_number is not None else None,
        canonical_artifact_hash=canonical_artifact_hash,
        is_latest_version=is_latest_version,
        zip_sha256=agent_hash.rjust(64, "0")[-64:],
    )


async def test_create_all_includes_submission_families():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        table_names = await connection.run_sync(
            lambda sync_connection: set(inspect(sync_connection).get_table_names())
        )
    await engine.dispose()

    assert "submission_families" in table_names


async def test_duplicate_family_normalized_name_fails(model_session):
    normalized_name = normalize_submission_name("Example Agent")
    model_session.add_all(
        [
            SubmissionFamily(
                public_family_id="family-a",
                owner_hotkey="owner-a",
                display_name="Example Agent",
                normalized_name=normalized_name,
            ),
            SubmissionFamily(
                public_family_id="family-b",
                owner_hotkey="owner-b",
                display_name="example agent",
                normalized_name=normalized_name,
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_duplicate_public_family_id_fails(model_session):
    model_session.add_all(
        [
            SubmissionFamily(
                public_family_id="family-a",
                owner_hotkey="owner-a",
                display_name="Agent A",
                normalized_name=normalize_submission_name("Agent A"),
            ),
            SubmissionFamily(
                public_family_id="family-a",
                owner_hotkey="owner-b",
                display_name="Agent B",
                normalized_name=normalize_submission_name("Agent B"),
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_family_accepts_three_versions_and_tracks_latest(model_session):
    family = SubmissionFamily(
        public_family_id="family-versioned",
        owner_hotkey="owner-hotkey",
        display_name="Versioned Agent",
        normalized_name=normalize_submission_name("Versioned Agent"),
    )
    model_session.add(family)
    await model_session.flush()

    versions = [
        _submission(
            family_id=family.id,
            version_number=number,
            agent_hash=f"agent-hash-{number}",
            canonical_artifact_hash=f"zip-hash-{number}",
            is_latest_version=number == 3,
        )
        for number in (1, 2, 3)
    ]
    model_session.add_all(versions)
    await model_session.flush()
    family.latest_submission_id = versions[2].id
    family.version_count = 3
    await model_session.commit()

    latest_count = await model_session.scalar(
        select(func.count(AgentSubmission.id)).where(
            AgentSubmission.submission_family_id == family.id,
            AgentSubmission.is_latest_version.is_(True),
        )
    )
    await model_session.refresh(family, attribute_names=["latest_submission", "submissions"])

    assert latest_count == 1
    assert family.latest_submission_id == versions[2].id
    assert family.latest_submission == versions[2]
    assert family.version_count == 3
    assert {submission.version_number for submission in family.submissions} == {1, 2, 3}


async def test_duplicate_family_version_number_fails(model_session):
    family = SubmissionFamily(
        public_family_id="family-duplicate-version",
        owner_hotkey="owner-hotkey",
        display_name="Duplicate Version Agent",
        normalized_name=normalize_submission_name("Duplicate Version Agent"),
    )
    model_session.add(family)
    await model_session.flush()
    model_session.add_all(
        [
            _submission(
                family_id=family.id,
                version_number=1,
                agent_hash="agent-hash-a",
                canonical_artifact_hash="zip-hash-a",
            ),
            _submission(
                family_id=family.id,
                version_number=1,
                agent_hash="agent-hash-b",
                canonical_artifact_hash="zip-hash-b",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_duplicate_canonical_artifact_hash_fails(model_session):
    family = SubmissionFamily(
        public_family_id="family-duplicate-artifact",
        owner_hotkey="owner-hotkey",
        display_name="Duplicate Artifact Agent",
        normalized_name=normalize_submission_name("Duplicate Artifact Agent"),
    )
    model_session.add(family)
    await model_session.flush()
    model_session.add_all(
        [
            _submission(
                family_id=family.id,
                version_number=1,
                agent_hash="agent-hash-a",
                canonical_artifact_hash="same-zip-hash",
            ),
            _submission(
                family_id=family.id,
                version_number=2,
                agent_hash="agent-hash-b",
                canonical_artifact_hash="same-zip-hash",
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await model_session.commit()


async def test_distinct_version_hashes_do_not_violate_agent_hash_constraint(model_session):
    family = SubmissionFamily(
        public_family_id="family-distinct-hashes",
        owner_hotkey="owner-hotkey",
        display_name="Distinct Hash Agent",
        normalized_name=normalize_submission_name("Distinct Hash Agent"),
    )
    model_session.add(family)
    await model_session.flush()
    model_session.add_all(
        [
            _submission(
                family_id=family.id,
                version_number=number,
                agent_hash=f"distinct-agent-hash-{number}",
                canonical_artifact_hash=f"distinct-zip-hash-{number}",
            )
            for number in (1, 2, 3)
        ]
    )
    await model_session.commit()

    assert await model_session.scalar(select(func.count(AgentSubmission.id))) == 3


async def test_legacy_agent_submission_constructor_still_works(model_session):
    submission = AgentSubmission(
        miner_hotkey="miner-hotkey",
        name="legacy-name",
        agent_hash="legacy-agent-hash",
        artifact_uri="/tmp/legacy-artifact.zip",
    )
    model_session.add(submission)
    await model_session.commit()
    await model_session.refresh(submission)

    assert submission.id is not None
    assert submission.submission_family_id is None
    assert submission.version_number is None
    assert submission.version_label is None
    assert submission.canonical_artifact_hash is None
    assert submission.is_latest_version is False


def test_normalize_submission_name_rules():
    assert normalize_submission_name("  Agent\tName  ") == "agent name"
    assert normalize_submission_name("Ａｇｅｎｔ＿１") == "agent_1"
    assert normalize_submission_name("Agent-1.2: Stable") == "agent-1.2: stable"

    for invalid in ("", "   ", "agent/one", "agent!", "a" * 129):
        with pytest.raises(ValueError):
            normalize_submission_name(invalid)


def test_version_label_rules():
    assert version_label(1) == "v1"
    assert version_label(3) == "v3"
    with pytest.raises(ValueError):
        version_label(0)
