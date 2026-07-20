from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import (
    AgentSubmission,
    EvaluationJob,
    SubmissionArtifact,
    SubmissionFamily,
)
from agent_challenge.security import SignedRequestAuth

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


@pytest.fixture
def signed_submission_override():
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey="signed-miner-hotkey",
            signature="test-signature",
            nonce="test-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="POST\n/submissions\n2026-05-22T12:00:00+00:00\ntest-nonce\ntest-body-sha256",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


def build_zip(files: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    archive_files = {"agent.py": ENTRYPOINT_SOURCE, **files}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_files.items():
            if filename == "agent.py":
                contents = agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def submission_payload(
    archive_bytes: bytes,
    *,
    name: str = "signed-agent",
    agent_hash: str | None = None,
) -> dict[str, str]:
    payload = {
        "name": name,
        "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
    }
    if agent_hash is not None:
        payload["agent_hash"] = agent_hash
    return payload


def zip_hash(archive_bytes: bytes) -> str:
    return hashlib.sha256(archive_bytes).hexdigest()


def set_signed_hotkey(hotkey: str) -> None:
    async def authenticate() -> SignedRequestAuth:
        return SignedRequestAuth(
            hotkey=hotkey,
            signature="test-signature",
            nonce=f"test-nonce-{hotkey}",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="test-body-sha256",
            canonical_request="POST\n/submissions\n2026-05-22T12:00:00+00:00\ntest-nonce\ntest-body-sha256",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate


@pytest.fixture
def disabled_submission_rate_limit(monkeypatch):
    async def reserve(**_kwargs):
        return SimpleNamespace(
            row=SimpleNamespace(status="reserved"),
            next_allowed_at=datetime(2026, 5, 22, 15, 0, tzinfo=UTC),
        )

    monkeypatch.setattr("agent_challenge.api.routes.reserve_submission_rate_limit", reserve)
    monkeypatch.setattr(
        "agent_challenge.api.routes.consume_submission_rate_limit",
        lambda reservation: setattr(reservation.row, "status", "consumed"),
    )


async def test_signed_submission_stores_immutable_zip_metadata(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    artifact_root = tmp_path / "agents"
    monkeypatch.setattr("agent_challenge.api.routes.settings.artifact_root", str(artifact_root))
    archive_bytes = build_zip({"agent.py": "print('ok')\n"})
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/submissions",
        json={
            "miner_hotkey": "body-hotkey-is-ignored",
            "name": "signed-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["submission_id"] is not None
    assert payload["name"] == "signed-agent"
    assert payload["agent_hash"] == zip_sha256
    assert payload["zip_sha256"] == zip_sha256
    assert payload["family_id"]
    assert payload["version_number"] == 1
    assert payload["version_label"] == "v1"
    assert payload["version_count"] == 1
    assert payload["is_latest_version"] is True
    assert payload["status"] == "queued"
    assert payload["effective_status"] == "queued"
    assert payload["submitted_at"] is not None
    assert payload["created_at"] is not None
    assert payload["latest_evaluation"] is None
    assert {
        "logs_ref",
        "report_json",
        "signature",
        "signature_nonce",
        "signature_payload_sha256",
        "signature_message",
        "raw_status",
        "artifact_path",
        "artifact_uri",
        "submission_family_id",
        "normalized_name",
        "canonical_artifact_hash",
    }.isdisjoint(payload)
    artifact_path = artifact_root / zip_sha256 / "agent.zip"
    assert artifact_path.read_bytes() == archive_bytes
    assert not (artifact_root / zip_sha256 / "agent.py").exists()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        stored_artifact = await session.scalar(select(SubmissionArtifact))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.id == payload["submission_id"]
    assert submission.miner_hotkey == "signed-miner-hotkey"
    assert submission.agent_name == "signed-agent"
    assert submission.agent_hash == zip_sha256
    assert submission.artifact_uri == str(artifact_path)
    assert submission.artifact_path == str(artifact_path)
    assert submission.zip_sha256 == zip_sha256
    assert submission.zip_size_bytes == len(archive_bytes)
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert submission.signature == "test-signature"
    assert submission.signature_nonce == "test-nonce"
    assert submission.signature_payload_sha256 == "test-body-sha256"
    assert stored_artifact is not None
    assert stored_artifact.submission_id == submission.id
    assert stored_artifact.artifact_kind == "source_zip"
    assert stored_artifact.uri == str(artifact_path)
    assert stored_artifact.sha256 == zip_sha256
    assert stored_artifact.size_bytes == len(archive_bytes)
    artifact_metadata = json.loads(stored_artifact.metadata_json)
    assert artifact_metadata["content_type"] == "application/zip"
    assert artifact_metadata["manifest_path"] == str(artifact_root / zip_sha256 / "manifest.json")
    assert artifact_metadata["manifest"]["zip_sha256"] == zip_sha256
    assert artifact_metadata["manifest"]["entries"][0]["normalized_path"] == "agent.py"
    assert job_count == 0

    evaluation = await client.get(f"/agents/{zip_sha256}/evaluation")
    assert evaluation.status_code == 404


async def test_unsigned_submission_is_rejected(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = build_zip({"agent.py": "print('ok')\n"})

    response = await client.post(
        "/submissions",
        json={
            "name": "unsigned-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid signed request"}


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "missing-artifact"},
        {
            "name": "too-many-artifacts",
            "artifact_uri": "/tmp/agent.zip",
            "artifact_zip_base64": base64.b64encode(build_zip({"agent.py": "ok"})).decode("ascii"),
        },
    ],
)
async def test_submission_requires_exactly_one_artifact_source(
    client,
    monkeypatch,
    payload,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    response = await client.post("/submissions", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_artifact_source_count"


async def test_submission_accepts_zip_artifact_uri_inside_artifact_root(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    artifact_root = tmp_path / "agents"
    artifact_root.mkdir()
    archive_bytes = build_zip({"agent.py": "print('ok')\n"})
    source_path = artifact_root / "upload.zip"
    source_path.write_bytes(archive_bytes)
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    monkeypatch.setattr("agent_challenge.api.routes.settings.artifact_root", str(artifact_root))

    response = await client.post(
        "/submissions",
        json={"name": "uri-agent", "artifact_uri": str(source_path)},
    )

    assert response.status_code == 201
    assert response.json()["zip_sha256"] == zip_sha256
    assert (artifact_root / zip_sha256 / "agent.zip").read_bytes() == archive_bytes


async def test_master_validator_submission_status_is_analysis_queued_without_evaluation_job(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    monkeypatch.setattr("agent_challenge.api.routes.settings.validator_role", "master")
    monkeypatch.setattr("agent_challenge.evaluation.runner.load_benchmark_tasks", lambda: [])
    monkeypatch.setattr("agent_challenge.evaluation.runner.settings.evaluation_task_count", 0)
    archive_bytes = build_zip({"agent.py": "print('ok')\n"})

    response = await client.post(
        "/submissions",
        json={
            "name": "master-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "queued"
    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        job_count = await session.scalar(select(func.count(EvaluationJob.id)))

    assert submission is not None
    assert submission.status == "queued"
    assert submission.raw_status == "analysis_queued"
    assert submission.effective_status == "queued"
    assert submission.latest_evaluation_job_id is None
    assert job_count == 0


async def test_oversized_submission_zip_returns_payload_too_large(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    response = await client.post(
        "/submissions",
        json={
            "name": "oversized-agent",
            "artifact_zip_base64": base64.b64encode(b"0" * 1_048_577).decode("ascii"),
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "zip_too_large"


async def test_unsafe_submission_zip_returns_bad_request(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = build_zip({"../agent.py": "print('escape')\n"})

    response = await client.post(
        "/submissions",
        json={
            "name": "unsafe-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "parent_path"


async def test_same_owner_name_unique_zips_create_v1_v2_v3_and_latest_state(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archives = [build_zip({"agent.py": f"print({version})\n"}) for version in (1, 2, 3)]

    responses = [
        await client.post("/submissions", json=submission_payload(archive, name="Versioned Agent"))
        for archive in archives
    ]

    assert [response.status_code for response in responses] == [201, 201, 201]
    payloads = [response.json() for response in responses]
    archive_hashes = [zip_hash(archive) for archive in archives]
    family_id = payloads[0]["family_id"]
    assert [payload["zip_sha256"] for payload in payloads] == archive_hashes
    assert [payload["agent_hash"] for payload in payloads] == archive_hashes
    assert [payload["family_id"] for payload in payloads] == [family_id, family_id, family_id]
    assert [payload["version_number"] for payload in payloads] == [1, 2, 3]
    assert [payload["version_label"] for payload in payloads] == ["v1", "v2", "v3"]
    assert [payload["version_count"] for payload in payloads] == [1, 2, 3]
    assert [payload["is_latest_version"] for payload in payloads] == [True, True, True]
    assert [payload["name"] for payload in payloads] == ["Versioned Agent"] * 3
    for payload in payloads:
        assert {
            "submission_family_id",
            "normalized_name",
            "canonical_artifact_hash",
            "artifact_path",
            "artifact_uri",
            "signature",
            "signature_nonce",
        }.isdisjoint(payload)
    async with database_session() as session:
        family = await session.scalar(select(SubmissionFamily))
        submissions = (
            await session.scalars(select(AgentSubmission).order_by(AgentSubmission.version_number))
        ).all()

    assert family is not None
    assert family.owner_hotkey == "signed-miner-hotkey"
    assert family.display_name == "Versioned Agent"
    assert family.normalized_name == "versioned agent"
    assert family.version_count == 3
    assert family.public_family_id == family_id
    assert family.latest_submission_id == submissions[2].id
    version_rows = [
        (submission.version_number, submission.version_label) for submission in submissions
    ]
    assert version_rows == [
        (1, "v1"),
        (2, "v2"),
        (3, "v3"),
    ]
    assert [submission.agent_hash for submission in submissions] == archive_hashes
    assert [submission.canonical_artifact_hash for submission in submissions] == archive_hashes
    assert [submission.is_latest_version for submission in submissions] == [False, False, True]


async def test_duplicate_hash_same_name_returns_duplicate_code_hash_without_new_version(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = build_zip({"agent.py": "print('same')\n"})

    first = await client.post(
        "/submissions",
        json=submission_payload(archive_bytes, name="Agent A"),
    )
    duplicate = await client.post(
        "/submissions",
        json=submission_payload(archive_bytes, name="Agent A", agent_hash="different-client-hash"),
    )

    assert first.status_code == 201
    assert first.json()["version_number"] == 1
    assert first.json()["version_label"] == "v1"
    assert first.json()["version_count"] == 1
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "duplicate_code_hash"
    async with database_session() as session:
        family = await session.scalar(select(SubmissionFamily))
        submission_count = await session.scalar(select(func.count(AgentSubmission.id)))

    assert family is not None
    assert family.version_count == 1
    assert submission_count == 1


async def test_rejected_duplicate_hash_does_not_skip_next_version(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archives = [build_zip({"agent.py": f"print({version})\n"}) for version in (1, 2, 3)]

    first = await client.post(
        "/submissions",
        json=submission_payload(archives[0], name="No Gap Agent"),
    )
    second = await client.post(
        "/submissions",
        json=submission_payload(archives[1], name="No Gap Agent"),
    )
    duplicate = await client.post(
        "/submissions",
        json=submission_payload(archives[0], name="No Gap Agent"),
    )
    third = await client.post(
        "/submissions",
        json=submission_payload(archives[2], name="No Gap Agent"),
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "duplicate_code_hash"
    assert third.status_code == 201
    assert [
        first.json()["version_number"],
        second.json()["version_number"],
        third.json()["version_number"],
    ] == [1, 2, 3]
    assert third.json()["version_label"] == "v3"
    assert third.json()["version_count"] == 3
    async with database_session() as session:
        family = await session.scalar(select(SubmissionFamily))
        submissions = (
            await session.scalars(select(AgentSubmission).order_by(AgentSubmission.version_number))
        ).all()

    assert family is not None
    assert family.version_count == 3
    assert [submission.version_number for submission in submissions] == [1, 2, 3]
    assert [submission.is_latest_version for submission in submissions] == [False, False, True]


async def test_duplicate_hash_different_name_and_owner_prioritizes_duplicate_code_hash(
    client,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = build_zip({"agent.py": "print('same')\n"})

    first = await client.post(
        "/submissions",
        json=submission_payload(archive_bytes, name="Agent A"),
    )
    set_signed_hotkey("another-signed-miner")
    duplicate = await client.post(
        "/submissions",
        json=submission_payload(archive_bytes, name="Agent B", agent_hash="client-hash"),
    )

    assert first.status_code == 201
    assert first.json()["version_number"] == 1
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "duplicate_code_hash"


async def test_name_claimed_by_another_owner_returns_name_taken(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    first = await client.post(
        "/submissions",
        json=submission_payload(build_zip({"agent.py": "print('a')\n"}), name="Claimed Agent"),
    )
    set_signed_hotkey("another-signed-miner")
    second = await client.post(
        "/submissions",
        json=submission_payload(build_zip({"agent.py": "print('b')\n"}), name=" claimed  agent "),
    )

    assert first.status_code == 201
    assert first.json()["version_number"] == 1
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "name_taken"
    async with database_session() as session:
        family_count = await session.scalar(select(func.count(SubmissionFamily.id)))
        submission_count = await session.scalar(select(func.count(AgentSubmission.id)))

    assert family_count == 1
    assert submission_count == 1


async def test_omitted_and_explicit_agent_hash_are_ignored_for_stored_zip_identity(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_a = build_zip({"agent.py": "print('a')\n"})
    archive_b = build_zip({"agent.py": "print('b')\n"})
    hash_a = zip_hash(archive_a)
    hash_b = zip_hash(archive_b)

    first = await client.post("/submissions", json=submission_payload(archive_a, name="Hash Agent"))
    second = await client.post(
        "/submissions",
        json=submission_payload(archive_b, name="Hash Agent", agent_hash="client-hash-123"),
    )
    duplicate = await client.post(
        "/submissions",
        json=submission_payload(archive_a, name="Hash Agent", agent_hash="another-client-hash"),
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["agent_hash"] == hash_a
    assert first.json()["zip_sha256"] == hash_a
    assert first.json()["version_number"] == 1
    assert first.json()["version_label"] == "v1"
    assert first.json()["version_count"] == 1
    assert second.json()["agent_hash"] == hash_b
    assert second.json()["zip_sha256"] == hash_b
    assert second.json()["version_number"] == 2
    assert second.json()["version_label"] == "v2"
    assert second.json()["version_count"] == 2
    assert second.json()["family_id"] == first.json()["family_id"]
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"]["code"] == "duplicate_code_hash"
    async with database_session() as session:
        submissions = (
            await session.scalars(select(AgentSubmission).order_by(AgentSubmission.version_number))
        ).all()
        family = await session.scalar(select(SubmissionFamily))

    assert family is not None
    assert family.version_count == 2
    version_rows = [
        (submission.version_number, submission.is_latest_version) for submission in submissions
    ]
    assert version_rows == [
        (1, False),
        (2, True),
    ]
    assert [submission.agent_hash for submission in submissions] == [hash_a, hash_b]
    assert [submission.canonical_artifact_hash for submission in submissions] == [hash_a, hash_b]
    assert second.json()["zip_sha256"] == hash_b


async def test_sequential_double_first_claim_same_name_creates_one_family_and_name_taken(
    client,
    database_session,
    monkeypatch,
    signed_submission_override,
    disabled_submission_rate_limit,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    first = await client.post(
        "/submissions",
        json=submission_payload(build_zip({"agent.py": "print('owner-a')\n"}), name="First Claim"),
    )
    set_signed_hotkey("another-signed-miner")
    second = await client.post(
        "/submissions",
        json=submission_payload(build_zip({"agent.py": "print('owner-b')\n"}), name="First Claim"),
    )

    assert first.status_code == 201
    assert first.json()["version_number"] == 1
    assert second.status_code == 409
    assert second.json()["detail"]["code"] == "name_taken"
    async with database_session() as session:
        family_count = await session.scalar(select(func.count(SubmissionFamily.id)))
        family = await session.scalar(select(SubmissionFamily))
        submission_count = await session.scalar(select(func.count(AgentSubmission.id)))

    assert family_count == 1
    assert family is not None
    assert family.version_count == 1
    assert submission_count == 1
