from __future__ import annotations

import base64
import hashlib
import io
import zipfile

import pytest
from sqlalchemy import select

from agent_challenge import routes
from agent_challenge.app import app
from agent_challenge.models import AgentSubmission, SubmissionArtifact
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
            hotkey="json-signed-hotkey",
            signature="json-signature",
            nonce="json-nonce",
            timestamp="2026-05-22T12:00:00+00:00",
            body_sha256="json-body-sha256",
            canonical_request="json-canonical-request",
        )

    app.dependency_overrides[routes.signed_submission_auth] = authenticate
    yield
    app.dependency_overrides.pop(routes.signed_submission_auth, None)


def build_zip(contents: str = "print('ok')\n") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", agent_source(contents))
    return buffer.getvalue()


def bridge_headers(**overrides: str) -> dict[str, str]:
    headers = {
        "Authorization": "Bearer test-token",
        "X-Base-Challenge-Slug": "agent-challenge",
        "X-Base-Verified-Hotkey": "verified-hotkey",
        "X-Base-Verified-Nonce": "verified-nonce",
        "X-Base-Request-Hash": "request-hash",
        "X-Hotkey": "spoofed-client-hotkey",
        "Content-Type": "application/zip",
    }
    headers.update(overrides)
    return headers


async def test_bridge_upload_stores_verified_hotkey_and_raw_zip(
    client,
    database_session,
    monkeypatch,
    tmp_path,
):
    artifact_root = tmp_path / "agents"
    monkeypatch.setattr("agent_challenge.api.routes.settings.artifact_root", str(artifact_root))
    archive_bytes = build_zip()
    zip_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=archive_bytes,
        headers=bridge_headers(**{"X-Submission-Filename": "../display-agent.zip"}),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["zip_sha256"] == zip_sha256
    assert payload["status"] == "queued"
    assert (artifact_root / zip_sha256 / "agent.zip").read_bytes() == archive_bytes
    assert not (artifact_root / zip_sha256 / "agent.py").exists()

    async with database_session() as session:
        submission = await session.scalar(select(AgentSubmission))
        artifact = await session.scalar(select(SubmissionArtifact))

    assert submission is not None
    assert submission.miner_hotkey == "verified-hotkey"
    assert submission.name == "display-agent.zip"
    assert submission.agent_hash == zip_sha256
    assert submission.signature == "base-verified"
    assert submission.signature_nonce == "verified-nonce"
    assert submission.signature_payload_sha256 == "request-hash"
    assert artifact is not None
    assert artifact.sha256 == zip_sha256


async def test_bridge_upload_requires_internal_token(client):
    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=build_zip(),
        headers={key: value for key, value in bridge_headers().items() if key != "Authorization"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "invalid bearer token"}


async def test_bridge_upload_rejects_mismatched_slug(client):
    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=build_zip(),
        headers=bridge_headers(**{"X-Base-Challenge-Slug": "other-challenge"}),
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "invalid challenge slug"}


async def test_bridge_upload_requires_verified_hotkey(client):
    headers = bridge_headers()
    headers.pop("X-Base-Verified-Hotkey")

    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=build_zip(),
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "missing X-Base-Verified-Hotkey"}


async def test_bridge_upload_rejects_invalid_zip(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )

    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=b"not-a-zip",
        headers=bridge_headers(),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_zip"


async def test_v1_submission_aliases_match_public_submission_routes(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    response = await client.post(
        "/internal/v1/bridge/submissions",
        content=build_zip(),
        headers=bridge_headers(),
    )
    assert response.status_code == 201
    submission_id = response.json()["submission_id"]

    public_detail = await client.get(f"/submissions/{submission_id}")
    alias_detail = await client.get(f"/v1/submissions/{submission_id}")
    public_status = await client.get(f"/submissions/{submission_id}/status")
    alias_status = await client.get(f"/v1/submissions/{submission_id}/status")

    assert alias_detail.status_code == 200
    assert alias_detail.json() == public_detail.json()
    assert alias_status.status_code == 200
    assert alias_status.json() == public_status.json()


async def test_json_base64_submission_route_still_works(
    client,
    monkeypatch,
    signed_submission_override,
    tmp_path,
):
    monkeypatch.setattr(
        "agent_challenge.api.routes.settings.artifact_root",
        str(tmp_path / "agents"),
    )
    archive_bytes = build_zip()

    response = await client.post(
        "/submissions",
        json={
            "name": "json-agent",
            "artifact_zip_base64": base64.b64encode(archive_bytes).decode("ascii"),
        },
    )

    assert response.status_code == 201
    assert response.json()["zip_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
