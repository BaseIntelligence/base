"""Token-bound eval artifact download endpoint.

Given / When / Then scenarios for GET /eval/v1/runs/{eval_run_id}/artifact:
1. absent token -> 401
2. token minted for another run -> 404
3. expired grant -> 401
4. valid grant -> 200 with exact stored ZIP bytes (sha256 == agent_hash)
5. response headers X-Agent-Hash / X-Package-Tree-Sha match submission
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from agent_challenge.app import app
from agent_challenge.models import AgentSubmission, EvalRun
from agent_challenge.submissions.artifacts import store_zip_bytes

_SHARED_SECRET = "test-token"
_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


def _zip_bytes(*, marker: str = "payload") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("agent.py", f"class Agent:\n    marker = {marker!r}\n")
    return buffer.getvalue()


async def _seed_run(
    session,
    tmp_path: Path,
    *,
    eval_run_id: str,
    zip_payload: bytes | None = None,
    suffix: str = "a",
) -> tuple[str, str, str | None, bytes]:
    """Persist one submission + EvalRun with a real on-disk ZIP.

    Returns (eval_run_id, agent_hash, package_tree_sha, zip_bytes).
    """

    payload = zip_payload if zip_payload is not None else _zip_bytes(marker=suffix)
    metadata = store_zip_bytes(zip_bytes=payload, artifact_root=str(tmp_path))
    agent_hash = metadata.zip_sha256
    submission = AgentSubmission(
        miner_hotkey=f"miner-{suffix}",
        name=f"agent-{suffix}",
        agent_name=f"agent-{suffix}",
        agent_hash=agent_hash,
        canonical_artifact_hash=agent_hash,
        artifact_uri=metadata.artifact_path,
        artifact_path=metadata.artifact_path,
        zip_sha256=metadata.zip_sha256,
        package_tree_sha=metadata.package_tree_sha,
        zip_size_bytes=metadata.zip_size_bytes,
        raw_status="eval_running",
        effective_status="eval_running",
        status="running",
    )
    session.add(submission)
    await session.flush()
    plan_json = f'{{"eval_run_id":"{eval_run_id}","agent_hash":"{agent_hash}"}}'
    run = EvalRun(
        eval_run_id=eval_run_id,
        submission_id=submission.id,
        submission_version=1,
        authorizing_review_digest="a" * 64,
        plan_json=plan_json,
        plan_sha256=hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        token_sha256=hashlib.sha256(f"run-token-{eval_run_id}".encode()).hexdigest(),
        phase="eval_running",
        issued_at=_NOW,
        expires_at=_NOW + timedelta(hours=2),
    )
    session.add(run)
    await session.commit()
    return eval_run_id, agent_hash, metadata.package_tree_sha, payload


def _path(eval_run_id: str) -> str:
    return f"/eval/v1/runs/{eval_run_id}/artifact"


def _mint(
    *,
    eval_run_id: str,
    agent_hash: str,
    expires_at: datetime,
    secret: str = _SHARED_SECRET,
) -> str:
    from agent_challenge.api.eval_artifact_routes import mint_eval_artifact_grant

    return mint_eval_artifact_grant(
        secret=secret,
        eval_run_id=eval_run_id,
        agent_hash=agent_hash,
        expires_at=expires_at,
    )


async def test_artifact_download_requires_token(database_session, tmp_path: Path) -> None:
    # Given: a seeded eval run with a stored ZIP
    async with database_session() as session:
        eval_run_id, _agent_hash, _tree, _zip = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_no_token", suffix="no-token"
        )

    # When: GET without Authorization
    with TestClient(app) as client:
        response = client.get(_path(eval_run_id))

    # Then: 401
    assert response.status_code == 401


async def test_artifact_download_rejects_token_for_other_run(
    database_session, tmp_path: Path
) -> None:
    # Given: two runs; grant minted only for run A
    async with database_session() as session:
        run_a, hash_a, _tree_a, _zip_a = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_scope_a", suffix="scope-a"
        )
        run_b, _hash_b, _tree_b, _zip_b = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_scope_b", suffix="scope-b"
        )

    token_for_a = _mint(
        eval_run_id=run_a,
        agent_hash=hash_a,
        expires_at=_NOW + timedelta(hours=1),
    )

    # When: present A's grant against B's path
    with TestClient(app) as client:
        response = client.get(
            _path(run_b),
            headers={"Authorization": f"Bearer {token_for_a}"},
        )

    # Then: 404 (no leak that the other run exists)
    assert response.status_code == 404


async def test_artifact_download_rejects_expired_grant(database_session, tmp_path: Path) -> None:
    # Given: grant already past expires_at
    async with database_session() as session:
        eval_run_id, agent_hash, _tree, _zip = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_expired", suffix="expired"
        )

    expired_token = _mint(
        eval_run_id=eval_run_id,
        agent_hash=agent_hash,
        # Far past wall clock so expiry is independent of fixture _NOW vs real now.
        expires_at=datetime(2000, 1, 1, tzinfo=UTC),
    )

    # When: GET with expired grant
    with TestClient(app) as client:
        response = client.get(
            _path(eval_run_id),
            headers={"Authorization": f"Bearer {expired_token}"},
        )

    # Then: 401
    assert response.status_code == 401


async def test_artifact_download_returns_exact_zip_bytes(database_session, tmp_path: Path) -> None:
    # Given: stored ZIP whose sha256 is the submission agent_hash
    async with database_session() as session:
        eval_run_id, agent_hash, _tree, zip_payload = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_bytes", suffix="bytes"
        )

    token = _mint(
        eval_run_id=eval_run_id,
        agent_hash=agent_hash,
        expires_at=_NOW + timedelta(hours=1),
    )

    # When: GET with valid grant
    with TestClient(app) as client:
        response = client.get(
            _path(eval_run_id),
            headers={"Authorization": f"Bearer {token}"},
        )

    # Then: 200 and body is the exact stored ZIP (sha256 == agent_hash)
    assert response.status_code == 200
    assert response.content == zip_payload
    assert hashlib.sha256(response.content).hexdigest() == agent_hash
    assert response.headers.get("content-type", "").startswith("application/zip")
    assert response.headers.get("content-length") == str(len(zip_payload))


async def test_artifact_download_sets_hash_headers(database_session, tmp_path: Path) -> None:
    # Given: submission with agent_hash + package_tree_sha
    async with database_session() as session:
        eval_run_id, agent_hash, package_tree_sha, _zip = await _seed_run(
            session, tmp_path, eval_run_id="eval_run_headers", suffix="headers"
        )

    token = _mint(
        eval_run_id=eval_run_id,
        agent_hash=agent_hash,
        expires_at=_NOW + timedelta(hours=1),
    )

    # When: successful download
    with TestClient(app) as client:
        response = client.get(
            _path(eval_run_id),
            headers={"Authorization": f"Bearer {token}"},
        )

    # Then: hash headers match the submission digests
    assert response.status_code == 200
    assert response.headers.get("X-Agent-Hash") == agent_hash
    assert package_tree_sha is not None
    assert response.headers.get("X-Package-Tree-Sha") == package_tree_sha
