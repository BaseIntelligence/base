"""S8 + checker HTTP surfaces for production constation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.attestation.payload import (
    AttestationPayload,
    derive_attestation_key,
    sign_attestation_payload,
)
from base.compute.attestation_nonce import NonceBinding
from base.db.models import Base
from base.master.constation.allowlist_repository import DigestAllowlistRepository
from base.master.constation.bundle_store import ConstationBundleStore
from base.master.constation.nonce_repository import DurableAttestationNonceService
from base.master.constation.routes import create_constation_test_app

TOKEN = "test-internal"
BINDING = NonceBinding(work_unit_id="wu-http", miner_hotkey="hk-1", pod_id="pod-1")
COMMIT = "a" * 40
TREE = "b" * 40
DIGEST = "sha256:" + ("c" * 64)
MANIFEST = {"harness.py": "d" * 64}


@pytest.fixture
async def harness(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    db_path = tmp_path / "constation_http.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    allowlist_repo = DigestAllowlistRepository(factory)
    nonce_svc = DurableAttestationNonceService(factory, ttl=timedelta(hours=1))
    store = ConstationBundleStore()
    key = derive_attestation_key(b"build-secret-fixture")
    app = create_constation_test_app(
        allowlist_repo=allowlist_repo,
        nonce_service=nonce_svc,
        internal_token=TOKEN,
        default_binding=BINDING,
        attestation_verify_key=key,
        bundle_store=store,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "headers": {"Authorization": f"Bearer {TOKEN}"},
            "key": key,
            "store": store,
        }
    await engine.dispose()


@pytest.mark.asyncio
async def test_challenge_answer_roundtrip_s8(harness: dict[str, Any]) -> None:
    client: AsyncClient = harness["client"]
    r = await client.get("/v1/attestation/challenge", params={"phase": "start"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["nonce"]
    assert body["phase"] == "start"
    # parse_challenge-compatible shape
    assert "nonce" in body and "phase" in body

    ans = await client.post(
        "/v1/attestation/answer",
        json={"nonce": body["nonce"], "phase": "start", "sig": "xx"},
    )
    assert ans.status_code == 200
    assert ans.json()["status"] == "accepted"


@pytest.mark.asyncio
async def test_register_and_check_allowlist(harness: dict[str, Any]) -> None:
    client = harness["client"]
    headers = harness["headers"]
    reg = await client.post(
        "/internal/v1/constation/register_digest",
        headers=headers,
        json={
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "variant": "cuda",
            "digest": DIGEST,
            "sealed_manifest_hashes": dict(MANIFEST),
        },
    )
    assert reg.status_code == 200, reg.text

    hit = await client.post(
        "/internal/v1/constation/check_allowlist",
        headers=headers,
        json={
            "digest": DIGEST,
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "variant": "cuda",
        },
    )
    assert hit.status_code == 200
    assert hit.json() == {"ok": True, "reason": "ok"}

    miss = await client.post(
        "/internal/v1/constation/check_allowlist",
        headers=headers,
        json={
            "digest": "sha256:" + ("f" * 64),
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "variant": "cuda",
        },
    )
    assert miss.json()["ok"] is False
    assert miss.json()["reason"] == "unknown_digest"


@pytest.mark.asyncio
async def test_check_nonce_consume_and_replay(harness: dict[str, Any]) -> None:
    client = harness["client"]
    headers = harness["headers"]
    ch = await client.get("/v1/attestation/challenge", params={"phase": "interval"})
    nonce = ch.json()["nonce"]
    body = {
        "nonce": nonce,
        "work_unit_id": BINDING.work_unit_id,
        "miner_hotkey": BINDING.miner_hotkey,
        "pod_id": BINDING.pod_id,
    }
    first = await client.post(
        "/internal/v1/constation/check_nonce", headers=headers, json=body
    )
    second = await client.post(
        "/internal/v1/constation/check_nonce", headers=headers, json=body
    )
    assert first.json() == {"ok": True, "reason": "ok"}
    assert second.json()["ok"] is False
    assert second.json()["reason"] == "already_consumed"


@pytest.mark.asyncio
async def test_bundle_put_get(harness: dict[str, Any]) -> None:
    client = harness["client"]
    headers = harness["headers"]
    bundle = {"digest": DIGEST, "nonce": "n1", "work_unit_id": "wu-http"}
    put = await client.put(
        "/internal/v1/constation/bundle/wu-http", headers=headers, json=bundle
    )
    assert put.status_code == 200
    got = await client.get("/internal/v1/constation/bundle/wu-http", headers=headers)
    assert got.status_code == 200
    assert got.json()["digest"] == DIGEST


@pytest.mark.asyncio
async def test_verify_attestation_ok(harness: dict[str, Any]) -> None:
    client = harness["client"]
    headers = harness["headers"]
    key: bytes = harness["key"]
    payload = AttestationPayload(
        nonce="nonce-1",
        digest=DIGEST,
        pod_id=BINDING.pod_id,
        variant="cuda",
        sealed_manifest_hashes=dict(MANIFEST),
        build_secret_response="a" * 64,
    )
    # build_secret_response must be valid hex from real helper for structure;
    # sign with derived key after computing real response.
    from base.attestation.payload import compute_build_secret_response

    secret = b"build-secret-fixture"
    key = derive_attestation_key(secret)
    payload = AttestationPayload(
        nonce="nonce-1",
        digest=DIGEST,
        pod_id=BINDING.pod_id,
        variant="cuda",
        sealed_manifest_hashes=dict(MANIFEST),
        build_secret_response=compute_build_secret_response(
            build_secret=secret, nonce="nonce-1"
        ),
    )
    signed = sign_attestation_payload(payload, signing_key=key)
    wire = {
        "payload": {
            "nonce": payload.nonce,
            "digest": payload.digest,
            "pod_id": payload.pod_id,
            "variant": payload.variant,
            "sealed_manifest_hashes": dict(payload.sealed_manifest_hashes),
            "build_secret_response": payload.build_secret_response,
        },
        "signature": signed.signature,
        "algorithm": signed.algorithm,
        "schema_version": signed.schema_version,
    }
    r = await client.post(
        "/internal/v1/constation/verify_attestation",
        headers=headers,
        json={"signed": wire},
    )
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True


@pytest.mark.asyncio
async def test_register_digest_http_rejects_empty_sealed_manifest(
    harness: dict[str, Any],
) -> None:
    """Given empty sealed hashes, When POST register_digest, Then 4xx not 500."""
    client = harness["client"]
    headers = harness["headers"]
    reg = await client.post(
        "/internal/v1/constation/register_digest",
        headers=headers,
        json={
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "variant": "cuda",
            "digest": DIGEST,
            "sealed_manifest_hashes": {},
        },
    )
    assert 400 <= reg.status_code < 500, reg.text
    assert reg.status_code != 500


# ---------------------------------------------------------------------------
# T6b register_miner_key
# ---------------------------------------------------------------------------


class _FakePodBinding:
    """Minimal stand-in for MinerPodBinding.register (HTTP layer only)."""

    def __init__(
        self,
        *,
        verdict: object | None = None,
        raise_value_error: str | None = None,
    ) -> None:
        from base.compute.constation_types import ConstationFailCode, ConstationVerdict

        self.calls: list[dict[str, str]] = []
        self._raise = raise_value_error
        self._verdict = (
            verdict
            if verdict is not None
            else ConstationVerdict(ok=True, reason=ConstationFailCode.OK)
        )

    async def register(
        self,
        *,
        miner_hotkey: str,
        api_key: str,
        instance_id: str,
    ) -> object:
        self.calls.append(
            {
                "miner_hotkey": miner_hotkey,
                "api_key": api_key,
                "instance_id": instance_id,
            }
        )
        if self._raise is not None:
            raise ValueError(self._raise)
        return self._verdict


@pytest.fixture
async def register_miner_harness(
    tmp_path: Path,
) -> AsyncIterator[dict[str, Any]]:
    from fastapi import FastAPI

    from base.master.constation.routes import build_constation_router

    db_path = tmp_path / "register_miner_http.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    allowlist_repo = DigestAllowlistRepository(factory)
    nonce_svc = DurableAttestationNonceService(factory, ttl=timedelta(hours=1))
    fake = _FakePodBinding()
    app = FastAPI()
    router = build_constation_router(
        allowlist_repo=allowlist_repo,
        nonce_service=nonce_svc,
        internal_token=TOKEN,
        pod_binding=fake,  # type: ignore[arg-type]
    )
    app.include_router(router)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "headers": {"Authorization": f"Bearer {TOKEN}"},
            "binding": fake,
            "app": app,
            "allowlist_repo": allowlist_repo,
            "nonce_svc": nonce_svc,
        }
    await engine.dispose()


@pytest.mark.asyncio
async def test_register_miner_key_ok(
    register_miner_harness: dict[str, Any],
) -> None:
    """Given valid body + binding ok, When POST, Then 200 registered."""
    client: AsyncClient = register_miner_harness["client"]
    headers = register_miner_harness["headers"]
    binding: _FakePodBinding = register_miner_harness["binding"]
    secret = "lium-http-test-key-NEVER-ECHO"

    r = await client.post(
        "/internal/v1/constation/register_miner_key",
        headers=headers,
        json={
            "miner_hotkey": "hk-miner-1",
            "api_key": secret,
            "instance_id": "pod-xyz",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {"status": "registered"}
    assert "api_key" not in body
    assert secret not in r.text
    assert binding.calls == [
        {
            "miner_hotkey": "hk-miner-1",
            "api_key": secret,
            "instance_id": "pod-xyz",
        }
    ]


@pytest.mark.asyncio
async def test_register_miner_key_verdict_fail_is_422(
    tmp_path: Path,
) -> None:
    """Given fail verdict, When POST, Then 422 with fail code not 500."""
    from fastapi import FastAPI

    from base.compute.constation_types import ConstationFailCode, ConstationVerdict
    from base.master.constation.routes import build_constation_router

    db_path = tmp_path / "register_miner_fail.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    fake = _FakePodBinding(
        verdict=ConstationVerdict(
            ok=False,
            reason=ConstationFailCode.LIUM_AUTH_REVOKED,
            detail="probe_401",
        )
    )
    app = FastAPI()
    app.include_router(
        build_constation_router(
            allowlist_repo=DigestAllowlistRepository(factory),
            nonce_service=DurableAttestationNonceService(
                factory, ttl=timedelta(hours=1)
            ),
            internal_token=TOKEN,
            pod_binding=fake,  # type: ignore[arg-type]
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/internal/v1/constation/register_miner_key",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "miner_hotkey": "hk-1",
                "api_key": "bad-key-secret",
                "instance_id": "pod-1",
            },
        )
    await engine.dispose()
    assert r.status_code == 422, r.text
    assert r.status_code != 500
    detail = r.json()["detail"]
    assert "lium_auth_revoked" in str(detail)
    assert "bad-key-secret" not in r.text


@pytest.mark.asyncio
async def test_register_miner_key_without_binding_is_503(
    tmp_path: Path,
) -> None:
    """Given pod_binding=None, When POST register_miner_key, Then 503."""
    from fastapi import FastAPI

    from base.master.constation.routes import build_constation_router

    db_path = tmp_path / "register_miner_503.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app = FastAPI()
    app.include_router(
        build_constation_router(
            allowlist_repo=DigestAllowlistRepository(factory),
            nonce_service=DurableAttestationNonceService(
                factory, ttl=timedelta(hours=1)
            ),
            internal_token=TOKEN,
            pod_binding=None,
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/internal/v1/constation/register_miner_key",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "miner_hotkey": "hk-1",
                "api_key": "k",
                "instance_id": "pod-1",
            },
        )
    await engine.dispose()
    assert r.status_code == 503, r.text


@pytest.mark.asyncio
async def test_register_miner_key_requires_internal_auth(
    register_miner_harness: dict[str, Any],
) -> None:
    """Given no bearer, When POST register_miner_key, Then 401."""
    client: AsyncClient = register_miner_harness["client"]
    r = await client.post(
        "/internal/v1/constation/register_miner_key",
        json={
            "miner_hotkey": "hk-1",
            "api_key": "k",
            "instance_id": "pod-1",
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_register_miner_key_value_error_is_422(
    tmp_path: Path,
) -> None:
    """Given binding raises ValueError, When POST, Then 422 not 500."""
    from fastapi import FastAPI

    from base.master.constation.routes import build_constation_router

    db_path = tmp_path / "register_miner_ve.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    fake = _FakePodBinding(raise_value_error="miner_hotkey must be a non-empty string")
    app = FastAPI()
    app.include_router(
        build_constation_router(
            allowlist_repo=DigestAllowlistRepository(factory),
            nonce_service=DurableAttestationNonceService(
                factory, ttl=timedelta(hours=1)
            ),
            internal_token=TOKEN,
            pod_binding=fake,  # type: ignore[arg-type]
        )
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/internal/v1/constation/register_miner_key",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "miner_hotkey": "   ",
                "api_key": "k",
                "instance_id": "pod-1",
            },
        )
    await engine.dispose()
    assert r.status_code == 422, r.text
    assert r.status_code != 500
