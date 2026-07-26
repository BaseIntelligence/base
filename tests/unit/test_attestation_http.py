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
