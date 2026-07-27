"""B1: master must load attestation_verify_key_hex and pass it to the router.

Prism BaseHttp verify_signature sends no body key_hex — production relies on
the router-held key. If main omits attestation_verify_key=, verify always
returns empty_key (the pre-fix bug).
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.attestation.payload import (
    AttestationPayload,
    compute_build_secret_response,
    derive_attestation_key,
    sign_attestation_payload,
)
from base.compute.attestation_nonce import NonceBinding
from base.config.settings import ConstationSettings, Settings
from base.db.models import Base
from base.master.constation.allowlist_repository import DigestAllowlistRepository
from base.master.constation.attestation_keys import load_attestation_verify_key
from base.master.constation.nonce_repository import DurableAttestationNonceService
from base.master.constation.routes import create_constation_test_app

TOKEN = "test-internal"
BINDING = NonceBinding(work_unit_id="wu-wire", miner_hotkey="hk-1", pod_id="pod-1")
DIGEST = "sha256:" + ("c" * 64)
MANIFEST = {"harness.py": "d" * 64}
BUILD_SECRET = b"build-secret-fixture"


def test_load_attestation_verify_key_missing_returns_none() -> None:
    """Given no hex setting, When load, Then None (fail-closed → empty_key)."""
    assert load_attestation_verify_key(Settings()) is None
    assert (
        load_attestation_verify_key(Settings(constation=ConstationSettings())) is None
    )
    assert (
        load_attestation_verify_key(
            Settings(constation=ConstationSettings(attestation_verify_key_hex=""))
        )
        is None
    )
    assert (
        load_attestation_verify_key(
            Settings(constation=ConstationSettings(attestation_verify_key_hex="   "))
        )
        is None
    )


def test_load_attestation_verify_key_hex_roundtrip() -> None:
    """Given hex key in settings, When load, Then raw key bytes."""
    key = derive_attestation_key(BUILD_SECRET)
    settings = Settings(
        constation=ConstationSettings(attestation_verify_key_hex=key.hex())
    )
    assert load_attestation_verify_key(settings) == key


def test_main_passes_attestation_verify_key_into_build_constation_router() -> None:
    """Given main.py call site, When AST-inspected, Then key kwarg is wired.

    Drops of ``attestation_verify_key=...`` at the production call site must
    fail this test (regression guard for the empty_key production bug).
    """
    import base.cli_app.main as main_mod

    source = inspect.getsource(main_mod)
    tree = ast.parse(source)
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "build_constation_router":
            continue
        hits.append(node)
    assert hits, "build_constation_router call missing from main"
    for call in hits:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "attestation_verify_key" in kw_names, (
            "build_constation_router must receive attestation_verify_key= "
            f"(got keywords {sorted(kw_names)})"
        )


@pytest.fixture
async def wired_harness(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    """Router built the production way: key from settings, not body key_hex."""
    key = derive_attestation_key(BUILD_SECRET)
    settings = Settings(
        constation=ConstationSettings(attestation_verify_key_hex=key.hex())
    )
    loaded = load_attestation_verify_key(settings)
    assert loaded == key

    db_path = tmp_path / "constation_wire.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app = create_constation_test_app(
        allowlist_repo=DigestAllowlistRepository(factory),
        nonce_service=DurableAttestationNonceService(factory, ttl=timedelta(hours=1)),
        internal_token=TOKEN,
        default_binding=BINDING,
        attestation_verify_key=loaded,
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "headers": {"Authorization": f"Bearer {TOKEN}"},
            "key": key,
        }
    await engine.dispose()


@pytest.fixture
async def unwired_harness(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    """Old bug: router built without attestation_verify_key."""
    db_path = tmp_path / "constation_unwire.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    app = create_constation_test_app(
        allowlist_repo=DigestAllowlistRepository(factory),
        nonce_service=DurableAttestationNonceService(factory, ttl=timedelta(hours=1)),
        internal_token=TOKEN,
        default_binding=BINDING,
        # intentionally omit attestation_verify_key
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {"client": client, "headers": {"Authorization": f"Bearer {TOKEN}"}}
    await engine.dispose()


def _signed_wire(*, key: bytes) -> dict[str, Any]:
    payload = AttestationPayload(
        nonce="nonce-wire-1",
        digest=DIGEST,
        pod_id=BINDING.pod_id,
        variant="cuda",
        sealed_manifest_hashes=dict(MANIFEST),
        build_secret_response=compute_build_secret_response(
            build_secret=BUILD_SECRET, nonce="nonce-wire-1"
        ),
    )
    signed = sign_attestation_payload(payload, signing_key=key)
    return {
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


@pytest.mark.asyncio
async def test_verify_attestation_ok_without_body_key_hex_when_router_wired(
    wired_harness: dict[str, Any],
) -> None:
    """Given settings hex key on router, When verify without key_hex, Then ok."""
    client: AsyncClient = wired_harness["client"]
    headers = wired_harness["headers"]
    key: bytes = wired_harness["key"]
    r = await client.post(
        "/internal/v1/constation/verify_attestation",
        headers=headers,
        json={"signed": _signed_wire(key=key)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["reason"] == "ok"


@pytest.mark.asyncio
async def test_verify_attestation_empty_key_when_router_unwired(
    unwired_harness: dict[str, Any],
) -> None:
    """Given router without key (old bug), When verify sans key_hex, Then empty_key."""
    client: AsyncClient = unwired_harness["client"]
    headers = unwired_harness["headers"]
    key = derive_attestation_key(BUILD_SECRET)
    r = await client.post(
        "/internal/v1/constation/verify_attestation",
        headers=headers,
        json={"signed": _signed_wire(key=key)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "empty_key"
