"""T12 surface prism + HTTP: S3 missing bundle, S6 legacy gate, S8 challenge route."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.attestation.payload import derive_attestation_key
from base.compute.attestation_nonce import NonceBinding
from base.db.models import Base
from base.master.constation.allowlist_repository import DigestAllowlistRepository
from base.master.constation.bundle_store import ConstationBundleStore
from base.master.constation.nonce_repository import DurableAttestationNonceService
from base.master.constation.routes import create_constation_test_app
from tests.surface.constation_surface_helpers import (
    BUILD_SECRET,
    DIGEST,
    HOTKEY,
    POD,
    TOKEN,
    WORK_UNIT,
    ensure_prism_on_path,
)


def test_s3_missing_bundle_prism_behavior() -> None:
    """S3: prism rejects elevation / ingest kwargs without constation_bundle."""
    if not ensure_prism_on_path():
        pytest.skip("prism_challenge not importable (packages/challenges/prism/src)")

    from prism_challenge.app import _constation_ingest_kwargs
    from prism_challenge.audit import effective_tier
    from prism_challenge.config import PrismSettings
    from prism_challenge.ingestion import miner_fault_reason
    from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature

    settings = PrismSettings(
        allow_insecure_signatures=False,
        constation_base_url="http://base.surface.test",
        constation_internal_token="tok",
    )
    assert _constation_ingest_kwargs(settings, {"executed": 1}) == {}

    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=DIGEST,
        provider=ProviderInfo(name="lium", pod_id=POD),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
    )
    assert effective_tier(proof, pinned_image_digest=DIGEST) == 0
    assert (
        effective_tier(proof, pinned_image_digest=DIGEST, constation_ok_result=False)
        == 0
    )
    assert miner_fault_reason("missing_constation_bundle") == (
        "miner_fault:missing_constation_bundle"
    )


def test_s6_legacy_gate_no_elevation_without_constation() -> None:
    """S6: self-report digest match alone cannot elevate (tier stays 0)."""
    if not ensure_prism_on_path():
        pytest.skip("prism_challenge not importable (packages/challenges/prism/src)")

    from prism_challenge.audit import effective_tier
    from prism_challenge.proof import ExecutionProof, ProviderInfo, WorkerSignature

    proof = ExecutionProof(
        version=1,
        tier=1,
        manifest_sha256="c" * 64,
        image_digest=DIGEST,
        provider=ProviderInfo(name="lium", pod_id=POD),
        worker_signature=WorkerSignature(worker_pubkey="wk", sig="0xab"),
    )
    assert effective_tier(proof, pinned_image_digest=DIGEST) == 0
    assert (
        effective_tier(proof, pinned_image_digest=DIGEST, constation_ok_result=None)
        == 0
    )
    assert (
        effective_tier(proof, pinned_image_digest=DIGEST, constation_ok_result=True)
        == 1
    )


@pytest.fixture
async def challenge_harness(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    db_path = tmp_path / "surface_challenge.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    binding = NonceBinding(work_unit_id=WORK_UNIT, miner_hotkey=HOTKEY, pod_id=POD)
    key = derive_attestation_key(BUILD_SECRET)
    app = create_constation_test_app(
        allowlist_repo=DigestAllowlistRepository(factory),
        nonce_service=DurableAttestationNonceService(factory, ttl=timedelta(hours=1)),
        internal_token=TOKEN,
        default_binding=binding,
        attestation_verify_key=key,
        bundle_store=ConstationBundleStore(),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield {
            "client": client,
            "headers": {"Authorization": f"Bearer {TOKEN}"},
            "key": key,
        }
    await engine.dispose()


@pytest.mark.asyncio
async def test_s8_challenge_route_smoke(challenge_harness: dict[str, Any]) -> None:
    """S8: GET challenge + POST answer + issue_nonce smoke."""
    client: AsyncClient = challenge_harness["client"]
    headers: dict[str, str] = challenge_harness["headers"]

    r = await client.get("/v1/attestation/challenge", params={"phase": "start"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["nonce"]
    assert body["phase"] == "start"
    assert body["work_unit_id"] == WORK_UNIT

    ans = await client.post(
        "/v1/attestation/answer",
        json={"nonce": body["nonce"], "phase": "start"},
    )
    assert ans.status_code == 200
    assert ans.json()["status"] == "accepted"

    issued = await client.post(
        "/internal/v1/constation/issue_nonce",
        headers=headers,
        json={
            "work_unit_id": WORK_UNIT,
            "miner_hotkey": HOTKEY,
            "pod_id": POD,
            "phase": "end",
        },
    )
    assert issued.status_code == 200, issued.text
    assert issued.json()["nonce"]
    assert issued.json()["phase"] == "end"
