"""Worker-plane unit-status read endpoint (master-unit-status-read-endpoint).

Makes the dispute -> audit -> invalidation -> fault chain OPERATOR-DISCOVERABLE
VIA API ALONE (VAL-CROSS-011). ``GET /v1/workers/units`` is a signed read-only
master surface (auth: ``CoordinationReadEligibility`` -- same as
``GET /v1/workers``, NOT the internal bridge bearer) that exposes, per primary
gpu unit:

* the unit id + status (INCLUDING ``disputed``);
* its replicas (worker_id, owner miner hotkey, posted manifest_sha256, proof
  presence);
* for a disputed unit, the linked validator AUDIT unit's id, executor kind
  (``validator``), and terminal outcome (pending/passed/mismatch-resolved).

Flag OFF => the router is unmounted (404). Signed-request auth is enforced
(bad/missing signature => 401/403) and the internal bridge bearer is NOT accepted
here (that acceptor is only on the narrow admission fleet-read).
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    WorkAssignment,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerFault,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.assignment import (
    EXECUTOR_KIND_PAYLOAD_KEY,
    EXECUTOR_KIND_VALIDATOR,
)
from base.master.worker_reconciliation import (
    AUDIT_OF_PAYLOAD_KEY,
    AUDIT_RESOLVED_PAYLOAD_KEY,
    audit_work_unit_id,
)
from base.master.worker_unit_status import (
    AUDIT_OUTCOME_MISMATCH_RESOLVED,
    AUDIT_OUTCOME_PASSED,
    AUDIT_OUTCOME_PENDING,
    WorkerUnitStatusService,
)
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
NOW_EPOCH = NOW.timestamp()
TTL = 120

MINER_A = "miner-A"
MINER_B = "miner-B"
MINER_H = "miner-H"
VALIDATOR = "val-permit"
STRANGER = "stranger-key"

HASH_A = "a" * 64
HASH_B = "b" * 64
INTERNAL_BRIDGE_TOKEN = "prism-bridge-shared-token"


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


class _TokenRegistry:
    def __init__(self, token: str) -> None:
        self._token = token

    def get_token(self, slug: str) -> str:
        if slug == "prism":
            return self._token
        raise RuntimeError(f"no token for {slug!r}")


class FakeNonceStore:
    async def reserve(self, **_kwargs: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _proof_payload(manifest: str) -> dict[str, Any]:
    return {
        PROOF_PAYLOAD_KEY: {
            "version": 1,
            "tier": 0,
            MANIFEST_SHA256_PAYLOAD_KEY: manifest,
        },
        MANIFEST_SHA256_PAYLOAD_KEY: manifest,
    }


async def _add_primary(
    factory: Any,
    *,
    work_unit_id: str,
    submitter: str,
    status: WorkAssignmentStatus,
    challenge_slug: str = "prism",
) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkAssignment(
                challenge_slug=challenge_slug,
                work_unit_id=work_unit_id,
                submission_ref=submitter,
                payload={"run_spec": {"image": "img"}},
                required_capability="gpu",
                status=status,
                attempt_count=1,
                max_attempts=3,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_audit_unit(
    factory: Any,
    *,
    original_id: str,
    status: WorkAssignmentStatus,
    resolved: bool,
    challenge_slug: str = "prism",
) -> None:
    payload: dict[str, Any] = {
        EXECUTOR_KIND_PAYLOAD_KEY: EXECUTOR_KIND_VALIDATOR,
        AUDIT_OF_PAYLOAD_KEY: original_id,
    }
    if resolved:
        payload[AUDIT_RESOLVED_PAYLOAD_KEY] = True
    async with session_scope(factory) as session:
        session.add(
            WorkAssignment(
                challenge_slug=challenge_slug,
                work_unit_id=audit_work_unit_id(original_id),
                submission_ref=MINER_H,
                payload=payload,
                required_capability="gpu",
                status=status,
                attempt_count=0,
                max_attempts=3,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_replica(
    factory: Any,
    *,
    work_unit_id: str,
    worker_id: str,
    miner_hotkey: str,
    manifest: str | None,
    status: WorkAssignmentStatus = WorkAssignmentStatus.COMPLETED,
    challenge_slug: str = "prism",
) -> None:
    payload = _proof_payload(manifest) if manifest is not None else None
    async with session_scope(factory) as session:
        session.add(
            WorkerAssignment(
                challenge_slug=challenge_slug,
                work_unit_id=work_unit_id,
                submission_ref=MINER_H,
                worker_id=worker_id,
                worker_pubkey=f"wp-{worker_id}",
                miner_hotkey=miner_hotkey,
                payload={},
                required_capability="gpu",
                status=status,
                attempt_count=1,
                max_attempts=3,
                result_success=manifest is not None,
                result_payload=payload,
                manifest_sha256=manifest,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_fault(
    factory: Any,
    *,
    worker_id: str,
    work_unit_id: str,
    challenge_slug: str = "prism",
) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkerFault(
                worker_id=worker_id,
                work_unit_id=work_unit_id,
                challenge_slug=challenge_slug,
                detail="manifest diverged from validator audit",
                created_at=NOW,
            )
        )


async def _build_app(
    *, mount: bool = True, internal_token: str | None = None
) -> tuple[AsyncClient, Any, WorkerUnitStatusService, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_A, MINER_B, MINER_H, VALIDATOR],
        validator_permits=[False, False, False, True],
        stakes=[0.0, 0.0, 0.0, 100.0],
    )
    verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(factory),
        eligibility=CoordinationReadEligibility(factory, cache),
        signature_verifier=_fake_verifier,
        ttl_seconds=300,
        now_fn=lambda: NOW_EPOCH,
    )
    service = WorkerUnitStatusService(factory)

    registry: Any = object()
    if internal_token is not None:
        registry = _TokenRegistry(internal_token)

    app = create_proxy_app(
        registry=registry,
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        worker_verifier=verifier if mount else None,
        worker_unit_status_service=service if mount else None,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return client, factory, service, engine


def _signed_headers(
    *, hotkey: str, nonce: str, sign_hotkey: str | None = None
) -> dict[str, str]:
    ts = str(int(NOW_EPOCH))
    canonical = canonical_validator_request(
        method="GET",
        path="/v1/workers/units",
        query_string="",
        timestamp=ts,
        nonce=nonce,
        body=b"",
    )
    signer = sign_hotkey if sign_hotkey is not None else hotkey
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(signer, canonical.encode()),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


@pytest.fixture
async def env() -> AsyncIterator[tuple[AsyncClient, Any, WorkerUnitStatusService]]:
    client, factory, service, engine = await _build_app()
    try:
        yield client, factory, service
    finally:
        await client.aclose()
        await engine.dispose()


# --- service-level assertions --------------------------------------------------


async def test_matched_unit_shows_completed_with_both_replica_hashes(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    _, factory, service = env
    await _add_primary(
        factory,
        work_unit_id="U-ok",
        submitter=MINER_H,
        status=WorkAssignmentStatus.COMPLETED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-ok",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )
    await _add_replica(
        factory,
        work_unit_id="U-ok",
        worker_id="wb",
        miner_hotkey=MINER_B,
        manifest=HASH_A,
    )

    units = await service.list_units()
    assert len(units) == 1
    unit = units[0]
    assert unit.work_unit_id == "U-ok"
    assert unit.status == WorkAssignmentStatus.COMPLETED.value
    assert unit.audit is None
    assert {r.miner_hotkey for r in unit.replicas} == {MINER_A, MINER_B}
    assert [r.manifest_sha256 for r in unit.replicas] == [HASH_A, HASH_A]
    assert all(r.has_proof for r in unit.replicas)
    assert {r.worker_id for r in unit.replicas} == {"wa", "wb"}


async def test_disputed_unit_shows_audit_pending_before_resolution(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    _, factory, service = env
    await _add_primary(
        factory,
        work_unit_id="U-dis",
        submitter=MINER_H,
        status=WorkAssignmentStatus.DISPUTED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wb",
        miner_hotkey=MINER_B,
        manifest=HASH_B,
    )
    await _add_audit_unit(
        factory,
        original_id="U-dis",
        status=WorkAssignmentStatus.PENDING,
        resolved=False,
    )

    units = await service.list_units()
    # The validator audit unit is nested, never a top-level primary entry.
    assert [u.work_unit_id for u in units] == ["U-dis"]
    unit = units[0]
    assert unit.status == WorkAssignmentStatus.DISPUTED.value
    assert unit.audit is not None
    assert unit.audit.work_unit_id == audit_work_unit_id("U-dis")
    assert unit.audit.executor_kind == EXECUTOR_KIND_VALIDATOR
    assert unit.audit.outcome == AUDIT_OUTCOME_PENDING
    assert {r.manifest_sha256 for r in unit.replicas} == {HASH_A, HASH_B}


async def test_disputed_unit_audit_mismatch_resolved_when_fault_recorded(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    _, factory, service = env
    await _add_primary(
        factory,
        work_unit_id="U-dis",
        submitter=MINER_H,
        status=WorkAssignmentStatus.DISPUTED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wb",
        miner_hotkey=MINER_B,
        manifest=HASH_B,
    )
    await _add_audit_unit(
        factory,
        original_id="U-dis",
        status=WorkAssignmentStatus.COMPLETED,
        resolved=True,
    )
    await _add_fault(factory, worker_id="wb", work_unit_id="U-dis")

    units = await service.list_units()
    unit = units[0]
    assert unit.audit is not None
    assert unit.audit.outcome == AUDIT_OUTCOME_MISMATCH_RESOLVED


async def test_disputed_unit_audit_passed_when_resolved_without_fault(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    _, factory, service = env
    await _add_primary(
        factory,
        work_unit_id="U-dis",
        submitter=MINER_H,
        status=WorkAssignmentStatus.DISPUTED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )
    await _add_replica(
        factory,
        work_unit_id="U-dis",
        worker_id="wb",
        miner_hotkey=MINER_B,
        manifest=HASH_B,
    )
    await _add_audit_unit(
        factory,
        original_id="U-dis",
        status=WorkAssignmentStatus.COMPLETED,
        resolved=True,
    )

    units = await service.list_units()
    unit = units[0]
    assert unit.audit is not None
    assert unit.audit.outcome == AUDIT_OUTCOME_PASSED


async def test_fault_keyed_by_challenge_and_unit_does_not_collide_across_slugs(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    """Two disputed units sharing a work_unit_id under DIFFERENT challenge slugs
    must not collide: a fault recorded for one slug's unit must not flip the
    other slug's (fault-free) audit to ``mismatch-resolved``."""

    _, factory, service = env
    shared_unit_id = "U-shared"
    for slug in ("prism", "other-challenge"):
        await _add_primary(
            factory,
            work_unit_id=shared_unit_id,
            submitter=MINER_H,
            status=WorkAssignmentStatus.DISPUTED,
            challenge_slug=slug,
        )
        await _add_replica(
            factory,
            work_unit_id=shared_unit_id,
            worker_id=f"wa-{slug}",
            miner_hotkey=MINER_A,
            manifest=HASH_A,
            challenge_slug=slug,
        )
        await _add_replica(
            factory,
            work_unit_id=shared_unit_id,
            worker_id=f"wb-{slug}",
            miner_hotkey=MINER_B,
            manifest=HASH_B,
            challenge_slug=slug,
        )
        await _add_audit_unit(
            factory,
            original_id=shared_unit_id,
            status=WorkAssignmentStatus.COMPLETED,
            resolved=True,
            challenge_slug=slug,
        )
    # Only the prism unit's diverging worker is faulted.
    await _add_fault(
        factory,
        worker_id="wb-prism",
        work_unit_id=shared_unit_id,
        challenge_slug="prism",
    )

    units = await service.list_units()
    by_slug = {u.challenge_slug: u for u in units}
    assert by_slug["prism"].audit is not None
    assert by_slug["prism"].audit.outcome == AUDIT_OUTCOME_MISMATCH_RESOLVED
    assert by_slug["other-challenge"].audit is not None
    assert by_slug["other-challenge"].audit.outcome == AUDIT_OUTCOME_PASSED


async def test_replica_without_proof_reports_absent_proof(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    _, factory, service = env
    await _add_primary(
        factory,
        work_unit_id="U-mix",
        submitter=MINER_H,
        status=WorkAssignmentStatus.ASSIGNED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-mix",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )
    await _add_replica(
        factory,
        work_unit_id="U-mix",
        worker_id="wb",
        miner_hotkey=MINER_B,
        manifest=None,
        status=WorkAssignmentStatus.RUNNING,
    )

    unit = (await service.list_units())[0]
    by_worker = {r.worker_id: r for r in unit.replicas}
    assert by_worker["wa"].has_proof is True
    assert by_worker["wa"].manifest_sha256 == HASH_A
    assert by_worker["wb"].has_proof is False
    assert by_worker["wb"].manifest_sha256 is None


# --- HTTP surface + auth -------------------------------------------------------


async def test_signed_request_returns_units(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    client, factory, _ = env
    await _add_primary(
        factory,
        work_unit_id="U-ok",
        submitter=MINER_H,
        status=WorkAssignmentStatus.COMPLETED,
    )
    await _add_replica(
        factory,
        work_unit_id="U-ok",
        worker_id="wa",
        miner_hotkey=MINER_A,
        manifest=HASH_A,
    )

    resp = await client.get(
        "/v1/workers/units",
        headers=_signed_headers(hotkey=VALIDATOR, nonce="units-1"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [u["work_unit_id"] for u in body["units"]] == ["U-ok"]
    assert body["units"][0]["status"] == "completed"


async def test_missing_signature_rejected(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    client, _, _ = env
    resp = await client.get("/v1/workers/units")
    assert resp.status_code in (401, 403)


async def test_bad_signature_rejected(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    client, _, _ = env
    resp = await client.get(
        "/v1/workers/units",
        headers=_signed_headers(
            hotkey=VALIDATOR, nonce="units-bad", sign_hotkey=STRANGER
        ),
    )
    assert resp.status_code in (401, 403)


async def test_unregistered_identity_rejected(
    env: tuple[AsyncClient, Any, WorkerUnitStatusService],
) -> None:
    client, _, _ = env
    resp = await client.get(
        "/v1/workers/units",
        headers=_signed_headers(hotkey=STRANGER, nonce="units-str"),
    )
    assert resp.status_code in (401, 403)


async def test_internal_bridge_bearer_not_accepted() -> None:
    client, factory, _service, engine = await _build_app(
        internal_token=INTERNAL_BRIDGE_TOKEN
    )
    try:
        resp = await client.get(
            "/v1/workers/units",
            headers={"Authorization": f"Bearer {INTERNAL_BRIDGE_TOKEN}"},
        )
        assert resp.status_code in (401, 403)
    finally:
        await client.aclose()
        await engine.dispose()


async def test_flag_off_router_unmounted_404() -> None:
    client, _factory, _service, engine = await _build_app(mount=False)
    try:
        resp = await client.get(
            "/v1/workers/units",
            headers=_signed_headers(hotkey=VALIDATOR, nonce="off-1"),
        )
        assert resp.status_code == 404
    finally:
        await client.aclose()
        await engine.dispose()
