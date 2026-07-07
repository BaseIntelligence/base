"""Worker-plane unit-status read endpoint over real postgres (15433).

master-unit-status-read-endpoint / VAL-CROSS-011: the full dispute -> audit ->
invalidation -> fault chain must be operator-discoverable via ``GET
/v1/workers/units`` alone. These tests drive the REAL reconciliation machinery
(``WorkerAssignmentEngine`` + ``WorkerReconciliationService`` + the validator
plane) against the mission test postgres, then read the chain back through the
service and the signed HTTP surface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Validator,
    ValidatorStatus,
    WorkAssignment,
    WorkAssignmentStatus,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import (
    WorkerReconciliationService,
    audit_work_unit_id,
)
from base.master.worker_unit_status import (
    AUDIT_OUTCOME_MISMATCH_RESOLVED,
    AUDIT_OUTCOME_PENDING,
    WorkerUnitStatusService,
)
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
)
from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY, PROOF_PAYLOAD_KEY

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
NOW_EPOCH = NOW.timestamp()
TTL = 120

MINER_A = "us-miner-A"
MINER_B = "us-miner-B"
MINER_H = "us-miner-H"
VALIDATOR = "us-val-permit"
STRANGER = "us-stranger"
GPU_VALIDATOR = "us-gpu-val"

HASH_A = "a" * 64
HASH_B = "b" * 64
INTERNAL_BRIDGE_TOKEN = "us-prism-bridge-token"


class _Clock:
    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def now(self) -> datetime:
        return self.moment


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


def _proof_payload(manifest: str) -> dict[str, Any]:
    return {
        PROOF_PAYLOAD_KEY: {
            "version": 1,
            "tier": 0,
            MANIFEST_SHA256_PAYLOAD_KEY: manifest,
        },
        MANIFEST_SHA256_PAYLOAD_KEY: manifest,
    }


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


async def _add_worker(factory: Any, *, worker_pubkey: str, miner_hotkey: str) -> str:
    worker_id = f"wid-{worker_pubkey}"
    async with session_scope(factory) as session:
        session.add(
            WorkerRegistration(
                worker_id=worker_id,
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                binding_signature="sig",
                binding_nonce=f"nonce-{worker_pubkey}",
                provider="local",
                provider_instance_ref="local-1",
                capabilities=["gpu"],
                status=WorkerStatus.ACTIVE,
                last_heartbeat_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    return worker_id


async def _add_gpu_unit(factory: Any, *, work_unit_id: str, submitter: str) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkAssignment(
                challenge_slug="prism",
                work_unit_id=work_unit_id,
                submission_ref=submitter,
                payload={"run_spec": {"image": "img"}},
                required_capability="gpu",
                status=WorkAssignmentStatus.PENDING,
                attempt_count=0,
                max_attempts=3,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_gpu_validator(factory: Any) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=GPU_VALIDATOR,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=["gpu"],
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _replicas(factory: Any, work_unit_id: str) -> list[Any]:
    from base.db import WorkerAssignment

    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment).where(
                        WorkerAssignment.work_unit_id == work_unit_id
                    )
                )
            )
            .scalars()
            .all()
        )
        return list(rows)


async def _post_replica(
    svc: WorkerAssignmentService,
    factory: Any,
    *,
    work_unit_id: str,
    miner_hotkey: str,
    manifest: str,
) -> None:
    replica = next(
        r
        for r in await _replicas(factory, work_unit_id)
        if r.miner_hotkey == miner_hotkey
    )
    await svc.post_result(
        assignment_id=str(replica.id),
        worker_pubkey=replica.worker_pubkey,
        success=True,
        payload=_proof_payload(manifest),
    )


def _build_services(factory: Any, clock: _Clock) -> dict[str, Any]:
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=clock.now,
    )
    worker_assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=clock.now
    )
    engine = WorkerAssignmentEngine(
        factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=clock.now,
    )
    reconciler = WorkerReconciliationService(factory, now_fn=clock.now)
    validator_plane = AssignmentService(
        factory,
        now_fn=clock.now,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    validator_coordination = AssignmentCoordinationService(factory, now_fn=clock.now)
    status_service = WorkerUnitStatusService(factory)
    return {
        "worker_service": worker_service,
        "worker_assignment_service": worker_assignment_service,
        "engine": engine,
        "reconciler": reconciler,
        "validator_plane": validator_plane,
        "validator_coordination": validator_coordination,
        "status_service": status_service,
    }


async def test_disputed_unit_discoverable_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine)
    clock = _Clock(NOW)
    svc = _build_services(factory, clock)
    try:
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
        await _add_gpu_unit(factory, work_unit_id="U", submitter=MINER_H)
        await svc["engine"].assign_pending(seed=1)

        await _post_replica(
            svc["worker_assignment_service"],
            factory,
            work_unit_id="U",
            miner_hotkey=MINER_A,
            manifest=HASH_A,
        )
        await _post_replica(
            svc["worker_assignment_service"],
            factory,
            work_unit_id="U",
            miner_hotkey=MINER_B,
            manifest=HASH_B,
        )
        await svc["reconciler"].reconcile_once()

        # Before audit resolution: disputed + linked validator audit unit pending.
        units = await svc["status_service"].list_units()
        assert [u.work_unit_id for u in units] == ["U"]
        unit = units[0]
        assert unit.status == WorkAssignmentStatus.DISPUTED.value
        assert {r.manifest_sha256 for r in unit.replicas} == {HASH_A, HASH_B}
        assert all(r.has_proof for r in unit.replicas)
        assert unit.audit is not None
        assert unit.audit.work_unit_id == audit_work_unit_id("U")
        assert unit.audit.executor_kind == "validator"
        assert unit.audit.outcome == AUDIT_OUTCOME_PENDING

        # Validator replays the audit (authoritative HASH_A); reconcile resolves it.
        await _add_gpu_validator(factory)
        await svc["validator_plane"].assign_pending(seed=1)
        async with factory() as session:
            audit = (
                await session.execute(
                    select(WorkAssignment).where(
                        WorkAssignment.work_unit_id == audit_work_unit_id("U")
                    )
                )
            ).scalar_one()
        await svc["validator_coordination"].post_result(
            assignment_id=str(audit.id),
            hotkey=GPU_VALIDATOR,
            success=True,
            payload=_proof_payload(HASH_A),
        )
        await svc["reconciler"].reconcile_once()

        units = await svc["status_service"].list_units()
        unit = units[0]
        assert unit.status == WorkAssignmentStatus.DISPUTED.value
        assert unit.audit is not None
        assert unit.audit.executor_kind == "validator"
        assert unit.audit.outcome == AUDIT_OUTCOME_MISMATCH_RESOLVED
    finally:
        await engine.dispose()


async def test_matched_unit_accepted_with_both_hashes_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine)
    clock = _Clock(NOW)
    svc = _build_services(factory, clock)
    try:
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey=MINER_A)
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey=MINER_B)
        await _add_gpu_unit(factory, work_unit_id="U", submitter=MINER_H)
        await svc["engine"].assign_pending(seed=1)

        await _post_replica(
            svc["worker_assignment_service"],
            factory,
            work_unit_id="U",
            miner_hotkey=MINER_A,
            manifest=HASH_A,
        )
        await _post_replica(
            svc["worker_assignment_service"],
            factory,
            work_unit_id="U",
            miner_hotkey=MINER_B,
            manifest=HASH_A,
        )
        await svc["reconciler"].reconcile_once()

        units = await svc["status_service"].list_units()
        assert [u.work_unit_id for u in units] == ["U"]
        unit = units[0]
        assert unit.status == WorkAssignmentStatus.COMPLETED.value
        assert unit.audit is None
        assert [r.manifest_sha256 for r in unit.replicas] == [HASH_A, HASH_A]
        assert all(r.has_proof for r in unit.replicas)
        assert {r.miner_hotkey for r in unit.replicas} == {MINER_A, MINER_B}
    finally:
        await engine.dispose()


def _signed_headers(*, hotkey: str, nonce: str) -> dict[str, str]:
    ts = str(int(NOW_EPOCH))
    canonical = canonical_validator_request(
        method="GET",
        path="/v1/workers/units",
        query_string="",
        timestamp=ts,
        nonce=nonce,
        body=b"",
    )
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(hotkey, canonical.encode()),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


async def test_unit_status_http_signed_auth_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine)
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_A, VALIDATOR],
        validator_permits=[False, True],
        stakes=[0.0, 100.0],
    )
    verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(factory),
        eligibility=CoordinationReadEligibility(factory, cache),
        signature_verifier=_fake_verifier,
        ttl_seconds=300,
        now_fn=lambda: NOW_EPOCH,
    )
    app = create_proxy_app(
        registry=_TokenRegistry(INTERNAL_BRIDGE_TOKEN),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        worker_verifier=verifier,
        worker_unit_status_service=WorkerUnitStatusService(factory),
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    try:
        await _add_gpu_unit(factory, work_unit_id="U", submitter=MINER_H)

        # (a) valid signed request from an eligible validator succeeds.
        ok = await client.get(
            "/v1/workers/units",
            headers=_signed_headers(hotkey=VALIDATOR, nonce="u-ok"),
        )
        assert ok.status_code == 200
        assert [u["work_unit_id"] for u in ok.json()["units"]] == ["U"]

        # (b) missing signature => 401/403.
        missing = await client.get("/v1/workers/units")
        assert missing.status_code in (401, 403)

        # (c) internal bridge bearer is NOT accepted on this surface.
        bearer = await client.get(
            "/v1/workers/units",
            headers={"Authorization": f"Bearer {INTERNAL_BRIDGE_TOKEN}"},
        )
        assert bearer.status_code in (401, 403)
    finally:
        await client.aclose()
        await engine.dispose()
