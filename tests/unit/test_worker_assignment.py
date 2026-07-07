"""Worker assignment surface: gpu-only pull, proof-carrying result, worker auth.

Covers the worker-plane pull/result routes against the real proxy app (mock
metagraph, ASGI transport):

* VAL-AGENT-007: an active worker pulls only gpu-capability replicas.
* VAL-AGENT-008: a posted result carries and persists the ExecutionProof.
* VAL-AGENT-018: pull/post authenticate as the WORKER identity and gate on
  registration/liveness, never on a metagraph validator permit.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.app_proxy import create_proxy_app
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_coordination import WorkerCoordinationService
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    RegisteredWorkerEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
    worker_binding_message,
)

NOW_EPOCH = 1_750_000_000.0
TTL_SECONDS = 120

MINER_H1 = "miner-H1"
MINER_H2 = "miner-H2"
VALIDATOR = "val-permit"
STRANGER = "stranger-key"


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)


class FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


def _signed_headers(
    *,
    method: str,
    path: str,
    body: bytes,
    hotkey: str,
    nonce: str,
    timestamp: float,
) -> dict[str, str]:
    ts = str(int(timestamp))
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string="",
        timestamp=ts,
        nonce=nonce,
        body=body,
    )
    return {
        "X-Hotkey": hotkey,
        "X-Signature": _sign(hotkey, canonical.encode()),
        "X-Nonce": nonce,
        "X-Timestamp": ts,
        "Content-Type": "application/json",
    }


class Harness:
    def __init__(
        self,
        client: AsyncClient,
        session_factory: Any,
        clock: FakeClock,
        service: WorkerCoordinationService,
        assignment_service: WorkerAssignmentService,
    ) -> None:
        self.client = client
        self.session_factory = session_factory
        self.clock = clock
        self.service = service
        self.assignment_service = assignment_service
        self._nonce = 0

    def _next_nonce(self, prefix: str) -> str:
        self._nonce += 1
        return f"{prefix}-{self._nonce}"

    async def enroll_active(self, *, worker_pubkey: str, miner_hotkey: str) -> str:
        """Register + heartbeat a worker so it is ACTIVE; return its worker_id."""

        nonce = self._next_nonce("bind")
        message = worker_binding_message(
            worker_pubkey=worker_pubkey, miner_hotkey=miner_hotkey, nonce=nonce
        )
        resp = await self.client.post(
            "/v1/workers/register",
            json={
                "worker_pubkey": worker_pubkey,
                "miner_hotkey": miner_hotkey,
                "binding_signature": _sign(miner_hotkey, message),
                "nonce": nonce,
                "provider": "local",
                "provider_instance_ref": "local-1",
            },
        )
        assert resp.status_code == 200, resp.text
        worker_id = resp.json()["worker"]["worker_id"]
        hb = await self.heartbeat(worker_id=worker_id, signer_pubkey=worker_pubkey)
        assert hb.status_code == 200, hb.text
        return worker_id

    async def register_pending(self, *, worker_pubkey: str, miner_hotkey: str) -> str:
        nonce = self._next_nonce("bind")
        message = worker_binding_message(
            worker_pubkey=worker_pubkey, miner_hotkey=miner_hotkey, nonce=nonce
        )
        resp = await self.client.post(
            "/v1/workers/register",
            json={
                "worker_pubkey": worker_pubkey,
                "miner_hotkey": miner_hotkey,
                "binding_signature": _sign(miner_hotkey, message),
                "nonce": nonce,
                "provider": "local",
                "provider_instance_ref": "local-1",
            },
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["worker"]["worker_id"]

    async def heartbeat(self, *, worker_id: str, signer_pubkey: str) -> Any:
        body = b"{}"
        path = f"/v1/workers/{worker_id}/heartbeat"
        headers = _signed_headers(
            method="POST",
            path=path,
            body=body,
            hotkey=signer_pubkey,
            nonce=self._next_nonce("hb"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def pull(self, *, signer_pubkey: str) -> Any:
        body = b"{}"
        path = "/v1/workers/assignments/pull"
        headers = _signed_headers(
            method="POST",
            path=path,
            body=body,
            hotkey=signer_pubkey,
            nonce=self._next_nonce("pull"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def post_result(
        self,
        *,
        assignment_id: str,
        signer_pubkey: str,
        success: bool,
        payload: dict[str, Any],
    ) -> Any:
        import json

        body = json.dumps(
            {"success": success, "payload": payload}, separators=(",", ":")
        ).encode()
        path = f"/v1/workers/assignments/{assignment_id}/result"
        headers = _signed_headers(
            method="POST",
            path=path,
            body=body,
            hotkey=signer_pubkey,
            nonce=self._next_nonce("res"),
            timestamp=self.clock.time(),
        )
        return await self.client.post(path, content=body, headers=headers)

    async def seed_assignment(
        self,
        *,
        work_unit_id: str,
        worker_id: str,
        worker_pubkey: str,
        miner_hotkey: str,
        required_capability: str = "gpu",
    ) -> WorkerAssignment:
        return await self.assignment_service.create_worker_assignment(
            work_unit_id=work_unit_id,
            challenge_slug="prism",
            submission_ref=f"ref-{work_unit_id}",
            worker_id=worker_id,
            worker_pubkey=worker_pubkey,
            miner_hotkey=miner_hotkey,
            payload={"run_spec": {"image": "img", "command": ["run"]}},
            required_capability=required_capability,
        )

    async def row(self, assignment_id: str) -> WorkerAssignment | None:
        import uuid

        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(WorkerAssignment).where(
                        WorkerAssignment.id == uuid.UUID(assignment_id)
                    )
                )
            ).scalar_one_or_none()

    async def set_status(self, worker_pubkey: str, status: WorkerStatus) -> None:
        from base.db import WorkerRegistration

        async with session_scope(self.session_factory) as session:
            row = (
                await session.execute(
                    select(WorkerRegistration).where(
                        WorkerRegistration.worker_pubkey == worker_pubkey
                    )
                )
            ).scalar_one()
            row.status = status


async def _build_harness() -> tuple[Harness, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER_H1, MINER_H2, VALIDATOR],
        validator_permits=[False, False, True],
        stakes=[0.0, 0.0, 100.0],
    )
    clock = FakeClock(NOW_EPOCH)
    service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        signature_verifier=_fake_verifier,
        heartbeat_ttl_seconds=TTL_SECONDS,
        now_fn=clock.now,
    )
    verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        eligibility=CoordinationReadEligibility(session_factory, cache),
        signature_verifier=_fake_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )
    assignment_service = WorkerAssignmentService(
        session_factory, worker_service=service, now_fn=clock.now
    )
    # The assignment surface uses a WORKER-ONLY verifier (no validator permit).
    assignment_verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        eligibility=RegisteredWorkerEligibility(session_factory),
        signature_verifier=_fake_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )

    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        worker_service=service,
        worker_verifier=verifier,
        worker_assignment_service=assignment_service,
        worker_assignment_verifier=assignment_verifier,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    return Harness(client, session_factory, clock, service, assignment_service), engine


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h, engine = await _build_harness()
    try:
        yield h
    finally:
        await h.client.aclose()
        await engine.dispose()


# VAL-AGENT-007
async def test_pull_returns_only_gpu_units(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    gpu = await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
        required_capability="gpu",
    )
    await harness.seed_assignment(
        work_unit_id="cpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
        required_capability="cpu",
    )

    resp = await harness.pull(signer_pubkey="wp-1")
    assert resp.status_code == 200, resp.text
    assignments = resp.json()["assignments"]
    assert [a["work_unit_id"] for a in assignments] == ["gpu-unit"]
    assert assignments[0]["required_capability"] == "gpu"
    assert assignments[0]["status"] == "running"

    # The gpu replica transitioned assigned -> running; the cpu one is untouched.
    gpu_row = await harness.row(str(gpu.id))
    assert gpu_row is not None
    assert WorkAssignmentStatus(gpu_row.status) == WorkAssignmentStatus.RUNNING


# VAL-AGENT-008
async def test_posted_result_carries_and_persists_execution_proof(
    harness: Harness,
) -> None:
    import bittensor as bt

    from base.validator.agent.signing import KeypairRequestSigner
    from base.worker.proof import build_execution_proof, verify_execution_proof

    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    unit = await harness.seed_assignment(
        work_unit_id="unit-proof",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    await harness.pull(signer_pubkey="wp-1")

    proof_signer = KeypairRequestSigner(bt.Keypair.create_from_uri("//Worker1"))
    proof = build_execution_proof(
        signer=proof_signer,
        manifest_sha256="d" * 64,
        unit_id="unit-proof",
    )
    resp = await harness.post_result(
        assignment_id=str(unit.id),
        signer_pubkey="wp-1",
        success=True,
        payload={"execution_proof": proof.model_dump(mode="json")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"

    row = await harness.row(str(unit.id))
    assert row is not None
    assert row.result_success is True
    assert row.result_payload is not None
    stored_proof = row.result_payload["execution_proof"]
    assert stored_proof["version"] == 1
    assert row.manifest_sha256 == "d" * 64
    # The stored worker signature verifies against the pinned message format.
    from base.schemas.worker import ExecutionProof

    parsed = ExecutionProof.model_validate(stored_proof)
    assert verify_execution_proof(parsed, unit_id="unit-proof") is True


# VAL-AGENT-018 (a): unregistered key cannot pull or post
async def test_unregistered_key_pull_rejected(harness: Harness) -> None:
    resp = await harness.pull(signer_pubkey=STRANGER)
    assert resp.status_code == 403


async def test_validator_permit_only_key_cannot_pull(harness: Harness) -> None:
    # VAL-AGENT-018 (d): a metagraph validator permit is NOT a worker credential.
    resp = await harness.pull(signer_pubkey=VALIDATOR)
    assert resp.status_code == 403


# VAL-AGENT-018 (b): stale/retired worker receives no NEW units
async def test_stale_worker_receives_no_units(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    # advance clock beyond TTL so the worker is effectively stale
    harness.clock.epoch = NOW_EPOCH + TTL_SECONDS + 10
    resp = await harness.pull(signer_pubkey="wp-1")
    assert resp.status_code == 200
    assert resp.json()["assignments"] == []


async def test_retired_worker_receives_no_units(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    await harness.set_status("wp-1", WorkerStatus.RETIRED)
    resp = await harness.pull(signer_pubkey="wp-1")
    assert resp.status_code == 200
    assert resp.json()["assignments"] == []


async def test_pending_worker_receives_no_units(harness: Harness) -> None:
    worker_id = await harness.register_pending(
        worker_pubkey="wp-1", miner_hotkey=MINER_H1
    )
    await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    resp = await harness.pull(signer_pubkey="wp-1")
    assert resp.status_code == 200
    assert resp.json()["assignments"] == []


# VAL-AGENT-018 (c): active registered worker pull succeeds without a permit
async def test_active_worker_pull_succeeds_without_permit(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    resp = await harness.pull(signer_pubkey="wp-1")
    assert resp.status_code == 200
    assert [a["work_unit_id"] for a in resp.json()["assignments"]] == ["gpu-unit"]


# Foreign/late post rejected (VAL-MASTER-019 baseline)
async def test_foreign_worker_post_rejected(harness: Harness) -> None:
    w1 = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    await harness.enroll_active(worker_pubkey="wp-2", miner_hotkey=MINER_H2)
    unit = await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=w1,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    resp = await harness.post_result(
        assignment_id=str(unit.id),
        signer_pubkey="wp-2",
        success=True,
        payload={"manifest_sha256": "e" * 64},
    )
    assert resp.status_code == 403
    row = await harness.row(str(unit.id))
    assert row is not None
    assert row.result_success is None


async def test_post_result_is_idempotent(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    unit = await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    first = await harness.post_result(
        assignment_id=str(unit.id),
        signer_pubkey="wp-1",
        success=True,
        payload={"manifest_sha256": "e" * 64},
    )
    assert first.status_code == 200
    assert first.json()["idempotent"] is False
    second = await harness.post_result(
        assignment_id=str(unit.id),
        signer_pubkey="wp-1",
        success=True,
        payload={"manifest_sha256": "f" * 64},
    )
    assert second.status_code == 200
    assert second.json()["idempotent"] is True
    row = await harness.row(str(unit.id))
    assert row is not None
    assert row.manifest_sha256 == "e" * 64


async def test_post_result_failure_records_success_false(harness: Harness) -> None:
    worker_id = await harness.enroll_active(worker_pubkey="wp-1", miner_hotkey=MINER_H1)
    unit = await harness.seed_assignment(
        work_unit_id="gpu-unit",
        worker_id=worker_id,
        worker_pubkey="wp-1",
        miner_hotkey=MINER_H1,
    )
    resp = await harness.post_result(
        assignment_id=str(unit.id),
        signer_pubkey="wp-1",
        success=False,
        payload={"error": "broker down"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "failed"
    row = await harness.row(str(unit.id))
    assert row is not None
    assert row.result_success is False
    assert row.manifest_sha256 is None
