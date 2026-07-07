"""WorkerAgent runtime behavior against the real master proxy app.

Drives the agent's :class:`WorkerCoordinationClient` + :class:`WorkerAgent`
through the in-process master (httpx ASGITransport) wired with the live worker
coordination + worker assignment services, using REAL sr25519 keypairs and the
production signature verifier. Covers the feature's expected behaviors:

* register under a miner-signed binding -> visible via the master (VAL-AGENT-002)
* forged binding rejected with a clear error (VAL-AGENT-003)
* replayed nonce rejected; fresh nonce re-enrolls (VAL-AGENT-004)
* heartbeats keep active + advance last-seen (VAL-AGENT-005); missing heartbeats
  go stale after the TTL (VAL-AGENT-006)
* pull only gpu units (VAL-AGENT-007); posted results carry the ExecutionProof
  (VAL-AGENT-008)
* unreachable broker -> clean failed unit, agent stays alive (VAL-AGENT-016)
* restart re-registers idempotently -> one fleet entry (VAL-AGENT-017)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerRegistration,
    create_engine,
    create_session_factory,
)
from base.master.app_proxy import create_proxy_app
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_coordination import WorkerCoordinationService
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    RegisteredWorkerEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
    worker_binding_message,
)
from base.validator.agent.executor import (
    AssignmentContext,
    BrokerConfig,
    ExecutionResult,
    ProgressCallback,
)
from base.validator.agent.signing import KeypairRequestSigner
from base.worker import (
    StubManifestExecutor,
    WorkerAgent,
    WorkerBinding,
    WorkerCoordinationClient,
    WorkerCoordinationClientError,
    WorkerProofExecutor,
    WorkerProvenance,
    verify_execution_proof,
)
from base.worker.runtime import BackoffPolicy

NOW_EPOCH = 1_750_000_000.0
TTL_SECONDS = 120
FAST_BACKOFF = BackoffPolicy(initial_seconds=0.0, max_seconds=0.0, multiplier=2.0)


class _LazyKeypair:
    """Defer sr25519 keypair creation until first real use.

    ``bittensor``'s import reconfigures the stdlib logging levels of every
    already-created logger, which breaks unrelated ``caplog`` suites. Deferring
    the import to test-execution time avoids that; the dunder guard is essential
    because pytest's collection probes module globals via ``getattr(obj,
    "__test__")`` and would otherwise trigger the import at collection time.
    """

    def __init__(self, uri: str) -> None:
        self._uri = uri
        self._kp: Any = None

    def _resolve(self) -> Any:
        if self._kp is None:
            import bittensor as bt

            self._kp = bt.Keypair.create_from_uri(self._uri)
        return self._kp

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)


MINER = _LazyKeypair("//AgentMiner1")
MINER2 = _LazyKeypair("//AgentMiner2")
WORKER = _LazyKeypair("//AgentWorker1")
BROKER = BrokerConfig(broker_url="http://127.0.0.1:65533", broker_token="t")


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


class ConnectionRefusedExecutor:
    """Executor that fails as an unreachable local broker would."""

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        raise ConnectionError("broker connection refused")


def _binding(miner: Any, *, worker_pubkey: str, nonce: str) -> WorkerBinding:
    message = worker_binding_message(
        worker_pubkey=worker_pubkey, miner_hotkey=miner.ss58_address, nonce=nonce
    )
    return WorkerBinding(
        miner_hotkey=miner.ss58_address,
        signature="0x" + bytes(miner.sign(message)).hex(),
        nonce=nonce,
    )


class Harness:
    def __init__(
        self,
        transport: ASGITransport,
        session_factory: Any,
        clock: FakeClock,
        service: WorkerCoordinationService,
        assignment_service: WorkerAssignmentService,
    ) -> None:
        self.transport = transport
        self.session_factory = session_factory
        self.clock = clock
        self.service = service
        self.assignment_service = assignment_service

    def make_agent(
        self,
        *,
        keypair: Any = WORKER,
        binding: WorkerBinding,
        executor: Any | None = None,
    ) -> WorkerAgent:
        signer = KeypairRequestSigner(keypair)
        client = WorkerCoordinationClient(
            "http://testserver",
            signer,
            transport=self.transport,
            now_fn=self.clock.time,
        )
        proof_executor = executor or WorkerProofExecutor(
            StubManifestExecutor(),
            signer=signer,
            provenance=WorkerProvenance(
                provider_name="local", miner_hotkey=binding.miner_hotkey
            ),
        )
        return WorkerAgent(
            client=client,
            executor=proof_executor,
            broker=BROKER,
            binding=binding,
            provider="local",
            provider_instance_ref="local-1",
            capabilities=["gpu"],
            backoff=FAST_BACKOFF,
        )

    async def fleet(self) -> list[dict[str, Any]]:
        async with self.session_factory() as session:
            rows = (await session.execute(select(WorkerRegistration))).scalars().all()
        now = self.clock.now()
        return [
            {
                "worker_pubkey": r.worker_pubkey,
                "miner_hotkey": r.miner_hotkey,
                "status": self.service.effective_status(r, now).value,
                "last_heartbeat_at": r.last_heartbeat_at,
            }
            for r in rows
        ]

    async def count_pubkey(self, worker_pubkey: str) -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(WorkerRegistration.id)).where(
                    WorkerRegistration.worker_pubkey == worker_pubkey
                )
            )

    async def assignment_row(self, work_unit_id: str) -> WorkerAssignment | None:
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(WorkerAssignment).where(
                        WorkerAssignment.work_unit_id == work_unit_id
                    )
                )
            ).scalar_one_or_none()


async def _build_harness() -> tuple[Harness, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        [MINER.ss58_address, MINER2.ss58_address],
        validator_permits=[False, False],
        stakes=[0.0, 0.0],
    )
    clock = FakeClock(NOW_EPOCH)
    service = WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        heartbeat_ttl_seconds=TTL_SECONDS,
        now_fn=clock.now,
    )
    verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        eligibility=CoordinationReadEligibility(session_factory, cache),
        ttl_seconds=300,
        now_fn=clock.time,
    )
    assignment_service = WorkerAssignmentService(
        session_factory, worker_service=service, now_fn=clock.now
    )
    assignment_verifier = WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
        eligibility=RegisteredWorkerEligibility(session_factory),
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
    return (
        Harness(transport, session_factory, clock, service, assignment_service),
        engine,
    )


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h, engine = await _build_harness()
    try:
        yield h
    finally:
        await engine.dispose()


# VAL-AGENT-002
async def test_register_makes_worker_visible(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    worker_id = await agent.register()
    assert worker_id
    fleet = await harness.fleet()
    assert len(fleet) == 1
    assert fleet[0]["worker_pubkey"] == WORKER.ss58_address
    assert fleet[0]["miner_hotkey"] == MINER.ss58_address


# VAL-AGENT-003
async def test_forged_binding_rejected(harness: Harness) -> None:
    # Sign the binding with the WRONG key (worker signs, claiming to be miner).
    forged = _binding(WORKER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    forged = WorkerBinding(
        miner_hotkey=MINER.ss58_address,
        signature=forged.signature,
        nonce="n1",
    )
    agent = harness.make_agent(binding=forged)
    with pytest.raises(WorkerCoordinationClientError) as exc:
        await agent.register()
    assert exc.value.status_code == 401
    assert await harness.count_pubkey(WORKER.ss58_address) == 0


# VAL-AGENT-004
async def test_replayed_nonce_rejected_fresh_nonce_reenrolls(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="dup")
    )
    await agent.register()

    replay = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="dup")
    )
    with pytest.raises(WorkerCoordinationClientError) as exc:
        await replay.register()
    assert exc.value.status_code == 409

    fresh = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="fresh")
    )
    await fresh.register()
    assert await harness.count_pubkey(WORKER.ss58_address) == 1


# VAL-AGENT-005
async def test_heartbeats_keep_active_and_advance_last_seen(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    await agent.register()

    harness.clock.epoch = NOW_EPOCH + 10
    await agent.heartbeat_once()
    first = (await harness.fleet())[0]
    assert first["status"] == "active"

    harness.clock.epoch = NOW_EPOCH + 20
    await agent.heartbeat_once()
    second = (await harness.fleet())[0]
    assert second["status"] == "active"
    assert second["last_heartbeat_at"] > first["last_heartbeat_at"]


# VAL-AGENT-006
async def test_missing_heartbeats_go_stale_after_ttl(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    await agent.register()
    await agent.heartbeat_once()
    assert (await harness.fleet())[0]["status"] == "active"

    harness.clock.epoch = NOW_EPOCH + TTL_SECONDS + 5
    assert (await harness.fleet())[0]["status"] == "stale"


# VAL-AGENT-007 + VAL-AGENT-008
async def test_pull_gpu_only_and_post_carries_proof(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    worker_id = await agent.register()
    await agent.heartbeat_once()

    await harness.assignment_service.create_worker_assignment(
        work_unit_id="gpu-unit",
        challenge_slug="prism",
        submission_ref="ref-gpu",
        worker_id=worker_id,
        worker_pubkey=WORKER.ss58_address,
        miner_hotkey=MINER.ss58_address,
        required_capability="gpu",
    )
    await harness.assignment_service.create_worker_assignment(
        work_unit_id="cpu-unit",
        challenge_slug="prism",
        submission_ref="ref-cpu",
        worker_id=worker_id,
        worker_pubkey=WORKER.ss58_address,
        miner_hotkey=MINER.ss58_address,
        required_capability="cpu",
    )

    summary = await agent.process_pending_assignments()
    assert summary.pulled == 1
    assert summary.completed == 1

    gpu_row = await harness.assignment_row("gpu-unit")
    assert gpu_row is not None
    assert WorkAssignmentStatus(gpu_row.status) == WorkAssignmentStatus.COMPLETED
    assert gpu_row.result_success is True
    from base.schemas.worker import ExecutionProof

    assert gpu_row.result_payload is not None
    proof = ExecutionProof.model_validate(gpu_row.result_payload["execution_proof"])
    assert proof.version == 1
    assert proof.provider is not None
    assert proof.provider.name == "local"
    assert proof.provider.miner_hotkey == MINER.ss58_address
    assert proof.worker_signature.worker_pubkey == WORKER.ss58_address
    assert verify_execution_proof(proof, unit_id="gpu-unit") is True

    cpu_row = await harness.assignment_row("cpu-unit")
    assert cpu_row is not None
    assert WorkAssignmentStatus(cpu_row.status) == WorkAssignmentStatus.ASSIGNED
    assert cpu_row.result_success is None


# VAL-AGENT-016
async def test_unreachable_broker_fails_cleanly_agent_survives(
    harness: Harness,
) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1"),
        executor=ConnectionRefusedExecutor(),
    )
    worker_id = await agent.register()
    await agent.heartbeat_once()
    await harness.assignment_service.create_worker_assignment(
        work_unit_id="gpu-unit",
        challenge_slug="prism",
        submission_ref="ref-gpu",
        worker_id=worker_id,
        worker_pubkey=WORKER.ss58_address,
        miner_hotkey=MINER.ss58_address,
    )

    summary = await agent.process_pending_assignments()
    assert summary.pulled == 1
    assert summary.failed == 1

    row = await harness.assignment_row("gpu-unit")
    assert row is not None
    assert WorkAssignmentStatus(row.status) == WorkAssignmentStatus.FAILED
    assert row.result_success is False
    assert "broker" in (row.result_payload or {}).get("error", "")

    # The agent did not crash: it keeps heartbeating and pulling.
    harness.clock.epoch = NOW_EPOCH + 10
    await agent.heartbeat_once()
    assert (await harness.fleet())[0]["status"] == "active"
    followup = await agent.process_pending_assignments()
    assert followup.pulled == 0  # the failed unit is terminal, not re-pulled


# VAL-AGENT-017
async def test_restart_reregisters_idempotently_single_entry(
    harness: Harness,
) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    worker_id = await agent.register()
    await agent.heartbeat_once()
    await harness.assignment_service.create_worker_assignment(
        work_unit_id="gpu-unit",
        challenge_slug="prism",
        submission_ref="ref-gpu",
        worker_id=worker_id,
        worker_pubkey=WORKER.ss58_address,
        miner_hotkey=MINER.ss58_address,
    )
    before = await harness.assignment_row("gpu-unit")
    assert before is not None
    before_status = WorkAssignmentStatus(before.status)
    before_attempts = before.attempt_count

    # Restart: same worker keypair, FRESH binding nonce.
    restarted = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n2")
    )
    restart_worker_id = await restarted.register()
    await restarted.heartbeat_once()

    assert await harness.count_pubkey(WORKER.ss58_address) == 1
    assert restart_worker_id == worker_id
    fleet = await harness.fleet()
    assert len(fleet) == 1
    assert fleet[0]["miner_hotkey"] == MINER.ss58_address
    assert fleet[0]["status"] == "active"

    after = await harness.assignment_row("gpu-unit")
    assert after is not None
    assert WorkAssignmentStatus(after.status) == before_status
    assert after.attempt_count == before_attempts


# VAL-AGENT-017 (cross-owner rebind never silently rebinds)
async def test_cross_owner_rebind_rejected(harness: Harness) -> None:
    agent = harness.make_agent(
        binding=_binding(MINER, worker_pubkey=WORKER.ss58_address, nonce="n1")
    )
    await agent.register()
    await agent.heartbeat_once()

    rebind = harness.make_agent(
        binding=_binding(MINER2, worker_pubkey=WORKER.ss58_address, nonce="n2")
    )
    with pytest.raises(WorkerCoordinationClientError) as exc:
        await rebind.register()
    assert exc.value.status_code == 409
    assert await harness.count_pubkey(WORKER.ss58_address) == 1
    assert (await harness.fleet())[0]["miner_hotkey"] == MINER.ss58_address


class _StubWorker:
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id


class _StubRegisterResponse:
    def __init__(self, worker_id: str, ttl: int) -> None:
        self.worker = _StubWorker(worker_id)
        self.heartbeat_ttl_seconds = ttl


class _FlakyClient:
    """In-memory client to exercise the agent's retry/loop resilience paths."""

    def __init__(self, *, fail_registers: int = 0) -> None:
        self.worker_pubkey = "wp-stub"
        self._fail_registers = fail_registers
        self.register_calls = 0
        self.heartbeat_calls = 0
        self.on_heartbeat: Any = None
        self.raise_heartbeats: set[int] = set()

    async def register(self, **_: Any) -> _StubRegisterResponse:
        self.register_calls += 1
        if self.register_calls <= self._fail_registers:
            raise WorkerCoordinationClientError("register down", status_code=503)
        return _StubRegisterResponse("w-1", 60)

    async def heartbeat(self, **_: Any) -> None:
        self.heartbeat_calls += 1
        if self.heartbeat_calls in self.raise_heartbeats:
            raise WorkerCoordinationClientError("hb down", status_code=503)
        if self.on_heartbeat is not None:
            self.on_heartbeat()

    async def pull(self) -> list[Any]:
        return []

    async def post_result(self, *_: Any, **__: Any) -> None:
        return None


def _stub_agent(client: _FlakyClient) -> WorkerAgent:
    return WorkerAgent(
        client=client,  # type: ignore[arg-type]
        executor=StubManifestExecutor(),
        broker=BROKER,
        binding=WorkerBinding(miner_hotkey="m", signature="0xsig", nonce="n"),
        provider="local",
        heartbeat_interval_seconds=0,
        poll_interval_seconds=0.0,
        backoff=FAST_BACKOFF,
    )


# VAL-AGENT-016: transient master failures are retried, not fatal.
async def test_register_retries_transient_then_succeeds() -> None:
    client = _FlakyClient(fail_registers=2)
    agent = _stub_agent(client)
    worker_id = await agent.register()
    assert worker_id == "w-1"
    assert client.register_calls == 3


async def test_register_fails_fast_on_permanent_error() -> None:
    class _Permanent(_FlakyClient):
        async def register(self, **_: Any):
            self.register_calls += 1
            raise WorkerCoordinationClientError("forged", status_code=401)

    client = _Permanent()
    agent = _stub_agent(client)
    with pytest.raises(WorkerCoordinationClientError) as exc:
        await agent.register()
    assert exc.value.status_code == 401
    assert client.register_calls == 1


# VAL-AGENT-016: a heartbeat failure never crashes the loop.
async def test_run_heartbeat_loop_survives_failures() -> None:
    import asyncio

    client = _FlakyClient()
    client.raise_heartbeats = {1}
    agent = _stub_agent(client)
    await agent.register()

    shutdown = asyncio.Event()

    def _stop_after_two() -> None:
        if client.heartbeat_calls >= 2:
            shutdown.set()

    client.on_heartbeat = _stop_after_two
    await asyncio.wait_for(agent.run_heartbeat_loop(shutdown), timeout=5.0)
    assert client.heartbeat_calls >= 2  # kept beating past the first failure


async def test_run_forever_registers_then_loops_until_shutdown() -> None:
    import asyncio

    client = _FlakyClient()
    agent = _stub_agent(client)
    shutdown = asyncio.Event()

    def _stop_after_two() -> None:
        if client.heartbeat_calls >= 2:
            shutdown.set()

    client.on_heartbeat = _stop_after_two
    await asyncio.wait_for(agent.run_forever(shutdown), timeout=5.0)
    assert client.register_calls == 1
    assert client.heartbeat_calls >= 2
    assert agent.worker_id == "w-1"
