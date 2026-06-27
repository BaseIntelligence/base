"""Behavioral tests for the validator agent runtime against the real master app.

Drives the agent's :class:`CoordinationClient` and :class:`ValidatorAgent` through
the in-process master proxy app (httpx ASGITransport) wired with the live
validator-coordination, assignment-coordination, and LLM-gateway services. Covers
the feature's expected behaviors: register + heartbeat on a configurable interval
with restart recovery, pull -> execute-on-own-broker -> post results, and routing
LLM calls through the master gateway with a per-assignment scoped token (no
provider key on the validator).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkResult,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.app_proxy import create_proxy_app
from base.master.assignment_coordination import (
    AssignmentCoordinationService,
    WorkAssignmentLifecycleResolver,
)
from base.master.llm_gateway import ProviderConfig, build_llm_gateway_service
from base.master.validator_coordination import ValidatorCoordinationService
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
)
from base.validator.agent import (
    AssignmentContext,
    BrokerConfig,
    CoordinationClient,
    ExecutionResult,
    ValidatorAgent,
)
from base.validator.agent.executor import ProgressCallback

NOW_EPOCH = 1_750_000_000.0
HEARTBEAT_INTERVAL = 45
HEARTBEAT_TIMEOUT = 100
ADMIN_TOKEN = "admin-secret-token"
DEEPSEEK_KEY = "ds-secret-key"
OPENROUTER_KEY = "or-secret-key"


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


def _sign(hotkey: str, canonical: str) -> str:
    return hashlib.sha256(f"{hotkey}:{canonical}".encode()).hexdigest()


def _signature_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message.decode())


class FakeSigner:
    """Client signer matching the server's stubbed signature verifier."""

    def __init__(self, hotkey: str) -> None:
        self._hotkey = hotkey

    @property
    def hotkey(self) -> str:
        return self._hotkey

    def sign(self, message: bytes) -> str:
        return _sign(self._hotkey, message.decode())


class RecordingExecutor:
    """Executor that records context and reports success."""

    def __init__(self) -> None:
        self.contexts: list[AssignmentContext] = []

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        self.contexts.append(context)
        return ExecutionResult(success=True, payload={"score": 1.0})


class GatewayCallingExecutor:
    """Executor that routes an LLM call through the master gateway using the
    per-assignment scoped token, proving validator-side gateway routing."""

    def __init__(self, transport: ASGITransport) -> None:
        self._transport = transport
        self.gateway_status: int | None = None
        self.held_provider_key: bool = False

    async def execute(
        self, context: AssignmentContext, *, progress: ProgressCallback
    ) -> ExecutionResult:
        env = context.gateway_env
        # The validator holds only a scoped gateway token, never a provider key.
        self.held_provider_key = any(key.endswith("_API_KEY") for key in env)
        token = env["BASE_GATEWAY_TOKEN"]
        async with AsyncClient(
            transport=self._transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/llm/deepseek/chat/completions",
                json={"model": "deepseek-v4-pro", "messages": []},
                headers={
                    "X-Gateway-Token": token,
                    "X-Gateway-Validator": context.assignment.payload["validator"],
                    "X-Gateway-Assignment": context.assignment.id,
                },
            )
        self.gateway_status = response.status_code
        return ExecutionResult(
            success=True, payload={"llm_status": response.status_code}
        )


class Harness:
    def __init__(
        self,
        *,
        app: Any,
        session_factory: Any,
        clock: FakeClock,
        gateway_service: Any,
        engine: Any,
    ) -> None:
        self.app = app
        self.transport = ASGITransport(app=app)
        self.session_factory = session_factory
        self.clock = clock
        self.gateway_service = gateway_service
        self.engine = engine

    def client(self, hotkey: str = "permitted") -> CoordinationClient:
        return CoordinationClient(
            "http://testserver",
            FakeSigner(hotkey),
            transport=self.transport,
            now_fn=self.clock.time,
        )

    def agent(
        self,
        *,
        executor: Any,
        hotkey: str = "permitted",
        capabilities: list[str] | None = None,
        heartbeat_interval_seconds: int | None = None,
    ) -> ValidatorAgent:
        return ValidatorAgent(
            client=self.client(hotkey),
            executor=executor,
            broker=BrokerConfig(broker_url="http://127.0.0.1:8082"),
            capabilities=capabilities or ["cpu"],
            version="0.1.0",
            gateway_url="http://testserver",
            heartbeat_interval_seconds=heartbeat_interval_seconds,
            poll_interval_seconds=0.01,
        )

    async def seed_assignment(
        self,
        *,
        hotkey: str = "permitted",
        capability: str = "cpu",
        payload: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        assignment_id = uuid.uuid4()
        async with session_scope(self.session_factory) as session:
            session.add(
                WorkAssignment(
                    id=assignment_id,
                    challenge_slug="agent-challenge",
                    work_unit_id=f"sub:{assignment_id.hex[:8]}",
                    submission_ref="sub",
                    payload=payload or {},
                    required_capability=capability,
                    assigned_validator_hotkey=hotkey,
                    status=WorkAssignmentStatus.ASSIGNED,
                    attempt_count=1,
                    max_attempts=3,
                    created_at=self.clock.now(),
                )
            )
        return assignment_id

    async def assignment_status(self, assignment_id: uuid.UUID) -> WorkAssignmentStatus:
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(WorkAssignment).where(WorkAssignment.id == assignment_id)
                )
            ).scalar_one()
            return WorkAssignmentStatus(row.status)

    async def result_count(self, assignment_id: uuid.UUID) -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(WorkResult.id)).where(
                    WorkResult.assignment_id == assignment_id
                )
            )

    async def validator_row(self, hotkey: str = "permitted") -> Validator | None:
        async with self.session_factory() as session:
            return (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()

    async def validator_count(self, hotkey: str = "permitted") -> int:
        async with self.session_factory() as session:
            return await session.scalar(
                select(func.count(Validator.id)).where(Validator.hotkey == hotkey)
            )


async def _build_harness() -> Harness:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    cache.update_from_metagraph(
        ["permitted", "permitted2"],
        validator_permits=[True, True],
        stakes=[100.0, 50.0],
    )
    clock = FakeClock(NOW_EPOCH)
    verifier = ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(session_factory),
        eligibility=MetagraphValidatorEligibility(cache),
        signature_verifier=_signature_verifier,
        ttl_seconds=300,
        now_fn=clock.time,
    )
    validator_service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL,
        heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT,
        now_fn=clock.now,
    )
    assignment_service = AssignmentCoordinationService(
        session_factory, now_fn=clock.now
    )
    gateway_service = build_llm_gateway_service(
        deepseek_api_key=DEEPSEEK_KEY,
        openrouter_api_key=OPENROUTER_KEY,
        token_secret="tok-secret",
        provider_config=ProviderConfig(mode="mock"),
        assignment_resolver=WorkAssignmentLifecycleResolver(session_factory),
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        validator_service=validator_service,
        validator_verifier=verifier,
        assignment_coordination_service=assignment_service,
        llm_gateway_service=gateway_service,
        admin_token_provider=lambda: ADMIN_TOKEN,
    )
    return Harness(
        app=app,
        session_factory=session_factory,
        clock=clock,
        gateway_service=gateway_service,
        engine=engine,
    )


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    h = await _build_harness()
    try:
        yield h
    finally:
        await h.engine.dispose()


async def test_agent_register_marks_validator_online_with_capabilities(
    harness: Harness,
) -> None:
    agent = harness.agent(executor=RecordingExecutor(), capabilities=["cpu", "gpu"])
    interval = await agent.register()

    assert interval == HEARTBEAT_INTERVAL
    assert agent.heartbeat_interval == HEARTBEAT_INTERVAL
    row = await harness.validator_row()
    assert row is not None
    assert row.status == ValidatorStatus.ONLINE
    assert row.capabilities == ["cpu", "gpu"]
    assert row.version == "0.1.0"


async def test_agent_configured_interval_overrides_server(harness: Harness) -> None:
    agent = harness.agent(executor=RecordingExecutor(), heartbeat_interval_seconds=7)
    interval = await agent.register()
    assert interval == 7
    assert agent.heartbeat_interval == 7


async def test_agent_heartbeat_refreshes_liveness(harness: Harness) -> None:
    agent = harness.agent(executor=RecordingExecutor())
    await agent.register()
    harness.clock.epoch = NOW_EPOCH + 20
    await agent.heartbeat_once()

    row = await harness.validator_row()
    assert row is not None
    assert row.status == ValidatorStatus.ONLINE
    last = row.last_heartbeat_at
    assert last is not None
    last = last if last.tzinfo else last.replace(tzinfo=UTC)
    assert last == harness.clock.now()


async def test_agent_pulls_executes_via_broker_and_posts_result(
    harness: Harness,
) -> None:
    executor = RecordingExecutor()
    agent = harness.agent(executor=executor)
    await agent.register()
    assignment_id = await harness.seed_assignment(payload={"score": 1.0})

    summary = await agent.process_pending_assignments()

    assert summary.pulled == 1
    assert summary.completed == 1
    assert summary.failed == 0
    assert len(executor.contexts) == 1
    # the executor ran against the validator's OWN broker, not a central one.
    assert executor.contexts[0].broker.broker_url == "http://127.0.0.1:8082"
    assert await harness.assignment_status(assignment_id) == (
        WorkAssignmentStatus.COMPLETED
    )
    assert await harness.result_count(assignment_id) == 1


async def test_agent_routes_llm_through_gateway_with_scoped_token(
    harness: Harness,
) -> None:
    assignment_id = await harness.seed_assignment(payload={})
    token = harness.gateway_service.issue_token(
        validator_hotkey="permitted", assignment_id=str(assignment_id)
    )
    async with session_scope(harness.session_factory) as session:
        row = (
            await session.execute(
                select(WorkAssignment).where(WorkAssignment.id == assignment_id)
            )
        ).scalar_one()
        row.payload = {"gateway_token": token, "validator": "permitted"}

    executor = GatewayCallingExecutor(harness.transport)
    agent = harness.agent(executor=executor)
    await agent.register()

    summary = await agent.process_pending_assignments()

    assert summary.completed == 1
    assert executor.gateway_status == 200
    assert executor.held_provider_key is False
    # the gateway injected the real provider key server-side (validator never saw it).
    deepseek_provider = harness.gateway_service.provider("deepseek")
    assert (
        deepseek_provider.requests[-1].header("authorization")
        == f"Bearer {DEEPSEEK_KEY}"
    )


async def test_agent_recovers_in_flight_work_across_restart(harness: Harness) -> None:
    # Agent A registers and pulls (unit -> running) but "crashes" before posting.
    assignment_id = await harness.seed_assignment(payload={"score": 1.0})
    agent_a = harness.agent(executor=RecordingExecutor())
    await agent_a.register()
    pulled = await harness.client().pull()
    assert any(a.id == str(assignment_id) for a in pulled)
    assert await harness.assignment_status(assignment_id) == (
        WorkAssignmentStatus.RUNNING
    )

    # A fresh agent (restart, same hotkey) re-registers (idempotent) and resumes.
    executor_b = RecordingExecutor()
    agent_b = harness.agent(executor=executor_b)
    await agent_b.register()
    assert await harness.validator_count() == 1

    summary = await agent_b.process_pending_assignments()
    assert summary.completed == 1
    assert len(executor_b.contexts) == 1
    assert await harness.assignment_status(assignment_id) == (
        WorkAssignmentStatus.COMPLETED
    )


async def test_agent_posts_failure_when_executor_raises(harness: Harness) -> None:
    class _BoomExecutor:
        async def execute(
            self, context: AssignmentContext, *, progress: ProgressCallback
        ) -> ExecutionResult:
            raise RuntimeError("execution exploded")

    assignment_id = await harness.seed_assignment(payload={})
    agent = harness.agent(executor=_BoomExecutor())
    await agent.register()

    summary = await agent.process_pending_assignments()
    assert summary.failed == 1
    assert summary.completed == 0
    assert await harness.assignment_status(assignment_id) == (
        WorkAssignmentStatus.FAILED
    )
    assert await harness.result_count(assignment_id) == 1


async def test_run_forever_registers_then_runs_loops_until_shutdown(
    harness: Harness,
) -> None:
    shutdown = asyncio.Event()
    assignment_id = await harness.seed_assignment(payload={"score": 1.0})

    class _StoppingExecutor(RecordingExecutor):
        async def execute(
            self, context: AssignmentContext, *, progress: ProgressCallback
        ) -> ExecutionResult:
            result = await super().execute(context, progress=progress)
            shutdown.set()
            return result

    executor = _StoppingExecutor()
    agent = harness.agent(executor=executor, heartbeat_interval_seconds=0)

    await asyncio.wait_for(agent.run_forever(shutdown), timeout=5)

    row = await harness.validator_row()
    assert row is not None and row.status == ValidatorStatus.ONLINE
    assert len(executor.contexts) >= 1
    assert await harness.assignment_status(assignment_id) == (
        WorkAssignmentStatus.COMPLETED
    )


async def test_progress_callback_persists_checkpoint_ref(harness: Harness) -> None:
    class _CheckpointingExecutor:
        async def execute(
            self, context: AssignmentContext, *, progress: ProgressCallback
        ) -> ExecutionResult:
            await progress(checkpoint_ref="hf://ckpt-1")
            return ExecutionResult(
                success=True, payload={"ok": True}, checkpoint_ref="hf://ckpt-1"
            )

    assignment_id = await harness.seed_assignment(
        capability="gpu", payload={"score": 1.0}
    )
    agent = harness.agent(
        executor=_CheckpointingExecutor(), capabilities=["cpu", "gpu"]
    )
    await agent.register()

    summary = await agent.process_pending_assignments()
    assert summary.completed == 1

    async with harness.session_factory() as session:
        row = (
            await session.execute(
                select(WorkAssignment).where(WorkAssignment.id == assignment_id)
            )
        ).scalar_one()
    assert row.checkpoint_ref == "hf://ckpt-1"
    assert row.last_progress_at is not None


async def test_register_persists_last_seen_meta_from_factory(
    harness: Harness,
) -> None:
    agent = ValidatorAgent(
        client=harness.client(),
        executor=RecordingExecutor(),
        broker=BrokerConfig(broker_url="http://127.0.0.1:8082"),
        capabilities=["cpu"],
        version="0.1.0",
        gateway_url="http://testserver",
        last_seen_meta_factory=lambda: {"broker": "ok", "concurrency": 4},
    )
    await agent.register()

    row = await harness.validator_row()
    assert row is not None
    assert row.last_seen_meta["broker"] == "ok"
    assert row.last_seen_meta["concurrency"] == 4
    assert row.last_seen_meta["broker_url"] == "http://127.0.0.1:8082"


async def test_heartbeat_for_unregistered_hotkey_raises(harness: Harness) -> None:
    from base.validator.agent import CoordinationClientError

    agent = harness.agent(executor=RecordingExecutor())
    with pytest.raises(CoordinationClientError) as excinfo:
        await agent.heartbeat_once()
    assert excinfo.value.status_code == 404


async def test_agent_post_result_is_idempotent(harness: Harness) -> None:
    executor = RecordingExecutor()
    agent = harness.agent(executor=executor)
    await agent.register()
    assignment_id = await harness.seed_assignment(payload={"score": 1.0})

    first = await agent.process_pending_assignments()
    assert first.completed == 1
    # A repeated pass is a safe no-op: completed units are not re-pulled.
    second = await agent.process_pending_assignments()
    assert second.pulled == 0
    assert await harness.result_count(assignment_id) == 1
