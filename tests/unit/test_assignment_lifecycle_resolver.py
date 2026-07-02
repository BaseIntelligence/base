"""Tests for the work_assignments-backed AssignmentLifecycleResolver.

The production resolver reads real ``work_assignments`` status so a scoped LLM
gateway token is rejected once its assignment is completed/failed/reassigned
(architecture.md sec 5; the M3 production replacement for the M2 in-memory
resolver). Covers the resolver in isolation and end-to-end through the gateway.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from base.db import (
    Base,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.app_proxy import create_proxy_app
from base.master.assignment_coordination import WorkAssignmentLifecycleResolver
from base.master.llm_gateway import (
    DEFAULT_PROVIDER_BASE_URL,
    GatewayTokenAuthority,
    LLMGatewayService,
    MockLLMProvider,
    SourceRoute,
)

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
YUNWU_KEY = "sk-yunwu-server-secret"
MODEL = "claude-opus-4-8"
TOKEN_SECRET = "gateway-secret"


async def _setup() -> tuple[Any, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, create_session_factory(engine)


async def _add_assignment(
    factory: Any,
    *,
    hotkey: str | None,
    status: WorkAssignmentStatus,
) -> uuid.UUID:
    unit_id = uuid.uuid4()
    async with session_scope(factory) as session:
        session.add(
            WorkAssignment(
                id=unit_id,
                challenge_slug="prism",
                work_unit_id=str(unit_id),
                submission_ref="hk",
                payload={},
                required_capability="gpu",
                assigned_validator_hotkey=hotkey,
                status=status,
                attempt_count=1,
                max_attempts=3,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    return unit_id


@pytest.mark.parametrize(
    "status,expected",
    [
        (WorkAssignmentStatus.ASSIGNED, True),
        (WorkAssignmentStatus.RUNNING, True),
        (WorkAssignmentStatus.COMPLETED, False),
        (WorkAssignmentStatus.FAILED, False),
    ],
)
async def test_resolver_active_only_for_non_terminal_owned(
    status: WorkAssignmentStatus, expected: bool
) -> None:
    engine, factory = await _setup()
    try:
        unit_id = await _add_assignment(factory, hotkey="val-1", status=status)
        resolver = WorkAssignmentLifecycleResolver(factory)
        active = await resolver.is_active(
            validator_hotkey="val-1", assignment_id=str(unit_id)
        )
        assert active is expected
    finally:
        await engine.dispose()


async def test_resolver_inactive_when_reassigned_to_other_validator() -> None:
    engine, factory = await _setup()
    try:
        unit_id = await _add_assignment(
            factory, hotkey="val-2", status=WorkAssignmentStatus.RUNNING
        )
        resolver = WorkAssignmentLifecycleResolver(factory)
        # The token's validator no longer owns the (reassigned) unit.
        assert not await resolver.is_active(
            validator_hotkey="val-1", assignment_id=str(unit_id)
        )
        # The current owner is still active.
        assert await resolver.is_active(
            validator_hotkey="val-2", assignment_id=str(unit_id)
        )
    finally:
        await engine.dispose()


async def test_resolver_inactive_for_reverted_pending_unit() -> None:
    engine, factory = await _setup()
    try:
        unit_id = await _add_assignment(
            factory, hotkey=None, status=WorkAssignmentStatus.PENDING
        )
        resolver = WorkAssignmentLifecycleResolver(factory)
        assert not await resolver.is_active(
            validator_hotkey="val-1", assignment_id=str(unit_id)
        )
    finally:
        await engine.dispose()


async def test_resolver_inactive_for_unknown_and_malformed_id() -> None:
    engine, factory = await _setup()
    try:
        resolver = WorkAssignmentLifecycleResolver(factory)
        assert not await resolver.is_active(
            validator_hotkey="val-1", assignment_id=str(uuid.uuid4())
        )
        assert not await resolver.is_active(
            validator_hotkey="val-1", assignment_id="not-a-uuid"
        )
    finally:
        await engine.dispose()


class FakeNonceStore:
    async def reserve(self, **_: Any) -> None:
        return None


class FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class Clock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def time(self) -> float:
        return self.epoch


@pytest.fixture
async def gateway() -> AsyncIterator[tuple[AsyncClient, Any, MockLLMProvider, str]]:
    engine, factory = await _setup()
    yunwu = MockLLMProvider(name="yunwu", base_url=DEFAULT_PROVIDER_BASE_URL)
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=Clock(NOW.timestamp()).time)
    service = LLMGatewayService(
        providers={"yunwu": yunwu},
        api_keys={"yunwu": YUNWU_KEY},
        token_authority=authority,
        sources={"agent": SourceRoute(provider="yunwu", model=MODEL)},
        assignment_resolver=WorkAssignmentLifecycleResolver(factory),
    )
    app = create_proxy_app(
        registry=object(),
        nonce_store=FakeNonceStore(),
        metagraph_cache=FakeCache(),  # type: ignore[arg-type]
        llm_gateway_service=service,
    )
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    token = authority.issue(
        validator_hotkey="val-1", assignment_id="placeholder", source="agent"
    )
    try:
        yield client, factory, yunwu, token
    finally:
        await client.aclose()
        await engine.dispose()


async def _set_status(
    factory: Any, unit_id: uuid.UUID, status: WorkAssignmentStatus
) -> None:
    async with session_scope(factory) as session:
        unit = (
            await session.execute(
                select(WorkAssignment).where(WorkAssignment.id == unit_id)
            )
        ).scalar_one()
        unit.status = status


# Production resolver end-to-end: a token is rejected once its real assignment
# transitions to a terminal/reassigned state.
async def test_gateway_rejects_token_after_assignment_completed(
    gateway: tuple[AsyncClient, Any, MockLLMProvider, str],
) -> None:
    client, factory, yunwu, _placeholder = gateway
    unit_id = await _add_assignment(
        factory, hotkey="val-1", status=WorkAssignmentStatus.RUNNING
    )
    authority = GatewayTokenAuthority(TOKEN_SECRET, now_fn=Clock(NOW.timestamp()).time)
    token = authority.issue(
        validator_hotkey="val-1", assignment_id=str(unit_id), source="agent"
    )

    body = json.dumps(
        {
            "model": "agent-sent-placeholder",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()

    active = await client.post(
        "/llm/v1/chat/completions",
        content=body,
        headers={"X-Gateway-Token": token},
    )
    assert active.status_code == 200
    assert yunwu.call_count == 1

    await _set_status(factory, unit_id, WorkAssignmentStatus.COMPLETED)

    rejected = await client.post(
        "/llm/v1/chat/completions",
        content=body,
        headers={"X-Gateway-Token": token},
    )
    assert rejected.status_code == 403
    # The provider was NOT invoked for the rejected call.
    assert yunwu.call_count == 1
    assert YUNWU_KEY not in rejected.text
    assert token not in rejected.text
