"""Non-authoritative validator submission observations on the master."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.challenge_sdk.roles import Role, activate_role
from base.db import Base
from base.master.submission_observation import (
    SubmissionObservationConflictError,
    ValidatorSubmissionObservationService,
)
from base.schemas.weights import ValidatorSubmissionObservationRequest


@pytest.fixture(autouse=True)
def _activate_master_role() -> Iterator[None]:
    with activate_role(Role.MASTER):
        yield


@pytest.fixture
async def session_factory(tmp_path: Any) -> AsyncIterator[async_sessionmaker]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'obs.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


async def test_observation_idempotent_exact_retry(
    session_factory: async_sessionmaker,
) -> None:
    service = ValidatorSubmissionObservationService(session_factory)
    request = ValidatorSubmissionObservationRequest(
        vector_id=str(uuid.uuid4()),
        vector_digest="a" * 64,
        netuid=100,
        chain_endpoint="wss://example",
        outcome="accepted",
        attempt=1,
        observed_at=datetime.now(UTC),
    )
    first = await service.record(validator_hotkey="validator-A", request=request)
    second = await service.record(validator_hotkey="validator-A", request=request)
    assert first.idempotent is False
    assert second.idempotent is True
    assert first.observation_id == second.observation_id


async def test_observation_conflict_on_changed_payload(
    session_factory: async_sessionmaker,
) -> None:
    service = ValidatorSubmissionObservationService(session_factory)
    vector_id = str(uuid.uuid4())
    base = ValidatorSubmissionObservationRequest(
        vector_id=vector_id,
        vector_digest="b" * 64,
        netuid=100,
        chain_endpoint="wss://example",
        outcome="accepted",
        attempt=1,
    )
    await service.record(validator_hotkey="validator-A", request=base)
    conflict = ValidatorSubmissionObservationRequest(
        vector_id=vector_id,
        vector_digest="b" * 64,
        netuid=101,
        chain_endpoint="wss://example",
        outcome="accepted",
        attempt=1,
    )
    with pytest.raises(SubmissionObservationConflictError):
        await service.record(validator_hotkey="validator-A", request=conflict)


async def test_observation_does_not_import_set_weights() -> None:
    import ast
    import inspect

    import base.master.submission_observation as mod

    tree = ast.parse(inspect.getsource(mod))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("weight_setter" in name or "bittensor" in name for name in imported)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            assert node.id != "WeightSetter"
        if isinstance(node, ast.Attribute):
            assert node.attr != "set_weights"
