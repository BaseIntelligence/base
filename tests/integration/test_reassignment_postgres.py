"""Postgres parity for crash/deadline reassignment (VAL-ASSIGN-029).

Mirrors the SQLite unit behavior of the reassignment cycle (assign -> pull ->
progress -> crash -> reclaim -> reassign, and max_attempts exhaustion) against
the throwaway Postgres at 127.0.0.1:15490, confirming identical observable
status transitions, ``attempt_count`` increments, and ``checkpoint_ref`` carry.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from base.db import (
    Validator,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.assignment import RESUME_CHECKPOINT_PAYLOAD_KEY, AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)
LATER = NOW + timedelta(seconds=1000)


async def _add_validator(
    factory,
    hotkey: str,
    capabilities: list[str],
    *,
    status: ValidatorStatus = ValidatorStatus.ONLINE,
) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=status,
                capabilities=list(capabilities),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _set_status(factory, hotkey: str, status: ValidatorStatus) -> None:
    async with session_scope(factory) as session:
        validator = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        validator.status = status


async def _one(factory) -> WorkAssignment:
    async with factory() as session:
        return (await session.execute(select(WorkAssignment))).scalar_one()


async def test_prism_crash_reassign_with_checkpoint_parity_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine)
    service = AssignmentService(factory, now_fn=lambda: NOW)
    coordination = AssignmentCoordinationService(factory, now_fn=lambda: NOW)
    try:
        await _add_validator(factory, "g1", ["gpu"])
        await service.create_prism_work_unit(
            submission_id="psub-pg", submission_ref="hk-pg"
        )
        await service.assign_pending(seed=1)
        assert (await _one(factory)).attempt_count == 1

        pulled = await coordination.pull(hotkey="g1")
        await coordination.progress(
            assignment_id=str(pulled[0].id),
            hotkey="g1",
            checkpoint_ref="hf://ckpt/pg-5",
        )

        await _set_status(factory, "g1", ValidatorStatus.OFFLINE)
        await _add_validator(factory, "g2", ["gpu"])

        outcome = await service.reclaim_stale_assignments()
        assert outcome.reverted == ["psub-pg"]
        row = await _one(factory)
        assert row.status == WorkAssignmentStatus.PENDING
        assert row.assigned_validator_hotkey is None
        assert row.checkpoint_ref == "hf://ckpt/pg-5"
        assert row.payload[RESUME_CHECKPOINT_PAYLOAD_KEY] == "hf://ckpt/pg-5"

        await service.assign_pending(seed=1)
        row = await _one(factory)
        assert row.status == WorkAssignmentStatus.ASSIGNED
        assert row.assigned_validator_hotkey == "g2"
        assert row.attempt_count == 2

        resumed = await coordination.pull(hotkey="g2")
        assert resumed[0].checkpoint_ref == "hf://ckpt/pg-5"
        assert resumed[0].payload[RESUME_CHECKPOINT_PAYLOAD_KEY] == "hf://ckpt/pg-5"
    finally:
        await engine.dispose()


async def test_max_attempts_exhaustion_parity_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine)
    service = AssignmentService(factory, now_fn=lambda: NOW, default_max_attempts=2)
    try:
        await _add_validator(factory, "v1", ["cpu"])
        await service.create_agent_challenge_work_units(
            submission_id="sub-pg", submission_ref="hk", task_ids=["a"]
        )

        await service.assign_pending(seed=1)
        assert (await _one(factory)).attempt_count == 1

        await _set_status(factory, "v1", ValidatorStatus.OFFLINE)
        await service.reclaim_stale_assignments()
        await _set_status(factory, "v1", ValidatorStatus.ONLINE)
        await service.assign_pending(seed=1)
        assert (await _one(factory)).attempt_count == 2

        await _set_status(factory, "v1", ValidatorStatus.OFFLINE)
        outcome = await service.reclaim_stale_assignments()
        assert outcome.failed == ["sub-pg:a"]
        row = await _one(factory)
        assert row.status == WorkAssignmentStatus.FAILED
        assert row.attempt_count == 2

        await _set_status(factory, "v1", ValidatorStatus.ONLINE)
        await service.assign_pending(seed=1)
        assert (await _one(factory)).status == WorkAssignmentStatus.FAILED
    finally:
        await engine.dispose()
