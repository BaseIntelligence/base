"""Tests for subscription-gated work assignment (VAL-VDIR-SUB-004).

A validator with a non-empty subscription set is assigned ONLY work for the
challenges it subscribed to (capability filtering still applies on top); a
validator with an empty/absent subscription set remains eligible for ALL
challenges (back-compat, never starved).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.assignment import AssignmentService

NOW = datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)


async def _setup():
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    return engine, factory


async def _add_validator(
    factory,
    hotkey: str,
    capabilities: list[str],
    *,
    subscriptions: list[str] | None = None,
    status: ValidatorStatus = ValidatorStatus.ONLINE,
) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=status,
                capabilities=list(capabilities),
                subscriptions=list(subscriptions or []),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _rows(factory) -> list[WorkAssignment]:
    async with factory() as session:
        result = await session.execute(
            select(WorkAssignment).order_by(WorkAssignment.work_unit_id)
        )
        return list(result.scalars().all())


def _service(factory, **kwargs) -> AssignmentService:
    return AssignmentService(factory, now_fn=lambda: NOW, **kwargs)


# VAL-VDIR-SUB-004
async def test_restricted_validator_only_gets_subscribed_challenge() -> None:
    engine, factory = await _setup()
    try:
        service = _service(factory)
        # restricted validator subscribes ONLY to prism
        await _add_validator(
            factory, "restricted", ["cpu", "gpu"], subscriptions=["prism"]
        )
        # unrestricted validator (empty subscriptions) takes everything
        await _add_validator(factory, "open", ["cpu", "gpu"], subscriptions=[])

        await service.create_agent_challenge_work_units(
            submission_id="sub-ac",
            submission_ref="hk-ac",
            task_ids=["t1", "t2", "t3", "t4"],
        )
        await service.assign_pending(seed=7)

        rows = await _rows(factory)
        assert len(rows) == 4
        # the restricted validator (subscribed only to prism) gets NO
        # agent-challenge work; all four units land on the open validator.
        assignees = {r.assigned_validator_hotkey for r in rows}
        assert assignees == {"open"}
    finally:
        await engine.dispose()


# VAL-VDIR-SUB-004
async def test_restricted_validator_receives_subscribed_challenge() -> None:
    engine, factory = await _setup()
    try:
        service = _service(factory)
        await _add_validator(
            factory, "prism-only", ["cpu", "gpu"], subscriptions=["prism"]
        )

        await service.create_prism_work_unit(
            submission_id="psub-1", submission_ref="hk-p"
        )
        await service.assign_pending(seed=1)

        rows = await _rows(factory)
        assert len(rows) == 1
        assert rows[0].challenge_slug == "prism"
        assert rows[0].assigned_validator_hotkey == "prism-only"
        assert rows[0].status == WorkAssignmentStatus.ASSIGNED
    finally:
        await engine.dispose()


# VAL-VDIR-SUB-004 (non-subscribed challenge stays pending if no eligible validator)
async def test_non_subscribed_challenge_stays_pending_when_only_restricted() -> None:
    engine, factory = await _setup()
    try:
        service = _service(factory)
        # the ONLY validator subscribes to prism; an agent-challenge unit has
        # no eligible validator and must stay pending (never lost, never forced).
        await _add_validator(
            factory, "prism-only", ["cpu", "gpu"], subscriptions=["prism"]
        )

        await service.create_agent_challenge_work_units(
            submission_id="sub-ac", submission_ref="hk-ac", task_ids=["t1"]
        )
        assigned = await service.assign_pending(seed=1)

        assert assigned == {}
        rows = await _rows(factory)
        assert len(rows) == 1
        assert rows[0].status == WorkAssignmentStatus.PENDING
        assert rows[0].assigned_validator_hotkey is None
    finally:
        await engine.dispose()


# VAL-VDIR-SUB-004 (empty subscription == all challenges; capability still applies)
async def test_empty_subscription_gets_all_challenges() -> None:
    engine, factory = await _setup()
    try:
        service = _service(factory)
        await _add_validator(factory, "open", ["cpu", "gpu"], subscriptions=[])

        await service.create_agent_challenge_work_units(
            submission_id="sub-ac", submission_ref="hk-ac", task_ids=["t1"]
        )
        await service.create_prism_work_unit(
            submission_id="psub-1", submission_ref="hk-p"
        )
        await service.assign_pending(seed=1)

        rows = await _rows(factory)
        assert len(rows) == 2
        assert all(r.assigned_validator_hotkey == "open" for r in rows)
        assert {r.challenge_slug for r in rows} == {"agent-challenge", "prism"}
    finally:
        await engine.dispose()


# VAL-VDIR-SUB-004 (capability filtering still applies on top of subscriptions)
async def test_subscription_does_not_override_capability() -> None:
    engine, factory = await _setup()
    try:
        service = _service(factory)
        # cpu-only validator subscribes to prism (a gpu challenge): capability
        # mismatch means it still cannot run the prism unit.
        await _add_validator(factory, "cpu-prism", ["cpu"], subscriptions=["prism"])

        await service.create_prism_work_unit(
            submission_id="psub-1", submission_ref="hk-p"
        )
        assigned = await service.assign_pending(seed=1)

        assert assigned == {}
        rows = await _rows(factory)
        assert rows[0].status == WorkAssignmentStatus.PENDING
    finally:
        await engine.dispose()
