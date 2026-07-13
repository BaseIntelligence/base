"""Postgres parity tests for the assignment coordination plane + work_results.

Covers the coordination lifecycle (assign -> pull -> progress -> result) and the
``work_results`` migration on the throwaway Postgres at 127.0.0.1:15490, mirroring
the SQLite unit-level behavior (VAL-ASSIGN-015..021 persistence parity).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text

from base.db import (
    Validator,
    ValidatorStatus,
    WorkResult,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment, WorkAssignmentStatus
from base.master.assignment import AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)


async def test_work_results_table_exists_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    try:
        async with engine.connect() as connection:
            columns = (
                (
                    await connection.execute(
                        text(
                            """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                              AND table_name = 'work_results'
                            """
                        )
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()

    assert {
        "id",
        "assignment_id",
        "challenge_slug",
        "work_unit_id",
        "submission_ref",
        "validator_hotkey",
        "success",
        "payload",
        "created_at",
    } <= set(columns)


async def test_coordination_lifecycle_parity_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    assignment_service = AssignmentService(session_factory, now_fn=lambda: NOW)
    coordination = AssignmentCoordinationService(
        session_factory, lease_seconds=900, now_fn=lambda: NOW
    )

    try:
        async with session_scope(session_factory) as session:
            session.add(
                Validator(
                    hotkey="vp-gpu",
                    uid=None,
                    status=ValidatorStatus.ONLINE,
                    capabilities=["cpu", "gpu"],
                    version="1.0.0",
                    registered_at=NOW,
                    last_heartbeat_at=NOW,
                )
            )

        await assignment_service.create_prism_work_unit(
            submission_id="psub-pg", submission_ref="hk-pg"
        )
        await assignment_service.assign_pending(seed=1)

        pulled = await coordination.pull(hotkey="vp-gpu")
        assert len(pulled) == 1
        assignment_id = str(pulled[0].id)

        # Pull transitioned the unit to running with a future lease deadline.
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(WorkAssignment).where(WorkAssignment.id == pulled[0].id)
                )
            ).scalar_one()
            assert row.status == WorkAssignmentStatus.RUNNING
            assert row.deadline_at is not None

        await coordination.progress(
            assignment_id=assignment_id,
            hotkey="vp-gpu",
            checkpoint_ref="hf://ckpt/pg",
        )

        outcome = await coordination.post_result(
            assignment_id=assignment_id,
            hotkey="vp-gpu",
            success=True,
            payload={"score": 0.77},
        )
        assert outcome.idempotent is False
        assert outcome.status == "completed"

        # Exact re-post is idempotent; differing payload is a conflict.
        repeat = await coordination.post_result(
            assignment_id=assignment_id,
            hotkey="vp-gpu",
            success=True,
            payload={"score": 0.77},
        )
        assert repeat.idempotent is True
        assert repeat.result_ref == outcome.result_ref

        async with session_factory() as session:
            final = (
                await session.execute(
                    select(WorkAssignment).where(WorkAssignment.id == pulled[0].id)
                )
            ).scalar_one()
            assert final.status == WorkAssignmentStatus.COMPLETED
            assert final.result_ref == outcome.result_ref
            assert final.checkpoint_ref == "hf://ckpt/pg"

            result_count = await session.scalar(
                select(func.count(WorkResult.id)).where(
                    WorkResult.assignment_id == pulled[0].id
                )
            )
            assert result_count == 1
    finally:
        await engine.dispose()
