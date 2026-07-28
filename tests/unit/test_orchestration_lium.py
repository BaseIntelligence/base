"""Lium capacity admission hooks on Prism GPU bridge (master-owned).

Fake scheduler only — no Lium network/money. Proves enqueue on prism unit
bridge when a scheduler is injected; default None leaves legacy path alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select

from base.db import Base, create_engine, create_session_factory
from base.db.models import WorkAssignment
from base.master.assignment import AssignmentService
from base.master.orchestration import ChallengePendingWork, MasterOrchestrationDriver
from base.master.validator_coordination import ValidatorCoordinationService

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)


@dataclass
class FakeLiumScheduler:
    """Records enqueue/tick; no provision or network."""

    enqueue_calls: list[tuple[str, str]] = field(default_factory=list)
    tick_calls: int = 0

    def enqueue(self, *, submission_id: str, job_id: str) -> object:
        self.enqueue_calls.append((submission_id, job_id))
        return object()

    async def tick(self) -> list[object]:
        self.tick_calls += 1
        return []


@dataclass
class FakeWorkSource:
    works: list[ChallengePendingWork] = field(default_factory=list)

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        return list(self.works)


def _prism_work(
    *,
    submission_id: str = "psub-lium-1",
    job_id: str | None = "job-lium-1",
) -> ChallengePendingWork:
    return ChallengePendingWork(
        challenge_slug="prism",
        submission_id=submission_id,
        submission_ref="miner-hk-p",
        checkpoint_ref="hf://ckpt/step-3",
        job_id=job_id,
    )


async def _setup():
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    return engine, factory


async def test_orchestration_enqueues_lium_lease_when_scheduler_present() -> None:
    """Prism GPU bridge success must admit via scheduler.enqueue (wait, not fail)."""
    engine, factory = await _setup()
    try:
        service = AssignmentService(factory, now_fn=lambda: NOW)
        validators = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
        source = FakeWorkSource(works=[_prism_work()])
        scheduler = FakeLiumScheduler()
        driver = MasterOrchestrationDriver(
            assignment_service=service,
            validator_service=validators,
            work_source=source,
            lium_scheduler=scheduler,
        )

        bridged = await driver.bridge_pending_work()
        assert bridged["prism"] == ["psub-lium-1"]
        assert scheduler.enqueue_calls == [("psub-lium-1", "job-lium-1")]

        async with factory() as session:
            rows = list(
                (
                    await session.execute(select(WorkAssignment))
                ).scalars().all()
            )
        assert len(rows) == 1
        assert rows[0].required_capability == "gpu"
    finally:
        await engine.dispose()


async def test_orchestration_skips_lium_when_scheduler_none() -> None:
    """Default (no scheduler) keeps prism bridge behavior unchanged."""
    engine, factory = await _setup()
    try:
        service = AssignmentService(factory, now_fn=lambda: NOW)
        validators = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
        source = FakeWorkSource(works=[_prism_work()])
        driver = MasterOrchestrationDriver(
            assignment_service=service,
            validator_service=validators,
            work_source=source,
        )

        bridged = await driver.bridge_pending_work()
        assert bridged["prism"] == ["psub-lium-1"]
        async with factory() as session:
            rows = list(
                (
                    await session.execute(select(WorkAssignment))
                ).scalars().all()
            )
        assert len(rows) == 1
        assert rows[0].work_unit_id == "psub-lium-1"
    finally:
        await engine.dispose()


async def test_orchestration_run_once_ticks_lium_scheduler() -> None:
    """Orchestration pass calls scheduler.tick when present (admission loop)."""
    engine, factory = await _setup()
    try:
        service = AssignmentService(factory, now_fn=lambda: NOW)
        validators = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
        source = FakeWorkSource(works=[_prism_work()])
        scheduler = FakeLiumScheduler()
        driver = MasterOrchestrationDriver(
            assignment_service=service,
            validator_service=validators,
            work_source=source,
            lium_scheduler=scheduler,
        )

        await driver.run_once()
        assert scheduler.enqueue_calls == [("psub-lium-1", "job-lium-1")]
        assert scheduler.tick_calls == 1
    finally:
        await engine.dispose()


async def test_orchestration_enqueue_uses_submission_id_when_job_id_missing() -> None:
    """Prism descriptors often omit job_id; enqueue still needs a non-empty id."""
    engine, factory = await _setup()
    try:
        service = AssignmentService(factory, now_fn=lambda: NOW)
        validators = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
        source = FakeWorkSource(works=[_prism_work(job_id=None)])
        scheduler = FakeLiumScheduler()
        driver = MasterOrchestrationDriver(
            assignment_service=service,
            validator_service=validators,
            work_source=source,
            lium_scheduler=scheduler,
        )

        await driver.bridge_pending_work()
        assert scheduler.enqueue_calls == [("psub-lium-1", "psub-lium-1")]
    finally:
        await engine.dispose()
