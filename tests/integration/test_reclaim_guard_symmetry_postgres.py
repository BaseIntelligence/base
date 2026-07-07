"""Flag-ON ``run_once`` integration proof for reclaim/assign guard symmetry.

A live :class:`MasterOrchestrationDriver.run_once` pass runs both the legacy
reassignment pass (detect -> reclaim -> assign) and the worker-plane engine.
Worker-owned prism PRIMARY units are ``ASSIGNED`` with a NULL
``assigned_validator_hotkey`` by design (owned by the ``worker_assignments``
replica plane). Before this fix the legacy ``reclaim_stale_assignments`` read a
null hotkey as "offline validator => reassignable" and churned those primaries
back to PENDING every pass. This test proves that under
``BASE_COMPUTE__WORKER_PLANE_ENABLED`` semantics (validator
``AssignmentService`` configured with ``worker_plane_capabilities={"gpu"}``) a
worker-owned primary stays ASSIGNED across repeated passes with NO PENDING churn,
while a genuinely stale validator-owned unit is still reclaimed (reclaim is not
globally disabled). Runs against the mission test Postgres (127.0.0.1:15433).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Validator,
    ValidatorStatus,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.orchestration import (
    ChallengePendingWork,
    MasterOrchestrationDriver,
)
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import WorkerReconciliationService
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)

pytestmark = pytest.mark.postgres

NOW = datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC)
TTL = 120


@dataclass
class _FakeWorkSource:
    works: list[ChallengePendingWork] = field(default_factory=list)

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        return list(self.works)


@dataclass
class _FakeFoldTrigger:
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)

    async def fold(
        self, *, challenge_slug: str, job_id: str, task_id: str, reason: str
    ) -> None:
        self.calls.append((challenge_slug, job_id, task_id, reason))


class _FakeForwarder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Any,
    ) -> None:
        self.calls.append(work_unit_id)


async def _add_worker(factory: Any, *, worker_pubkey: str, miner_hotkey: str) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkerRegistration(
                worker_id=f"wid-{worker_pubkey}",
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


async def _add_validator(factory: Any, hotkey: str, capabilities: list[str]) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=list(capabilities),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _set_validator_status(
    factory: Any, hotkey: str, status: ValidatorStatus
) -> None:
    async with session_scope(factory) as session:
        validator = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        validator.status = status


async def _rows(factory: Any) -> dict[str, WorkAssignment]:
    async with factory() as session:
        result = await session.execute(select(WorkAssignment))
        return {r.work_unit_id: r for r in result.scalars().all()}


async def _active_replicas(factory: Any, work_unit_id: str) -> list[WorkerAssignment]:
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(WorkerAssignment).where(
                        WorkerAssignment.work_unit_id == work_unit_id,
                        WorkerAssignment.status.in_(
                            (
                                WorkAssignmentStatus.ASSIGNED,
                                WorkAssignmentStatus.RUNNING,
                            )
                        ),
                    )
                )
            )
            .scalars()
            .all()
        )
        return list(rows)


def _build_flag_on_driver(factory: Any) -> tuple[MasterOrchestrationDriver, Any, Any]:
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=lambda: NOW,
    )
    worker_assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=lambda: NOW
    )
    worker_engine = WorkerAssignmentEngine(
        factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=lambda: NOW,
    )
    forwarder = _FakeForwarder()
    reconciler = WorkerReconciliationService(
        factory, result_forwarder=forwarder, now_fn=lambda: NOW
    )
    # Flag-ON semantics: gpu units are owned by the worker plane, so the validator
    # AssignmentService is configured to skip them in BOTH assign and reclaim.
    assignment_service = AssignmentService(
        factory,
        now_fn=lambda: NOW,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    validator_service = ValidatorCoordinationService(factory, now_fn=lambda: NOW)
    fold = _FakeFoldTrigger()
    driver = MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=validator_service,
        work_source=_FakeWorkSource(
            works=[
                ChallengePendingWork(
                    challenge_slug="prism",
                    submission_id="psub",
                    submission_ref="miner-H",
                ),
                ChallengePendingWork(
                    challenge_slug="agent-challenge",
                    submission_id="sub",
                    submission_ref="miner-C",
                    task_ids=("a",),
                    job_id="job",
                ),
            ]
        ),
        fold_trigger=fold,
        worker_assignment_engine=worker_engine,
        worker_reconciler=reconciler,
        seed=1,
    )
    return driver, forwarder, fold


async def test_flag_on_run_once_keeps_worker_primary_assigned_no_churn(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine_db = create_engine(migrated_postgres_database)
    factory = create_session_factory(engine_db)
    driver, forwarder, _fold = _build_flag_on_driver(factory)
    try:
        # Two workers of distinct owners, neither owned by the submitter miner-H.
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey="miner-A")
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey="miner-B")
        # A cpu validator holds the agent-challenge (validator-owned) unit.
        await _add_validator(factory, "v1", ["cpu"])

        # Pass 1: bridge + replicate the gpu primary across two workers and assign
        # the cpu unit to the validator.
        await driver.run_once()
        rows = await _rows(factory)
        assert rows["psub"].status == WorkAssignmentStatus.ASSIGNED
        assert rows["psub"].assigned_validator_hotkey is None  # worker-owned
        assert len(await _active_replicas(factory, "psub")) == 2
        assert rows["sub:a"].status == WorkAssignmentStatus.ASSIGNED
        assert rows["sub:a"].assigned_validator_hotkey == "v1"

        # The cpu validator crashes; its (validator-owned) unit must still be
        # reclaimed by the legacy reclaim path on the next passes.
        await _set_validator_status(factory, "v1", ValidatorStatus.OFFLINE)

        # Repeated passes must NOT churn the worker-owned primary to PENDING.
        for _ in range(3):
            await driver.run_once()
            rows = await _rows(factory)
            assert rows["psub"].status == WorkAssignmentStatus.ASSIGNED
            assert rows["psub"].assigned_validator_hotkey is None
            assert rows["psub"].attempt_count == 1  # never re-incremented
            assert len(await _active_replicas(factory, "psub")) == 2

        # Reconciliation never fired (no worker posted a result), so the primary
        # was held ASSIGNED purely by the reclaim guard, not by acceptance.
        assert forwarder.calls == []

        # Reclaim is NOT globally disabled: the stale validator-owned cpu unit was
        # reverted off the offline validator (reclaimed to pending, no re-assign
        # target online).
        rows = await _rows(factory)
        assert rows["sub:a"].status == WorkAssignmentStatus.PENDING
        assert rows["sub:a"].assigned_validator_hotkey is None
    finally:
        await engine_db.dispose()
