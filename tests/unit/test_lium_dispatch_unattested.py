"""T14 — unattested master-owned Lium dispatch path (mocked; no billable rent).

Proves the code path that submits Prism GPU work onto Lium capacity without
constation elevation or live provider calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from base.compute.lium_capacity import (
    InMemoryLeaseStore,
    LeaseState,
    LiumCapacityScheduler,
)
from base.compute.lium_training_wiring import (
    build_lium_capacity_scheduler,
    try_build_lium_capacity_scheduler,
)
from base.compute.provider import Instance, InstanceSpec, Offer
from base.config.settings import LiumTrainingSettings, Settings
from base.db import Base, create_engine, create_session_factory
from base.db.models import WorkAssignment
from base.master.assignment import AssignmentService
from base.master.orchestration import ChallengePendingWork, MasterOrchestrationDriver
from base.master.validator_coordination import ValidatorCoordinationService

NOW = datetime(2026, 6, 27, 12, 0, 0, tzinfo=UTC)

_BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


@dataclass
class FakeLiumClient:
    """In-memory Lium surface — never hits the network."""

    offers: list[Offer] = field(default_factory=list)
    provisioned: list[InstanceSpec] = field(default_factory=list)
    terminated: list[str] = field(default_factory=list)
    pods: list[dict[str, Any]] = field(default_factory=list)
    _seq: int = 0

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]:
        out: list[Offer] = []
        for offer in self.offers:
            if (
                max_price_per_hour is not None
                and offer.price_per_hour > max_price_per_hour
            ):
                continue
            out.append(offer)
        return out

    async def list_pods(self) -> list[dict[str, Any]]:
        return list(self.pods)

    async def provision(
        self, spec: InstanceSpec, *, offer: Offer | None = None
    ) -> Instance:
        self.provisioned.append(spec)
        self._seq += 1
        pod_id = f"pod-fake-{self._seq}"
        self.pods.append(
            {
                "id": pod_id,
                "pod_name": spec.name,
                "name": spec.name,
                "status": "RUNNING",
                "provider": "lium",
            }
        )
        if offer is not None:
            self.offers = [o for o in self.offers if o.id != offer.id]
        elif self.offers:
            self.offers = self.offers[1:]
        return Instance(id=pod_id, status="RUNNING", provider="lium")

    async def terminate(self, instance_id: str) -> None:
        self.terminated.append(instance_id)
        self.pods = [p for p in self.pods if str(p.get("id")) != str(instance_id)]


def _blackwell_offer(*, offer_id: str = "offer-bw-1", price: float = 1.0) -> Offer:
    return Offer(
        id=offer_id,
        gpu_type=_BLACKWELL,
        gpu_count=1,
        price_per_hour=price,
        provider="lium",
        raw={"machine_name": "rtx-pro-6000-blackwell-server"},
    )


def test_try_build_scheduler_none_when_disabled() -> None:
    """Given lium_training.enabled=False, When try_build, Then None (default off)."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=False))
    assert try_build_lium_capacity_scheduler(settings) is None


def test_try_build_scheduler_fail_closed_missing_key_returns_none() -> None:
    """Given enabled without key, When try_build, Then None (master still boots)."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=True))
    assert try_build_lium_capacity_scheduler(settings) is None


def test_build_scheduler_fail_closed_raises_without_key() -> None:
    """Given enabled without key, When hard build, Then ValueError names key fields."""
    settings = Settings(lium_training=LiumTrainingSettings(enabled=True))
    with pytest.raises(ValueError, match="api_key|api_key_file|lium_training"):
        build_lium_capacity_scheduler(settings)


async def test_dispatch_enqueue_and_tick_provisions_mocked_pod() -> None:
    """Given Prism pending work + fake Lium inventory, When bridge+tick, Then pod provisioned.

    End-to-end master-owned dispatch without constation and without live API.
    """
    client = FakeLiumClient(offers=[_blackwell_offer()])
    store = InMemoryLeaseStore()
    scheduler = LiumCapacityScheduler(
        lambda: client,
        store=store,
        ssh_public_keys=("ssh-ed25519 AAAA test-dispatch",),
    )

    engine = create_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = create_session_factory(engine)
        service = AssignmentService(factory, now_fn=lambda: NOW)
        validators = ValidatorCoordinationService(factory, now_fn=lambda: NOW)

        @dataclass
        class _Src:
            async def fetch_pending_work(self) -> list[ChallengePendingWork]:
                return [
                    ChallengePendingWork(
                        challenge_slug="prism",
                        submission_id="sub-t14-1",
                        submission_ref="miner-hk",
                        checkpoint_ref="hf://ckpt/1",
                        job_id="job-t14-1",
                    )
                ]

        driver = MasterOrchestrationDriver(
            assignment_service=service,
            validator_service=validators,
            work_source=_Src(),
            lium_scheduler=scheduler,
        )

        # bridge enqueues; run_once ticks Lium and provisions
        result = await driver.run_once()
        assert result.bridged["prism"] == ["sub-t14-1"]

        assert len(client.provisioned) == 1
        assert client.provisioned[0].name == "prism-train-sub-t14-1"
        active = store.get("sub-t14-1")
        assert active is not None
        assert active.state is LeaseState.ACTIVE
        assert active.pod_id is not None

        async with factory() as session:
            rows = list((await session.execute(select(WorkAssignment))).scalars().all())
        assert len(rows) == 1
        assert rows[0].work_unit_id == "sub-t14-1"
        assert rows[0].required_capability == "gpu"
    finally:
        await engine.dispose()


async def test_dispatch_queues_when_inventory_empty_no_terminal_fail() -> None:
    """Given empty Lium inventory, When tick, Then lease stays queued (capacity_wait)."""
    client = FakeLiumClient(offers=[])
    store = InMemoryLeaseStore()
    scheduler = LiumCapacityScheduler(
        lambda: client,
        store=store,
        ssh_public_keys=("ssh-ed25519 AAAA test-dispatch",),
    )
    scheduler.enqueue(submission_id="sub-wait", job_id="job-wait")
    await scheduler.tick()
    lease = store.get("sub-wait")
    assert lease is not None
    assert lease.state is LeaseState.QUEUED
    assert lease.reason == "capacity_wait"
    assert client.provisioned == []
