"""Unit tests for master-owned Lium capacity scheduler.

Capacity is never a terminal failure: empty 1-GPU Blackwell inventory leaves
leases ``queued`` (user lock: attendre). Fake client only — no network/money.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from base.compute.lium_capacity import (
    InMemoryLeaseStore,
    LeaseState,
    LiumCapacityScheduler,
    LiumLease,
)
from base.compute.provider import Instance, InstanceSpec, Offer

_BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
_ID = itertools.count(1)


def _offer(*, offer_id: str | None = None, price: float = 1.0) -> Offer:
    oid = offer_id or f"exec-{next(_ID)}"
    return Offer(
        id=oid,
        gpu_type=_BLACKWELL,
        gpu_count=1,
        price_per_hour=price,
    )


@dataclass
class FakeLiumClient:
    """Mutable fake of the Lium surface the scheduler calls."""

    offers: list[Offer] = field(default_factory=list)
    pods: list[dict[str, Any]] = field(default_factory=list)
    provision_calls: list[InstanceSpec] = field(default_factory=list)
    terminate_calls: list[str] = field(default_factory=list)
    training_gpu_lock: bool = True
    _pod_seq: itertools.count = field(default_factory=lambda: itertools.count(1))

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
        self.provision_calls.append(spec)
        if not self.offers and offer is None:
            raise RuntimeError("no capacity — scheduler must not call provision")
        selected = offer or self.offers[0]
        pod_id = f"pod-{next(self._pod_seq)}"
        self.pods.append(
            {
                "id": pod_id,
                "pod_name": spec.name,
                "status": "RUNNING",
                "executor_id": selected.id,
            }
        )
        # Consume one offer slot (inventory shrinks when rented).
        if offer is None and self.offers:
            self.offers = self.offers[1:]
        elif offer is not None:
            self.offers = [o for o in self.offers if o.id != offer.id]
        return Instance(id=pod_id, status="RUNNING", provider="lium")

    async def terminate(self, instance_id: str) -> None:
        self.terminate_calls.append(instance_id)
        self.pods = [p for p in self.pods if str(p.get("id")) != str(instance_id)]


def _scheduler(
    client: FakeLiumClient,
    *,
    concurrency_cap: int = 3,
    store: InMemoryLeaseStore | None = None,
) -> LiumCapacityScheduler:
    return LiumCapacityScheduler(
        lambda: client,
        concurrency_cap=concurrency_cap,
        pod_name_prefix="prism-train-",
        max_price_per_hour=1.50,
        max_lifetime_hours=4.0,
        store=store or InMemoryLeaseStore(),
    )


# -- S1 enqueue idempotent ----------------------------------------------------


def test_enqueue_idempotent() -> None:
    sched = _scheduler(FakeLiumClient())
    first = sched.enqueue(submission_id="sub-1", job_id="job-a")
    second = sched.enqueue(submission_id="sub-1", job_id="job-a")
    assert first.lease_id == second.lease_id
    assert first.submission_id == "sub-1"
    assert first.state is LeaseState.QUEUED
    assert len(sched.store.list_all()) == 1


# -- S2 FIFO admission --------------------------------------------------------


async def test_fifo_admission_order() -> None:
    client = FakeLiumClient(offers=[_offer(), _offer()])
    sched = _scheduler(client, concurrency_cap=1)
    a = sched.enqueue(submission_id="sub-a", job_id="j-a")
    b = sched.enqueue(submission_id="sub-b", job_id="j-b")
    assert a.enqueued_at <= b.enqueued_at

    admitted = await sched.tick()
    assert len(admitted) == 1
    assert admitted[0].submission_id == "sub-a"
    assert admitted[0].state is LeaseState.ACTIVE
    assert admitted[0].pod_id is not None

    still_queued = sched.store.get("sub-b")
    assert still_queued is not None
    assert still_queued.state is LeaseState.QUEUED

    # Free the active slot and admit B.
    client.offers = [_offer()]
    active = sched.store.get("sub-a")
    assert active is not None and active.pod_id is not None
    await client.terminate(active.pod_id)
    sched.store.put(
        LiumLease(
            lease_id=active.lease_id,
            submission_id=active.submission_id,
            job_id=active.job_id,
            state=LeaseState.RELEASED,
            enqueued_at=active.enqueued_at,
            pod_id=active.pod_id,
            reason=None,
        )
    )
    admitted_b = await sched.tick()
    assert len(admitted_b) == 1
    assert admitted_b[0].submission_id == "sub-b"
    assert admitted_b[0].state is LeaseState.ACTIVE


# -- S3 queue when inventory empty --------------------------------------------


async def test_queues_when_inventory_empty() -> None:
    client = FakeLiumClient(offers=[])
    sched = _scheduler(client)
    lease = sched.enqueue(submission_id="sub-wait", job_id="j-wait")
    assert lease.state is LeaseState.QUEUED

    changed = await sched.tick()
    assert changed == []
    after = sched.store.get("sub-wait")
    assert after is not None
    assert after.state is LeaseState.QUEUED
    assert after.reason == "capacity_wait"
    assert client.provision_calls == []


# -- S4 recover reattaches existing pod ---------------------------------------


async def test_recover_reattaches_existing_pod() -> None:
    client = FakeLiumClient(
        pods=[
            {
                "id": "pod-live-9",
                "pod_name": "prism-train-sub-rec",
                "status": "RUNNING",
            }
        ]
    )
    store = InMemoryLeaseStore()
    # Pre-seed a lease that lost process memory of pod_id (queued after crash).
    store.put(
        LiumLease(
            lease_id="lease-rec",
            submission_id="sub-rec",
            job_id="j-rec",
            state=LeaseState.QUEUED,
            enqueued_at=1.0,
            pod_id=None,
            reason="capacity_wait",
        )
    )
    sched = _scheduler(client, store=store)
    recovered = await sched.recover()
    assert len(recovered) == 1
    assert recovered[0].submission_id == "sub-rec"
    assert recovered[0].state is LeaseState.ACTIVE
    assert recovered[0].pod_id == "pod-live-9"
    assert recovered[0].reason is None


# -- S5 cancel queued ---------------------------------------------------------


def test_cancel_queued() -> None:
    sched = _scheduler(FakeLiumClient())
    sched.enqueue(submission_id="sub-c", job_id="j-c")
    cancelled = sched.cancel("sub-c")
    assert cancelled is not None
    assert cancelled.state is LeaseState.CANCELLED
    assert sched.store.get("sub-c") is not None
    assert sched.store.get("sub-c").state is LeaseState.CANCELLED  # type: ignore[union-attr]


def test_cancel_unknown_returns_none() -> None:
    sched = _scheduler(FakeLiumClient())
    assert sched.cancel("missing") is None


async def test_tick_admits_up_to_offer_count_and_cap() -> None:
    client = FakeLiumClient(offers=[_offer(), _offer()])
    sched = _scheduler(client, concurrency_cap=3)
    for i in range(4):
        sched.enqueue(submission_id=f"sub-{i}", job_id=f"j-{i}")
    admitted = await sched.tick()
    assert len(admitted) == 2  # only 2 offers
    assert all(lease.state is LeaseState.ACTIVE for lease in admitted)
    queued = [
        lease for lease in sched.store.list_all() if lease.state is LeaseState.QUEUED
    ]
    assert len(queued) == 2
    assert {lease.submission_id for lease in queued} == {"sub-2", "sub-3"}
