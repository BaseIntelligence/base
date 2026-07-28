"""Master-owned Lium capacity scheduler for Prism training pods.

Queues when no natively 1-GPU Blackwell offer is free — never fails a job for
lack of capacity (user lock: *attendre*). Inventory is always read through a
training-locked client (``LiumClient.for_prism_training`` /
``training_gpu_lock=True``).

Persistence: v1 uses :class:`InMemoryLeaseStore` + :meth:`recover` (reattach
pods named ``{pod_name_prefix}{submission_id}``). Queued-only leases are lost
on process restart without an external store; production should swap SQLite or
Postgres behind :class:`LeaseStore`.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from base.compute.provider import Instance, InstanceSpec, Offer

logger = logging.getLogger(__name__)

REASON_CAPACITY_WAIT = "capacity_wait"
REASON_SPEND_CEILING = "spend_ceiling"


class LeaseState(StrEnum):
    """Lifecycle of one Lium training capacity lease."""

    QUEUED = "queued"
    ADMITTING = "admitting"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    CANCELLED = "cancelled"


_SLOT_HOLDING: frozenset[LeaseState] = frozenset(
    {LeaseState.ADMITTING, LeaseState.ACTIVE, LeaseState.RELEASING}
)
_TERMINAL: frozenset[LeaseState] = frozenset(
    {LeaseState.CANCELLED, LeaseState.RELEASED, LeaseState.RELEASING}
)


@dataclass(frozen=True, slots=True)
class LiumLease:
    """One capacity reservation keyed by ``submission_id``."""

    lease_id: str
    submission_id: str
    job_id: str
    state: LeaseState
    enqueued_at: float
    pod_id: str | None = None
    reason: str | None = None


@runtime_checkable
class LeaseStore(Protocol):
    """Lease map by ``submission_id``. Prod: SQLite/Postgres; tests: in-memory."""

    def get(self, submission_id: str) -> LiumLease | None: ...
    def put(self, lease: LiumLease) -> None: ...
    def list_all(self) -> list[LiumLease]: ...


class InMemoryLeaseStore:
    """Process-local store (not durable across restart)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_submission: dict[str, LiumLease] = {}

    def get(self, submission_id: str) -> LiumLease | None:
        with self._lock:
            return self._by_submission.get(submission_id)

    def put(self, lease: LiumLease) -> None:
        with self._lock:
            self._by_submission[lease.submission_id] = lease

    def list_all(self) -> list[LiumLease]:
        with self._lock:
            return list(self._by_submission.values())


@runtime_checkable
class LiumCapacityClient(Protocol):
    """Minimal async Lium surface the scheduler needs (real or fake)."""

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]: ...

    async def list_pods(self) -> list[dict[str, Any]]: ...

    async def provision(
        self, spec: InstanceSpec, *, offer: Offer | None = None
    ) -> Instance: ...

    async def terminate(self, instance_id: str) -> None: ...


ClientFactory = Callable[[], LiumCapacityClient]
SpendGate = Callable[[], bool]


class LiumCapacityScheduler:
    """FIFO queue + admit loop for 1-GPU Blackwell Lium training pods.

    ``client_factory`` MUST return a training-locked client. Empty inventory
    keeps leases :attr:`LeaseState.QUEUED` with ``reason=capacity_wait``.
    """

    def __init__(
        self,
        client_factory: ClientFactory,
        *,
        concurrency_cap: int = 3,
        pod_name_prefix: str = "prism-train-",
        max_price_per_hour: float = 1.50,
        max_lifetime_hours: float = 4.0,
        store: LeaseStore | None = None,
        spend_gate: SpendGate | None = None,
        template_ref: str = "prism-train",
        image: str = "ghcr.io/base/prism-train:latest",
        ssh_public_keys: Sequence[str] = ("ssh-ed25519 AAAA capacity-scheduler",),
    ) -> None:
        if (
            isinstance(concurrency_cap, bool)
            or not isinstance(concurrency_cap, int)
            or concurrency_cap < 1
        ):
            raise ValueError("concurrency_cap must be a positive integer")
        if max_price_per_hour <= 0 or max_lifetime_hours <= 0:
            raise ValueError(
                "max_price_per_hour and max_lifetime_hours must be positive"
            )
        self._client_factory = client_factory
        self._concurrency_cap = concurrency_cap
        self._pod_name_prefix = pod_name_prefix
        self._max_price_per_hour = max_price_per_hour
        self._max_lifetime_hours = max_lifetime_hours
        self._store: LeaseStore = store if store is not None else InMemoryLeaseStore()
        self._spend_gate = spend_gate
        self._template_ref = template_ref
        self._image = image
        self._ssh_public_keys = tuple(ssh_public_keys)
        self._lock = threading.Lock()

    @property
    def store(self) -> LeaseStore:
        """Backing lease store."""
        return self._store

    def pod_name_for(self, submission_id: str) -> str:
        """Stable Lium ``pod_name`` for a submission (used by recover)."""
        return f"{self._pod_name_prefix}{submission_id}"

    def enqueue(self, *, submission_id: str, job_id: str) -> LiumLease:
        """Enqueue a capacity request. Idempotent on ``submission_id``."""
        if not submission_id:
            raise ValueError("submission_id must be non-empty")
        if not job_id:
            raise ValueError("job_id must be non-empty")
        with self._lock:
            existing = self._store.get(submission_id)
            if existing is not None:
                return existing
            lease = LiumLease(
                lease_id=f"lium-lease-{uuid.uuid4().hex}",
                submission_id=submission_id,
                job_id=job_id,
                state=LeaseState.QUEUED,
                enqueued_at=time.time(),
            )
            self._store.put(lease)
            return lease

    def cancel(self, submission_id: str) -> LiumLease | None:
        """Cancel a queued lease. Unknown → ``None``; non-queued left unchanged."""
        with self._lock:
            lease = self._store.get(submission_id)
            if lease is None:
                return None
            if lease.state is LeaseState.CANCELLED:
                return lease
            if lease.state is not LeaseState.QUEUED:
                return lease
            cancelled = replace(lease, state=LeaseState.CANCELLED, reason=None)
            self._store.put(cancelled)
            return cancelled

    async def tick(self) -> list[LiumLease]:
        """Admit FIFO queued leases up to free slots. Never raises for capacity."""
        client = self._client_factory()
        if self._spend_gate is not None and not self._spend_gate():
            self._mark_queued_reason(REASON_SPEND_CEILING)
            return []

        offers = list(
            await client.list_offers(max_price_per_hour=self._max_price_per_hour)
        )
        free = self._free_slots(len(offers))
        if free <= 0:
            self._mark_queued_reason(REASON_CAPACITY_WAIT)
            return []

        admitted: list[LiumLease] = []
        for lease in self._queued_fifo():
            if len(admitted) >= free or not offers:
                if not offers:
                    self._set_reason(lease.submission_id, REASON_CAPACITY_WAIT)
                break
            offer = offers.pop(0)
            result = await self._admit_one(client, lease, offer)
            if result is None:
                self._set_reason(lease.submission_id, REASON_CAPACITY_WAIT)
                break
            admitted.append(result)
        return admitted

    async def recover(self) -> list[LiumLease]:
        """Reattach prefix-matching live pods to stored leases as ACTIVE."""
        client = self._client_factory()
        pods = await client.list_pods()
        recovered: list[LiumLease] = []
        prefix = self._pod_name_prefix
        with self._lock:
            for pod in pods:
                pod_id = pod.get("id")
                name = pod.get("pod_name") or pod.get("name")
                if not pod_id or not name:
                    continue
                name_s = str(name)
                if not name_s.startswith(prefix):
                    continue
                submission_id = name_s[len(prefix) :]
                if not submission_id:
                    continue
                lease = self._store.get(submission_id)
                if lease is None or lease.state in _TERMINAL:
                    continue
                if lease.state is LeaseState.ACTIVE and lease.pod_id == str(pod_id):
                    continue
                updated = replace(
                    lease,
                    state=LeaseState.ACTIVE,
                    pod_id=str(pod_id),
                    reason=None,
                )
                self._store.put(updated)
                recovered.append(updated)
        return recovered

    def _active_count(self) -> int:
        return sum(
            1 for lease in self._store.list_all() if lease.state in _SLOT_HOLDING
        )

    def _free_slots(self, offer_count: int) -> int:
        remaining = self._concurrency_cap - self._active_count()
        if remaining <= 0:
            return 0
        return min(remaining, max(0, offer_count))

    def _queued_fifo(self) -> list[LiumLease]:
        queued = [
            lease
            for lease in self._store.list_all()
            if lease.state is LeaseState.QUEUED
        ]
        return sorted(queued, key=lambda lease: (lease.enqueued_at, lease.lease_id))

    def _mark_queued_reason(self, reason: str) -> None:
        with self._lock:
            for lease in self._store.list_all():
                if lease.state is LeaseState.QUEUED and lease.reason != reason:
                    self._store.put(replace(lease, reason=reason))

    def _set_reason(self, submission_id: str, reason: str) -> None:
        with self._lock:
            lease = self._store.get(submission_id)
            if lease is None or lease.state is not LeaseState.QUEUED:
                return
            if lease.reason != reason:
                self._store.put(replace(lease, reason=reason))

    def _build_spec(self, submission_id: str) -> InstanceSpec:
        return InstanceSpec(
            name=self.pod_name_for(submission_id),
            template_ref=self._template_ref,
            image=self._image,
            ssh_public_keys=self._ssh_public_keys,
            max_lifetime_hours=self._max_lifetime_hours,
            max_price_per_hour=self._max_price_per_hour,
            gpu_count=1,
        )

    async def _admit_one(
        self,
        client: LiumCapacityClient,
        lease: LiumLease,
        offer: Offer,
    ) -> LiumLease | None:
        with self._lock:
            current = self._store.get(lease.submission_id)
            if current is None or current.state is not LeaseState.QUEUED:
                return None
            self._store.put(replace(current, state=LeaseState.ADMITTING, reason=None))

        try:
            instance = await client.provision(
                self._build_spec(lease.submission_id), offer=offer
            )
        except Exception:
            logger.exception(
                "lium capacity admit failed for submission_id=%s; re-queue",
                lease.submission_id,
            )
            with self._lock:
                current = self._store.get(lease.submission_id)
                if current is not None and current.state is LeaseState.ADMITTING:
                    self._store.put(
                        replace(
                            current,
                            state=LeaseState.QUEUED,
                            reason=REASON_CAPACITY_WAIT,
                        )
                    )
            return None

        with self._lock:
            current = self._store.get(lease.submission_id)
            if current is None:
                return None
            if current.state is LeaseState.CANCELLED:
                orphan_pod_id: str | None = instance.id
            else:
                orphan_pod_id = None
                active = replace(
                    current,
                    state=LeaseState.ACTIVE,
                    pod_id=instance.id,
                    reason=None,
                )
                self._store.put(active)
                return active

        try:
            await client.terminate(orphan_pod_id)
        except Exception:  # noqa: BLE001 - cancel race must not raise
            logger.warning(
                "terminate after cancel race failed for pod %s", orphan_pod_id
            )
        return None
