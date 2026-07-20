from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast
from uuid import uuid4

import aiosqlite

from .config import PrismSettings
from .db import Database, dumps, loads
from .repository import now_iso
from .runtime_config import RuntimePolicy

GpuLeaseStatus = Literal["queued", "active", "released"]


@dataclass(frozen=True)
class BaseGpuTarget:
    id: str
    server: str | None = None
    enabled: bool = True
    draining: bool = False
    gpu_count: int = 0


@dataclass(frozen=True)
class GpuLeaseRequest:
    submission_id: str
    job_id: str | None
    mode: str
    tier: str
    score_eligible: bool
    min_gpu_count: int
    max_gpu_count: int
    requested_gpu_count: int
    autosplit_allowed: bool
    official_fixed_profile: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.min_gpu_count < 1:
            raise ValueError("min_gpu_count must be >= 1")
        if self.max_gpu_count > 8:
            raise ValueError("max_gpu_count must be <= 8")
        if self.max_gpu_count < self.min_gpu_count:
            raise ValueError("max_gpu_count must be >= min_gpu_count")
        if not self.min_gpu_count <= self.requested_gpu_count <= self.max_gpu_count:
            raise ValueError("requested_gpu_count must be in [min_gpu_count, max_gpu_count]")


@dataclass(frozen=True)
class GpuLease:
    id: str
    submission_id: str
    job_id: str | None
    target_id: str | None
    target_server: str | None
    device_ids: tuple[str, ...]
    gpu_count: int
    min_gpu_count: int
    max_gpu_count: int
    requested_gpu_count: int
    mode: str
    tier: str
    score_eligible: bool
    autosplit_allowed: bool
    official_fixed_profile: bool
    status: GpuLeaseStatus
    reason: str
    created_at: str
    released_at: str | None = None

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass(frozen=True)
class GpuCapacitySnapshot:
    total_devices: int
    active_devices: int
    active_leases: int
    queued_leases: int

    @property
    def free_devices(self) -> int:
        return max(0, self.total_devices - self.active_devices)

    @property
    def at_capacity(self) -> bool:
        return self.free_devices == 0

    @property
    def oversubscribed(self) -> bool:
        return self.active_devices > self.total_devices


class GpuLeaseScheduler:
    def __init__(self, database: Database, targets: tuple[BaseGpuTarget, ...]) -> None:
        self.database = database
        self.targets = targets

    async def enqueue_or_allocate(self, request: GpuLeaseRequest) -> GpuLease:
        async with self.database.connect() as conn:
            await conn.execute("BEGIN IMMEDIATE")
            existing = await self._active_or_queued_lease(conn, request.submission_id)
            if existing is None:
                await self._insert_queued(conn, request)
            await self._schedule_fifo(conn)
            lease = await self._active_or_queued_lease(conn, request.submission_id)
            if lease is None:
                raise RuntimeError("GPU lease was not persisted")
            return lease

    async def release_for_submission(self, submission_id: str, reason: str) -> None:
        now = now_iso()
        async with self.database.connect() as conn:
            await conn.execute(
                "UPDATE gpu_leases SET status='released', released_at=?, updated_at=?, reason=? "
                "WHERE submission_id=? AND status='active'",
                (now, now, reason, submission_id),
            )
            await self._schedule_fifo(conn)

    async def active_lease_for_submission(self, submission_id: str) -> GpuLease | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM gpu_leases WHERE submission_id=? AND status='active' "
                "ORDER BY created_at LIMIT 1",
                (submission_id,),
            )
        row_list = list(rows)
        return _lease_from_row(dict(row_list[0])) if row_list else None

    async def leases(self) -> list[GpuLease]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall("SELECT * FROM gpu_leases ORDER BY created_at, id")
        return [_lease_from_row(dict(row)) for row in rows]

    async def capacity_snapshot(self) -> GpuCapacitySnapshot:
        async with self.database.connect() as conn:
            leased = await self._leased_devices(conn)
            queued_rows = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM gpu_leases WHERE status='queued'"
            )
            active_rows = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM gpu_leases WHERE status='active'"
            )
        total_devices = sum(
            target.gpu_count for target in self.targets if target.enabled and not target.draining
        )
        active_devices = sum(len(devices) for devices in leased.values())
        queued_list = list(queued_rows)
        active_list = list(active_rows)
        return GpuCapacitySnapshot(
            total_devices=total_devices,
            active_devices=active_devices,
            active_leases=int(active_list[0]["n"]) if active_list else 0,
            queued_leases=int(queued_list[0]["n"]) if queued_list else 0,
        )

    async def _active_or_queued_lease(
        self, conn: aiosqlite.Connection, submission_id: str
    ) -> GpuLease | None:
        rows = await conn.execute_fetchall(
            "SELECT * FROM gpu_leases WHERE submission_id=? AND status IN ('active', 'queued') "
            "ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, created_at LIMIT 1",
            (submission_id,),
        )
        row_list = list(rows)
        return _lease_from_row(dict(row_list[0])) if row_list else None

    async def _insert_queued(self, conn: aiosqlite.Connection, request: GpuLeaseRequest) -> None:
        now = now_iso()
        await conn.execute(
            "INSERT INTO gpu_leases("
            "id, submission_id, job_id, target_id, target_server, device_ids, gpu_count, "
            "min_gpu_count, max_gpu_count, requested_gpu_count, mode, tier, score_eligible, "
            "autosplit_allowed, official_fixed_profile, status, created_at, updated_at, reason) "
            "VALUES (?, ?, ?, NULL, NULL, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
            (
                str(uuid4()),
                request.submission_id,
                request.job_id,
                dumps([]),
                request.min_gpu_count,
                request.max_gpu_count,
                request.requested_gpu_count,
                request.mode,
                request.tier,
                int(request.score_eligible),
                int(request.autosplit_allowed),
                int(request.official_fixed_profile),
                now,
                now,
                request.reason or "queued for GPU allocation",
            ),
        )

    async def _schedule_fifo(self, conn: aiosqlite.Connection) -> None:
        rows = await conn.execute_fetchall(
            "SELECT * FROM gpu_leases WHERE status='queued' ORDER BY created_at, id"
        )
        leased = await self._leased_devices(conn)
        for row in rows:
            lease = _lease_from_row(dict(row))
            allocation = _choose_allocation(lease, self.targets, leased)
            if allocation is None:
                await conn.execute(
                    "UPDATE gpu_leases SET reason=?, updated_at=? WHERE id=?",
                    (_unavailable_reason(lease), now_iso(), lease.id),
                )
                continue
            target, device_ids = allocation
            for device_id in device_ids:
                leased.setdefault(target.id, set()).add(device_id)
            now = now_iso()
            await conn.execute(
                "UPDATE gpu_leases SET target_id=?, target_server=?, device_ids=?, gpu_count=?, "
                "status='active', reason=?, updated_at=? WHERE id=?",
                (
                    target.id,
                    target.server or target.id,
                    dumps(list(device_ids)),
                    len(device_ids),
                    "allocated",
                    now,
                    lease.id,
                ),
            )

    async def _leased_devices(self, conn: aiosqlite.Connection) -> dict[str, set[str]]:
        rows = await conn.execute_fetchall(
            "SELECT target_id, device_ids FROM gpu_leases "
            "WHERE status='active' AND target_id IS NOT NULL"
        )
        leased: dict[str, set[str]] = {}
        for row in rows:
            target_id = str(row["target_id"])
            device_ids = loads(str(row["device_ids"]))
            if isinstance(device_ids, list):
                leased.setdefault(target_id, set()).update(
                    str(device_id) for device_id in device_ids
                )
        return leased


def lease_request_from_runtime(
    *,
    submission_id: str,
    job_id: str | None,
    runtime_policy: RuntimePolicy,
    mode: str,
    score_eligible: bool | None = None,
) -> GpuLeaseRequest:
    mode_target = runtime_policy.execution_mode_targets.model_dump().get(mode, {})
    configured_gpu_count = int(runtime_policy.gpu_policy.actual_gpu_count)
    is_score_eligible = bool(
        mode_target.get("official_score", False) if score_eligible is None else score_eligible
    )
    max_gpu_count = int(runtime_policy.gpu_policy.max_gpu_count)
    requested_gpu_count = configured_gpu_count if is_score_eligible else max_gpu_count
    return GpuLeaseRequest(
        submission_id=submission_id,
        job_id=job_id,
        mode=mode,
        tier=mode,
        score_eligible=is_score_eligible,
        min_gpu_count=1,
        max_gpu_count=max_gpu_count,
        requested_gpu_count=max(1, requested_gpu_count),
        autosplit_allowed=not is_score_eligible,
        official_fixed_profile=bool(runtime_policy.gpu_policy.official_fixed_profile),
        reason="runtime GPU policy",
    )


def targets_from_settings(
    settings: PrismSettings, runtime_policy: RuntimePolicy
) -> tuple[BaseGpuTarget, ...]:
    raw_targets = settings.base_gpu_targets
    if raw_targets:
        payload = loads(raw_targets)
        if not isinstance(payload, list):
            raise ValueError("base_gpu_targets must be a JSON list")
        return tuple(_target_from_mapping(cast(dict[str, Any], item)) for item in payload)
    return (
        BaseGpuTarget(
            id="local-base",
            server="local-base",
            enabled=True,
            draining=False,
            gpu_count=max(1, int(runtime_policy.gpu_policy.actual_gpu_count)),
        ),
    )


def _target_from_mapping(item: dict[str, Any]) -> BaseGpuTarget:
    return BaseGpuTarget(
        id=str(item["id"]),
        server=str(item.get("server") or item.get("api_url") or item["id"]),
        enabled=bool(item.get("enabled", True)),
        draining=bool(item.get("draining", False)),
        gpu_count=int(item.get("gpu_count", 0)),
    )


def _choose_allocation(
    lease: GpuLease,
    targets: tuple[BaseGpuTarget, ...],
    leased: dict[str, set[str]],
) -> tuple[BaseGpuTarget, tuple[str, ...]] | None:
    if lease.score_eligible or not lease.autosplit_allowed:
        candidate_counts: tuple[int, ...] = (lease.requested_gpu_count,)
    else:
        candidate_counts = tuple(range(lease.requested_gpu_count, lease.min_gpu_count - 1, -1))
    for count in candidate_counts:
        if count < 1 or count > lease.max_gpu_count:
            continue
        for target in targets:
            if not target.enabled or target.draining or target.gpu_count < count:
                continue
            used = leased.get(target.id, set())
            free = tuple(str(index) for index in range(target.gpu_count) if str(index) not in used)
            if len(free) >= count:
                return target, free[:count]
    return None


def _unavailable_reason(lease: GpuLease) -> str:
    if lease.score_eligible:
        return "official resource profile unavailable"
    return "GPU capacity unavailable"


def _lease_from_row(row: dict[str, Any]) -> GpuLease:
    device_ids = loads(str(row.get("device_ids") or "[]"))
    devices = (
        tuple(str(device_id) for device_id in device_ids) if isinstance(device_ids, list) else ()
    )
    gpu_count = int(row.get("gpu_count") or len(devices))
    if gpu_count < 0 or gpu_count > 8:
        raise ValueError("persisted gpu_count must be in [0, 8]")
    if gpu_count and not 1 <= gpu_count <= int(row["max_gpu_count"]):
        raise ValueError("actual_gpu_count must be in [1, max_gpu_count]")
    return GpuLease(
        id=str(row["id"]),
        submission_id=str(row["submission_id"]),
        job_id=cast(str | None, row.get("job_id")),
        target_id=cast(str | None, row.get("target_id")),
        target_server=cast(str | None, row.get("target_server")),
        device_ids=devices,
        gpu_count=gpu_count,
        min_gpu_count=int(row["min_gpu_count"]),
        max_gpu_count=int(row["max_gpu_count"]),
        requested_gpu_count=int(row["requested_gpu_count"]),
        mode=str(row["mode"]),
        tier=str(row["tier"]),
        score_eligible=bool(row["score_eligible"]),
        autosplit_allowed=bool(row["autosplit_allowed"]),
        official_fixed_profile=bool(row["official_fixed_profile"]),
        status=cast(GpuLeaseStatus, str(row["status"])),
        reason=str(row.get("reason") or ""),
        created_at=str(row["created_at"]),
        released_at=cast(str | None, row.get("released_at")),
    )
