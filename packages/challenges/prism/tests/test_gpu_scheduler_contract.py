from __future__ import annotations

import aiosqlite
import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.db import Database
from prism_challenge.gpu_scheduler import (
    BaseGpuTarget,
    GpuLeaseRequest,
    GpuLeaseScheduler,
    lease_request_from_runtime,
)
from prism_challenge.repository import PrismRepository


@pytest.fixture
async def repository(tmp_path):
    database = Database(tmp_path / "gpu-scheduler.sqlite3")
    await database.init()
    return PrismRepository(database, epoch_seconds=60)


async def test_autosplit_to_one_gpu_for_non_scoring_path(repository: PrismRepository) -> None:
    await repository.store_runtime_config(
        config_key="gpu_policy",
        value={"max_gpu_count": 8, "actual_gpu_count": 8},
        updated_by="ops",
    )
    runtime_policy = await repository.runtime_config(PrismSettings(), official=True)
    scheduler = GpuLeaseScheduler(
        repository.database,
        (BaseGpuTarget(id="target-a", server="server-a", gpu_count=1),),
    )

    lease = await scheduler.enqueue_or_allocate(
        lease_request_from_runtime(
            submission_id="dev-submission",
            job_id="job-dev",
            runtime_policy=runtime_policy,
            mode="gpu_proxy_eval",
            score_eligible=False,
        )
    )

    assert lease.active
    assert lease.gpu_count == 1
    assert lease.requested_gpu_count == 8
    assert lease.target_id == "target-a"


async def test_official_profile_requires_exact_resources(repository: PrismRepository) -> None:
    await repository.store_runtime_config(
        config_key="gpu_policy",
        value={"max_gpu_count": 8, "actual_gpu_count": 4, "official_fixed_profile": True},
        updated_by="ops",
    )
    await repository.store_runtime_config(
        config_key="execution_mode_targets",
        value={"gpu_proxy_eval": {"official_score": True, "max_tokens": 10_000, "gpu_count": 4}},
        updated_by="ops",
    )
    runtime_policy = await repository.runtime_config(PrismSettings(), official=True)
    scheduler = GpuLeaseScheduler(
        repository.database,
        (BaseGpuTarget(id="target-a", server="server-a", gpu_count=1),),
    )

    lease = await scheduler.enqueue_or_allocate(
        lease_request_from_runtime(
            submission_id="official-submission",
            job_id="job-official",
            runtime_policy=runtime_policy,
            mode="gpu_proxy_eval",
        )
    )

    assert not lease.active
    assert lease.status == "queued"
    assert lease.gpu_count == 0
    assert lease.requested_gpu_count == 4
    assert lease.reason == "official resource profile unavailable"


async def test_fifo_constrained_queue(repository: PrismRepository) -> None:
    scheduler = GpuLeaseScheduler(
        repository.database,
        (BaseGpuTarget(id="target-a", server="server-a", gpu_count=1),),
    )
    first_request = GpuLeaseRequest(
        submission_id="submission-1",
        job_id="job-1",
        mode="gpu_proxy_eval",
        tier="dev",
        score_eligible=False,
        min_gpu_count=1,
        max_gpu_count=1,
        requested_gpu_count=1,
        autosplit_allowed=True,
        official_fixed_profile=False,
    )
    second_request = GpuLeaseRequest(
        submission_id="submission-2",
        job_id="job-2",
        mode="gpu_proxy_eval",
        tier="dev",
        score_eligible=False,
        min_gpu_count=1,
        max_gpu_count=1,
        requested_gpu_count=1,
        autosplit_allowed=True,
        official_fixed_profile=False,
    )

    first_lease = await scheduler.enqueue_or_allocate(first_request)
    second_lease = await scheduler.enqueue_or_allocate(second_request)

    assert first_lease.active
    assert second_lease.status == "queued"

    await scheduler.release_for_submission("submission-1", "completed")
    leases = {lease.submission_id: lease for lease in await scheduler.leases()}

    assert leases["submission-1"].status == "released"
    assert leases["submission-2"].active
    assert leases["submission-2"].device_ids == ("0",)


async def test_release_on_failed_job_frees_capacity(repository: PrismRepository) -> None:
    scheduler = GpuLeaseScheduler(
        repository.database,
        (BaseGpuTarget(id="target-a", server="server-a", gpu_count=1),),
    )
    first = await scheduler.enqueue_or_allocate(
        GpuLeaseRequest(
            submission_id="failed-submission",
            job_id="job-failed",
            mode="gpu_proxy_eval",
            tier="proxy",
            score_eligible=True,
            min_gpu_count=1,
            max_gpu_count=1,
            requested_gpu_count=1,
            autosplit_allowed=False,
            official_fixed_profile=True,
        )
    )
    second = await scheduler.enqueue_or_allocate(
        GpuLeaseRequest(
            submission_id="next-submission",
            job_id="job-next",
            mode="gpu_proxy_eval",
            tier="proxy",
            score_eligible=True,
            min_gpu_count=1,
            max_gpu_count=1,
            requested_gpu_count=1,
            autosplit_allowed=False,
            official_fixed_profile=True,
        )
    )

    assert first.active
    assert second.status == "queued"

    await scheduler.release_for_submission("failed-submission", "failed")

    promoted = await scheduler.active_lease_for_submission("next-submission")
    assert promoted is not None
    assert promoted.gpu_count == 1


async def test_legacy_database_without_gpu_leases_migrates(tmp_path) -> None:
    db_path = tmp_path / "legacy-no-gpu-leases.sqlite3"
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(
            "CREATE TABLE eval_jobs ("
            "id TEXT PRIMARY KEY, submission_id TEXT NOT NULL, level TEXT NOT NULL, "
            "status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, external_job_id TEXT, "
            "metrics TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL"
            ");"
        )
        await conn.commit()

    database = Database(db_path)
    await database.init()
    scheduler = GpuLeaseScheduler(
        database,
        (BaseGpuTarget(id="legacy-target", server="legacy-server", gpu_count=1),),
    )
    lease = await scheduler.enqueue_or_allocate(
        GpuLeaseRequest(
            submission_id="legacy-submission",
            job_id="legacy-job",
            mode="gpu_proxy_eval",
            tier="dev",
            score_eligible=False,
            min_gpu_count=1,
            max_gpu_count=1,
            requested_gpu_count=1,
            autosplit_allowed=True,
            official_fixed_profile=False,
        )
    )

    async with database.connect() as conn:
        table_rows = await conn.execute_fetchall("PRAGMA table_info(gpu_leases)")
        index_rows = await conn.execute_fetchall("PRAGMA index_list(gpu_leases)")
        eval_job_rows = await conn.execute_fetchall("PRAGMA table_info(eval_jobs)")

    assert lease.active
    assert {str(row[1]) for row in table_rows} >= {
        "autosplit_allowed",
        "official_fixed_profile",
        "gpu_count",
    }
    assert {str(row[1]) for row in index_rows} >= {
        "idx_gpu_leases_one_active_submission",
        "idx_gpu_leases_fifo",
    }
    assert {str(row[1]) for row in eval_job_rows} >= {
        "gpu_lease_id",
        "actual_gpu_count",
        "gpu_device_ids",
    }
