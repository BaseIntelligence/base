from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from prism_challenge.db import Database
from prism_challenge.models import SubmissionCreate, SubmissionStatus
from prism_challenge.repository import PrismRepository, now_iso

CODE = "def build_model(ctx):\n    return None\n"


@pytest.fixture
async def repository(tmp_path):
    database = Database(tmp_path / "claim-race.sqlite3")
    await database.init()
    # Tiny timeout so a claimed_at a few seconds in the past counts as stale.
    return PrismRepository(database, epoch_seconds=60, worker_claim_timeout_seconds=1)


async def _seed_pending(repository: PrismRepository, hotkey: str = "miner-1") -> str:
    created = await repository.create_submission(
        hotkey, SubmissionCreate(code=CODE, filename="model.py", metadata={})
    )
    return created.id


async def _status_and_claimed_at(repository: PrismRepository, submission_id: str):
    async with repository.database.connect() as conn:
        rows = await conn.execute_fetchall(
            "SELECT status, claimed_at FROM submissions WHERE id=?", (submission_id,)
        )
    row = list(rows)[0]
    return str(row["status"]), row["claimed_at"]


async def test_orphaned_running_is_requeued_then_reclaimed(repository: PrismRepository) -> None:
    """DEFECT B anchor: a stale `running` row (worker died mid-claim) must be
    requeued by the reaper and become claimable again."""
    submission_id = await _seed_pending(repository)

    # Simulate a worker that claimed the row and then died: status=running with a
    # claimed_at well in the past (older than worker_claim_timeout_seconds).
    stale = (datetime.now(UTC) - timedelta(seconds=3600)).isoformat()
    async with repository.database.connect() as conn:
        await conn.execute(
            "UPDATE submissions SET status=?, claimed_at=?, updated_at=? WHERE id=?",
            (SubmissionStatus.RUNNING.value, stale, stale, submission_id),
        )

    before = datetime.now(UTC)
    claimed = await repository.claim_next()

    assert claimed is not None, "stale running row was never requeued/reclaimed"
    assert claimed["id"] == submission_id

    status, claimed_at = await _status_and_claimed_at(repository, submission_id)
    assert status == SubmissionStatus.RUNNING.value
    # A *fresh* claimed_at proves it went pending -> running again, not left stale.
    assert claimed_at is not None
    assert datetime.fromisoformat(str(claimed_at)) >= before


async def test_fresh_running_is_not_requeued(repository: PrismRepository) -> None:
    """The reaper must only touch *stale* running rows; a recently claimed row
    stays running and is never handed out again."""
    submission_id = await _seed_pending(repository)
    fresh = now_iso()
    async with repository.database.connect() as conn:
        await conn.execute(
            "UPDATE submissions SET status=?, claimed_at=?, updated_at=? WHERE id=?",
            (SubmissionStatus.RUNNING.value, fresh, fresh, submission_id),
        )

    claimed = await repository.claim_next()

    assert claimed is None, "a fresh running row must not be requeued/reclaimed"
    status, claimed_at = await _status_and_claimed_at(repository, submission_id)
    assert status == SubmissionStatus.RUNNING.value
    assert str(claimed_at) == fresh


async def test_pending_row_claimed_exactly_once(repository: PrismRepository) -> None:
    """DEFECT A invariant: a single pending row yields the row ONCE then None;
    a non-pending row can never be transitioned to running by claim."""
    submission_id = await _seed_pending(repository)

    first = await repository.claim_next()
    second = await repository.claim_next()

    assert first is not None and first["id"] == submission_id
    assert second is None, "the same pending row was claimed twice"

    status, _ = await _status_and_claimed_at(repository, submission_id)
    assert status == SubmissionStatus.RUNNING.value


async def test_concurrent_claims_never_double_claim(repository: PrismRepository) -> None:
    """DEFECT A concurrency anchor: two concurrent claims on a single pending row
    must not BOTH return the same non-None submission."""
    submission_id = await _seed_pending(repository)

    results = await asyncio.gather(repository.claim_next(), repository.claim_next())

    non_none = [r for r in results if r is not None]
    assert len(non_none) == 1, f"expected exactly one claim to win, got {results}"
    assert non_none[0]["id"] == submission_id
