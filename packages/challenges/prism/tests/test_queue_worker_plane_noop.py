"""Worker-plane ownership: master PrismWorker must not claim/eval when plane is ON.

Host landmine (VAL-PRISM-037): when ``worker_plane.enabled`` and not
``cpu_reexec_test_mode``, ``process_next`` is a no-op and ``_run_container_eval``
refuses Docker on master so Lium/miners own GPU execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.queue import PrismWorker


def _ctx() -> PrismContext:
    return PrismContext()


@pytest.mark.asyncio
async def test_process_next_noop_when_worker_plane_enabled() -> None:
    """Given worker plane ON without cpu_reexec, When process_next, Then None and no claim."""
    repo = SimpleNamespace(claim_next=AsyncMock(return_value={"id": "should-not-claim"}))
    settings = PrismSettings(
        worker_plane=WorkerPlaneConfig(enabled=True, cpu_reexec_test_mode=False),
        docker_enabled=False,
        plagiarism_enabled=False,
    )
    worker = PrismWorker(
        repository=repo,  # type: ignore[arg-type]
        ctx=_ctx(),
        execution_backend="base_gpu",
        settings=settings,
    )

    result = await worker.process_next()

    assert result is None
    repo.claim_next.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_next_claims_when_worker_plane_disabled() -> None:
    """Given worker plane OFF, When process_next and empty queue, Then claim_next is called."""
    repo = SimpleNamespace(claim_next=AsyncMock(return_value=None))
    settings = PrismSettings(
        worker_plane=WorkerPlaneConfig(enabled=False),
        docker_enabled=False,
        plagiarism_enabled=False,
    )
    worker = PrismWorker(
        repository=repo,  # type: ignore[arg-type]
        ctx=_ctx(),
        execution_backend="base_gpu",
        settings=settings,
    )

    result = await worker.process_next()

    assert result is None
    repo.claim_next.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_container_refuses_when_worker_plane_owns_gpu() -> None:
    """Given worker plane ON, When _process_container, Then RuntimeError worker_plane_enabled."""
    repo = SimpleNamespace()
    settings = PrismSettings(
        worker_plane=WorkerPlaneConfig(enabled=True, cpu_reexec_test_mode=False),
        docker_enabled=True,
        plagiarism_enabled=False,
    )
    worker = PrismWorker(
        repository=repo,  # type: ignore[arg-type]
        ctx=_ctx(),
        execution_backend="base_gpu",
        settings=settings,
    )

    with pytest.raises(RuntimeError, match="worker_plane_enabled"):
        await worker._process_container(  # noqa: SLF001
            "sub-1",
            "print(1)",
            "main.py",
            {},
            "hk",
            "deadbeef",
            resume_checkpoint_ref=None,
        )
