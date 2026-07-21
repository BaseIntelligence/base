"""In-process continuous master weight sealer (auto-weights always-200).

Production freshness for ``GET /v1/weights/latest`` is owned by the master
proxy process: a background lifespan task seals on
``master.epoch_interval_seconds`` (disable with ``<=0``), and the serve path
may lazy-seal under lock when the latest durable vector is missing or past
``expires_at``.

This module never calls on-chain ``set_weights``. CLI
``base master weights`` is emergency/debug only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI

from base.challenge_sdk.roles import Role, activate_role
from base.master.service import MasterWeightService, active_challenge_inputs
from base.schemas.weights import FinalWeights, MasterWeightsResponse

logger = logging.getLogger(__name__)


def resolve_master_weight_epoch(
    *,
    epoch_interval_seconds: int | float,
    now: datetime | None = None,
    epoch: int | None = None,
) -> int:
    """Wall-clock epoch bucket shared with CLI seal identity."""

    if epoch is not None:
        return int(epoch)
    raw_interval = epoch_interval_seconds or 360
    interval = max(1, int(raw_interval))
    clock = now if now is not None else datetime.now(UTC)
    return int(clock.timestamp()) // interval


class MasterWeightsSealer:
    """One-tick seal driver used by the proxy lifespan loop.

    Each tick loads ACTIVE challenges and calls
    :meth:`MasterWeightService.seal_fresh_if_needed` (durable
    ``AggregationService.seal_epoch`` / zero-miner retained).
    """

    def __init__(
        self,
        *,
        weight_service: MasterWeightService,
        registry: Any,
        netuid: int,
        chain_endpoint: str = "",
        epoch_interval_seconds: int | float = 360,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._weight_service = weight_service
        self._registry = registry
        self._netuid = int(netuid)
        self._chain_endpoint = chain_endpoint or ""
        self._epoch_interval_seconds = epoch_interval_seconds
        self._now_fn = now_fn

    async def tick_once(self) -> MasterWeightsResponse | FinalWeights | None:
        """Run one continuous-sealer tick (best-effort; errors log and return)."""

        challenges, tokens = await active_challenge_inputs(self._registry)
        with activate_role(Role.MASTER):
            return await self._weight_service.seal_fresh_if_needed(
                challenges,
                tokens,
                netuid=self._netuid,
                chain_endpoint=self._chain_endpoint,
                epoch_interval_seconds=self._epoch_interval_seconds,
                now_fn=self._now_fn,
                force=True,
            )


async def run_master_weights_sealer_loop(
    sealer: MasterWeightsSealer,
    *,
    interval_seconds: float,
    shutdown_event: asyncio.Event,
) -> None:
    """Run :meth:`MasterWeightsSealer.tick_once` until shutdown.

    A failing pass is logged and the loop continues so one transient error
    never stops continuous seal freshness.
    """

    while not shutdown_event.is_set():
        try:
            await sealer.tick_once()
        except Exception:
            logger.exception("master weights sealer pass failed")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


def build_master_weights_sealer_lifespan(
    sealer: MasterWeightsSealer | None,
    interval_seconds: float | None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]] | None:
    """Build a FastAPI lifespan that runs the continuous weights sealer.

    Returns ``None`` when the sealer is not configured or the interval is
    non-positive (``epoch_interval_seconds <= 0`` disables continuous seal).
    """

    if sealer is None or interval_seconds is None or interval_seconds <= 0:
        return None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        shutdown = asyncio.Event()
        # Seal immediately on startup so first GET is 200 without CLI.
        try:
            await sealer.tick_once()
        except Exception:
            logger.exception("master weights sealer startup tick failed")
        task = asyncio.create_task(
            run_master_weights_sealer_loop(
                sealer,
                interval_seconds=float(interval_seconds),
                shutdown_event=shutdown,
            )
        )
        try:
            yield
        finally:
            shutdown.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return lifespan


__all__ = [
    "MasterWeightsSealer",
    "build_master_weights_sealer_lifespan",
    "resolve_master_weight_epoch",
    "run_master_weights_sealer_loop",
]
