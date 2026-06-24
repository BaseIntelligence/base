from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_epoch_loop(
    interval_seconds: int, callback: Callable[[], Awaitable[None]]
) -> None:
    while True:
        try:
            await callback()
        except Exception:
            logger.exception("epoch loop failed")
        await asyncio.sleep(interval_seconds)
