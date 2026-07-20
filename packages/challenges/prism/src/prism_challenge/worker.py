from __future__ import annotations

import argparse
import asyncio
import logging

from .app import create_app
from .config import PrismSettings, configure_logging
from .queue import PrismWorker

logger = logging.getLogger("prism.worker")


async def run_worker_loop(
    worker: PrismWorker,
    *,
    interval_seconds: float,
    resilient: bool = False,
) -> None:
    """Drain the eval queue one submission at a time until cancelled.

    Shared by the standalone ``prism-worker`` CLI (``resilient=False``) and combined mode
    (``resilient=True``). When resilient, an unexpected ``process_next`` error is logged and the
    loop continues, so a transient failure never permanently stops the drain or takes down the
    co-hosted API; ``CancelledError`` is always re-raised so the lifespan can cancel the task
    cleanly (it is a ``BaseException`` in 3.12, so ``except Exception`` never swallows it).
    """
    while True:
        try:
            submission_id = await worker.process_next()
        except asyncio.CancelledError:
            raise
        except Exception:
            if not resilient:
                raise
            logger.exception("worker iteration failed; continuing")
        else:
            if submission_id:
                logger.info(
                    "worker iteration completed",
                    extra={"submission_id": submission_id},
                )
        await asyncio.sleep(interval_seconds)


async def run_worker(settings: PrismSettings, *, interval_seconds: float) -> None:
    app = create_app(settings)
    await app.state.database.init()
    try:
        await run_worker_loop(app.state.worker, interval_seconds=interval_seconds)
    finally:
        await app.state.database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prism evaluation worker")
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    worker_settings = PrismSettings()
    configure_logging(worker_settings)
    asyncio.run(run_worker(worker_settings, interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
