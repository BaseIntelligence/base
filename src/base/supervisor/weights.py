"""Weights schedule port (plan Task 21) — compute-only `master weights --once`.

Each tick performs ONE master weight epoch exactly as the CLI command
``base master weights --once`` does (the scheduled weights job it
replaces), by calling the SAME ``cli_app.main`` helpers, with no duplicated
logic. The cycle: startup migrations (idempotent alembic upgrade, identical
to every task invocation), registry/challenge-token reads, metagraph
fetch (chain READ), per-challenge ``get_weights`` HTTP collection, then
aggregation into the final UID vector.

ZERO on-chain effects — the master can never submit weights:

1. ``create_bittensor_runtime`` (bittensor/factory.py) never constructs a
   ``WeightSetter`` (only ``create_bittensor_submit_runtime`` does).
2. ``MasterWeightService`` has NO ``weight_setter`` and ``run_epoch`` only
   computes/aggregates the vector — there is no ``set_weights`` call anywhere
   in the master weight path (master/service.py).

On-chain submission lives entirely in the per-validator submitter
(``base.validator.weight_submitter``), each validator committing the
master-aggregated vector under its OWN hotkey.

The broker health gate is accepted per the Task-16 builder recipe but NOT
consulted: the weights compute path touches the control-plane DB, the chain
endpoint (read-only), and challenge HTTP APIs — never the Docker broker.
"""

from __future__ import annotations

import asyncio
import logging

from base.config.settings import Settings
from base.schemas.weights import FinalWeights
from base.supervisor.health import BrokerHealthGate
from base.supervisor.scheduler import ScheduledTask

logger = logging.getLogger(__name__)

WEIGHTS_TASK_NAME = "weights-compute"


def compute_weights_once(settings: Settings) -> FinalWeights:
    """Run one compute-only master weight epoch (no chain submission).

    Mirrors ``cli_app.main.master_weights(once=True)`` minus the CLI-only
    pieces (``configure_logging``, typer echo). ``MasterWeightService`` has no
    ``weight_setter`` and no submit path, so this can only compute/aggregate.
    """
    # Lazy import: keeps the supervisor package import light and immune to
    # any future cli_app <-> supervisor import cycle (cli_app.main already
    # imports the supervisor lazily inside `master supervisor`).
    from base.cli_app import main as cli_main

    cli_main._run_startup_migrations(settings)
    registry = cli_main._master_registry(settings)
    runtime = cli_main.create_bittensor_runtime(settings)
    service = cli_main._master_weight_service(
        settings,
        metagraph_cache=runtime.metagraph_cache,
    )
    final = asyncio.run(cli_main._run_master_weight_epoch(service, registry))
    logger.info(
        "supervisor weights tick: compute-only, %d uids",
        len(final.uids),
        extra={"uids": len(final.uids)},
    )
    return final


def build_weights_task(
    settings: Settings,
    *,
    health_gate: BrokerHealthGate | None = None,
) -> ScheduledTask:
    """Build the scheduled compute-only weights task (Task-16 recipe).

    Interval follows the CLI loop's cadence (`settings.master.
    epoch_interval_seconds`). ``health_gate`` is part of the shared builder
    signature but deliberately unused — see module docstring.
    """
    del health_gate  # weights compute never touches the broker

    def run() -> None:
        # Module-level lookup so tests can monkeypatch compute_weights_once.
        compute_weights_once(settings)

    return ScheduledTask(
        name=WEIGHTS_TASK_NAME,
        interval_seconds=float(settings.master.epoch_interval_seconds),
        run=run,
    )
