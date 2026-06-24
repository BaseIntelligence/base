"""Control-plane supervisor for the Docker-Swarm backend (plan Task 16).

A single systemd-managed (``Type=notify`` + ``WatchdogSec=``) long-running
process that runs the control-plane scheduled tasks. Module layout:

- ``loop.py``      — :class:`Supervisor` core (lifecycle, heartbeat). FROZEN.
- ``scheduler.py`` — :class:`ScheduledTask` + per-task worker threads. FROZEN.
- ``sd_notify.py`` — stdlib ``NOTIFY_SOCKET`` datagram protocol.
- ``health.py``    — broker ``/health`` gate (short timeout, consecutive
  failures threshold; Task 15 contract).
- ``tasks.py``     — the ONLY registration seam; Tasks 17-22 add one module
  each plus one line here.

No kubernetes/helm imports anywhere in this package — keep it that way.
"""

from __future__ import annotations

from base.config.settings import Settings
from base.supervisor.health import BrokerHealthGate, http_health_prober
from base.supervisor.loop import Supervisor
from base.supervisor.scheduler import ScheduledTask, TaskWorker
from base.supervisor.sd_notify import (
    SystemdNotifier,
    watchdog_interval_seconds,
)
from base.supervisor.tasks import build_scheduled_tasks

__all__ = [
    "BrokerHealthGate",
    "ScheduledTask",
    "Supervisor",
    "SystemdNotifier",
    "TaskWorker",
    "build_scheduled_tasks",
    "build_supervisor",
    "http_health_prober",
    "watchdog_interval_seconds",
]


def build_supervisor(settings: Settings) -> Supervisor:
    """Compose a fully-wired supervisor from validated settings.

    The heartbeat interval is derived from systemd's ``WATCHDOG_USEC`` when
    present (half the watchdog window), falling back to the loop default.
    """
    from base.supervisor.loop import (
        DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    )

    supervisor = Supervisor(
        notifier=SystemdNotifier(),
        heartbeat_interval_seconds=watchdog_interval_seconds(
            DEFAULT_HEARTBEAT_INTERVAL_SECONDS
        ),
    )
    tasks, _gate = build_scheduled_tasks(settings)
    supervisor.register_all(tasks)
    return supervisor
