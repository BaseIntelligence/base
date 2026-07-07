"""Orphan own-runner sandbox sweep — supervisor scheduled task.

The agent-challenge own-runner backend creates per-trial sandbox containers
named ``own-runner-task-<hex>`` and ``own-runner-exec-<hex>`` via ``docker
run`` directly on the HOST docker daemon (Docker-out-of-Docker). They are
host-daemon SIBLINGS of the runner/job container and are normally torn down
by the runner's in-process Python ``finally``. When the job is killed
EXTERNALLY (broker timeout ``docker service rm``, or the supervisor's timeout
reaper), that in-process teardown never runs and the sandboxes LEAK. They
carry no recoverable timeout, so the ledger-based :mod:`base.supervisor.reaper`
never reaps them (it only handles swarm services and ``base.challenge``-labeled
escape-hatch containers).

This task is the ledger-independent, age-based backstop: each tick it
force-removes own-runner sandboxes whose age STRICTLY EXCEEDS a TTL. All
runner jobs (and thus their sandboxes) run on the SAME node/daemon as the
supervisor, so the sweep operates on the LOCAL docker daemon (an optional
``host`` override is included for future multi-node use, defaulting to local).

Safety: a legitimate sandbox lives at most one job lease
(``evaluation_timeout_seconds`` default 3600s + ~900s lease ≈ 4500s / 75 min),
so the TTL default (7200s / 2h) sits comfortably above the max legit age;
only containers strictly older than the TTL are removed.

Clock handling mirrors the reaper: "now" is read from the docker daemon's own
clock (``docker info`` ``SystemTime``) and each container's start from the
same daemon's ``.State.StartedAt`` — NEVER the host wall clock compared
against a container ``StartedAt``. If the daemon clock can't be read this
tick the tick is skipped; a container without a valid ``StartedAt`` is
skipped.

Health gate: like the reaper, this task performs daemon-scoped cleanup with
NO broker HTTP dependency, so an unhealthy :class:`BrokerHealthGate` does not
disable it — it is precisely the crash-recovery backstop for sandboxes leaked
when a job's in-process teardown died with an external kill.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime

from base.config.settings import Settings
from base.master.docker_broker import (
    EscapeHatchCliRunner,
    EscapeHatchCommandRunner,
)
from base.master.swarm_backend import _parse_docker_timestamp
from base.supervisor.health import BrokerHealthGate
from base.supervisor.scheduler import ScheduledTask

logger = logging.getLogger(__name__)

ORPHAN_SWEEP_TASK_NAME = "orphan-sandbox-sweep"
DEFAULT_ORPHAN_SWEEP_INTERVAL_SECONDS = 300.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0

#: Recognized own-runner sandbox name prefixes. Docker's ``--filter name=``
#: is an UNANCHORED substring match, so candidates are re-checked against
#: these prefixes (defense in depth) before any removal.
OWN_RUNNER_NAME_PREFIXES = ("own-runner-task-", "own-runner-exec-")

#: stderr markers meaning "the container is already gone" — removal of an
#: already-removed sandbox counts as success (the sweep is idempotent).
_GONE_MARKERS = ("no such", "not found")


def _removal_succeeded(returncode: int, stderr: str) -> bool:
    if returncode == 0:
        return True
    lowered = stderr.lower()
    return any(marker in lowered for marker in _GONE_MARKERS)


def _zero_timestamp_to_none(value: datetime | None) -> datetime | None:
    """Docker reports ``0001-01-01T00:00:00Z`` for never-started containers."""
    if value is not None and value.year <= 1:
        return None
    return value


def _load_json_object(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json_timestamp(raw: str) -> datetime | None:
    text = raw.strip()
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = text
    return _parse_docker_timestamp(decoded)


class OrphanSandboxSweeper:
    """Age-based, ledger-independent sweeper for leaked own-runner sandboxes.

    ``host`` (optional ``docker -H <host>`` target) selects a remote daemon
    for future multi-node use; ``None`` targets the LOCAL daemon (where every
    runner job — and thus its sandboxes — runs today). ``clock`` is injectable
    for tests; when ``None`` "now" derives from ``docker info`` ``SystemTime``.
    """

    def __init__(
        self,
        *,
        runner: EscapeHatchCommandRunner,
        ttl_seconds: float,
        docker_bin: str = "docker",
        host: str | None = None,
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        clock: Callable[[], datetime | None] | None = None,
    ) -> None:
        self._runner = runner
        self._ttl_seconds = ttl_seconds
        self._docker_bin = docker_bin
        self._host = host
        self._command_timeout_seconds = command_timeout_seconds
        self._clock = clock

    def _run(self, *args: str) -> tuple[int, str, str]:
        argv = [self._docker_bin]
        if self._host:
            argv += ["-H", self._host]
        argv.extend(args)
        result = self._runner.run(argv, timeout_seconds=self._command_timeout_seconds)
        return result.returncode, result.stdout, result.stderr

    def tick(self) -> None:
        now = self._clock() if self._clock is not None else self._daemon_now()
        if now is None:
            logger.warning(
                "orphan-sandbox-sweep: no daemon-derived clock available; "
                "skipping sweep this tick"
            )
            return
        returncode, stdout, stderr = self._run(
            "ps",
            "-a",
            "--no-trunc",
            "--format",
            "{{.ID}}\t{{.Names}}",
            "--filter",
            "name=own-runner-",
        )
        if returncode != 0:
            logger.warning(
                "orphan-sandbox-sweep: docker ps failed (rc=%d): %s; "
                "skipping sweep this tick",
                returncode,
                stderr.strip(),
            )
            return
        removed = 0
        for container_id, name in self._parse_candidates(stdout):
            try:
                if self._sweep_container(container_id, name, now):
                    removed += 1
            except Exception:
                logger.warning(
                    "orphan-sandbox-sweep: error handling candidate %s (%s); "
                    "continuing",
                    name,
                    container_id,
                    exc_info=True,
                )
        if removed:
            logger.info(
                "orphan-sandbox-sweep: removed %d orphaned own-runner "
                "sandbox(es) older than %ss",
                removed,
                self._ttl_seconds,
            )

    def _parse_candidates(self, stdout: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        for line in stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            container_id, _, name = stripped.partition("\t")
            container_id = container_id.strip()
            name = name.strip()
            # Docker's name filter is an UNANCHORED substring match; keep only
            # names that actually START with an own-runner prefix.
            if not container_id or not name.startswith(OWN_RUNNER_NAME_PREFIXES):
                continue
            candidates.append((container_id, name))
        return candidates

    def _sweep_container(self, container_id: str, name: str, now: datetime) -> bool:
        started_at = self._container_started_at(container_id)
        if started_at is None:
            return False
        age_seconds = (now - started_at).total_seconds()
        if age_seconds <= self._ttl_seconds:
            return False
        returncode, _, stderr = self._run("rm", "-f", container_id)
        if not _removal_succeeded(returncode, stderr):
            logger.warning(
                "orphan-sandbox-sweep: failed to remove orphaned sandbox "
                "%s (%s); retrying next tick",
                name,
                container_id,
            )
            return False
        logger.info(
            "orphan-sandbox-sweep: removed orphaned own-runner sandbox %s "
            "(id=%s, age=%.0fs)",
            name,
            container_id,
            age_seconds,
        )
        return True

    def _container_started_at(self, container_id: str) -> datetime | None:
        returncode, stdout, _ = self._run(
            "inspect", "--format", "{{json .State}}", container_id
        )
        if returncode != 0:
            return None
        state = _load_json_object(stdout)
        if state is None:
            return None
        return _zero_timestamp_to_none(_parse_docker_timestamp(state.get("StartedAt")))

    def _daemon_now(self) -> datetime | None:
        returncode, stdout, _ = self._run("info", "--format", "{{json .SystemTime}}")
        if returncode != 0:
            return None
        return _parse_json_timestamp(stdout)


def build_orphan_sweep_task(
    settings: Settings,
    *,
    health_gate: BrokerHealthGate | None = None,
    runner: EscapeHatchCommandRunner | None = None,
    docker_host: str | None = None,
    clock: Callable[[], datetime | None] | None = None,
    interval_seconds: float | None = None,
) -> ScheduledTask:
    """Build the supervisor's orphan-sandbox-sweep :class:`ScheduledTask`.

    Like the reaper, this task is daemon-scoped and intentionally NOT disabled
    by an unhealthy broker gate (it is the crash-recovery backstop), so
    ``health_gate`` is accepted for signature parity only. Runner, host, clock,
    and interval are injectable for tests; defaults are the real docker CLI
    runner, the LOCAL daemon, the daemon-derived clock, and the configured
    interval.
    """
    del health_gate  # signature parity; sweep is daemon-scoped, no broker gate
    sweeper = OrphanSandboxSweeper(
        runner=runner or EscapeHatchCliRunner(),
        ttl_seconds=settings.supervisor.orphan_sweep_ttl_seconds,
        host=docker_host,
        clock=clock,
    )
    return ScheduledTask(
        name=ORPHAN_SWEEP_TASK_NAME,
        interval_seconds=(
            interval_seconds or settings.supervisor.orphan_sweep_interval_seconds
        ),
        run=sweeper.tick,
    )
