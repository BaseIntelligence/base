"""Unit tests for the supervisor orphan own-runner sandbox sweep.

Everything runs against a fake escape-hatch docker CLI runner (mirroring
``test_reaper.py``'s ``FakeEscapeRunner``) and an injected clock — no dockerd
required. Covers the age-based TTL semantics, defense-in-depth prefix
anchoring, zero/missing-StartedAt skipping, docker-ps-failure tolerance, both
sandbox prefixes, and the builder wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from base.config.settings import Settings, SupervisorSettings
from base.master.docker_broker import EscapeHatchCommandResult
from base.supervisor.orphan_sweep import (
    ORPHAN_SWEEP_TASK_NAME,
    OrphanSandboxSweeper,
    build_orphan_sweep_task,
)
from base.supervisor.scheduler import ScheduledTask
from base.supervisor.tasks import build_scheduled_tasks

UTC = UTC
NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
TTL_SECONDS = 7200.0

_Handler = Callable[[tuple[str, ...]], tuple[int, str, str]]


class FakeEscapeRunner:
    """Argv-capturing EscapeHatchCommandRunner fake (mirrors test_reaper)."""

    def __init__(self, handler: _Handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> EscapeHatchCommandResult:
        call = tuple(argv)
        self.calls.append(call)
        returncode, stdout, stderr = self.handler(call)
        return EscapeHatchCommandResult(
            argv=call, returncode=returncode, stdout=stdout, stderr=stderr
        )


def _state_json(started_at: str | None) -> str:
    if started_at is None:
        return '{"Status": "running"}'
    return f'{{"Status": "running", "StartedAt": "{started_at}"}}'


def _iso(delta: timedelta) -> str:
    """RFC3339 StartedAt for a container that started ``delta`` before NOW."""
    return (NOW - delta).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _sweeper(
    runner: FakeEscapeRunner,
    *,
    ttl_seconds: float = TTL_SECONDS,
    clock: Callable[[], datetime | None] | None = lambda: NOW,
    host: str | None = None,
) -> OrphanSandboxSweeper:
    return OrphanSandboxSweeper(
        runner=runner,
        ttl_seconds=ttl_seconds,
        host=host,
        clock=clock,
    )


def _subcommand(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Strip the docker bin and any leading ``-H <host>`` for matching."""
    rest = argv[1:]  # drop docker bin
    if rest[:1] == ("-H",):
        rest = rest[2:]  # drop -H <host>
    return rest


def _ps_handler(
    listing: str,
    inspects: dict[str, str],
    *,
    ps_rc: int = 0,
) -> _Handler:
    """Build a handler answering ps / inspect / rm for the given fixtures."""

    def handler(argv: tuple[str, ...]) -> tuple[int, str, str]:
        cmd = _subcommand(argv)
        if cmd[:1] == ("ps",):
            return ps_rc, listing, "" if ps_rc == 0 else "daemon down"
        if cmd[:1] == ("inspect",):
            container_id = cmd[-1]
            return 0, inspects.get(container_id, "{}"), ""
        if cmd[:2] == ("rm", "-f"):
            return 0, cmd[-1], ""
        return 1, "", f"unexpected argv: {argv!r}"

    return handler


def _rm_ids(runner: FakeEscapeRunner) -> list[str]:
    return [call[-1] for call in runner.calls if _subcommand(call)[:2] == ("rm", "-f")]


# ---------------------------------------------------------------------------
# TTL semantics
# ---------------------------------------------------------------------------


def test_sandbox_older_than_ttl_is_removed() -> None:
    listing = "task_old_id\town-runner-task-deadbeef\n"
    inspects = {"task_old_id": _state_json(_iso(timedelta(hours=3)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == ["task_old_id"]


def test_sandbox_younger_than_ttl_is_not_removed() -> None:
    listing = "task_young_id\town-runner-task-cafe\n"
    inspects = {"task_young_id": _state_json(_iso(timedelta(hours=1)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []


def test_age_exactly_ttl_is_not_removed() -> None:
    """TTL is a STRICT bound: age == ttl must not be removed."""
    listing = "task_edge_id\town-runner-task-edge\n"
    inspects = {"task_edge_id": _state_json(_iso(timedelta(seconds=TTL_SECONDS)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []


# ---------------------------------------------------------------------------
# Defense-in-depth prefix anchoring
# ---------------------------------------------------------------------------


def test_non_own_runner_prefix_is_ignored() -> None:
    # docker's `--filter name=own-runner-` is an UNANCHORED substring match, so
    # a container merely CONTAINING the fragment can be returned; it must be
    # ignored (never inspected, never removed).
    listing = "alien_id\tmy-own-runner-task-123\n"
    inspects = {"alien_id": _state_json(_iso(timedelta(hours=5)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []
    # never even inspected
    assert not any(call[:2] == ("docker", "inspect") for call in runner.calls)


# ---------------------------------------------------------------------------
# StartedAt handling
# ---------------------------------------------------------------------------


def test_zero_started_at_is_skipped() -> None:
    listing = "zero_id\town-runner-task-zero\n"
    inspects = {"zero_id": _state_json("0001-01-01T00:00:00Z")}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []


def test_missing_started_at_is_skipped() -> None:
    listing = "nostart_id\town-runner-exec-nostart\n"
    inspects = {"nostart_id": _state_json(None)}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []


# ---------------------------------------------------------------------------
# Failure tolerance
# ---------------------------------------------------------------------------


def test_docker_ps_failure_returns_without_removals() -> None:
    runner = FakeEscapeRunner(_ps_handler("", {}, ps_rc=1))
    # Must not raise and must not remove anything.
    _sweeper(runner).tick()
    assert _rm_ids(runner) == []


def test_no_daemon_clock_skips_tick() -> None:
    # No injected clock; docker info returns nonzero so _daemon_now() is None.
    def handler(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("docker", "info"):
            return 1, "", "info failed"
        return 1, "", "should not reach"

    runner = FakeEscapeRunner(handler)
    OrphanSandboxSweeper(runner=runner, ttl_seconds=TTL_SECONDS, clock=None).tick()
    # Only the docker info probe was attempted; no ps/inspect/rm.
    assert all(call[:2] == ("docker", "info") for call in runner.calls)
    assert _rm_ids(runner) == []


def test_one_bad_container_does_not_abort_sweep() -> None:
    # First candidate's inspect raises inside the runner; the sweep must still
    # process (and remove) the second, older candidate.
    listing = "bad_id\town-runner-task-bad\ngood_id\town-runner-task-good\n"

    def handler(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("docker", "ps"):
            return 0, listing, ""
        if argv[:2] == ("docker", "inspect"):
            if argv[-1] == "bad_id":
                raise RuntimeError("boom")
            return 0, _state_json(_iso(timedelta(hours=4))), ""
        if argv[:3] == ("docker", "rm", "-f"):
            return 0, argv[-1], ""
        return 1, "", "unexpected"

    runner = FakeEscapeRunner(handler)
    _sweeper(runner).tick()
    assert _rm_ids(runner) == ["good_id"]


# ---------------------------------------------------------------------------
# Both prefixes + host targeting
# ---------------------------------------------------------------------------


def test_both_task_and_exec_prefixes_handled() -> None:
    listing = "task_id\town-runner-task-aaaa\nexec_id\town-runner-exec-bbbb\n"
    inspects = {
        "task_id": _state_json(_iso(timedelta(hours=3))),
        "exec_id": _state_json(_iso(timedelta(hours=6))),
    }
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner).tick()
    assert sorted(_rm_ids(runner)) == ["exec_id", "task_id"]


def test_host_override_targets_remote_daemon() -> None:
    listing = "task_id\town-runner-task-aaaa\n"
    inspects = {"task_id": _state_json(_iso(timedelta(hours=3)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    _sweeper(runner, host="ssh://worker.example").tick()
    # Every issued command is prefixed with `-H <host>`.
    assert all(
        call[:3] == ("docker", "-H", "ssh://worker.example") for call in runner.calls
    )
    assert _rm_ids(runner) == ["task_id"]


def test_already_gone_removal_counts_as_success() -> None:
    listing = "task_id\town-runner-task-aaaa\n"

    def handler(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("docker", "ps"):
            return 0, listing, ""
        if argv[:2] == ("docker", "inspect"):
            return 0, _state_json(_iso(timedelta(hours=3))), ""
        if argv[:3] == ("docker", "rm", "-f"):
            return 1, "", "Error: No such container: task_id"
        return 1, "", "unexpected"

    runner = FakeEscapeRunner(handler)
    # Must not raise; the "no such container" stderr is treated as success.
    _sweeper(runner).tick()
    assert _rm_ids(runner) == ["task_id"]


def test_daemon_clock_used_when_no_clock_injected() -> None:
    listing = "task_id\town-runner-task-aaaa\n"

    def handler(argv: tuple[str, ...]) -> tuple[int, str, str]:
        if argv[:2] == ("docker", "info"):
            return 0, '"2026-01-01T12:00:00.000000000Z"', ""
        if argv[:2] == ("docker", "ps"):
            return 0, listing, ""
        if argv[:2] == ("docker", "inspect"):
            return 0, _state_json(_iso(timedelta(hours=3))), ""
        if argv[:3] == ("docker", "rm", "-f"):
            return 0, argv[-1], ""
        return 1, "", "unexpected"

    runner = FakeEscapeRunner(handler)
    OrphanSandboxSweeper(runner=runner, ttl_seconds=TTL_SECONDS, clock=None).tick()
    assert _rm_ids(runner) == ["task_id"]


# ---------------------------------------------------------------------------
# Builder + registration
# ---------------------------------------------------------------------------


def test_build_orphan_sweep_task_returns_scheduled_task() -> None:
    task = build_orphan_sweep_task(Settings())
    assert isinstance(task, ScheduledTask)
    assert task.name == ORPHAN_SWEEP_TASK_NAME
    assert task.interval_seconds == Settings().supervisor.orphan_sweep_interval_seconds
    assert callable(task.run)


def test_build_orphan_sweep_task_honours_ttl_setting() -> None:
    settings = Settings(supervisor=SupervisorSettings(orphan_sweep_ttl_seconds=1234))
    # A container aged between the default TTL and the custom low TTL is removed
    # only because the builder wired the configured (lower) TTL.
    listing = "task_id\town-runner-task-aaaa\n"
    inspects = {"task_id": _state_json(_iso(timedelta(seconds=2000)))}
    runner = FakeEscapeRunner(_ps_handler(listing, inspects))
    task = build_orphan_sweep_task(settings, runner=runner, clock=lambda: NOW)
    task.run()
    assert _rm_ids(runner) == ["task_id"]


def test_default_ttl_is_two_hours() -> None:
    assert Settings().supervisor.orphan_sweep_ttl_seconds == 7200


def test_orphan_sweep_registered_when_enabled() -> None:
    tasks, _gate = build_scheduled_tasks(Settings())
    assert ORPHAN_SWEEP_TASK_NAME in {task.name for task in tasks}


def test_orphan_sweep_not_registered_when_disabled() -> None:
    settings = Settings(supervisor=SupervisorSettings(orphan_sweep_enabled=False))
    tasks, _gate = build_scheduled_tasks(settings)
    assert ORPHAN_SWEEP_TASK_NAME not in {task.name for task in tasks}
