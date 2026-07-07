"""Tests for the Docker-backend control-plane supervisor (plan Task 16)."""

from __future__ import annotations

import signal
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from base.config.settings import Settings
from base.supervisor.health import BrokerHealthGate, http_health_prober
from base.supervisor.image_updater import image_updater_from_task
from base.supervisor.loop import Supervisor
from base.supervisor.scheduler import ScheduledTask
from base.supervisor.sd_notify import (
    READY,
    STOPPING,
    WATCHDOG,
    SystemdNotifier,
    watchdog_interval_seconds,
)
from base.supervisor.tasks import build_broker_health_task, build_scheduled_tasks

ROOT = Path(__file__).resolve().parents[2]


class RecordingNotifier(SystemdNotifier):
    """Notifier double recording every state instead of touching a socket."""

    def __init__(self) -> None:
        super().__init__(socket_path="")
        self.states: list[str] = []
        self._lock = threading.Lock()

    def notify(self, state: str) -> bool:
        with self._lock:
            self.states.append(state)
        return True

    def count(self, state: str) -> int:
        with self._lock:
            return self.states.count(state)


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _run_in_thread(supervisor: Supervisor) -> tuple[threading.Thread, dict[str, int]]:
    result: dict[str, int] = {}

    def _target() -> None:
        result["exit"] = supervisor.run()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, result


def test_scheduled_task_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ScheduledTask(name="  ", interval_seconds=1.0, run=lambda: None)


def test_scheduled_task_rejects_nonpositive_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScheduledTask(name="t", interval_seconds=0, run=lambda: None)


def test_supervisor_rejects_duplicate_task_names() -> None:
    supervisor = Supervisor(notifier=RecordingNotifier())
    supervisor.register(ScheduledTask(name="t", interval_seconds=1.0, run=lambda: None))
    with pytest.raises(ValueError, match="duplicate"):
        supervisor.register(
            ScheduledTask(name="t", interval_seconds=2.0, run=lambda: None)
        )


def test_supervisor_runs_registered_tasks_and_shuts_down_cleanly() -> None:
    notifier = RecordingNotifier()
    ticks = {"a": 0, "b": 0}
    lock = threading.Lock()

    def _tick(name: str) -> Callable[[], None]:
        def run() -> None:
            with lock:
                ticks[name] += 1

        return run

    supervisor = Supervisor(
        notifier=notifier,
        heartbeat_interval_seconds=0.01,
        shutdown_grace_seconds=1.0,
    )
    supervisor.register(ScheduledTask(name="a", interval_seconds=0.01, run=_tick("a")))
    supervisor.register(ScheduledTask(name="b", interval_seconds=0.01, run=_tick("b")))
    thread, result = _run_in_thread(supervisor)
    assert _wait_until(lambda: ticks["a"] >= 3 and ticks["b"] >= 3)
    supervisor.request_shutdown()
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert result["exit"] == 0
    assert notifier.count(READY) == 1
    assert notifier.count(STOPPING) == 1


def test_task_exception_does_not_kill_loop_or_siblings() -> None:
    notifier = RecordingNotifier()
    counts = {"bad": 0, "good": 0}
    lock = threading.Lock()

    def bad() -> None:
        with lock:
            counts["bad"] += 1
        raise RuntimeError("boom")

    def good() -> None:
        with lock:
            counts["good"] += 1

    supervisor = Supervisor(
        notifier=notifier,
        heartbeat_interval_seconds=0.01,
        shutdown_grace_seconds=1.0,
    )
    supervisor.register(ScheduledTask(name="bad", interval_seconds=0.01, run=bad))
    supervisor.register(ScheduledTask(name="good", interval_seconds=0.01, run=good))
    thread, result = _run_in_thread(supervisor)
    # The raising task keeps being rescheduled AND its sibling keeps ticking.
    assert _wait_until(lambda: counts["bad"] >= 3 and counts["good"] >= 3)
    supervisor.request_shutdown()
    thread.join(timeout=5.0)
    assert result["exit"] == 0


def test_blocked_task_does_not_starve_heartbeat() -> None:
    notifier = RecordingNotifier()
    release = threading.Event()
    entered = threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(timeout=10.0)

    supervisor = Supervisor(
        notifier=notifier,
        heartbeat_interval_seconds=0.01,
        shutdown_grace_seconds=1.0,
    )
    supervisor.register(
        ScheduledTask(name="blocked", interval_seconds=0.01, run=blocked)
    )
    thread, result = _run_in_thread(supervisor)
    assert entered.wait(timeout=5.0)
    # While the only scheduled task is blocked inside run(), heartbeats
    # keep flowing because they live on the supervisor's own thread.
    assert _wait_until(lambda: notifier.count(WATCHDOG) >= 5)
    release.set()
    supervisor.request_shutdown()
    thread.join(timeout=5.0)
    assert result["exit"] == 0


def test_slow_task_ticks_do_not_stack() -> None:
    notifier = RecordingNotifier()
    runs = {"n": 0}
    lock = threading.Lock()

    def slow() -> None:
        with lock:
            runs["n"] += 1
        time.sleep(0.05)

    supervisor = Supervisor(
        notifier=notifier,
        heartbeat_interval_seconds=0.01,
        shutdown_grace_seconds=1.0,
    )
    # Interval (10ms) is far shorter than the run time (50ms): fixed-delay
    # scheduling means each cycle costs ~60ms, so runs cannot burst.
    supervisor.register(ScheduledTask(name="slow", interval_seconds=0.01, run=slow))
    thread, result = _run_in_thread(supervisor)
    time.sleep(0.3)
    supervisor.request_shutdown()
    thread.join(timeout=5.0)
    assert result["exit"] == 0
    # 0.3s / 0.06s per cycle ~= 5; a stacking scheduler would hit ~30.
    assert 1 <= runs["n"] <= 10


def test_sigterm_triggers_clean_shutdown_exit_zero() -> None:
    notifier = RecordingNotifier()
    supervisor = Supervisor(
        notifier=notifier,
        heartbeat_interval_seconds=0.02,
        shutdown_grace_seconds=1.0,
    )
    supervisor.register(
        ScheduledTask(name="noop", interval_seconds=0.02, run=lambda: None)
    )
    timer = threading.Timer(0.15, signal.raise_signal, args=(signal.SIGTERM,))
    timer.start()
    try:
        # Runs on the test's main thread so the real SIGTERM handler installs.
        exit_code = supervisor.run()
    finally:
        timer.cancel()
    assert exit_code == 0
    assert notifier.count(READY) == 1
    assert notifier.count(STOPPING) == 1


def test_sd_notify_sends_datagrams_to_real_unix_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket_path = tmp_path / "notify.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(socket_path))
    server.settimeout(2.0)
    try:
        monkeypatch.setenv("NOTIFY_SOCKET", str(socket_path))
        notifier = SystemdNotifier()
        assert notifier.enabled
        assert notifier.ready()
        assert notifier.watchdog()
        assert notifier.stopping()
        received = [server.recv(4096).decode("utf-8") for _ in range(3)]
        assert received == [READY, WATCHDOG, STOPPING]
    finally:
        server.close()


def test_sd_notify_noop_when_notify_socket_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    notifier = SystemdNotifier()
    assert not notifier.enabled
    assert notifier.ready() is False
    assert notifier.watchdog() is False
    assert notifier.stopping() is False


def test_sd_notify_send_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "missing.sock"))
    notifier = SystemdNotifier()
    assert notifier.enabled
    assert notifier.notify("READY=1") is False  # no listener — logged, not raised


def test_watchdog_interval_derived_from_watchdog_usec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WATCHDOG_USEC", "10000000")  # 10s window
    assert watchdog_interval_seconds(5.0) == 5.0  # 10s / 2
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    assert watchdog_interval_seconds(5.0) == 15.0
    monkeypatch.setenv("WATCHDOG_USEC", "not-a-number")
    assert watchdog_interval_seconds(5.0) == 5.0
    monkeypatch.setenv("WATCHDOG_USEC", "0")
    assert watchdog_interval_seconds(5.0) == 5.0
    monkeypatch.delenv("WATCHDOG_USEC")
    assert watchdog_interval_seconds(7.0) == 7.0


def test_health_gate_requires_consecutive_failures() -> None:
    gate = BrokerHealthGate(lambda: True, failure_threshold=3)
    assert gate.healthy
    gate.record(False)
    gate.record(False)
    assert gate.healthy  # 2 < threshold of 3
    gate.record(False)
    assert not gate.healthy
    assert gate.consecutive_failures == 3
    gate.record(True)  # one success resets the streak
    assert gate.healthy
    gate.record(False)
    gate.record(False)
    assert gate.healthy  # streak restarted from zero


def test_health_gate_probe_once_records_prober_result() -> None:
    results = iter([True, False, False])
    gate = BrokerHealthGate(lambda: next(results), failure_threshold=2)
    gate.probe_once()
    assert gate.healthy
    gate.probe_once()
    gate.probe_once()
    assert not gate.healthy


def test_http_health_prober_returns_false_on_refused_connection() -> None:
    # Reserve a port, then close it so the probe hits a dead endpoint.
    placeholder = socket.socket()
    placeholder.bind(("127.0.0.1", 0))
    port = placeholder.getsockname()[1]
    placeholder.close()
    probe = http_health_prober(f"http://127.0.0.1:{port}/health", 1.0)
    assert probe() is False


def test_build_scheduled_tasks_registers_health_probe_with_shared_gate() -> None:
    settings = Settings()
    tasks, gate = build_scheduled_tasks(settings)
    names = [task.name for task in tasks]
    assert "broker-health-probe" in names
    assert isinstance(gate, BrokerHealthGate)
    # The gate returned IS the one driven by the probe task: a tripped gate
    # must be observable by future Task 17-22 job builders sharing it.
    injected = BrokerHealthGate(lambda: False, failure_threshold=1)
    tasks_with_gate, same_gate = build_scheduled_tasks(settings, health_gate=injected)
    assert same_gate is injected
    probe_task = next(t for t in tasks_with_gate if t.name == "broker-health-probe")
    probe_task.run()
    assert not injected.healthy


def test_broker_health_probe_prefers_supervisor_override_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The host supervisor cannot resolve the overlay name in docker.broker_url;
    # a set supervisor.broker_health_url must win so the probe hits the
    # host-published broker port instead of failing forever.
    captured: dict[str, str] = {}

    def fake_prober(url: str, timeout_seconds: float) -> Callable[[], bool]:
        captured["url"] = url
        return lambda: True

    monkeypatch.setattr("base.supervisor.tasks.http_health_prober", fake_prober)
    settings = Settings.model_validate(
        {
            "docker": {"broker_url": "http://base-docker-broker:8082"},
            "supervisor": {"broker_health_url": "http://127.0.0.1:8082"},
        }
    )
    build_broker_health_task(settings)
    assert captured["url"] == "http://127.0.0.1:8082/health"


def test_broker_health_probe_falls_back_to_docker_broker_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_prober(url: str, timeout_seconds: float) -> Callable[[], bool]:
        captured["url"] = url
        return lambda: True

    monkeypatch.setattr("base.supervisor.tasks.http_health_prober", fake_prober)
    settings = Settings.model_validate(
        {"docker": {"broker_url": "http://base-docker-broker:8082"}}
    )
    build_broker_health_task(settings)
    assert captured["url"] == "http://base-docker-broker:8082/health"


def test_build_scheduled_tasks_targets_canonical_docker_broker() -> None:
    tasks, _gate = build_scheduled_tasks(Settings())

    image_updater = next(t for t in tasks if t.name == "image-updater")
    updater_services = {
        target.service for target in image_updater_from_task(image_updater).targets
    }
    assert updater_services == {
        "base-master-proxy",
        "base-docker-broker",
    }
    assert "base-admin" not in updater_services
    assert "base-proxy" not in updater_services
    assert "base-broker" not in updater_services
    assert "base-config-sync" not in updater_services

    config_sync = next(t for t in tasks if t.name == "config-sync")
    rollout_services = set(config_sync.run.__self__._rollout_services)  # type: ignore[attr-defined]
    assert rollout_services == {
        "base-master-proxy",
        "base-docker-broker",
    }
    assert "base-admin" not in rollout_services
    assert "base-proxy" not in rollout_services
    assert "base-broker" not in rollout_services
    assert "base-config-sync" not in rollout_services


def test_config_sync_and_image_updater_share_service_lock_registry() -> None:
    # Recommendation H: a SHARED per-service update lock registry is threaded
    # into BOTH the image-updater and config-sync so the two 60s loops never
    # issue overlapping `docker service update` on the same shared service.
    tasks, _gate = build_scheduled_tasks(Settings())
    image_updater = image_updater_from_task(
        next(t for t in tasks if t.name == "image-updater")
    )
    config_sync = next(t for t in tasks if t.name == "config-sync").run.__self__  # type: ignore[attr-defined]

    assert image_updater._service_locks is not None
    assert image_updater._service_locks is config_sync._service_locks
    # The same service name resolves to the SAME lock object across both loops.
    lock = image_updater._service_locks.get("base-master-proxy")
    assert config_sync._service_locks.get("base-master-proxy") is lock


def test_systemd_unit_template_is_notify_with_watchdog() -> None:
    unit = (ROOT / "deploy" / "swarm" / "base-supervisor.service").read_text()
    assert "Type=notify" in unit
    # `uv run` forks the supervisor as a child, so systemd must accept sd_notify
    # from any cgroup process, not just the main (uv) PID.
    assert "NotifyAccess=all" in unit
    assert "WatchdogSec=" in unit
    assert "Restart=always" in unit
    assert "master supervisor" in unit
