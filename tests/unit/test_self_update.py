"""Tests for the supervisor self-update task (plan Task 22)."""

from __future__ import annotations

import io
import json
import os
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from base.config.settings import Settings, SupervisorSettings
from base.supervisor.health import BrokerHealthGate
from base.supervisor.self_update import (
    STATE_ABORTED,
    STATE_COMMITTED,
    STATE_PENDING,
    STATE_ROLLED_BACK,
    AvailableRelease,
    ReleasePaths,
    SelfUpdater,
    SelfUpdateRollback,
    UpdateState,
    build_self_update_task,
    http_manifest_detector,
    load_state,
    run_startup_rollback_check,
    save_state,
    tarball_stager,
)

ROOT = Path(__file__).resolve().parents[2]

V1 = "v1.0.0"
V2 = "v2.0.0"


def _gate(healthy: bool) -> BrokerHealthGate:
    gate = BrokerHealthGate(lambda: healthy, failure_threshold=1)
    gate.record(healthy)
    return gate


def _make_paths(tmp_path: Path, *, current: str | None = V1) -> ReleasePaths:
    paths = ReleasePaths(root=tmp_path / "supervisor")
    paths.releases.mkdir(parents=True)
    for version in (V1,):
        (paths.release_dir(version)).mkdir()
        (paths.release_dir(version) / "VERSION").write_text(version)
    if current is not None:
        os.symlink(f"releases/{current}", paths.current)
    return paths


def _copy_stager(
    payload: str = "new release",
) -> Callable[[AvailableRelease, Path], None]:
    def stage(release: AvailableRelease, target_dir: Path) -> None:
        target_dir.mkdir(parents=True)
        (target_dir / "VERSION").write_text(release.version)
        (target_dir / "payload.txt").write_text(payload)

    return stage


def _updater(
    paths: ReleasePaths,
    *,
    detector_version: str | None = V2,
    healthy: bool = True,
    probe_ok: bool = True,
    running: str | None = V1,
    clock: Callable[[], float] | None = None,
    min_uptime_seconds: float = 0.0,
) -> tuple[SelfUpdater, list[str]]:
    restarts: list[str] = []

    def detector() -> AvailableRelease | None:
        if detector_version is None:
            return None
        return AvailableRelease(version=detector_version)

    updater = SelfUpdater(
        paths,
        version_detector=detector,
        stager=_copy_stager(),
        release_prober=lambda _release_dir: probe_ok,
        health_gate=_gate(healthy),
        restart_requester=lambda: restarts.append("restart"),
        running_version=lambda: running,
        clock=clock if clock is not None else (lambda: 0.0),
        min_uptime_seconds=min_uptime_seconds,
    )
    return updater, restarts


def test_no_new_version_is_a_noop(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, restarts = _updater(paths, detector_version=None)
    updater.tick()
    assert paths.current_version() == V1
    assert restarts == []
    assert load_state(paths).status == "idle"

    same_version, restarts_same = _updater(paths, detector_version=V1)
    same_version.tick()
    assert paths.current_version() == V1
    assert restarts_same == []
    assert not paths.release_dir(V1 + ".staging").exists()


def test_staged_but_unhealthy_gate_never_swaps(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, restarts = _updater(paths, healthy=False)
    updater.tick()
    assert paths.release_dir(V2).exists(), "staging side-by-side is allowed"
    assert paths.current_version() == V1, "swap must be blocked by the gate"
    assert restarts == []
    assert load_state(paths).status == "idle"


def test_staged_but_failing_probe_never_swaps(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, restarts = _updater(paths, probe_ok=False)
    updater.tick()
    assert paths.release_dir(V2).exists()
    assert paths.current_version() == V1
    assert restarts == []


def test_healthy_swap_flips_current_and_requests_restart(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, restarts = _updater(paths)
    updater.tick()
    assert paths.current_version() == V2
    assert restarts == ["restart"]
    state = load_state(paths)
    assert state.status == STATE_PENDING
    assert state.previous == V1
    assert state.new == V2
    assert paths.release_dir(V1).exists(), "previous release must be retained"
    assert (paths.release_dir(V1) / "VERSION").read_text() == V1


def test_previous_release_survives_through_commit(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, _ = _updater(paths)
    updater.tick()
    assert paths.release_dir(V1).exists()

    new_process, _ = _updater(paths, running=V2)
    new_process.tick()
    assert load_state(paths).status == STATE_COMMITTED
    assert paths.release_dir(V1).exists(), "previous retained even after commit"
    assert paths.release_dir(V2).exists()


def test_commit_waits_for_min_uptime_and_healthy_gate(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    swapper, _ = _updater(paths)
    swapper.tick()

    early, _ = _updater(paths, running=V2, clock=lambda: 0.0, min_uptime_seconds=30.0)
    early.tick()
    assert load_state(paths).status == STATE_PENDING

    unhealthy, _ = _updater(paths, running=V2, healthy=False)
    unhealthy.tick()
    assert load_state(paths).status == STATE_PENDING

    ticks = iter([0.0, 31.0])
    healthy = SelfUpdater(
        paths,
        version_detector=lambda: None,
        stager=_copy_stager(),
        release_prober=lambda _d: True,
        health_gate=_gate(True),
        restart_requester=lambda: None,
        running_version=lambda: V2,
        clock=lambda: next(ticks),
        min_uptime_seconds=30.0,
    )
    healthy.tick()
    assert load_state(paths).status == STATE_COMMITTED


def test_pending_with_old_process_rerequests_restart(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    swapper, restarts = _updater(paths)
    swapper.tick()
    assert restarts == ["restart"]

    still_old, old_restarts = _updater(paths, running=V1)
    still_old.tick()
    assert old_restarts == ["restart"], "crash between flip and exit → re-request"
    assert paths.current_version() == V2


def test_startup_rollback_after_boot_storm(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    swapper, _ = _updater(paths)
    swapper.tick()
    assert paths.current_version() == V2

    for boot in (1, 2, 3):
        run_startup_rollback_check(
            paths, running_version=lambda: V2, max_boot_attempts=3
        )
        assert load_state(paths).boot_attempts == boot
        assert paths.current_version() == V2

    with pytest.raises(SelfUpdateRollback):
        run_startup_rollback_check(
            paths, running_version=lambda: V2, max_boot_attempts=3
        )
    state = load_state(paths)
    assert state.status == STATE_ROLLED_BACK
    assert paths.current_version() == V1, "rollback must flip current back"
    assert paths.release_dir(V2).exists(), "failed release kept for forensics"

    old_serving, restarts = _updater(paths, running=V1, detector_version=None)
    old_serving.tick()
    assert restarts == []
    assert paths.current_version() == V1


def test_rolled_back_version_is_retried_before_blacklist(tmp_path: Path) -> None:
    # A rolled-back version whose swap-attempt budget is NOT yet spent is
    # re-swapped (a transient boot failure must not permanently blacklist a
    # possibly-good version); the swap-attempt counter increments.
    paths = _make_paths(tmp_path)
    save_state(
        paths,
        UpdateState(
            status=STATE_ROLLED_BACK,
            previous=V1,
            new=V2,
            boot_attempts=4,
            swap_attempts=1,
        ),
    )
    updater, restarts = _updater(paths)  # max_swap_attempts defaults to 3
    updater.tick()
    assert paths.current_version() == V2
    assert restarts == ["restart"]
    state = load_state(paths)
    assert state.status == STATE_PENDING
    assert state.new == V2
    assert state.swap_attempts == 2  # incremented on the retry swap


def test_rolled_back_version_blacklisted_after_max_swap_attempts(
    tmp_path: Path,
) -> None:
    # Once the swap-attempt budget is spent the rolled-back version is refused
    # until a DIFFERENT version is published.
    paths = _make_paths(tmp_path)
    save_state(
        paths,
        UpdateState(
            status=STATE_ROLLED_BACK,
            previous=V1,
            new=V2,
            boot_attempts=4,
            swap_attempts=3,
        ),
    )
    updater, restarts = _updater(paths)  # max_swap_attempts defaults to 3
    updater.tick()
    assert paths.current_version() == V1
    assert restarts == []

    retry_newer, newer_restarts = _updater(paths, detector_version="v3.0.0")
    retry_newer.tick()
    assert paths.current_version() == "v3.0.0"
    assert newer_restarts == ["restart"]


def test_first_swap_records_swap_attempt_one(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    updater, _ = _updater(paths)
    updater.tick()
    state = load_state(paths)
    assert state.status == STATE_PENDING
    assert state.swap_attempts == 1


def test_startup_hook_noop_paths(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    run_startup_rollback_check(paths, running_version=lambda: V1)
    assert load_state(paths).status == "idle"

    save_state(paths, UpdateState(status=STATE_PENDING, previous=V1, new=V2))
    run_startup_rollback_check(paths, running_version=lambda: V1)
    assert load_state(paths).boot_attempts == 0, "old version booting must not count"

    missing_root = ReleasePaths(root=tmp_path / "absent")
    run_startup_rollback_check(missing_root, running_version=lambda: None)


def test_pending_without_flip_is_marked_aborted(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    save_state(paths, UpdateState(status=STATE_PENDING, previous=V1, new=V2))
    updater, restarts = _updater(paths, running=V1)
    updater.tick()
    assert load_state(paths).status == STATE_ABORTED
    assert restarts == []
    assert paths.current_version() == V1


def test_tick_never_raises(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)

    def exploding_detector() -> AvailableRelease | None:
        raise RuntimeError("registry down")

    def exploding_stager(release: AvailableRelease, target_dir: Path) -> None:
        raise RuntimeError("disk full")

    bad_detector = SelfUpdater(
        paths,
        version_detector=exploding_detector,
        stager=_copy_stager(),
        release_prober=lambda _d: True,
        restart_requester=lambda: None,
        running_version=lambda: V1,
    )
    bad_detector.tick()
    assert paths.current_version() == V1

    bad_stager = SelfUpdater(
        paths,
        version_detector=lambda: AvailableRelease(version=V2),
        stager=exploding_stager,
        release_prober=lambda _d: True,
        restart_requester=lambda: None,
        running_version=lambda: V1,
    )
    bad_stager.tick()
    assert paths.current_version() == V1
    assert not paths.release_dir(V2).exists()
    assert not paths.staging_dir(V2).exists()


def test_idempotent_retick_after_commit(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    swapper, _ = _updater(paths)
    swapper.tick()
    committer, _ = _updater(paths, running=V2)
    committer.tick()
    assert load_state(paths).status == STATE_COMMITTED

    again, restarts = _updater(paths, running=V2)
    again.tick()
    again.tick()
    assert restarts == [], "committed update must not re-swap or re-restart"
    assert paths.current_version() == V2
    assert load_state(paths).status == STATE_COMMITTED


def test_builder_returns_named_task_and_inert_default_detector(
    tmp_path: Path,
) -> None:
    paths = _make_paths(tmp_path)
    restarts: list[str] = []
    task = build_self_update_task(
        Settings(),
        paths=paths,
        stager=_copy_stager(),
        release_prober=lambda _d: True,
        restart_requester=lambda: restarts.append("restart"),
        running_version=lambda: V1,
    )
    assert task.name == "self-update"
    task.run()
    assert restarts == [], "no manifest configured → inert no-op"
    assert paths.current_version() == V1


class _FakeResponse:
    """Minimal urlopen response double (context manager + ``read``)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def test_manifest_detector_retries_transient_download_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps(
        {"version": V2, "source_url": "https://example/v2.tar.gz"}
    ).encode("utf-8")
    calls = {"n": 0}

    def flaky(url: str, timeout: float | None = None) -> _FakeResponse:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("transient blip")
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    detector = http_manifest_detector(
        "https://example/manifest.json", attempts=3, sleep=lambda _d: None
    )
    release = detector()
    assert calls["n"] == 3  # two transient failures, then success
    assert release is not None
    assert release.version == V2
    assert release.source_url == "https://example/v2.tar.gz"


def test_manifest_detector_returns_none_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def always_fail(url: str, timeout: float | None = None) -> _FakeResponse:
        calls["n"] += 1
        raise urllib.error.URLError("registry down")

    monkeypatch.setattr(urllib.request, "urlopen", always_fail)
    detector = http_manifest_detector(
        "https://example/manifest.json", attempts=3, sleep=lambda _d: None
    )
    assert detector() is None  # contract preserved: exhausted retries → None
    assert calls["n"] == 3


def test_tarball_stager_retries_transient_download_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"payload"
        info = tarfile.TarInfo("base-abc123/VERSION")  # single top-level dir stripped
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    tar_bytes = buffer.getvalue()
    calls = {"n": 0}

    def flaky(url: str, timeout: float | None = None) -> io.BytesIO:
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.URLError("transient blip")
        return io.BytesIO(tar_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    stage = tarball_stager(uv_sync=False, attempts=3, sleep=lambda _d: None)
    target = tmp_path / "releases" / V2
    stage(AvailableRelease(version=V2, source_url="https://example/v2.tar.gz"), target)
    assert calls["n"] == 2  # one transient failure, then success
    assert (target / "VERSION").read_text() == "payload"
    assert not target.with_name(target.name + ".tar.gz").exists()  # archive cleaned


def test_build_self_update_task_reads_config_knobs() -> None:
    settings = Settings(
        supervisor=SupervisorSettings(
            self_update_interval_seconds=123.0,
            self_update_min_uptime_seconds=45.0,
            self_update_max_boot_attempts=7,
            self_update_max_swap_attempts=9,
        )
    )
    task = build_self_update_task(settings)
    assert task.interval_seconds == 123.0
    updater = task.run.__self__  # type: ignore[attr-defined]
    assert updater._min_uptime_seconds == 45.0
    assert updater._max_boot_attempts == 7
    assert updater._max_swap_attempts == 9


def test_systemd_unit_launches_via_current_with_restart_always() -> None:
    unit = (ROOT / "deploy" / "swarm" / "base-supervisor.service").read_text()
    assert "Restart=always" in unit
    assert "/var/lib/base/supervisor/current" in unit
    assert "master supervisor" in unit
