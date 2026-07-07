"""Supervisor auto-update wiring in ``build_scheduled_tasks`` (G4).

Covers:
- VAL-CODE-UPD-001 (wiring side): the image-updater uses the AUTHENTICATED
  resolver when registry credentials are configured, and the anonymous resolver
  otherwise. (The challenge-image-updater moved into the master proxy — see
  architecture.md sec 9.1 — so it is no longer registered on the host supervisor.)
- VAL-CODE-UPD-003: master self-update is wired when enabled+manifest_url, and
  EXPLICITLY DISABLED (task absent — not registered-but-inert) by default; an
  enabled-without-manifest config is rejected (no silent half-state).
- VAL-CODE-UPD-004: the image-updater rolls a service only on a digest change,
  pins the immutable ``tag@sha256:<digest>``, and is a no-op when current.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from base.config.settings import Settings, SupervisorSettings
from base.master.swarm_backend import SwarmCommandResult
from base.supervisor.image_ref import ImageReference, resolve_remote_digest
from base.supervisor.image_updater import (
    DEFAULT_FIRST_PARTY_TARGETS,
    DEFAULT_MASTER_IMAGE,
    SwarmImageUpdater,
)
from base.supervisor.tasks import build_scheduled_tasks

MANIFEST_URL = "https://raw.example/base/release/supervisor-manifest.json"


def _settings(**supervisor: object) -> Settings:
    return Settings(supervisor=SupervisorSettings(**supervisor))  # type: ignore[arg-type]


def _docker_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    auth = base64.b64encode(b"ci-bot:ghp_secret").decode("ascii")
    config.write_text(json.dumps({"auths": {"ghcr.io": {"auth": auth}}}))
    return config


def _task(tasks: Sequence[object], name: str) -> object:
    return next(t for t in tasks if t.name == name)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Authenticated resolver wiring (VAL-CODE-UPD-001)
# ---------------------------------------------------------------------------


def test_updaters_use_authenticated_resolver_when_credentials_present(
    tmp_path: Path,
) -> None:
    settings = _settings(registry_docker_config_path=str(_docker_config(tmp_path)))
    tasks, _gate = build_scheduled_tasks(settings)

    image_updater = _task(tasks, "image-updater")

    # A wrapped (authenticated) resolver, NOT the bare anonymous function.
    assert image_updater.run.__self__._resolver is not resolve_remote_digest  # type: ignore[attr-defined]


def test_updaters_fall_back_to_anonymous_resolver_without_credentials(
    tmp_path: Path,
) -> None:
    settings = _settings(
        registry_docker_config_path=str(tmp_path / "absent.json"),
    )
    tasks, _gate = build_scheduled_tasks(settings)

    image_updater = _task(tasks, "image-updater")

    assert image_updater.run.__self__._resolver is resolve_remote_digest  # type: ignore[attr-defined]


def test_challenge_image_updater_not_registered_on_host_supervisor(
    tmp_path: Path,
) -> None:
    # The challenge-image-updater moved into the master proxy (architecture.md
    # sec 9.1): the host supervisor default task set must NOT register it (it
    # cannot resolve the overlay registry DB DNS from the host).
    settings = _settings(registry_docker_config_path=str(tmp_path / "absent.json"))
    tasks, _gate = build_scheduled_tasks(settings)

    names = [t.name for t in tasks]  # type: ignore[attr-defined]
    assert "challenge-image-updater" not in names


# ---------------------------------------------------------------------------
# Self-update wired / explicitly disabled (VAL-CODE-UPD-003)
# ---------------------------------------------------------------------------


def test_self_update_disabled_by_default_is_not_registered(tmp_path: Path) -> None:
    settings = _settings(registry_docker_config_path=str(tmp_path / "absent.json"))
    tasks, _gate = build_scheduled_tasks(settings)

    names = [t.name for t in tasks]  # type: ignore[attr-defined]
    # No inert half-state: the task is simply absent when disabled.
    assert "self-update" not in names


def test_self_update_registered_and_wired_when_enabled(tmp_path: Path) -> None:
    settings = _settings(
        registry_docker_config_path=str(tmp_path / "absent.json"),
        self_update_enabled=True,
        self_update_manifest_url=MANIFEST_URL,
    )
    tasks, _gate = build_scheduled_tasks(settings)

    self_update = _task(tasks, "self-update")
    detector = self_update.run.__self__._detector  # type: ignore[attr-defined]
    # The WIRED manifest detector (http_manifest_detector inner is named
    # ``detect``); the inert default detector is named ``version_detector``.
    assert detector.__name__ == "detect"


def test_self_update_enabled_without_manifest_is_rejected(tmp_path: Path) -> None:
    settings = _settings(
        registry_docker_config_path=str(tmp_path / "absent.json"),
        self_update_enabled=True,
    )
    with pytest.raises(ValueError, match="self_update_manifest_url"):
        build_scheduled_tasks(settings)


def test_self_update_config_knobs_feed_the_task(tmp_path: Path) -> None:
    # The interval/uptime/boot/swap knobs flow from SupervisorSettings into the
    # built self-update task (defaults equal the historical constants).
    settings = _settings(
        registry_docker_config_path=str(tmp_path / "absent.json"),
        self_update_enabled=True,
        self_update_manifest_url=MANIFEST_URL,
        self_update_interval_seconds=111.0,
        self_update_min_uptime_seconds=22.0,
        self_update_max_boot_attempts=4,
        self_update_max_swap_attempts=5,
    )
    tasks, _gate = build_scheduled_tasks(settings)

    self_update = _task(tasks, "self-update")
    assert self_update.interval_seconds == 111.0  # type: ignore[attr-defined]
    updater = self_update.run.__self__  # type: ignore[attr-defined]
    assert updater._min_uptime_seconds == 22.0
    assert updater._max_boot_attempts == 4
    assert updater._max_swap_attempts == 5


# ---------------------------------------------------------------------------
# Digest-compare roll / no-op + immutable pin (VAL-CODE-UPD-004)
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self, current_images: dict[str, str]) -> None:
        self._current_images = dict(current_images)
        self.update_calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> SwarmCommandResult:
        call = tuple(argv)
        if call[1:3] == ("service", "inspect"):
            fmt = call[call.index("--format") + 1]
            if "UpdateStatus.State" in fmt:
                return SwarmCommandResult(call, 0, "completed\n", "")
            image = self._current_images[call[-1]]
            return SwarmCommandResult(call, 0, f"{image}\n", "")
        if call[1:3] == ("service", "update"):
            self.update_calls.append(call)
            return SwarmCommandResult(call, 0, "", "")
        raise AssertionError(f"unexpected docker command: {call}")


def _updater(runner: _FakeRunner, digest: str) -> SwarmImageUpdater:
    def resolver(reference: ImageReference) -> str:
        return digest

    return SwarmImageUpdater(
        DEFAULT_FIRST_PARTY_TARGETS, runner=runner, resolver=resolver
    )


def test_no_op_when_running_digest_matches() -> None:
    digest = "sha256:" + "c" * 64
    pinned = {
        t.service: f"{DEFAULT_MASTER_IMAGE}@{digest}"
        for t in DEFAULT_FIRST_PARTY_TARGETS
    }
    runner = _FakeRunner(pinned)
    _updater(runner, digest).run_once()
    assert runner.update_calls == []


def test_rolls_with_immutable_pin_on_digest_change() -> None:
    old = "sha256:" + "c" * 64
    new = "sha256:" + "d" * 64
    pinned = {
        t.service: f"{DEFAULT_MASTER_IMAGE}@{old}" for t in DEFAULT_FIRST_PARTY_TARGETS
    }
    runner = _FakeRunner(pinned)
    _updater(runner, new).run_once()

    assert len(runner.update_calls) == len(DEFAULT_FIRST_PARTY_TARGETS)
    for call in runner.update_calls:
        # Immutable pin: tag@sha256:<digest> (NOT a bare tag).
        assert f"{DEFAULT_MASTER_IMAGE}@{new}" in call
        image_arg = call[call.index("--image") + 1]
        assert image_arg == f"{DEFAULT_MASTER_IMAGE}@{new}"
        assert image_arg.startswith(
            "ghcr.io/baseintelligence/base-master:latest@sha256:"
        )
