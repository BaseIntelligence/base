"""Unit tests for the supervisor image-updater (Task 18).

Fake resolver + fake runner only — no network, no dockerd.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from base.config.settings import Settings
from base.master.swarm_backend import SwarmCommandResult
from base.supervisor.image_ref import ImageReference
from base.supervisor.image_updater import (
    DEFAULT_FIRST_PARTY_TARGETS,
    IMAGE_UPDATER_INTERVAL_SECONDS,
    ImageUpdateTarget,
    SwarmImageUpdater,
    build_image_updater_task,
)

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
IMAGE = "ghcr.io/baseintelligence/base-master:latest"
IMAGE_UPDATER_LOGGER = "base.supervisor.image_updater"


class _AttachedHandler:
    """Capture records on a named logger, immune to root-logging churn.

    Mirrors the pattern in test_supervisor_weights.py: other tests reconfigure
    root logging / raise logger levels (importing ``bittensor`` bumps every
    already-created logger to CRITICAL) or disable loggers (alembic env.py),
    which breaks ``caplog`` in a full-suite run, so we attach directly to the
    target logger and reset its level/disabled flag.
    """

    def __init__(self, logger_name: str) -> None:
        self._logger = logging.getLogger(logger_name)
        self.messages: list[str] = []
        self._lock = threading.Lock()
        self._was_disabled = False

        class _H(logging.Handler):
            def __init__(inner) -> None:
                super().__init__(level=logging.DEBUG)

            def emit(inner, record: logging.LogRecord) -> None:
                with self._lock:
                    self.messages.append(record.getMessage())

        self._handler = _H()

    def __enter__(self) -> _AttachedHandler:
        self._was_disabled = self._logger.disabled
        self._logger.disabled = False
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc: object) -> None:
        self._logger.removeHandler(self._handler)
        self._logger.disabled = self._was_disabled


class FakeRunner:
    def __init__(
        self,
        current_images: dict[str, str] | None = None,
        *,
        update_returncode: int = 0,
    ) -> None:
        self.current_images = dict(current_images or {})
        self.update_returncode = update_returncode
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> SwarmCommandResult:
        call = tuple(argv)
        self.calls.append(call)
        if call[1:3] == ("service", "inspect"):
            image = self.current_images.get(call[-1])
            if image is None:
                return SwarmCommandResult(call, 1, "", "no such service")
            return SwarmCommandResult(call, 0, f"{image}\n", "")
        if call[1:3] == ("service", "update"):
            return SwarmCommandResult(call, self.update_returncode, "", "boom")
        raise AssertionError(f"unexpected docker command: {call}")

    @property
    def update_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[1:3] == ("service", "update")]


def make_resolver(digest: str):
    def resolver(reference: ImageReference) -> str:
        return digest

    return resolver


def make_updater(
    runner: FakeRunner,
    resolver,
    targets: tuple[ImageUpdateTarget, ...] = (
        ImageUpdateTarget(service="base-admin", image=IMAGE),
    ),
) -> SwarmImageUpdater:
    return SwarmImageUpdater(targets, runner=runner, resolver=resolver)


def test_same_digest_is_a_noop() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})
    make_updater(runner, make_resolver(DIGEST_A)).run_once()
    assert runner.update_calls == []


def test_new_digest_issues_exactly_one_update_per_service() -> None:
    services = ("base-admin", "base-proxy")
    runner = FakeRunner({name: f"{IMAGE}@{DIGEST_A}" for name in services})
    targets = tuple(ImageUpdateTarget(service=name, image=IMAGE) for name in services)
    make_updater(runner, make_resolver(DIGEST_B), targets).run_once()
    assert len(runner.update_calls) == len(services)
    for call, name in zip(runner.update_calls, services, strict=True):
        assert call == (
            "docker",
            "service",
            "update",
            "--detach",
            "--image",
            f"{IMAGE}@{DIGEST_B}",
            name,
        )


def test_resolver_failure_logs_and_skips_update() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})

    def resolver(reference: ImageReference) -> str:
        raise RuntimeError("registry unreachable")

    updater = make_updater(runner, resolver)
    with _AttachedHandler(IMAGE_UPDATER_LOGGER) as handler:
        updater.run_once()
    assert runner.update_calls == []
    assert any("digest resolution failed" in msg for msg in handler.messages)


def test_resolver_failure_does_not_block_other_targets() -> None:
    other_image = "ghcr.io/baseintelligence/other:latest"
    runner = FakeRunner(
        {
            "base-admin": f"{IMAGE}@{DIGEST_A}",
            "base-other": f"{other_image}@{DIGEST_A}",
        }
    )

    def resolver(reference: ImageReference) -> str:
        if reference.repository.endswith("base-master"):
            raise RuntimeError("registry unreachable")
        return DIGEST_B

    targets = (
        ImageUpdateTarget(service="base-admin", image=IMAGE),
        ImageUpdateTarget(service="base-other", image=other_image),
    )
    make_updater(runner, resolver, targets).run_once()
    assert len(runner.update_calls) == 1
    assert runner.update_calls[0][-1] == "base-other"


def test_untagged_image_rejected_without_any_docker_calls() -> None:
    runner = FakeRunner()
    targets = (
        ImageUpdateTarget(
            service="base-admin",
            image="ghcr.io/baseintelligence/base-master",
        ),
    )
    with _AttachedHandler(IMAGE_UPDATER_LOGGER) as handler:
        make_updater(runner, make_resolver(DIGEST_B), targets).run_once()
    assert runner.calls == []
    assert any("untagged image" in msg for msg in handler.messages)


def test_non_sha256_resolver_result_rejected() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})
    with _AttachedHandler(IMAGE_UPDATER_LOGGER) as handler:
        make_updater(runner, make_resolver("md5:deadbeef")).run_once()
    assert runner.update_calls == []
    assert any("refusing un-pinned update" in msg for msg in handler.messages)


def test_missing_service_skipped_without_update() -> None:
    runner = FakeRunner({})
    with _AttachedHandler(IMAGE_UPDATER_LOGGER) as handler:
        make_updater(runner, make_resolver(DIGEST_B)).run_once()
    assert runner.update_calls == []
    assert any("cannot inspect service" in msg for msg in handler.messages)


def test_failed_service_update_logged_not_raised() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"}, update_returncode=1)
    with _AttachedHandler(IMAGE_UPDATER_LOGGER) as handler:
        make_updater(runner, make_resolver(DIGEST_B)).run_once()
    assert len(runner.update_calls) == 1
    assert any("docker service update failed" in msg for msg in handler.messages)


def test_unpinned_current_image_is_updated() -> None:
    runner = FakeRunner({"base-admin": IMAGE})
    make_updater(runner, make_resolver(DIGEST_B)).run_once()
    assert len(runner.update_calls) == 1
    assert runner.update_calls[0][-2] == f"{IMAGE}@{DIGEST_B}"


def test_builder_returns_wired_scheduled_task() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})
    targets = (ImageUpdateTarget(service="base-admin", image=IMAGE),)
    task = build_image_updater_task(
        Settings(),
        targets=targets,
        resolver=make_resolver(DIGEST_B),
        runner=runner,
    )
    assert task.name == "image-updater"
    assert task.interval_seconds == IMAGE_UPDATER_INTERVAL_SECONDS
    task.run()
    assert len(runner.update_calls) == 1


def test_default_targets_cover_first_party_services() -> None:
    names = {target.service for target in DEFAULT_FIRST_PARTY_TARGETS}
    assert names == {
        "base-master-proxy",
        "base-docker-broker",
    }
    assert "base-proxy" not in names
    assert "base-admin" not in names
    # Realigned to the installer-created service names: the stale placeholders must
    # never come back (base-broker is not a service; config-sync is a task).
    assert "base-broker" not in names
    assert "base-config-sync" not in names
    assert all(
        target.image == IMAGE and "@" not in target.image
        for target in DEFAULT_FIRST_PARTY_TARGETS
    )
