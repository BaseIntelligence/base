"""Settings-driven image-updater targets + validator-agent auto-roll (G-A5).

Covers VAL-CODE-AUTO-003: the image-updater targets are configurable (not
hardcoded to only the master services), the default preserves the two master
services (back-compat), and a configured validator-agent target tracking the
mutable validator runtime image is registered and rolled on a digest change.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from base.config.settings import (
    DEFAULT_VALIDATOR_AGENT_SERVICE,
    DEFAULT_VALIDATOR_RUNTIME_IMAGE,
    ImageUpdateTargetSetting,
    Settings,
    SupervisorSettings,
)
from base.master.swarm_backend import SwarmCommandResult
from base.supervisor.image_ref import ImageReference
from base.supervisor.image_updater import (
    DEFAULT_FIRST_PARTY_TARGETS,
    ImageUpdateTarget,
    SwarmImageUpdater,
    image_updater_from_task,
    resolve_image_update_targets,
)
from base.supervisor.scheduler import ScheduledTask
from base.supervisor.tasks import build_scheduled_tasks

DIGEST_OLD = "sha256:" + "a" * 64
DIGEST_NEW = "sha256:" + "b" * 64
MASTER_SERVICES = {"base-master-proxy", "base-docker-broker"}


def _settings(**supervisor: object) -> Settings:
    return Settings(supervisor=SupervisorSettings(**supervisor))  # type: ignore[arg-type]


def _image_updater_targets(settings: Settings) -> tuple[ImageUpdateTarget, ...]:
    tasks, _gate = build_scheduled_tasks(settings)
    image_updater = next(t for t in tasks if t.name == "image-updater")
    return image_updater_from_task(image_updater).targets


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
            image = self._current_images.get(call[-1])
            if image is None:
                return SwarmCommandResult(call, 1, "", "no such service")
            return SwarmCommandResult(call, 0, f"{image}\n", "")
        if call[1:3] == ("service", "update"):
            self.update_calls.append(call)
            return SwarmCommandResult(call, 0, "", "")
        raise AssertionError(f"unexpected docker command: {call}")


# ---------------------------------------------------------------------------
# Default / back-compat: the two master services are preserved when unset.
# ---------------------------------------------------------------------------


def test_default_targets_preserve_the_two_master_services() -> None:
    targets = resolve_image_update_targets(Settings())
    assert targets == DEFAULT_FIRST_PARTY_TARGETS
    assert {t.service for t in targets} == MASTER_SERVICES


def test_build_scheduled_tasks_default_image_updater_targets_unchanged() -> None:
    assert {t.service for t in _image_updater_targets(_settings())} == MASTER_SERVICES


# ---------------------------------------------------------------------------
# Configurable: an explicit list drives the targets.
# ---------------------------------------------------------------------------


def test_targets_are_settings_driven() -> None:
    custom = [
        ImageUpdateTargetSetting(
            service="custom-svc",
            image="ghcr.io/baseintelligence/base-master:latest",
        ),
    ]
    targets = resolve_image_update_targets(_settings(image_updater_targets=custom))
    assert targets == (
        ImageUpdateTarget(
            service="custom-svc",
            image="ghcr.io/baseintelligence/base-master:latest",
        ),
    )


def test_build_scheduled_tasks_honours_configured_targets() -> None:
    custom = [ImageUpdateTargetSetting(service="custom-svc", image="ghcr.io/x:latest")]
    targets = _image_updater_targets(_settings(image_updater_targets=custom))
    assert [t.service for t in targets] == ["custom-svc"]


# ---------------------------------------------------------------------------
# Validator-agent target: present/derivable and tracks the runtime image.
# ---------------------------------------------------------------------------


def test_validator_runtime_image_default_is_the_mutable_runtime_tag() -> None:
    assert DEFAULT_VALIDATOR_RUNTIME_IMAGE == (
        "ghcr.io/baseintelligence/base-validator-runtime:latest"
    )


def test_validator_agent_target_appended_when_enabled() -> None:
    targets = resolve_image_update_targets(
        _settings(validator_agent_target_enabled=True)
    )
    services = {t.service for t in targets}
    # Back-compat master services stay present...
    assert MASTER_SERVICES <= services
    # ...plus the validator-agent target tracking the runtime image.
    validator = next(t for t in targets if t.service == DEFAULT_VALIDATOR_AGENT_SERVICE)
    assert validator.image == DEFAULT_VALIDATOR_RUNTIME_IMAGE


def test_validator_only_targets_on_a_validator_node() -> None:
    # A validator NODE watches ONLY its agent: empty master list + toggle on.
    targets = resolve_image_update_targets(
        _settings(image_updater_targets=[], validator_agent_target_enabled=True)
    )
    assert targets == (
        ImageUpdateTarget(
            service=DEFAULT_VALIDATOR_AGENT_SERVICE,
            image=DEFAULT_VALIDATOR_RUNTIME_IMAGE,
        ),
    )


def test_empty_targets_with_validator_toggle_off_is_empty_tuple() -> None:
    # A node that watches NOTHING: empty explicit list AND validator toggle off
    # resolves to NO targets (it must NOT fall back to the master defaults, which
    # only happens when image_updater_targets is unset/None).
    targets = resolve_image_update_targets(
        _settings(image_updater_targets=[], validator_agent_target_enabled=False)
    )
    assert targets == ()


def test_validator_target_not_duplicated_when_already_listed() -> None:
    explicit = [
        ImageUpdateTargetSetting(
            service=DEFAULT_VALIDATOR_AGENT_SERVICE,
            image=DEFAULT_VALIDATOR_RUNTIME_IMAGE,
        )
    ]
    targets = resolve_image_update_targets(
        _settings(image_updater_targets=explicit, validator_agent_target_enabled=True)
    )
    assert [t.service for t in targets] == [DEFAULT_VALIDATOR_AGENT_SERVICE]


def test_build_scheduled_tasks_registers_validator_agent_target() -> None:
    targets = _image_updater_targets(_settings(validator_agent_target_enabled=True))
    assert any(t.service == DEFAULT_VALIDATOR_AGENT_SERVICE for t in targets)


def test_custom_validator_service_name_and_image_respected() -> None:
    targets = resolve_image_update_targets(
        _settings(
            image_updater_targets=[],
            validator_agent_target_enabled=True,
            validator_agent_service="base-validator-prod",
            validator_agent_image="ghcr.io/baseintelligence/base:latest",
        )
    )
    assert targets == (
        ImageUpdateTarget(
            service="base-validator-prod",
            image="ghcr.io/baseintelligence/base:latest",
        ),
    )


# ---------------------------------------------------------------------------
# Roll: a configured validator-agent target is rolled on a digest change.
# ---------------------------------------------------------------------------


def test_validator_agent_service_rolled_on_digest_change() -> None:
    runner = _FakeRunner(
        {
            DEFAULT_VALIDATOR_AGENT_SERVICE: (
                f"{DEFAULT_VALIDATOR_RUNTIME_IMAGE}@{DIGEST_OLD}"
            )
        }
    )

    def resolver(reference: ImageReference) -> str:
        return DIGEST_NEW

    targets = resolve_image_update_targets(
        _settings(image_updater_targets=[], validator_agent_target_enabled=True)
    )
    SwarmImageUpdater(targets, runner=runner, resolver=resolver).run_once()

    assert len(runner.update_calls) == 1
    call = runner.update_calls[0]
    assert call[-1] == DEFAULT_VALIDATOR_AGENT_SERVICE
    image_arg = call[call.index("--image") + 1]
    assert image_arg == f"{DEFAULT_VALIDATOR_RUNTIME_IMAGE}@{DIGEST_NEW}"


def test_validator_agent_service_noop_when_digest_matches() -> None:
    runner = _FakeRunner(
        {
            DEFAULT_VALIDATOR_AGENT_SERVICE: (
                f"{DEFAULT_VALIDATOR_RUNTIME_IMAGE}@{DIGEST_NEW}"
            )
        }
    )

    def resolver(reference: ImageReference) -> str:
        return DIGEST_NEW

    targets = resolve_image_update_targets(
        _settings(image_updater_targets=[], validator_agent_target_enabled=True)
    )
    SwarmImageUpdater(targets, runner=runner, resolver=resolver).run_once()
    assert runner.update_calls == []


# ---------------------------------------------------------------------------
# No targets: run_once is a clean no-op (no docker command, no digest resolve).
# ---------------------------------------------------------------------------


class _RecordingRunner:
    """Records every docker command so we can assert ZERO were issued."""

    def __init__(self) -> None:
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
        raise AssertionError(f"no docker command expected for empty targets: {call}")


def test_image_updater_no_targets_is_a_clean_noop() -> None:
    # A validator-off node with an empty target list resolves to () (above); the
    # updater built from those targets must issue ZERO docker commands and NEVER
    # call the resolver - a clean no-op, not an error or a stray inspect/update.
    settings = _settings(image_updater_targets=[], validator_agent_target_enabled=False)
    targets = resolve_image_update_targets(settings)
    assert targets == ()

    runner = _RecordingRunner()
    resolver_calls: list[ImageReference] = []

    def resolver(reference: ImageReference) -> str:
        resolver_calls.append(reference)
        return DIGEST_NEW

    SwarmImageUpdater(targets, runner=runner, resolver=resolver).run_once()
    assert runner.calls == []
    assert resolver_calls == []


def test_build_scheduled_tasks_empty_validator_off_updater_is_noop() -> None:
    # End-to-end wiring: build_scheduled_tasks with an empty target list and the
    # validator toggle off produces an image-updater with no targets, so its
    # run() iterates nothing and spawns no docker subprocess (clean no-op).
    settings = _settings(image_updater_targets=[], validator_agent_target_enabled=False)
    tasks, _gate = build_scheduled_tasks(settings)
    image_updater = next(t for t in tasks if t.name == "image-updater")
    assert image_updater_from_task(image_updater).targets == ()
    image_updater.run()  # empty targets -> no docker subprocess, must not raise


# ---------------------------------------------------------------------------
# Stable public seam: image_updater_from_task exposes the wired updater/targets.
# ---------------------------------------------------------------------------


def test_image_updater_from_task_exposes_wired_targets() -> None:
    settings = _settings(validator_agent_target_enabled=True)
    tasks, _gate = build_scheduled_tasks(settings)
    image_updater = next(t for t in tasks if t.name == "image-updater")
    updater = image_updater_from_task(image_updater)
    assert isinstance(updater, SwarmImageUpdater)
    assert updater.targets == resolve_image_update_targets(settings)


def test_image_updater_from_task_rejects_a_foreign_task() -> None:
    foreign = ScheduledTask(
        name="not-an-updater", interval_seconds=1.0, run=lambda: None
    )
    with pytest.raises(TypeError):
        image_updater_from_task(foreign)
