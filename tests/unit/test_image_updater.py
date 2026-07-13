"""Unit tests for the supervisor image-updater (Task 18).

Fake resolver + fake runner only — no network, no dockerd. The fake runner
models Swarm's convergence: a ``service update --image`` accepts the spec, and a
``service inspect --format '{{.UpdateStatus.State}}'`` reports the (scriptable)
rollout state so the updater can convergence-verify, roll back, and retry.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from base.config.settings import Settings
from base.master.swarm_backend import SwarmCommandResult
from base.supervisor.alerts import ALERT_IMAGE_UPDATE_FAILED
from base.supervisor.image_ref import ImageReference
from base.supervisor.image_updater import (
    DEFAULT_FIRST_PARTY_TARGETS,
    IMAGE_UPDATER_INTERVAL_SECONDS,
    ImageUpdateTarget,
    SwarmImageUpdater,
    build_image_updater_task,
)
from base.supervisor.retry import RetryPolicy
from base.supervisor.weight_submit import WeightsAlert

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
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


class FakeClock:
    """Deterministic monotonic clock whose ``sleep`` advances virtual time."""

    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeRunner:
    def __init__(
        self,
        current_images: dict[str, str] | None = None,
        *,
        update_returncode: int = 0,
        convergence: str | list[str] = "completed",
    ) -> None:
        self.current_images = dict(current_images or {})
        self.update_returncode = update_returncode
        self._convergence = convergence
        self.calls: list[tuple[str, ...]] = []

    def _next_convergence(self) -> str:
        if isinstance(self._convergence, list):
            return self._convergence.pop(0) if self._convergence else "completed"
        return self._convergence

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
            fmt = call[call.index("--format") + 1]
            service = call[-1]
            if "UpdateStatus.State" in fmt:
                return SwarmCommandResult(call, 0, f"{self._next_convergence()}\n", "")
            image = self.current_images.get(service)
            if image is None:
                return SwarmCommandResult(call, 1, "", "no such service")
            return SwarmCommandResult(call, 0, f"{image}\n", "")
        if call[1:3] == ("service", "update"):
            service = call[-1]
            if "--rollback" not in call:
                image = call[call.index("--image") + 1]
                if self.update_returncode == 0:
                    self.current_images[service] = image
            stderr = "boom" if self.update_returncode != 0 else ""
            return SwarmCommandResult(call, self.update_returncode, "", stderr)
        raise AssertionError(f"unexpected docker command: {call}")

    @property
    def update_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.calls if call[1:3] == ("service", "update")]

    def image_update_calls(self, pinned: str) -> list[tuple[str, ...]]:
        """``service update --image <pinned>`` calls (forward roll or re-pin)."""
        return [
            call for call in self.update_calls if "--image" in call and pinned in call
        ]

    @property
    def rollback_flag_calls(self) -> list[tuple[str, ...]]:
        return [call for call in self.update_calls if "--rollback" in call]


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
    **kwargs,
) -> SwarmImageUpdater:
    return SwarmImageUpdater(targets, runner=runner, resolver=resolver, **kwargs)


# ---------------------------------------------------------------------------
# Digest compare / no-op / immutable pin (existing behaviour, minus --detach)
# ---------------------------------------------------------------------------


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
        # Convergence-verified update: no more --detach fire-and-forget.
        assert call == (
            "docker",
            "service",
            "update",
            "--image",
            f"{IMAGE}@{DIGEST_B}",
            name,
        )


def test_unpinned_current_image_is_updated() -> None:
    runner = FakeRunner({"base-admin": IMAGE})
    make_updater(runner, make_resolver(DIGEST_B)).run_once()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1


# ---------------------------------------------------------------------------
# Graceful skips (resolver / inspect / policy) — no retry budget consumed
# ---------------------------------------------------------------------------


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
    other_image = (
        "ghcr.io/baseintelligence/other:latest@sha256:"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
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


# ---------------------------------------------------------------------------
# Convergence poll: success, paused -> rollback, timeout -> rollback
# ---------------------------------------------------------------------------


def test_convergence_poll_success_path() -> None:
    clock = FakeClock()
    runner = FakeRunner(
        {"base-admin": f"{IMAGE}@{DIGEST_A}"},
        convergence=["updating", "completed"],
    )
    make_updater(
        runner,
        make_resolver(DIGEST_B),
        clock=clock,
        sleep=clock.sleep,
    ).run_once()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1
    # Two convergence inspects were polled (updating -> completed), no rollback.
    convergence_polls = [
        call
        for call in runner.calls
        if call[1:3] == ("service", "inspect") and "{{.UpdateStatus.State}}" in call
    ]
    assert len(convergence_polls) == 2
    assert runner.rollback_flag_calls == []


def test_paused_convergence_rolls_back_to_last_known_good(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = FakeRunner(
        {"base-admin": f"{IMAGE}@{DIGEST_A}"},
        convergence="paused",
    )
    with caplog.at_level(logging.ERROR):
        make_updater(runner, make_resolver(DIGEST_B)).run_once()
    # Forward roll to the new digest, then a rollback re-pin to the pre-update
    # last-known-good (DIGEST_A).
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1
    rollback = runner.image_update_calls(f"{IMAGE}@{DIGEST_A}")
    assert len(rollback) == 1
    assert rollback[0][-1] == "base-admin"
    assert any("rolling back" in rec.message for rec in caplog.records)


def test_timeout_convergence_rolls_back(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    runner = FakeRunner(
        {"base-admin": f"{IMAGE}@{DIGEST_A}"},
        convergence="updating",  # never terminal -> timeout
    )
    updater = make_updater(
        runner,
        make_resolver(DIGEST_B),
        clock=clock,
        sleep=clock.sleep,
        convergence_timeout_seconds=5.0,
        convergence_poll_interval_seconds=2.0,
    )
    with caplog.at_level(logging.WARNING):
        updater.run_once()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_A}")) == 1  # rollback
    assert any("did not converge" in rec.message for rec in caplog.records)


def test_failed_service_update_rolled_back_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = FakeRunner(
        {"base-admin": f"{IMAGE}@{DIGEST_A}"},
        update_returncode=1,
    )
    with caplog.at_level(logging.ERROR):
        make_updater(runner, make_resolver(DIGEST_B)).run_once()
    # The forward update was attempted (rc!=0), then a rollback was issued.
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_A}")) == 1
    assert any("docker service update failed" in rec.message for rec in caplog.records)


def test_rollback_uses_swarm_rollback_when_no_last_known_good() -> None:
    # Current image is an un-pinned tag (no digest) -> no last-known-good digest,
    # so the rollback falls back to Swarm's --rollback (previous spec).
    runner = FakeRunner({"base-admin": IMAGE}, convergence="paused")
    make_updater(runner, make_resolver(DIGEST_B)).run_once()
    assert len(runner.rollback_flag_calls) == 1
    assert runner.rollback_flag_calls[0][-1] == "base-admin"


# ---------------------------------------------------------------------------
# Backoff + retry budget + exhaustion alert
# ---------------------------------------------------------------------------


def _paused_updater(
    runner: FakeRunner, clock: FakeClock, **kwargs
) -> SwarmImageUpdater:
    return make_updater(
        runner,
        make_resolver(DIGEST_B),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,
    )


def test_backoff_skips_target_until_eligible() -> None:
    clock = FakeClock()
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"}, convergence="paused")
    policy = RetryPolicy(
        max_attempts=5, base_delay=60.0, max_delay=1800.0, jitter=False
    )
    updater = _paused_updater(runner, clock, retry_policy=policy)

    updater.run_once()  # attempt 1 fails -> next-eligible at +60s
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1

    updater.run_once()  # still within backoff window -> skipped (no new roll)
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1

    clock.advance(61.0)  # past the backoff -> eligible again
    updater.run_once()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 2


def test_max_attempts_emits_alert_once() -> None:
    clock = FakeClock()
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"}, convergence="paused")
    alerts: list[WeightsAlert] = []
    policy = RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=10.0, jitter=False)
    updater = _paused_updater(
        runner, clock, retry_policy=policy, alert_emit=alerts.append
    )

    for _ in range(6):  # more ticks than attempts; each becomes eligible
        updater.run_once()
        clock.advance(100.0)

    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 3  # capped
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.kind == ALERT_IMAGE_UPDATE_FAILED
    assert alert.details["service"] == "base-admin"
    assert alert.details["attempts"] == 3


def test_new_digest_resets_exhausted_state() -> None:
    clock = FakeClock()
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"}, convergence="paused")
    alerts: list[WeightsAlert] = []
    policy = RetryPolicy(max_attempts=2, base_delay=1.0, max_delay=10.0, jitter=False)

    digest = {"value": DIGEST_B}

    def resolver(reference: ImageReference) -> str:
        return digest["value"]

    updater = make_updater(
        runner,
        resolver,
        clock=clock,
        sleep=clock.sleep,
        retry_policy=policy,
        alert_emit=alerts.append,
    )

    for _ in range(4):  # exhaust the budget for DIGEST_B
        updater.run_once()
        clock.advance(100.0)
    assert len(alerts) == 1
    b_rolls = len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}"))
    assert b_rolls == 2  # capped at max_attempts

    # A NEW desired digest resets the exhausted state: a fresh roll is attempted.
    digest["value"] = DIGEST_C
    updater.run_once()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_C}")) == 1
    # DIGEST_B was not retried again (still capped at 2).
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == b_rolls


# ---------------------------------------------------------------------------
# Operator freeze / hold (global + per-target)
# ---------------------------------------------------------------------------


def test_global_hold_skips_every_target(caplog: pytest.LogCaptureFixture) -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})
    updater = make_updater(runner, make_resolver(DIGEST_B), global_hold=True)
    with caplog.at_level(logging.INFO):
        updater.run_once()
    assert runner.calls == []  # no resolve, no inspect, no update
    assert any("skipped-held" in rec.message for rec in caplog.records)


def test_per_target_hold_skips_only_that_target() -> None:
    runner = FakeRunner(
        {
            "held-svc": f"{IMAGE}@{DIGEST_A}",
            "live-svc": f"{IMAGE}@{DIGEST_A}",
        }
    )
    targets = (
        ImageUpdateTarget(service="held-svc", image=IMAGE, hold=True),
        ImageUpdateTarget(service="live-svc", image=IMAGE),
    )
    make_updater(runner, make_resolver(DIGEST_B), targets).run_once()
    rolled = {call[-1] for call in runner.update_calls}
    assert rolled == {"live-svc"}


# ---------------------------------------------------------------------------
# Last-known-good persistence across a simulated supervisor restart
# ---------------------------------------------------------------------------


def test_last_known_good_persisted_and_reloaded_for_rollback(tmp_path: Path) -> None:
    state_path = tmp_path / "lkg.json"

    # First process: a successful roll records the pre-update digest as LKG.
    runner1 = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"}, convergence="completed")
    make_updater(runner1, make_resolver(DIGEST_B), state_path=state_path).run_once()
    assert json.loads(state_path.read_text()) == {"base-admin": DIGEST_A}

    # Simulated restart: a fresh updater loads the persisted LKG from disk.
    runner2 = FakeRunner({"base-admin": IMAGE}, convergence="paused")
    updater2 = make_updater(runner2, make_resolver(DIGEST_C), state_path=state_path)
    assert updater2._last_known_good == {"base-admin": DIGEST_A}

    # A failed roll now rolls back to the LKG that survived the restart. The
    # current image is un-pinned (no fresh digest recorded), so the loaded
    # DIGEST_A is the rollback target.
    updater2.run_once()
    assert len(runner2.image_update_calls(f"{IMAGE}@{DIGEST_A}")) == 1


# ---------------------------------------------------------------------------
# Builder wiring
# ---------------------------------------------------------------------------


def test_builder_returns_wired_scheduled_task() -> None:
    runner = FakeRunner({"base-admin": f"{IMAGE}@{DIGEST_A}"})
    targets = (ImageUpdateTarget(service="base-admin", image=IMAGE),)
    task = build_image_updater_task(
        Settings(),
        targets=targets,
        resolver=make_resolver(DIGEST_B),
        runner=runner,
        state_path=None,
    )
    assert task.name == "image-updater"
    assert task.interval_seconds == IMAGE_UPDATER_INTERVAL_SECONDS
    task.run()
    assert len(runner.image_update_calls(f"{IMAGE}@{DIGEST_B}")) == 1


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
