"""Unit tests for the supervisor challenge-image-updater (Task 19).

Fake registry + fake resolver + fake controller only — no network, no
dockerd, no real database.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from base.config.settings import Settings
from base.master.docker_orchestrator import DockerOrchestrationError
from base.schemas.challenge import ChallengeStatus, ChallengeUpdate
from base.supervisor.challenge_image_updater import (
    CHALLENGE_IMAGE_UPDATER_INTERVAL_SECONDS,
    ChallengeImageUpdater,
    build_challenge_image_update_lifespan,
    build_challenge_image_updater_task,
    run_challenge_image_update_loop,
)
from base.supervisor.image_ref import ImageReference
from base.supervisor.retry import RetryPolicy
from base.supervisor.weight_submit import WeightsAlert

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
BASE = "ghcr.io/baseintelligence/demo:latest"


def record(
    slug: str,
    image: str,
    status: ChallengeStatus = ChallengeStatus.ACTIVE,
) -> SimpleNamespace:
    return SimpleNamespace(slug=slug, image=image, status=status)


class FakeRegistry:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records
        self.updates: list[tuple[str, ChallengeUpdate]] = []

    async def list(self) -> list[SimpleNamespace]:
        return list(self.records)

    async def update(self, slug: str, update: ChallengeUpdate) -> None:
        self.updates.append((slug, update))


class FakeController:
    def __init__(self) -> None:
        self.restarts: list[str] = []

    async def restart(self, slug: str) -> dict[str, str]:
        self.restarts.append(slug)
        return {"slug": slug, "operation": "restart", "status": "ok"}


class ServiceAwareController(FakeController):
    """A controller that can introspect the running service image.

    Mirrors the production ``DockerRuntimeController.running_image`` seam so the
    updater gates a roll on the SERVICE's actually-running digest (not the DB
    record).
    """

    def __init__(self, running: str | None = None) -> None:
        super().__init__()
        self.running = running
        self.running_image_calls: list[str] = []

    async def running_image(self, slug: str) -> str | None:
        self.running_image_calls.append(slug)
        return self.running


def make_resolver(digest: str):
    def resolver(reference: ImageReference) -> str:
        return digest

    return resolver


def make_updater(
    registry: FakeRegistry,
    controller: FakeController,
    resolver: Any,
    *,
    retry_policy: RetryPolicy | None = None,
    alert_emit: Any = None,
    clock: Any = None,
    jitter_source: Any = None,
) -> ChallengeImageUpdater:
    extra: dict[str, Any] = {}
    if retry_policy is not None:
        extra["retry_policy"] = retry_policy
    if alert_emit is not None:
        extra["alert_emit"] = alert_emit
    if clock is not None:
        extra["clock"] = clock
    if jitter_source is not None:
        extra["jitter_source"] = jitter_source
    return ChallengeImageUpdater(
        registry_factory=lambda: registry,
        controller_factory=lambda _registry: controller,
        resolver=resolver,
        **extra,
    )


def test_changed_digest_updates_record_and_restarts_active() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


def test_unchanged_digest_is_a_noop() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_A)).run_once()
    assert registry.updates == []
    assert controller.restarts == []


def test_unpinned_record_is_updated_to_pinned_reference() -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_A)).run_once()
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_A}"))]
    assert controller.restarts == ["demo"]


@pytest.mark.parametrize("status", [ChallengeStatus.DRAFT, ChallengeStatus.DISABLED])
def test_draft_and_disabled_are_skipped_entirely(status: ChallengeStatus) -> None:
    calls: list[ImageReference] = []

    def resolver(reference: ImageReference) -> str:
        calls.append(reference)
        return DIGEST_B

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}", status)])
    controller = FakeController()
    make_updater(registry, controller, resolver).run_once()
    assert calls == []
    assert registry.updates == []
    assert controller.restarts == []


def test_inactive_is_updated_but_never_restarted() -> None:
    registry = FakeRegistry(
        [record("demo", f"{BASE}@{DIGEST_A}", ChallengeStatus.INACTIVE)]
    )
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == []


@pytest.mark.parametrize(
    "image",
    [
        "docker.io/library/redis:7",
        "ghcr.io/baseintelligence/demo:sha-abc1234",
    ],
)
def test_non_ghcr_or_sha_tagged_images_are_skipped(image: str) -> None:
    registry = FakeRegistry([record("demo", image)])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []
    assert controller.restarts == []


def test_resolver_failure_skips_challenge_but_siblings_proceed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def resolver(reference: ImageReference) -> str:
        if "broken" in reference.repository:
            raise RuntimeError("registry unreachable")
        return DIGEST_B

    registry = FakeRegistry(
        [
            record("broken", f"ghcr.io/baseintelligence/broken:latest@{DIGEST_A}"),
            record("demo", f"{BASE}@{DIGEST_A}"),
        ]
    )
    controller = FakeController()
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level("WARNING", logger=logger_name):
        make_updater(registry, controller, resolver).run_once()
    assert [slug for slug, _ in registry.updates] == ["demo"]
    assert controller.restarts == ["demo"]
    assert any("digest resolution failed" in message for message in caplog.messages)


def test_non_sha256_digest_is_refused() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    make_updater(registry, controller, make_resolver("md5:nope")).run_once()
    assert registry.updates == []
    assert controller.restarts == []


def test_restart_failure_does_not_block_sibling_challenges() -> None:
    class ExplodingController(FakeController):
        async def restart(self, slug: str) -> dict[str, str]:
            if slug == "broken":
                raise RuntimeError("dockerd unavailable")
            return await super().restart(slug)

    registry = FakeRegistry(
        [
            record("broken", f"ghcr.io/baseintelligence/broken:latest@{DIGEST_A}"),
            record("demo", f"{BASE}@{DIGEST_A}"),
        ]
    )
    controller = ExplodingController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert [slug for slug, _ in registry.updates] == ["broken", "demo"]
    assert controller.restarts == ["demo"]


def test_tick_never_raises_even_when_registry_listing_fails() -> None:
    class ExplodingRegistry(FakeRegistry):
        async def list(self) -> list[SimpleNamespace]:
            raise RuntimeError("database down")

    registry = ExplodingRegistry([])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []
    assert controller.restarts == []


def test_builder_wires_task_name_interval_and_seams() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    task = build_challenge_image_updater_task(
        Settings(),
        registry_factory=lambda: registry,
        controller_factory=lambda _registry: controller,
        resolver=make_resolver(DIGEST_B),
    )
    assert task.name == "challenge-image-updater"
    assert task.interval_seconds == CHALLENGE_IMAGE_UPDATER_INTERVAL_SECONDS
    task.run()
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


def test_builder_tag_override_retargets_mutable_base() -> None:
    registry = FakeRegistry(
        [record("demo", f"ghcr.io/baseintelligence/demo:main@{DIGEST_A}")]
    )
    controller = FakeController()
    task = build_challenge_image_updater_task(
        Settings(),
        registry_factory=lambda: registry,
        controller_factory=lambda _registry: controller,
        resolver=make_resolver(DIGEST_B),
        tag="main",
    )
    task.run()
    expected = f"ghcr.io/baseintelligence/demo:main@{DIGEST_B}"
    assert registry.updates == [("demo", ChallengeUpdate(image=expected))]


# ---------------------------------------------------------------------------
# Service convergence gated on the SERVICE's actually-running digest, decoupled
# from the record update (VAL-CODE-AUTO-008). These use a controller that
# exposes ``running_image`` (the production seam), so the roll decision is made
# against the running service digest, not the DB record.
# ---------------------------------------------------------------------------


def test_service_behind_rolls_even_when_record_already_current() -> None:
    # The desync this fixes: the record ALREADY equals the resolved digest (a
    # prior tick advanced it), but the running service is stuck on an older
    # digest -> the service is STILL rolled to desired this tick.
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_A}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []  # record already current -> no DB write
    assert controller.restarts == ["demo"]  # service converged anyway
    assert controller.running_image_calls == ["demo"]


def test_service_current_is_a_noop() -> None:
    # Idempotent: the running service already equals desired -> no roll, no churn.
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_B}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []
    assert controller.restarts == []
    assert controller.running_image_calls == ["demo"]


def test_record_behind_updates_record_and_rolls_service() -> None:
    # Record AND service behind desired -> both the record is updated and the
    # service is rolled.
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_A}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


def test_service_image_unknown_triggers_roll() -> None:
    # running_image returns None (service absent) -> roll to converge
    # (restart_challenge starts the service when it does not exist).
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = ServiceAwareController(running=None)
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []
    assert controller.restarts == ["demo"]


def test_transient_inspect_failure_is_skipped_roll_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A transient inspect failure (running_image RAISES DockerOrchestrationError,
    # not "absent") must NOT be treated as absence: the challenge is skipped this
    # tick (logged CONCISELY at WARNING with NO stack trace, retried next tick),
    # so an already-current service is never spuriously --force redeployed on a
    # momentary dockerd error and the logs are not flooded every tick.
    class _RaisingController(ServiceAwareController):
        async def running_image(self, slug: str) -> str | None:
            self.running_image_calls.append(slug)
            raise DockerOrchestrationError("transient dockerd inspect error")

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = _RaisingController()
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert controller.running_image_calls == ["demo"]
    assert controller.restarts == []  # no redeploy on a transient inspect failure
    inspect_warnings = [
        rec
        for rec in caplog.records
        if rec.name == logger_name
        and rec.levelno == logging.WARNING
        and "transient service-inspect failure" in rec.getMessage()
    ]
    assert len(inspect_warnings) == 1
    # Concise warning: no stack trace attached.
    assert inspect_warnings[0].exc_info is None
    # The transient path is NOT logged at exception/stack-trace (ERROR+) level.
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


def test_transient_inspect_failure_reports_skipped_action_in_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _RaisingController(ServiceAwareController):
        async def running_image(self, slug: str) -> str | None:
            self.running_image_calls.append(slug)
            raise DockerOrchestrationError("transient dockerd inspect error")

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = _RaisingController()
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level(logging.INFO, logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert any(
        f"demo: desired={BASE}@{DIGEST_B} action=skipped-inspect-error" in message
        for message in caplog.messages
    )


def test_unexpected_inspect_error_is_logged_at_exception_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A genuinely unexpected error (NOT a DockerOrchestrationError) is not the
    # expected self-correcting transient case: it still surfaces at exception /
    # stack-trace (ERROR) level, and the roll is skipped for that challenge.
    class _RaisingController(ServiceAwareController):
        async def running_image(self, slug: str) -> str | None:
            self.running_image_calls.append(slug)
            raise RuntimeError("unexpected dockerd crash")

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = _RaisingController()
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert controller.running_image_calls == ["demo"]
    assert controller.restarts == []
    exception_records = [
        rec
        for rec in caplog.records
        if rec.name == logger_name and rec.levelno >= logging.ERROR
    ]
    assert len(exception_records) == 1
    # logger.exception attaches the stack trace.
    assert exception_records[0].exc_info is not None


def test_inactive_challenge_is_never_rolled_even_if_service_behind() -> None:
    registry = FakeRegistry(
        [record("demo", f"{BASE}@{DIGEST_B}", ChallengeStatus.INACTIVE)]
    )
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_A}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert controller.restarts == []
    # An INACTIVE challenge never queries/rolls the running service.
    assert controller.running_image_calls == []


def test_info_summary_reports_already_current(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_B}")
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level("INFO", logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert any(
        f"demo: desired={BASE}@{DIGEST_B} action=already-current" in message
        for message in caplog.messages
    )


def test_info_summary_reports_rolled(caplog: pytest.LogCaptureFixture) -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = ServiceAwareController(running=f"{BASE}@{DIGEST_A}")
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level("INFO", logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert any(
        f"demo: desired={BASE}@{DIGEST_B} action=rolled" in message
        for message in caplog.messages
    )


def test_info_summary_reports_skipped_not_tracked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = FakeRegistry([record("demo", "docker.io/library/redis:7")])
    controller = ServiceAwareController()
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level("INFO", logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert any(
        "demo: desired=<untracked> action=skipped-not-tracked" in message
        for message in caplog.messages
    )


def test_no_introspection_seam_degrades_to_record_change_gate() -> None:
    # A controller WITHOUT running_image cannot introspect the service, so the
    # updater degrades to record-change gating: record already current -> no roll.
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = FakeController()
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert registry.updates == []
    assert controller.restarts == []


# ---------------------------------------------------------------------------
# Proxy-hosted async loop + FastAPI lifespan (VAL-CODE-AUTO-007).
#
# The challenge-image-updater moved into the master proxy (architecture.md
# sec 9.1): a resilient, cancellable async loop that reuses
# ``ChallengeImageUpdater._refresh`` on a settings-driven interval, wired as a
# FastAPI lifespan gated so ``interval<=0`` disables it.
# ---------------------------------------------------------------------------


async def test_loop_rolls_on_digest_change_then_stops() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    updater = make_updater(registry, controller, make_resolver(DIGEST_B))
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_challenge_image_update_loop(
            updater, interval_seconds=0.01, shutdown_event=shutdown
        )
    )
    for _ in range(200):
        await asyncio.sleep(0.005)
        if registry.updates:
            break
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


async def test_loop_is_noop_when_already_current() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    updater = make_updater(registry, controller, make_resolver(DIGEST_A))
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_challenge_image_update_loop(
            updater, interval_seconds=0.01, shutdown_event=shutdown
        )
    )
    # Let several ticks run: the digest is unchanged, so no update/restart fires.
    await asyncio.sleep(0.05)
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert registry.updates == []
    assert controller.restarts == []


async def test_loop_continues_after_a_failing_tick() -> None:
    class OnceExplodingRegistry(FakeRegistry):
        def __init__(self, records: list[SimpleNamespace]) -> None:
            super().__init__(records)
            self.list_calls = 0

        async def list(self) -> list[SimpleNamespace]:
            self.list_calls += 1
            if self.list_calls == 1:
                raise RuntimeError("registry blip")
            return await super().list()

    registry = OnceExplodingRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    updater = make_updater(registry, controller, make_resolver(DIGEST_B))
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_challenge_image_update_loop(
            updater, interval_seconds=0.01, shutdown_event=shutdown
        )
    )
    for _ in range(200):
        await asyncio.sleep(0.005)
        if registry.updates:
            break
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)
    # The first tick raised but the loop kept going and rolled on a later tick.
    assert registry.list_calls >= 2
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


def test_lifespan_is_none_when_disabled() -> None:
    # No settings, or a non-positive interval, disables the loop (parity with the
    # registry-reconcile-interval gate; disabled -> lifespan returns None).
    assert build_challenge_image_update_lifespan(None, 60.0) is None
    assert build_challenge_image_update_lifespan(Settings(), 0) is None
    assert build_challenge_image_update_lifespan(Settings(), None) is None
    assert build_challenge_image_update_lifespan(Settings(), -1.0) is None


async def test_lifespan_starts_and_cancels_challenge_image_update_loop() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    lifespan = build_challenge_image_update_lifespan(
        Settings(),
        0.01,
        registry_factory=lambda: registry,
        controller_factory=lambda _registry: controller,
        resolver=make_resolver(DIGEST_B),
    )
    assert lifespan is not None

    async with lifespan(FastAPI()):
        for _ in range(200):
            await asyncio.sleep(0.005)
            if registry.updates:
                break
    # After the lifespan exits the loop task is cancelled+awaited cleanly, and the
    # digest change rolled the challenge while it ran.
    assert registry.updates == [("demo", ChallengeUpdate(image=f"{BASE}@{DIGEST_B}"))]
    assert controller.restarts == ["demo"]


async def test_lifespan_noop_when_already_current() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_A}")])
    controller = FakeController()
    lifespan = build_challenge_image_update_lifespan(
        Settings(),
        0.01,
        registry_factory=lambda: registry,
        controller_factory=lambda _registry: controller,
        resolver=make_resolver(DIGEST_A),
    )
    assert lifespan is not None

    async with lifespan(FastAPI()):
        await asyncio.sleep(0.05)
    assert registry.updates == []
    assert controller.restarts == []


class _FakeCache:
    def get(self) -> dict[str, int]:
        return {}


class _FakeNonceStore:
    async def reserve(self, **_kwargs: Any) -> None:
        return None


def test_create_proxy_app_wires_challenge_image_update_lifespan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proxy factory composes the challenge-image-update lifespan, forwarding
    the settings + interval so the loop runs INSIDE the proxy."""

    from base.master import app_proxy
    from base.master.registry import ChallengeRegistry

    calls: list[tuple[Any, Any]] = []

    def spy(settings: Any, interval: Any) -> None:
        calls.append((settings, interval))
        return None

    monkeypatch.setattr(app_proxy, "build_challenge_image_update_lifespan", spy)

    settings = Settings()
    app_proxy.create_proxy_app(
        registry=ChallengeRegistry(),
        nonce_store=_FakeNonceStore(),  # type: ignore[arg-type]
        metagraph_cache=_FakeCache(),  # type: ignore[arg-type]
        challenge_image_updater_settings=settings,
        challenge_image_update_interval_seconds=42.0,
    )
    assert calls == [(settings, 42.0)]


# ---------------------------------------------------------------------------
# Retry-with-rollback for an UNHEALTHY challenge roll (recommendation F).
#
# ``restart_challenge`` waits for readiness and RAISES a
# ``DockerOrchestrationError`` when the post-roll service is unhealthy, so the
# updater rolls the service back to the pre-roll digest and records a per-slug
# ``RetryState`` failure (backoff between ticks, exhaustion alert, new-digest
# reset — mirroring the master image-updater).
# ---------------------------------------------------------------------------


class UnhealthyRollController(ServiceAwareController):
    """A controller whose roll comes up UNHEALTHY (restart raises) and that can
    revert to a previous image via ``rollback`` (the production seam)."""

    def __init__(self, running: str | None = None) -> None:
        super().__init__(running=running)
        self.rollbacks: list[tuple[str, str]] = []

    async def restart(self, slug: str) -> dict[str, str]:
        self.restarts.append(slug)
        raise DockerOrchestrationError(f"challenge {slug!r} failed health checks")

    async def rollback(self, slug: str, image: str) -> dict[str, str]:
        self.rollbacks.append((slug, image))
        return {"slug": slug, "operation": "rollback", "status": "ok"}


class _AlertCollector:
    def __init__(self) -> None:
        self.alerts: list[WeightsAlert] = []

    def __call__(self, alert: WeightsAlert) -> None:
        self.alerts.append(alert)


def test_unhealthy_roll_rolls_back_to_previous_digest() -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = UnhealthyRollController(running=f"{BASE}@{DIGEST_A}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    # The roll was attempted, came up unhealthy, and the service was reverted to
    # the digest it was running BEFORE the roll.
    assert controller.restarts == ["demo"]
    assert controller.rollbacks == [("demo", f"{BASE}@{DIGEST_A}")]


def test_unhealthy_roll_reports_rolled_back_action(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = UnhealthyRollController(running=f"{BASE}@{DIGEST_A}")
    logger_name = "base.supervisor.challenge_image_updater"
    with caplog.at_level(logging.INFO, logger=logger_name):
        make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert any(
        f"demo: desired={BASE}@{DIGEST_B} action=rolled-back" in message
        for message in caplog.messages
    )


def test_repeated_unhealthy_roll_backs_off_between_ticks() -> None:
    clock = {"t": 0.0}
    policy = RetryPolicy(max_attempts=5, base_delay=60.0, jitter=False)
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = UnhealthyRollController(running=f"{BASE}@{DIGEST_A}")
    updater = make_updater(
        registry,
        controller,
        make_resolver(DIGEST_B),
        retry_policy=policy,
        clock=lambda: clock["t"],
    )
    # Tick 1: roll fails → rollback → backoff scheduled (next eligible at 60s).
    updater.run_once()
    assert controller.restarts == ["demo"]
    assert controller.rollbacks == [("demo", f"{BASE}@{DIGEST_A}")]
    # Tick 2 (t=30, still backing off): the roll is skipped, not hammered.
    clock["t"] = 30.0
    updater.run_once()
    assert controller.restarts == ["demo"]
    # Tick 3 (t=60, eligible again): the roll is retried.
    clock["t"] = 60.0
    updater.run_once()
    assert controller.restarts == ["demo", "demo"]


def test_max_attempts_emits_alert_and_stops_hammering() -> None:
    clock = {"t": 0.0}
    policy = RetryPolicy(max_attempts=2, base_delay=1.0, jitter=False)
    alerts = _AlertCollector()
    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = UnhealthyRollController(running=f"{BASE}@{DIGEST_A}")
    updater = make_updater(
        registry,
        controller,
        make_resolver(DIGEST_B),
        retry_policy=policy,
        clock=lambda: clock["t"],
        alert_emit=alerts,
    )
    # Tick 1 (t=0): first failure, budget not yet spent → no alert.
    updater.run_once()
    assert alerts.alerts == []
    # Tick 2 (t=1): second failure exhausts the budget → alert fired once.
    clock["t"] = 1.0
    updater.run_once()
    assert len(alerts.alerts) == 1
    assert alerts.alerts[0].kind == "challenge_image_update_failed"
    assert alerts.alerts[0].details["slug"] == "demo"
    assert alerts.alerts[0].details["attempts"] == 2
    # Tick 3 (t=3): budget spent → no further roll and no duplicate alert.
    clock["t"] = 3.0
    updater.run_once()
    assert controller.restarts == ["demo", "demo"]
    assert len(alerts.alerts) == 1


def test_new_digest_resets_retry_state() -> None:
    clock = {"t": 0.0}
    policy = RetryPolicy(max_attempts=2, base_delay=100.0, jitter=False)
    resolved = {"digest": DIGEST_B}

    def resolver(_reference: ImageReference) -> str:
        return resolved["digest"]

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = UnhealthyRollController(running=f"{BASE}@{DIGEST_A}")
    updater = make_updater(
        registry,
        controller,
        resolver,
        retry_policy=policy,
        clock=lambda: clock["t"],
    )
    # Tick 1: roll fails → long backoff (next eligible at 100s).
    updater.run_once()
    assert controller.restarts == ["demo"]
    # Tick 2 (still backing off): no roll.
    updater.run_once()
    assert controller.restarts == ["demo"]
    # A NEW desired digest resets the state, so the roll is retried immediately
    # despite the outstanding backoff for the previous digest.
    resolved["digest"] = DIGEST_C
    updater.run_once()
    assert controller.restarts == ["demo", "demo"]


def test_no_rollback_seam_records_failure_without_reverting() -> None:
    # A controller WITHOUT a rollback seam still records the failure/backoff (it
    # simply cannot revert); the tick never raises.
    class _NoRollbackController(ServiceAwareController):
        async def restart(self, slug: str) -> dict[str, str]:
            self.restarts.append(slug)
            raise DockerOrchestrationError("unhealthy")

    registry = FakeRegistry([record("demo", f"{BASE}@{DIGEST_B}")])
    controller = _NoRollbackController(running=f"{BASE}@{DIGEST_A}")
    make_updater(registry, controller, make_resolver(DIGEST_B)).run_once()
    assert controller.restarts == ["demo"]
    assert not hasattr(controller, "rollbacks")
