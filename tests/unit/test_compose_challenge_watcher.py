"""Unit tests for the Compose challenge watcher (VAL-COMPOSE-022..041, CROSS-071).

Fake registry + controller + resolver only — no dockerd, no network.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from base.challenge_sdk.roles import Role, activate_role
from base.config.settings import MasterSettings, Settings
from base.master.docker_orchestrator import DockerOrchestrationError
from base.schemas.challenge import ChallengeStatus, ChallengeUpdate
from base.supervisor.challenge_watcher import (
    ChallengeWatcher,
    WatcherPhase,
    WatcherStateStore,
    build_challenge_watcher_lifespan,
    run_challenge_watcher_loop,
)
from base.supervisor.image_ref import ImageReference
from base.supervisor.retry import RetryPolicy


@pytest.fixture(autouse=True)
def _activate_master_role() -> Iterator[None]:
    with activate_role(Role.MASTER):
        yield


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64
BASE = "ghcr.io/baseintelligence/demo:latest"
PINNED_A = f"{BASE}@{DIGEST_A}"
PINNED_B = f"{BASE}@{DIGEST_B}"
PINNED_C = f"{BASE}@{DIGEST_C}"


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
        for item in self.records:
            if item.slug == slug:
                item.image = update.image


class FakeController:
    def __init__(self, running: dict[str, str] | None = None) -> None:
        self.running = dict(running or {})
        self.restarts: list[str] = []
        self.verifies: list[str] = []
        self.rollbacks: list[tuple[str, str]] = []
        self.pulls: list[str] = []
        self.fail_restart_with: Exception | None = None
        self.fail_verify_with: Exception | None = None
        self.fail_pull_with: Exception | None = None
        self.restart_delay_ticks = 0

    async def running_image(self, slug: str) -> str | None:
        return self.running.get(slug)

    async def pull(self, slug: str) -> dict[str, str]:
        self.pulls.append(slug)
        if self.fail_pull_with is not None:
            raise self.fail_pull_with
        return {"slug": slug, "operation": "pull", "status": "ok"}

    async def restart(self, slug: str) -> dict[str, str]:
        self.restarts.append(slug)
        if self.fail_restart_with is not None:
            raise self.fail_restart_with
        # After a healthy restart, desire is applied from the registry image.
        for uid, image in list(self.running.items()):
            del uid, image
        # Controller leaves update of running image to the test/registry position.
        return {"slug": slug, "operation": "restart", "status": "ok"}

    async def verify(self, slug: str) -> dict[str, str]:
        """Re-probe readiness without force-recreate (mid-rollout resume)."""

        self.verifies.append(slug)
        if self.fail_verify_with is not None:
            raise self.fail_verify_with
        if self.fail_restart_with is not None:
            # Preserve older tests that only set fail_restart_with.
            raise self.fail_restart_with
        return {"slug": slug, "operation": "verify", "status": "ok"}

    async def rollback(self, slug: str, image: str) -> dict[str, str]:
        self.rollbacks.append((slug, image))
        self.running[slug] = image
        return {"slug": slug, "operation": "rollback", "status": "ok"}


def make_resolver(digest: str):
    def resolver(reference: ImageReference) -> str:
        del reference
        return digest

    return resolver


def make_watcher(
    tmp_path: Path,
    registry: FakeRegistry,
    controller: FakeController,
    resolver: Any,
    *,
    retry_policy: RetryPolicy | None = None,
    clock: Any = None,
    wall_clock: Any = None,
    jitter_source: Any = None,
) -> ChallengeWatcher:
    extra: dict[str, Any] = {
        "retry_policy": retry_policy
        or RetryPolicy(max_attempts=3, base_delay=10.0, max_delay=40.0, jitter=False),
    }
    if clock is not None:
        extra["clock"] = clock
    if wall_clock is not None:
        extra["wall_clock"] = wall_clock
    if jitter_source is not None:
        extra["jitter_source"] = jitter_source
    return ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _reg: controller,
        resolver=resolver,
        state_store=WatcherStateStore(tmp_path / "watcher.json"),
        project_name="mission-watcher-test",
        **extra,
    )


@pytest.mark.asyncio
async def test_stable_digest_is_strict_noop(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", PINNED_A)])
    controller = FakeController(running={"demo": PINNED_A})
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_A))
    actions = await watcher.run_once()
    assert actions["demo"] == "already-current"
    assert controller.restarts == []
    assert controller.pulls == []
    assert controller.rollbacks == []
    actions2 = await watcher.run_once()
    assert actions2["demo"] == "already-current"
    assert controller.restarts == []


@pytest.mark.asyncio
async def test_healthy_digest_update_is_targeted(tmp_path: Path) -> None:
    # Mutable tracking tag in the registry; desired digest comes from resolver.
    registry = FakeRegistry(
        [
            record("demo", BASE),
            record("sibling", BASE),
        ]
    )
    controller = FakeController(running={"demo": PINNED_A, "sibling": PINNED_A})

    def resolver(reference: ImageReference) -> str:
        del reference
        return DIGEST_B

    watcher = make_watcher(tmp_path, registry, controller, resolver)
    registry.records[1].status = ChallengeStatus.INACTIVE
    actions = await watcher.run_once()
    assert actions["demo"] == "rolled"
    assert actions["sibling"] == "skipped-inactive"
    assert controller.restarts == ["demo"]
    assert "sibling" not in controller.restarts
    assert controller.pulls == ["demo"]
    assert registry.updates and registry.updates[0][1].image == PINNED_B


@pytest.mark.asyncio
async def test_mutable_or_malformed_digest_refused(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", PINNED_A)])
    controller = FakeController(running={"demo": PINNED_A})

    def bad_resolver(reference: ImageReference) -> str:
        del reference
        return "not-a-digest"

    watcher = make_watcher(tmp_path, registry, controller, bad_resolver)
    # Mutable-tracking mode always re-resolves; malformed digests refuse mutation.
    actions = await watcher.run_once()
    assert actions["demo"] == "skipped-untracked-or-invalid"
    assert controller.restarts == []

    # Strict pin mode accepts existing digest-pinned records without resolver.
    strict = ChallengeWatcher(
        registry_factory=lambda: FakeRegistry([record("demo", PINNED_A)]),
        controller_factory=lambda _r: controller,
        resolver=bad_resolver,
        state_store=WatcherStateStore(tmp_path / "strict.json"),
        project_name="mission-watcher-test",
        allow_mutable_tracking=False,
    )
    actions = await strict.run_once()
    assert actions["demo"] == "already-current"
    assert controller.restarts == []


@pytest.mark.asyncio
async def test_health_failure_triggers_rollback(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError(
        "Challenge 'demo' failed health/version checks"
    )
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_B))
    actions = await watcher.run_once()
    assert actions["demo"] == "health-or-version-failed"
    assert controller.restarts == ["demo"]
    assert controller.rollbacks == [("demo", PINNED_A)]
    # Running remains previous digest after rollback.
    assert controller.running["demo"] == PINNED_A
    state = WatcherStateStore(tmp_path / "watcher.json").load()["demo"]
    assert state.phase in {WatcherPhase.BACKOFF, WatcherPhase.EXHAUSTED}
    assert state.rollback_digest == DIGEST_A
    assert state.desired_digest == DIGEST_B


@pytest.mark.asyncio
async def test_version_failure_triggers_rollback(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError(
        "Challenge 'demo' version/capability mismatch: api_version='0.0'"
    )
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_B))
    actions = await watcher.run_once()
    assert actions["demo"] == "health-or-version-failed"
    assert controller.rollbacks == [("demo", PINNED_A)]
    state = WatcherStateStore(tmp_path / "watcher.json").load()["demo"]
    assert state.last_version_ok is False


@pytest.mark.asyncio
async def test_pull_failure_is_non_disruptive(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_pull_with = DockerOrchestrationError("pull denied")
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_B))
    actions = await watcher.run_once()
    assert actions["demo"] == "pull-failed"
    assert controller.restarts == []
    assert controller.running["demo"] == PINNED_A


@pytest.mark.asyncio
async def test_failed_digest_uses_bounded_backoff(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError("unhealthy")
    clock = {"t": 0.0}

    def now() -> float:
        return clock["t"]

    policy = RetryPolicy(max_attempts=3, base_delay=10.0, max_delay=40.0, jitter=False)
    watcher = make_watcher(
        tmp_path,
        registry,
        controller,
        make_resolver(DIGEST_B),
        retry_policy=policy,
        clock=now,
    )
    assert (await watcher.run_once())["demo"] == "health-or-version-failed"
    # Immediate next attempt while still within backoff window is skipped.
    assert (await watcher.run_once())["demo"] == "skipped-backoff"
    # Advance past first delay (10s).
    clock["t"] = 10.0
    assert (await watcher.run_once())["demo"] == "health-or-version-failed"
    clock["t"] = 30.0  # next delay=20 -> eligible
    assert (await watcher.run_once())["demo"] == "health-or-version-failed"
    # Exhausted after max_attempts=3.
    clock["t"] = 100.0
    assert (await watcher.run_once())["demo"] == "skipped-exhausted"
    state = WatcherStateStore(tmp_path / "watcher.json").load()["demo"]
    assert state.attempts == 3
    assert state.phase == WatcherPhase.EXHAUSTED


@pytest.mark.asyncio
async def test_new_digest_resets_backoff(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError("unhealthy")
    policy = RetryPolicy(max_attempts=2, base_delay=60.0, max_delay=600.0, jitter=False)
    clock = {"t": 0.0}
    watcher = make_watcher(
        tmp_path,
        registry,
        controller,
        make_resolver(DIGEST_B),
        retry_policy=policy,
        clock=lambda: clock["t"],
    )
    assert (await watcher.run_once())["demo"] == "health-or-version-failed"
    assert (await watcher.run_once())["demo"] == "skipped-backoff"
    shared = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_C),
        state_store=WatcherStateStore(tmp_path / "watcher.json"),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: clock["t"],
    )
    controller.fail_restart_with = None
    controller.running["demo"] = PINNED_A
    actions = await shared.run_once()
    assert actions["demo"] == "rolled"
    state = WatcherStateStore(tmp_path / "watcher.json").load()["demo"]
    assert state.desired_digest == DIGEST_C
    assert state.attempts == 0
    assert state.phase == WatcherPhase.IDLE
    del watcher


@pytest.mark.asyncio
async def test_watcher_resumes_after_restart(tmp_path: Path) -> None:
    store_path = tmp_path / "watcher.json"
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError("unhealthy")
    policy = RetryPolicy(
        max_attempts=5, base_delay=100.0, max_delay=400.0, jitter=False
    )
    wall = {"t": 1_000.0}
    first = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: 0.0,
        wall_clock=lambda: wall["t"],
    )
    assert (await first.run_once())["demo"] == "health-or-version-failed"
    loaded = WatcherStateStore(store_path).load()["demo"]
    assert loaded.desired_digest == DIGEST_B
    assert loaded.rollback_digest == DIGEST_A
    assert loaded.attempts == 1
    assert loaded.next_eligible_at == pytest.approx(1_100.0)
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    # Durable JSON must not store authoritative process-local mono timestamps.
    assert "next_eligible_monotonic" not in raw["challenges"]["demo"]
    assert raw["challenges"]["demo"]["next_eligible_at"] == pytest.approx(1_100.0)

    # Master restart: new process loads durable state. Model monotonic clock
    # RESET to a small value while wall-clock still within the backoff window.
    second = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: 0.5,  # reset mono; wall source decides residual delay
        wall_clock=lambda: 1_050.0,
    )
    assert (await second.run_once())["demo"] == "skipped-backoff"
    # New healthy digest converges immediately after restart load.
    third = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_C),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: 0.5,
        wall_clock=lambda: 1_050.0,
    )
    controller.fail_restart_with = None
    assert (await third.run_once())["demo"] == "rolled"


@pytest.mark.asyncio
async def test_backoff_survives_monotonic_clock_reset(tmp_path: Path) -> None:
    """Persisted mono deadlines must not stick forever after process restart.

    VAL-COMPOSE-039 / VAL-CROSS-071: wall-clock next_eligible_at (or last_failure
    + delay) drives residual backoff after the monotonic clock resets.
    """

    store_path = tmp_path / "watcher.json"
    registry = FakeRegistry([record("demo", BASE)])
    controller = FakeController(running={"demo": PINNED_A})
    controller.fail_restart_with = DockerOrchestrationError("unhealthy")
    policy = RetryPolicy(
        max_attempts=5, base_delay=100.0, max_delay=400.0, jitter=False
    )
    mono = {"t": 5_000.0}
    wall = {"t": 2_000.0}
    first = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: mono["t"],
        wall_clock=lambda: wall["t"],
    )
    assert (await first.run_once())["demo"] == "health-or-version-failed"
    persisted = WatcherStateStore(store_path).load()["demo"]
    assert persisted.next_eligible_at == pytest.approx(2_100.0)
    assert persisted.last_failure_at == pytest.approx(2_000.0)

    # Process restarts: mono clocks start near 0 again, wall barely advanced.
    mono_reset = 1.0
    wall["t"] = 2_050.0
    second = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: mono_reset,
        wall_clock=lambda: wall["t"],
    )
    assert (await second.run_once())["demo"] == "skipped-backoff"
    # Wall advances past next_eligible_at → residual delay expires.
    wall["t"] = 2_100.0
    third = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=policy,
        clock=lambda: mono_reset + 5.0,
        wall_clock=lambda: wall["t"],
    )
    assert (await third.run_once())["demo"] == "health-or-version-failed"
    reloaded = WatcherStateStore(store_path).load()["demo"]
    assert reloaded.attempts == 2
    assert reloaded.next_eligible_at == pytest.approx(2_300.0)  # +200s delay


@pytest.mark.asyncio
async def test_mid_verifying_resume_reprobes_and_rolls_back(tmp_path: Path) -> None:
    """Digest equality must not skip re-verify while phase is mid-rollout.

    Restart mid VERIFYING with already-current desired digest re-probes
    /health+/version and finishes rollback on failure (VAL-CROSS-071).
    """

    store_path = tmp_path / "watcher.json"
    # Durable mid-rollout snapshot: desired already offlined onto the container,
    # but verify never completed and rollback still points at previous good.
    store_path.write_text(
        json.dumps(
            {
                "version": 1,
                "challenges": {
                    "demo": {
                        "slug": "demo",
                        "desired_digest": DIGEST_B,
                        "current_digest": DIGEST_B,
                        "rollback_digest": DIGEST_A,
                        "desired_image": PINNED_B,
                        "rollback_image": PINNED_A,
                        "phase": WatcherPhase.VERIFYING.value,
                        "attempts": 0,
                        "last_error": None,
                        "last_result": None,
                        "last_health_ok": None,
                        "last_version_ok": None,
                        "alerted": False,
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    registry = FakeRegistry([record("demo", PINNED_B)])
    controller = FakeController(running={"demo": PINNED_B})
    controller.fail_verify_with = DockerOrchestrationError(
        "Challenge 'demo' failed health/version checks"
    )
    watcher = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=RetryPolicy(
            max_attempts=3, base_delay=10.0, max_delay=40.0, jitter=False
        ),
        clock=lambda: 10.0,
        wall_clock=lambda: 5_000.0,
    )
    actions = await watcher.run_once()
    assert actions["demo"] == "health-or-version-failed"
    assert controller.verifies == ["demo"]
    assert controller.restarts == []  # verify-only path, no force recreate
    assert controller.rollbacks == [("demo", PINNED_A)]
    assert controller.running["demo"] == PINNED_A
    state = WatcherStateStore(store_path).load()["demo"]
    assert state.phase in {WatcherPhase.BACKOFF, WatcherPhase.EXHAUSTED}
    assert state.current_digest == DIGEST_A
    assert state.rollback_digest == DIGEST_A
    assert state.last_result == "rolled-back"
    # Failure message contains "version" so the failure flag is last_version_ok;
    # successful rollback itself marks the restored service health True.
    assert state.last_version_ok is False
    assert state.attempts == 1


@pytest.mark.asyncio
async def test_mid_verifying_resume_healthy_commits(tmp_path: Path) -> None:
    store_path = tmp_path / "watcher.json"
    store_path.write_text(
        json.dumps(
            {
                "version": 1,
                "challenges": {
                    "demo": {
                        "slug": "demo",
                        "desired_digest": DIGEST_B,
                        "current_digest": DIGEST_B,
                        "rollback_digest": DIGEST_A,
                        "desired_image": PINNED_B,
                        "rollback_image": PINNED_A,
                        "phase": WatcherPhase.VERIFYING.value,
                        "attempts": 0,
                        "last_error": None,
                        "last_result": None,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = FakeRegistry([record("demo", PINNED_B)])
    controller = FakeController(running={"demo": PINNED_B})
    watcher = ChallengeWatcher(
        registry_factory=lambda: registry,
        controller_factory=lambda _r: controller,
        resolver=make_resolver(DIGEST_B),
        state_store=WatcherStateStore(store_path),
        project_name="mission-watcher-test",
        retry_policy=RetryPolicy(
            max_attempts=3, base_delay=10.0, max_delay=40.0, jitter=False
        ),
        clock=lambda: 10.0,
        wall_clock=lambda: 5_000.0,
    )
    actions = await watcher.run_once()
    assert actions["demo"] == "resumed-verified"
    assert controller.verifies == ["demo"]
    assert controller.rollbacks == []
    state = WatcherStateStore(store_path).load()["demo"]
    assert state.phase == WatcherPhase.IDLE
    assert state.current_digest == DIGEST_B
    assert state.last_result == "resumed-verified"


@pytest.mark.asyncio
async def test_per_challenge_serialization_lock(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", BASE), record("other", BASE)])
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowController(FakeController):
        async def restart(self, slug: str) -> dict[str, str]:
            self.restarts.append(slug)
            if slug == "demo":
                started.set()
                await release.wait()
            return {"slug": slug, "operation": "restart", "status": "ok"}

    slow = SlowController(running={"demo": PINNED_A, "other": PINNED_A})
    watcher = make_watcher(tmp_path, registry, slow, make_resolver(DIGEST_B))

    task1 = asyncio.create_task(watcher.run_once())
    await asyncio.wait_for(started.wait(), timeout=2)
    # Concurrent second pass for same slug should not create nested restarts
    # beyond the lock-protected first tick; sibling may continue when first
    # pass processes it if ordered after. Force a direct concurrent refresh
    # for demo only.
    task2 = asyncio.create_task(
        watcher._refresh_challenge(registry, slow, registry.records[0])
    )
    await asyncio.sleep(0.05)
    # While first holds the lock, second is waiting; only one restart so far.
    assert slow.restarts.count("demo") == 1
    release.set()
    await task1
    await task2
    assert slow.restarts.count("demo") == 2  # second proceeds only after first


@pytest.mark.asyncio
async def test_inactive_and_draft_never_roll(tmp_path: Path) -> None:
    registry = FakeRegistry(
        [
            record("a", PINNED_A, ChallengeStatus.INACTIVE),
            record("b", PINNED_A, ChallengeStatus.DRAFT),
            record("c", PINNED_A, ChallengeStatus.DISABLED),
        ]
    )
    controller = FakeController()
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_B))
    actions = await watcher.run_once()
    assert actions["a"] == "skipped-inactive"
    assert actions["b"] == "skipped-not-active"
    assert actions["c"] == "skipped-not-active"
    assert controller.restarts == []


def test_state_store_atomic_roundtrip(tmp_path: Path) -> None:
    store = WatcherStateStore(tmp_path / "state.json")
    watcher = ChallengeWatcher(
        registry_factory=lambda: FakeRegistry([]),
        controller_factory=lambda _r: FakeController(),
        resolver=make_resolver(DIGEST_A),
        state_store=store,
        project_name="p",
    )
    rec = watcher._record("demo")
    rec.desired_digest = DIGEST_B
    rec.current_digest = DIGEST_A
    rec.phase = WatcherPhase.BACKOFF
    rec.attempts = 2
    watcher._persist()
    reloaded = WatcherStateStore(tmp_path / "state.json").load()
    assert reloaded["demo"].desired_digest == DIGEST_B
    assert reloaded["demo"].current_digest == DIGEST_A
    assert reloaded["demo"].phase == WatcherPhase.BACKOFF
    assert reloaded["demo"].attempts == 2
    raw = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert raw["version"] == 1


def test_compose_backend_refuses_unpinned_and_foreign_project(tmp_path: Path) -> None:
    from base.master.compose_backend import ComposeChallengeOrchestrator
    from base.master.docker_orchestrator import ChallengeSpec

    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n  challenge-prism:\n    image: example\n",
        encoding="utf-8",
    )
    orch = ComposeChallengeOrchestrator(
        project_name="mission-p",
        compose_file=compose,
        override_dir=tmp_path / "ovr",
    )
    with pytest.raises(DockerOrchestrationError):
        orch.pull_image("ghcr.io/baseintelligence/demo:latest")
    with pytest.raises(DockerOrchestrationError):
        orch.pull_image("ghcr.io/baseintelligence/demo@sha256:deadbeef")
    # Static base service: image-pin override only.
    path = orch._write_service_override(
        "challenge-prism", ChallengeSpec(slug="prism", image=PINNED_A)
    )
    text = path.read_text(encoding="utf-8")
    assert DIGEST_A in text
    assert "base.managed_by: master-watcher" in text
    assert "volumes:" not in text


def test_compose_backend_dynamic_challenge_override_is_full_service(
    tmp_path: Path,
) -> None:
    """VAL-COMPOSE-008/025: second active challenge is installable via override."""

    from base.master.compose_backend import ComposeChallengeOrchestrator
    from base.master.docker_orchestrator import ChallengeSpec

    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n  challenge-prism:\n    image: example\n",
        encoding="utf-8",
    )
    orch = ComposeChallengeOrchestrator(
        project_name="mission-p",
        compose_file=compose,
        override_dir=tmp_path / "ovr",
    )
    spec = ChallengeSpec(
        slug="challenge-b",
        image=PINNED_B,
        env={"PRISM_COMBINED_MODE": "true"},
        port=8080,
    )
    path = orch._write_service_override("challenge-challenge-b", spec)
    text = path.read_text(encoding="utf-8")
    assert "challenge-challenge-b:" in text
    assert DIGEST_B in text
    assert "base.compose.lifecycle: managed" in text
    assert "source: challenge-challenge-b_data" in text
    assert "networks:" in text
    assert path.stat().st_mode & 0o777 == 0o600


def test_watcher_lifespan_none_when_disabled(tmp_path: Path) -> None:
    # Disabled interval never constructs watcher state under /var/lib/base.
    assert build_challenge_watcher_lifespan(None, 60.0) is None
    settings = Settings(
        master=MasterSettings(
            challenge_watcher_state_path=str(tmp_path / "watcher.json")
        )
    )
    assert build_challenge_watcher_lifespan(settings, 0) is None
    assert build_challenge_watcher_lifespan(settings, -1) is None
    # Enabled path with an explicit writable state path is non-None.
    lifespan = build_challenge_watcher_lifespan(
        settings, 0.01, state_path=tmp_path / "watcher.json"
    )
    assert lifespan is not None


@pytest.mark.asyncio
async def test_watcher_loop_logs_and_continues(tmp_path: Path) -> None:
    registry = FakeRegistry([record("demo", PINNED_A)])
    controller = FakeController(running={"demo": PINNED_A})
    watcher = make_watcher(tmp_path, registry, controller, make_resolver(DIGEST_A))
    shutdown = asyncio.Event()

    async def stop_soon() -> None:
        await asyncio.sleep(0.05)
        shutdown.set()

    asyncio.create_task(stop_soon())
    await run_challenge_watcher_loop(
        watcher, interval_seconds=0.02, shutdown_event=shutdown
    )


@pytest.mark.asyncio
async def test_lifespan_starts_and_cancels(tmp_path: Path) -> None:
    settings = Settings()
    lifespan_factory = build_challenge_watcher_lifespan(
        settings,
        0.05,
        registry_factory=lambda: FakeRegistry([]),
        controller_factory=lambda _r: FakeController(),
        resolver=make_resolver(DIGEST_A),
        state_path=tmp_path / "w.json",
        project_name="mission-watcher-test",
    )
    assert lifespan_factory is not None
    app = FastAPI()
    async with lifespan_factory(app):
        await asyncio.sleep(0.08)


def test_compose_project_boundary_helpers() -> None:
    from base.master.compose_backend import _parse_compose_ps_json, _parse_label_string

    entries = _parse_compose_ps_json(
        json.dumps(
            [
                {
                    "Service": "challenge-prism",
                    "Project": "mission-p",
                    "Labels": (
                        "base.challenge.slug=prism,com.docker.compose.project=mission-p"
                    ),
                }
            ]
        )
    )
    assert entries[0]["Service"] == "challenge-prism"
    labels = _parse_label_string(
        "base.challenge.slug=prism,com.docker.compose.project=mission-p"
    )
    assert labels["base.challenge.slug"] == "prism"
