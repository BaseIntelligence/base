"""Tests for the master registry-driven challenge reconciler.

Covers the m7 registry-driven-deploy feature (architecture.md sec 4 + sec 9.2):
a master-side control loop that turns every ACTIVE registry challenge into a
running challenge service (idempotent) and tears down services for challenges
that are no longer ACTIVE. Both the registry and the challenge-service
orchestrator are faked here. Fulfills VAL-CODE-REG-001 / VAL-CODE-REG-002.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from fastapi import FastAPI

from base.master.docker_orchestrator import ChallengeSpec
from base.master.orchestration import (
    MasterChallengeReconciler,
    build_master_registry_reconcile_lifespan,
    run_registry_reconcile_loop,
)
from base.schemas.challenge import ChallengeRecord, ChallengeStatus


def _record(
    slug: str,
    status: ChallengeStatus = ChallengeStatus.ACTIVE,
    *,
    resources: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    required_capabilities: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> ChallengeRecord:
    return ChallengeRecord(
        slug=slug,
        name=slug.title(),
        image=f"ghcr.io/o/{slug}:1",
        version="1",
        emission_percent=Decimal("0"),
        status=status,
        token_hash="hash",
        token_hint="hint",
        internal_base_url=f"http://challenge-{slug}:8000",
        public_proxy_base_path=f"/challenges/{slug}",
        required_capabilities=required_capabilities or ["get_weights", "proxy_routes"],
        resources=resources or {},
        env=env or {},
        metadata=metadata or {},
    )


class FakeRegistry:
    """Sync faked registry that honors ``active_only`` like the DB registry."""

    def __init__(
        self, records: list[ChallengeRecord], *, honor_active_only: bool = True
    ) -> None:
        self.records = list(records)
        self.honor_active_only = honor_active_only
        self.active_only_calls: list[bool] = []

    def list(self, *, active_only: bool = False) -> list[ChallengeRecord]:
        self.active_only_calls.append(active_only)
        if active_only and self.honor_active_only:
            return [r for r in self.records if r.status == ChallengeStatus.ACTIVE]
        return list(self.records)


class FakeAsyncRegistry(FakeRegistry):
    """Async faked registry mirroring ``DatabaseChallengeRegistry.list``."""

    async def list(  # type: ignore[override]
        self, *, active_only: bool = False
    ) -> list[ChallengeRecord]:
        return FakeRegistry.list(self, active_only=active_only)


class FakeOrchestrator:
    def __init__(
        self,
        *,
        fail_slugs: set[str] | None = None,
        fail_stop_slugs: set[str] | None = None,
    ) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.specs: list[ChallengeSpec] = []
        self.fail_slugs = fail_slugs or set()
        self.fail_stop_slugs = fail_stop_slugs or set()

    def start_challenge(self, spec: ChallengeSpec, *, recreate: bool = False) -> object:
        if spec.slug in self.fail_slugs:
            raise RuntimeError(f"start failed for {spec.slug}")
        self.started.append(spec.slug)
        self.specs.append(spec)
        return object()

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        if slug in self.fail_stop_slugs:
            raise RuntimeError(f"stop failed for {slug}")
        self.stopped.append(slug)


async def test_starts_all_active_challenges_once_idempotent() -> None:
    registry = FakeRegistry(
        [
            _record("agent-challenge"),
            _record("prism"),
            _record("draft-one", ChallengeStatus.DRAFT),
        ]
    )
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    first = await reconciler.reconcile_once()
    assert sorted(first.started) == ["agent-challenge", "prism"]
    assert first.stopped == []
    assert sorted(orchestrator.started) == ["agent-challenge", "prism"]

    # Idempotent on re-run: each ACTIVE challenge is started exactly once.
    second = await reconciler.reconcile_once()
    assert second.started == []
    assert second.stopped == []
    assert sorted(orchestrator.started) == ["agent-challenge", "prism"]
    # It asks the registry for ACTIVE challenges only.
    assert registry.active_only_calls == [True, True]


async def test_non_active_challenges_are_never_started() -> None:
    registry = FakeRegistry(
        [
            _record("draft-one", ChallengeStatus.DRAFT),
            _record("inactive-one", ChallengeStatus.INACTIVE),
            _record("disabled-one", ChallengeStatus.DISABLED),
        ]
    )
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    result = await reconciler.reconcile_once()
    assert result.started == []
    assert orchestrator.started == []


async def test_non_active_filtered_even_if_registry_ignores_flag() -> None:
    # Defensive: a registry that ignores ``active_only`` must not cause a
    # DRAFT/INACTIVE/DISABLED challenge to be deployed.
    registry = FakeRegistry(
        [
            _record("prism"),
            _record("draft-one", ChallengeStatus.DRAFT),
            _record("disabled-one", ChallengeStatus.DISABLED),
        ],
        honor_active_only=False,
    )
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    result = await reconciler.reconcile_once()
    assert result.started == ["prism"]
    assert orchestrator.started == ["prism"]


async def test_new_active_challenge_deploys_next_pass() -> None:
    registry = FakeRegistry([_record("prism")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()
    assert orchestrator.started == ["prism"]

    # Register a new ACTIVE challenge; it deploys on the next pass, and the
    # already-deployed one is not re-started.
    registry.records.append(_record("agent-challenge"))
    result = await reconciler.reconcile_once()
    assert result.started == ["agent-challenge"]
    assert result.stopped == []
    assert orchestrator.started == ["prism", "agent-challenge"]


async def test_deactivated_challenge_is_stopped() -> None:
    prism = _record("prism")
    registry = FakeRegistry([prism, _record("agent-challenge")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()
    assert sorted(orchestrator.started) == ["agent-challenge", "prism"]

    # Flip prism away from ACTIVE -> its service is torn down next pass.
    registry.records = [
        _record("prism", ChallengeStatus.INACTIVE),
        _record("agent-challenge"),
    ]
    result = await reconciler.reconcile_once()
    assert result.started == []
    assert result.stopped == ["prism"]
    assert orchestrator.stopped == ["prism"]


async def test_removed_challenge_is_stopped() -> None:
    registry = FakeRegistry([_record("prism"), _record("agent-challenge")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()

    # Remove prism from the registry entirely -> torn down next pass.
    registry.records = [_record("agent-challenge")]
    result = await reconciler.reconcile_once()
    assert result.stopped == ["prism"]
    assert orchestrator.stopped == ["prism"]


async def test_reactivated_challenge_is_started_again() -> None:
    registry = FakeRegistry([_record("prism")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()
    registry.records = [_record("prism", ChallengeStatus.INACTIVE)]
    await reconciler.reconcile_once()
    assert orchestrator.stopped == ["prism"]

    # Re-activate: it is (re)deployed on the following pass.
    registry.records = [_record("prism")]
    result = await reconciler.reconcile_once()
    assert result.started == ["prism"]
    assert orchestrator.started == ["prism", "prism"]


async def test_spec_is_built_like_the_legacy_runner() -> None:
    registry = FakeRegistry(
        [
            _record(
                "agent-challenge",
                resources={"cpu": "2", "memory": "1g"},
                env={"FOO": "bar"},
                metadata={"worker_command": ["agent-challenge-worker"]},
            )
        ]
    )
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()
    spec = orchestrator.specs[0]
    assert spec.slug == "agent-challenge"
    assert spec.image == "ghcr.io/o/agent-challenge:1"
    assert spec.workload_class == "service"
    assert spec.resources.cpu == 2.0
    assert spec.resources.memory == "1g"
    assert spec.env == {"FOO": "bar"}
    assert spec.worker_command == ("agent-challenge-worker",)


async def test_start_failure_is_retried_next_pass() -> None:
    orchestrator = FakeOrchestrator(fail_slugs={"prism"})
    registry = FakeRegistry([_record("prism")])
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    first = await reconciler.reconcile_once()
    assert first.started == []
    assert orchestrator.started == []

    # The transient failure clears; the next pass deploys it.
    orchestrator.fail_slugs.clear()
    second = await reconciler.reconcile_once()
    assert second.started == ["prism"]
    assert orchestrator.started == ["prism"]


async def test_stop_failure_is_retried_next_pass() -> None:
    orchestrator = FakeOrchestrator(fail_stop_slugs={"prism"})
    registry = FakeRegistry([_record("prism")])
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    await reconciler.reconcile_once()
    assert orchestrator.started == ["prism"]

    # Deactivate prism; the stop raises, so it stays tracked and is retried.
    registry.records = [_record("prism", ChallengeStatus.INACTIVE)]
    first = await reconciler.reconcile_once()
    assert first.stopped == []
    assert orchestrator.stopped == []

    # The transient failure clears; the next pass tears it down.
    orchestrator.fail_stop_slugs.clear()
    second = await reconciler.reconcile_once()
    assert second.stopped == ["prism"]
    assert orchestrator.stopped == ["prism"]


async def test_supports_async_registry() -> None:
    registry = FakeAsyncRegistry([_record("prism"), _record("agent-challenge")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)

    result = await reconciler.reconcile_once()
    assert sorted(result.started) == ["agent-challenge", "prism"]


async def test_run_registry_reconcile_loop_runs_then_stops() -> None:
    registry = FakeRegistry([_record("prism")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_registry_reconcile_loop(
            reconciler, interval_seconds=0.01, shutdown_event=shutdown
        )
    )
    for _ in range(200):
        await asyncio.sleep(0.005)
        if orchestrator.started:
            break
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert orchestrator.started == ["prism"]


def test_lifespan_is_none_when_disabled() -> None:
    reconciler = MasterChallengeReconciler(
        registry=FakeRegistry([]), orchestrator=FakeOrchestrator()
    )
    assert build_master_registry_reconcile_lifespan(None, 60.0) is None
    assert build_master_registry_reconcile_lifespan(reconciler, 0) is None
    assert build_master_registry_reconcile_lifespan(reconciler, None) is None
    assert build_master_registry_reconcile_lifespan(reconciler, -1.0) is None


async def test_lifespan_runs_reconcile_loop() -> None:
    registry = FakeRegistry([_record("prism")])
    orchestrator = FakeOrchestrator()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orchestrator)
    lifespan = build_master_registry_reconcile_lifespan(reconciler, 0.01)
    assert lifespan is not None

    async with lifespan(FastAPI()):
        for _ in range(200):
            await asyncio.sleep(0.005)
            if orchestrator.started:
                break
    assert orchestrator.started == ["prism"]


async def test_loop_continues_after_a_failing_pass() -> None:
    class FlakyReconciler:
        def __init__(self) -> None:
            self.calls = 0

        async def reconcile_once(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient reconcile failure")

    reconciler = FlakyReconciler()
    shutdown = asyncio.Event()
    task = asyncio.create_task(
        run_registry_reconcile_loop(
            reconciler,  # type: ignore[arg-type]
            interval_seconds=0.01,
            shutdown_event=shutdown,
        )
    )
    for _ in range(200):
        await asyncio.sleep(0.005)
        if reconciler.calls >= 2:
            break
    shutdown.set()
    await asyncio.wait_for(task, timeout=2.0)
    # The first pass raised but the loop kept going and ran a second pass.
    assert reconciler.calls >= 2
