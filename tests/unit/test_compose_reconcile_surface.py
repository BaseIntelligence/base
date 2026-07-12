"""Unit coverage for multi-challenge compose reconcile surfaces (008, 025-029, 024)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from base.challenge_sdk.roles import Role, activate_role
from base.master.compose_backend import ComposeChallengeOrchestrator
from base.master.docker_orchestrator import ChallengeSpec, DockerOrchestrationError
from base.master.orchestration import MasterChallengeReconciler
from base.schemas.challenge import ChallengeStatus

PINNED = "ghcr.io/baseintelligence/demo@sha256:" + ("b" * 64)


@pytest.fixture(autouse=True)
def _activate_master_role():
    with activate_role(Role.MASTER):
        yield


def _orch(
    tmp_path: Path, *, base_services: str = "challenge-prism"
) -> ComposeChallengeOrchestrator:
    services = "\n".join(f"  {name}:\n    image: x\n" for name in base_services.split())
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(f"services:\n{services}", encoding="utf-8")
    return ComposeChallengeOrchestrator(
        project_name="mission-reconcile",
        compose_file=compose,
        override_dir=tmp_path / "ovr",
    )


def test_dynamic_slug_writes_full_override_and_static_pin_only(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    static = orch._write_service_override(
        "challenge-prism", ChallengeSpec(slug="prism", image=PINNED)
    )
    dynamic = orch._write_service_override(
        "challenge-challenge-b",
        ChallengeSpec(slug="challenge-b", image=PINNED, env={"A": "1"}),
    )
    static_text = static.read_text(encoding="utf-8")
    dynamic_text = dynamic.read_text(encoding="utf-8")
    assert "challenge-prism:" in static_text
    assert "volumes:" not in static_text
    assert "challenge-challenge-b:" in dynamic_text
    assert "base.compose.lifecycle: managed" in dynamic_text
    assert "mission-reconcile_challenge-challenge-b_data" in dynamic_text
    assert dynamic.stat().st_mode & 0o777 == 0o600


def test_stop_challenge_refuses_static_lifecycle(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    orch._inspect_service_container = MagicMock(  # type: ignore[method-assign]
        return_value={
            "Config": {
                "Labels": {
                    "base.compose.lifecycle": "static",
                    "com.docker.compose.project": "mission-reconcile",
                }
            }
        }
    )
    run_mock = MagicMock()
    object.__setattr__(orch.runner, "run", run_mock)
    orch.stop_challenge("prism", remove=True)
    run_mock.assert_not_called()


def test_stop_challenge_removes_managed_and_override(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    override = orch._write_service_override(
        "challenge-challenge-b",
        ChallengeSpec(slug="challenge-b", image=PINNED),
    )
    assert override.is_file()
    orch._inspect_service_container = MagicMock(  # type: ignore[method-assign]
        return_value={
            "Config": {
                "Labels": {
                    "base.compose.lifecycle": "managed",
                    "com.docker.compose.project": "mission-reconcile",
                }
            }
        }
    )
    run_mock = MagicMock()
    object.__setattr__(orch.runner, "run", run_mock)
    orch.stop_challenge("challenge-b")
    run_mock.assert_called_once()
    assert run_mock.call_args.args[0][:2] == ["rm", "-sf"]
    assert not override.is_file()


@pytest.mark.asyncio
async def test_reconciler_starts_second_active_and_skips_inactive(
    tmp_path: Path,
) -> None:
    """VAL-COMPOSE-008/025/026: active starts once; inactive never starts."""

    class Registry:
        def __init__(self) -> None:
            self.records = [
                SimpleNamespace(
                    slug="prism",
                    image=PINNED,
                    version="0.1.0",
                    status=ChallengeStatus.ACTIVE,
                    env={},
                    resources={},
                    required_capabilities=["get_weights", "proxy_routes"],
                    metadata={"combined_mode_env": "PRISM_COMBINED_MODE"},
                    secrets=[],
                    internal_base_url="http://challenge-prism:8080",
                ),
                SimpleNamespace(
                    slug="challenge-b",
                    image=PINNED,
                    version="0.1.0",
                    status=ChallengeStatus.ACTIVE,
                    env={},
                    resources={},
                    required_capabilities=["get_weights", "proxy_routes"],
                    metadata={},
                    secrets=[],
                    internal_base_url="http://challenge-challenge-b:8080",
                ),
                SimpleNamespace(
                    slug="drafty",
                    image=PINNED,
                    version="0.1.0",
                    status=ChallengeStatus.DRAFT,
                    env={},
                    resources={},
                    required_capabilities=["get_weights", "proxy_routes"],
                    metadata={},
                    secrets=[],
                    internal_base_url="http://challenge-drafty:8080",
                ),
            ]

        async def list(self, active_only: bool = False):  # noqa: ANN201
            if active_only:
                return [r for r in self.records if r.status == ChallengeStatus.ACTIVE]
            return list(self.records)

    class Orch:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.stopped: list[str] = []
            self.running: set[str] = {"prism"}

        def start_challenge(self, spec: ChallengeSpec, *, recreate: bool = False):
            del recreate
            self.started.append(spec.slug)
            self.running.add(spec.slug)
            return SimpleNamespace(slug=spec.slug)

        def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
            del remove
            self.stopped.append(slug)
            self.running.discard(slug)

        def list_running_challenge_slugs(self) -> frozenset[str]:
            return frozenset(self.running)

    registry = Registry()
    orch = Orch()
    reconciler = MasterChallengeReconciler(registry=registry, orchestrator=orch)
    first = await reconciler.reconcile_once()
    assert "prism" in first.adopted
    assert "challenge-b" in first.started
    assert "drafty" not in first.started
    second = await reconciler.reconcile_once()
    assert second.started == []
    assert second.adopted == []

    # Deactivate challenge-b: managed stop once (VAL-COMPOSE-027).
    for rec in registry.records:
        if rec.slug == "challenge-b":
            rec.status = ChallengeStatus.INACTIVE
    third = await reconciler.reconcile_once()
    assert "challenge-b" in third.stopped
    assert "prism" not in third.stopped

    # Reactivate: starts again on same slug (VAL-COMPOSE-029 path entry).
    for rec in registry.records:
        if rec.slug == "challenge-b":
            rec.status = ChallengeStatus.ACTIVE
    fourth = await reconciler.reconcile_once()
    assert "challenge-b" in fourth.started


@pytest.mark.asyncio
async def test_reconciler_orphan_cleanup_cross_restart() -> None:
    """VAL-COMPOSE-028: orphan discovered from Docker, stopped after restart."""

    class Registry:
        async def list(self, active_only: bool = False):  # noqa: ANN201
            del active_only
            return [
                SimpleNamespace(
                    slug="prism",
                    image=PINNED,
                    version="0.1.0",
                    status=ChallengeStatus.ACTIVE,
                    env={},
                    resources={},
                    required_capabilities=["get_weights", "proxy_routes"],
                    metadata={},
                    secrets=[],
                    internal_base_url="http://challenge-prism:8080",
                )
            ]

    class Orch:
        def __init__(self) -> None:
            self.stopped: list[str] = []
            self.running = {"prism", "orphan-old"}

        def start_challenge(self, spec: ChallengeSpec, *, recreate: bool = False):
            del recreate
            return SimpleNamespace(slug=spec.slug)

        def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
            del remove
            self.stopped.append(slug)
            self.running.discard(slug)

        def list_running_challenge_slugs(self) -> frozenset[str]:
            return frozenset(self.running)

    orch = Orch()
    reconciler = MasterChallengeReconciler(registry=Registry(), orchestrator=orch)
    result = await reconciler.reconcile_once()
    assert "prism" in result.adopted
    assert "orphan-old" in result.stopped
    assert "prism" not in result.stopped


def test_unpinned_image_refused(tmp_path: Path) -> None:
    orch = _orch(tmp_path)
    with pytest.raises(DockerOrchestrationError):
        orch._write_service_override(
            "challenge-x",
            ChallengeSpec(slug="x", image="repo:tag"),
        )
