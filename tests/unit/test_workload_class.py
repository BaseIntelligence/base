"""Tests for ChallengeSpec.workload_class (job vs service scheduling class)."""

from __future__ import annotations

from typing import Any

import pytest

from base.master.docker_orchestrator import (
    ChallengeSpec,
    DockerOrchestrationError,
)
from base.master.workload_ledger import WorkloadEntry


def test_challenge_spec_defaults_to_job() -> None:
    spec = ChallengeSpec(slug="eval-run", image="ghcr.io/baseintelligence/eval:1")
    assert spec.workload_class == "job"


def test_challenge_spec_accepts_explicit_service() -> None:
    spec = ChallengeSpec(
        slug="prism",
        image="ghcr.io/baseintelligence/prism:1",
        workload_class="service",
    )
    assert spec.workload_class == "service"


def test_challenge_spec_rejects_invalid_workload_class() -> None:
    with pytest.raises(DockerOrchestrationError, match="workload_class"):
        ChallengeSpec(
            slug="prism",
            image="ghcr.io/baseintelligence/prism:1",
            workload_class="daemon",  # type: ignore[arg-type]
        )


def test_challenge_spec_default_matches_workload_ledger_default() -> None:
    entry = WorkloadEntry(key="svc-1", kind="swarm_service", challenge_slug="prism")
    spec = ChallengeSpec(slug="eval-run", image="ghcr.io/baseintelligence/eval:1")
    assert spec.workload_class == entry.workload_class == "job"


@pytest.mark.asyncio
async def test_cli_runtime_controller_spec_is_service() -> None:
    from base.cli_app.main import DockerRuntimeController

    class _Record:
        slug = "prism"
        image = "ghcr.io/baseintelligence/prism:1"
        version = "1.0.0"
        env: dict[str, str] = {}
        resources: dict[str, str] = {}
        required_capabilities: tuple[str, ...] = ()
        internal_base_url = "http://challenge-prism:8080"
        metadata: dict[str, Any] = {}

    class _Registry:
        async def get(self, slug: str) -> _Record:
            return _Record()

        def get_token(self, slug: str) -> str:
            return "token"

    controller = DockerRuntimeController(registry=_Registry(), orchestrator=None)
    spec = await controller._spec("prism")
    assert spec.workload_class == "service"
