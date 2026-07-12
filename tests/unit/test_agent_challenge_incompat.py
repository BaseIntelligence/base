"""Agent-challenge remains reference-only and never launches after gateway removal."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from base.master.agent_challenge_compat import (
    AGENT_CHALLENGE_INCOMPATIBLE_CODE,
    agent_challenge_incompatibility,
)
from base.challenge_sdk.roles import Role, activate_role
from base.master.orchestration import MasterChallengeReconciler, RegistryReconcilePassResult


class _Registry:
    def __init__(self, challenges) -> None:
        self._challenges = challenges

    async def list(self):
        return list(self._challenges)


class _Orchestrator:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start_challenge(self, spec) -> None:
        self.started.append(spec.slug)

    def stop_challenge(self, slug: str) -> None:
        self.stopped.append(slug)

    def list_running_challenge_slugs(self) -> frozenset[str]:
        return frozenset()


@pytest.mark.asyncio
async def test_reconciler_refuses_agent_challenge(caplog: pytest.LogCaptureFixture) -> None:
    orchestrator = _Orchestrator()
    challenges = [
        SimpleNamespace(
            slug="agent-challenge",
            status="active",
            image="ghcr.io/example/agent-challenge:latest",
            internal_base_url="http://challenge-agent-challenge:8000",
            env={},
            metadata={},
            resources={},
            secrets=[],
            required_capabilities=[],
            volumes={},
            version="1",
            name="agent-challenge",
            emission_percent=15,
        ),
        SimpleNamespace(
            slug="prism",
            status="active",
            image="ghcr.io/example/prism:latest",
            internal_base_url="http://challenge-prism:8080",
            env={},
            metadata={},
            resources={},
            secrets=[],
            required_capabilities=[],
            volumes={},
            version="1",
            name="prism",
            emission_percent=85,
        ),
    ]
    # challenge_spec_from_registry may need more fields; wrap reconcile path by
    # only exercising the gate before start only for agent-challenge.
    reconciler = MasterChallengeReconciler(registry=_Registry(challenges), orchestrator=orchestrator)

    class _FixedReconciler(MasterChallengeReconciler):
        async def _active_challenges(self):  # type: ignore[override]
            return challenges

        def _running_challenge_slugs(self) -> set[str]:  # type: ignore[override]
            return set()

    reconciler = _FixedReconciler(registry=_Registry(challenges), orchestrator=orchestrator)
    with caplog.at_level(logging.ERROR):
        # start_challenge for prism will likely fail with incomplete namespace; intercept
        def _start(spec):  # noqa: ANN001
            if getattr(spec, "slug", None) == "prism":
                orchestrator.started.append("prism")
                return
            orchestrator.started.append(getattr(spec, "slug", "unknown"))

        orchestrator.start_challenge = _start  # type: ignore[method-assign]
        # monkeypatch challenge_spec_from_registry to identity-like object
        import base.master.orchestration as orch

        original = orch.challenge_spec_from_registry
        orch.challenge_spec_from_registry = lambda challenge: SimpleNamespace(slug=challenge.slug)
        try:
            with activate_role(Role.MASTER):
                result = await reconciler.reconcile_once()
        finally:
            orch.challenge_spec_from_registry = original
    assert "agent-challenge" not in orchestrator.started
    assert "prism" in orchestrator.started
    assert AGENT_CHALLENGE_INCOMPATIBLE_CODE in caplog.text
    assert isinstance(result, RegistryReconcilePassResult)


def test_bypass_legacy_vars_does_not_change_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISM_LLM_GATEWAY_URL", "http://gateway/llm/v1")
    monkeypatch.setenv("BASE_GATEWAY_TOKEN", "legacy")
    diagnostic = agent_challenge_incompatibility()
    assert diagnostic.code == AGENT_CHALLENGE_INCOMPATIBLE_CODE
    assert "adapter" in diagnostic.message.lower()
