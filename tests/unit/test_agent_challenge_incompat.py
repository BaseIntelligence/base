"""Agent-challenge digest gate: unlock gateway-free digests only.

Covers VAL-ACAT-019, 041, 042, 045, 046 (Base unit surface).
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from types import SimpleNamespace

import pytest

from base.challenge_sdk.roles import Role, activate_role
from base.master.agent_challenge_compat import (
    AGENT_CHALLENGE_INCOMPATIBLE_CODE,
    FORBIDDEN_GATEWAY_ENV_NAMES,
    FORBIDDEN_MASTER_OPENROUTER_ENV_NAMES,
    GATEWAY_FREE_DIGESTS_ENV,
    agent_challenge_compose_env_is_gateway_free,
    agent_challenge_image_digest,
    agent_challenge_incompatibility,
    clear_gateway_free_digest_registry,
    decide_agent_challenge_activation,
    env_declares_gateway_contract,
    filter_forbidden_gateway_env,
    is_gateway_free_agent_challenge_image,
    master_env_holds_openrouter_keys,
    register_gateway_free_digest,
    should_refuse_agent_challenge,
)
from base.master.orchestration import (
    MasterChallengeReconciler,
    RegistryReconcilePassResult,
)

OLD_DIGEST = "sha256:" + ("a" * 64)
NEW_DIGEST = "sha256:" + ("b" * 64)
OTHER_DIGEST = "sha256:" + ("c" * 64)

OLD_IMAGE = f"ghcr.io/example/agent-challenge:legacy@{OLD_DIGEST}"
NEW_IMAGE = f"ghcr.io/example/agent-challenge:attestation-only@{NEW_DIGEST}"
OTHER_IMAGE = f"ghcr.io/example/agent-challenge:other@{OTHER_DIGEST}"


@pytest.fixture(autouse=True)
def _clear_digest_registry() -> Generator[None, None, None]:
    clear_gateway_free_digest_registry()
    yield
    clear_gateway_free_digest_registry()


class _Registry:
    def __init__(self, challenges) -> None:
        self._challenges = challenges

    async def list(self, *, active_only: bool = False):
        del active_only
        return list(self._challenges)


class _Orchestrator:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start_challenge(self, spec, *, recreate: bool = False) -> None:
        del recreate
        self.started.append(spec.slug)

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        del remove
        self.stopped.append(slug)

    def list_running_challenge_slugs(self) -> frozenset[str]:
        return frozenset()


def _challenge(
    slug: str,
    *,
    image: str,
    env: dict | None = None,
    emission: int = 15,
) -> SimpleNamespace:
    return SimpleNamespace(
        slug=slug,
        status="active",
        image=image,
        internal_base_url=f"http://challenge-{slug}:8000",
        env=env or {},
        metadata={},
        resources={},
        secrets=[],
        required_capabilities=[],
        volumes={},
        version="1",
        name=slug,
        emission_percent=emission,
    )


async def _reconcile(
    challenges: list[SimpleNamespace],
    orchestrator: _Orchestrator,
) -> RegistryReconcilePassResult:
    import base.master.orchestration as orch

    class _FixedReconciler(MasterChallengeReconciler):
        async def _active_challenges(self):  # type: ignore[override]
            return challenges

        def _running_challenge_slugs(self) -> set[str]:  # type: ignore[override]
            return set()

    reconciler = _FixedReconciler(
        registry=_Registry(challenges),
        orchestrator=orchestrator,
    )
    original = orch.challenge_spec_from_registry

    def _spec_from_registry(challenge):  # noqa: ANN001
        return SimpleNamespace(slug=challenge.slug, image=challenge.image)

    orch.challenge_spec_from_registry = _spec_from_registry
    try:
        with activate_role(Role.MASTER):
            return await reconciler.reconcile_once()
    finally:
        orch.challenge_spec_from_registry = original


def test_digest_helpers_normalize_image_pin() -> None:
    assert agent_challenge_image_digest(NEW_IMAGE) == NEW_DIGEST
    assert agent_challenge_image_digest("ghcr.io/x/y:tag") is None
    assert agent_challenge_image_digest(None) is None
    assert agent_challenge_image_digest(f"repo@SHA256:{'B' * 64}") == NEW_DIGEST


def test_register_and_env_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not is_gateway_free_agent_challenge_image(NEW_IMAGE)
    register_gateway_free_digest(NEW_DIGEST)
    assert is_gateway_free_agent_challenge_image(NEW_IMAGE)
    assert not is_gateway_free_agent_challenge_image(OLD_IMAGE)

    clear_gateway_free_digest_registry()
    monkeypatch.setenv(
        GATEWAY_FREE_DIGESTS_ENV,
        f"{NEW_DIGEST}, not-a-digest, sha256:{'c' * 64}",
    )
    assert is_gateway_free_agent_challenge_image(NEW_IMAGE)
    assert is_gateway_free_agent_challenge_image(OTHER_IMAGE)
    assert not is_gateway_free_agent_challenge_image(OLD_IMAGE)


def test_gateway_env_residue_blocks_even_allowlisted_digest() -> None:
    register_gateway_free_digest(NEW_DIGEST)
    assert env_declares_gateway_contract({"BASE_GATEWAY_TOKEN": "x"})
    assert env_declares_gateway_contract({"BASE_LLM_GATEWAY_URL": "http://x/llm/v1"})
    assert not is_gateway_free_agent_challenge_image(
        NEW_IMAGE,
        env={"BASE_GATEWAY_TOKEN": "legacy"},
    )
    decision = decide_agent_challenge_activation(
        image=NEW_IMAGE,
        env={"BASE_LLM_GATEWAY_URL": "http://master/llm/v1"},
    )
    assert decision.allowed is False
    assert decision.code == AGENT_CHALLENGE_INCOMPATIBLE_CODE


def test_decide_activation_unlocks_only_gateway_free_digest() -> None:
    refused = decide_agent_challenge_activation(image=OLD_IMAGE, env={})
    assert refused.allowed is False
    assert refused.digest == OLD_DIGEST
    assert refused.incompatibility is not None
    assert refused.incompatibility.code == AGENT_CHALLENGE_INCOMPATIBLE_CODE

    unpinned = decide_agent_challenge_activation(
        image="ghcr.io/example/agent-challenge:floating",
        env={},
    )
    assert unpinned.allowed is False

    register_gateway_free_digest(NEW_DIGEST)
    allowed = decide_agent_challenge_activation(image=NEW_IMAGE, env={})
    assert allowed.allowed is True
    assert allowed.digest == NEW_DIGEST
    assert allowed.incompatibility is None
    assert should_refuse_agent_challenge(image=NEW_IMAGE, env={}) is None
    assert should_refuse_agent_challenge(image=OLD_IMAGE, env={}) is not None


def test_compose_env_filter_and_gateway_free_checks() -> None:
    dirty = {
        "CHALLENGE_DOCKER_ENABLED": "true",
        "BASE_GATEWAY_TOKEN": "must-drop",
        "BASE_LLM_GATEWAY_URL": "http://must-drop/llm/v1",
        "OPENROUTER_API_KEY": "must-not-be-on-master",
        "CHALLENGE_EVALUATION_CONCURRENCY": "13",
    }
    assert not agent_challenge_compose_env_is_gateway_free(dirty)
    cleaned = filter_forbidden_gateway_env(dirty)
    assert "BASE_GATEWAY_TOKEN" not in cleaned
    assert "BASE_LLM_GATEWAY_URL" not in cleaned
    # OpenRouter key is not a Base gateway bag; filter only gateway names.
    # Master must still refuse to hold it via separate check.
    assert master_env_holds_openrouter_keys(dirty)
    assert master_env_holds_openrouter_keys(
        {name: "x" for name in FORBIDDEN_MASTER_OPENROUTER_ENV_NAMES}
    )
    assert agent_challenge_compose_env_is_gateway_free(
        filter_forbidden_gateway_env(
            {
                "CHALLENGE_DOCKER_ENABLED": "true",
                "CHALLENGE_EVALUATION_CONCURRENCY": "13",
            }
        )
    )
    for name in FORBIDDEN_GATEWAY_ENV_NAMES:
        assert env_declares_gateway_contract({name: "x"})


def test_bypass_legacy_vars_does_not_change_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRISM_LLM_GATEWAY_URL", "http://gateway/llm/v1")
    monkeypatch.setenv("BASE_GATEWAY_TOKEN", "legacy")
    diagnostic = agent_challenge_incompatibility()
    assert diagnostic.code == AGENT_CHALLENGE_INCOMPATIBLE_CODE
    assert "adapter" in diagnostic.message.lower()


@pytest.mark.asyncio
async def test_reconciler_refuses_pre_upgrade_agent_challenge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """VAL-ACAT-042: old digests stay refused with stable diagnostic."""

    orchestrator = _Orchestrator()
    challenges = [
        _challenge("agent-challenge", image=OLD_IMAGE, emission=15),
        _challenge(
            "prism",
            image=f"ghcr.io/example/prism:latest@{OLD_DIGEST}",
            emission=85,
        ),
    ]
    with caplog.at_level(logging.ERROR):
        result = await _reconcile(challenges, orchestrator)
    assert "agent-challenge" not in orchestrator.started
    assert "prism" in orchestrator.started
    assert AGENT_CHALLENGE_INCOMPATIBLE_CODE in caplog.text
    assert isinstance(result, RegistryReconcilePassResult)


@pytest.mark.asyncio
async def test_reconciler_starts_gateway_free_agent_challenge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """VAL-ACAT-041: upgraded gateway-free digests may start without gateway env."""

    register_gateway_free_digest(NEW_DIGEST)
    orchestrator = _Orchestrator()
    challenges = [
        _challenge(
            "agent-challenge",
            image=NEW_IMAGE,
            env={
                "CHALLENGE_DOCKER_ENABLED": "true",
                "CHALLENGE_EVALUATION_CONCURRENCY": "13",
            },
            emission=15,
        ),
        _challenge(
            "prism",
            image=f"ghcr.io/example/prism:latest@{OLD_DIGEST}",
            emission=85,
        ),
    ]
    with caplog.at_level(logging.ERROR):
        result = await _reconcile(challenges, orchestrator)
    assert "agent-challenge" in orchestrator.started
    assert "prism" in orchestrator.started
    assert AGENT_CHALLENGE_INCOMPATIBLE_CODE not in caplog.text
    assert isinstance(result, RegistryReconcilePassResult)


@pytest.mark.asyncio
async def test_reconciler_refuses_allowlisted_digest_with_gateway_env(
    caplog: pytest.LogCaptureFixture,
) -> None:
    register_gateway_free_digest(NEW_DIGEST)
    orchestrator = _Orchestrator()
    challenges = [
        _challenge(
            "agent-challenge",
            image=NEW_IMAGE,
            env={"BASE_GATEWAY_TOKEN": "must-not-start"},
        )
    ]
    with caplog.at_level(logging.ERROR):
        await _reconcile(challenges, orchestrator)
    assert "agent-challenge" not in orchestrator.started
    assert AGENT_CHALLENGE_INCOMPATIBLE_CODE in caplog.text


@pytest.mark.asyncio
async def test_seed_marks_gateway_free_compatible() -> None:
    """VAL-ACAT-041: seed reports compatible for upgraded digests."""

    from base.cli_app.main import seed_prism_challenges
    from base.config.settings import Settings
    from base.master.registry import ChallengeNotFoundError

    register_gateway_free_digest(NEW_DIGEST)

    class Registry:
        def __init__(self) -> None:
            self.created = 0
            self.updated = 0

        async def get(self, slug: str):
            if slug == "prism":
                return SimpleNamespace(
                    metadata={},
                    env={},
                    secrets=[],
                    required_capabilities=[],
                    image=f"ghcr.io/example/prism:1@{OLD_DIGEST}",
                )
            if slug == "agent-challenge":
                return SimpleNamespace(
                    metadata={},
                    env={},
                    secrets=[],
                    required_capabilities=[],
                    image=NEW_IMAGE,
                )
            raise ChallengeNotFoundError(slug)

        async def create(self, payload):  # pragma: no cover
            self.created += 1
            return payload

        async def update(self, slug, payload):  # pragma: no cover
            del slug
            self.updated += 1
            return payload

    result = await seed_prism_challenges(Registry(), settings=Settings())
    assert result["agent-challenge"] == "compatible"
    assert result.get("prism") in {"created", "updated"}


def test_own_runner_env_omits_gateway_and_openrouter() -> None:
    """VAL-ACAT-045/046: AC compose helper has no gateway or OpenRouter keys."""

    from base.cli_app.main import _agent_challenge_own_runner_env
    from base.config.settings import Settings

    env = _agent_challenge_own_runner_env(Settings())
    assert agent_challenge_compose_env_is_gateway_free(env)
    assert not master_env_holds_openrouter_keys(env)
    for name in FORBIDDEN_GATEWAY_ENV_NAMES:
        assert name not in env
    for name in FORBIDDEN_MASTER_OPENROUTER_ENV_NAMES:
        assert name not in env
    assert "BASE_GATEWAY_TOKEN" not in env
    assert "BASE_LLM_GATEWAY_URL" not in env
    assert "OPENROUTER_API_KEY" not in env
