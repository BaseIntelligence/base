"""Registry seed renders the LLM gateway-ROUTING env (VAL-CODE-REG-005).

The registry SEED sets the challenge combined-mode env but historically NOT the
LLM gateway-ROUTING env, so a challenge the master reconciler creates PURELY from
the registry (no static install-swarm env then adopted) would not route its LLM
calls through the master gateway. These guards lock that the seed renders the
gateway-routing URL into the record env, sourced from the master
``gateway.public_base_url`` and BYTE-MATCHING install-swarm.sh:

- agent-challenge -> ``CHALLENGE_LLM_GATEWAY_BASE_URL=<gateway.public_base_url>``
  (install-swarm.sh:1536)
- prism -> ``PRISM_LLM_GATEWAY_URL=<gateway.public_base_url>/llm/openrouter``
  (install-swarm.sh:1653)

and that a spec built by ``challenge_spec_from_registry`` from the seeded record
carries the same routing env onto the reconciler-built service. No raw provider
key is added; the gateway token FILE env stays ``/run/secrets/base_gateway_token``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import base.cli_app.main as cli_module
from base.master.docker_orchestrator import challenge_spec_from_registry
from base.master.registry import ChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus

GATEWAY_PUBLIC_BASE_URL = "http://88.216.198.199:19080"
GATEWAY_OPENROUTER_ROUTE = f"{GATEWAY_PUBLIC_BASE_URL}/llm/openrouter"


def _settings() -> SimpleNamespace:
    """Master-shaped settings carrying a gateway public base URL."""

    return SimpleNamespace(
        docker=SimpleNamespace(broker_url="http://base-docker-broker:8082"),
        gateway=SimpleNamespace(public_base_url=GATEWAY_PUBLIC_BASE_URL),
        master=SimpleNamespace(registry_url="https://chain.joinbase.ai"),
    )


def _seed(registry: ChallengeRegistry) -> None:
    # agent-challenge must already exist for the seed to render its env (the seed
    # updates an existing agent-challenge record; prism is created fresh).
    registry.create(
        ChallengeCreate(
            slug="agent-challenge",
            name="Agent Challenge",
            image="ghcr.io/baseintelligence/agent-challenge:latest",
            version="0.1.0",
            status=ChallengeStatus.ACTIVE,
        )
    )
    asyncio.run(cli_module.seed_prism_challenges(registry, _settings()))


def test_prism_challenge_create_env_carries_gateway_routing_url() -> None:
    payload = cli_module.prism_challenge_create(_settings())
    assert payload.env["PRISM_LLM_GATEWAY_URL"] == GATEWAY_OPENROUTER_ROUTE
    # No raw provider key smuggled into the record env.
    assert "PRISM_OPENROUTER_API_KEY" not in payload.env
    assert not any(key.endswith("_API_KEY") for key in payload.env)


def test_agent_challenge_own_runner_env_carries_gateway_routing_url() -> None:
    env = cli_module._agent_challenge_own_runner_env(_settings())
    assert env["CHALLENGE_LLM_GATEWAY_BASE_URL"] == GATEWAY_PUBLIC_BASE_URL
    assert not any(key.endswith("_API_KEY") for key in env)


def test_seeded_prism_record_env_and_reconciler_spec_carry_routing() -> None:
    registry = ChallengeRegistry()
    _seed(registry)

    prism = registry.get("prism")
    assert prism.env["PRISM_LLM_GATEWAY_URL"] == GATEWAY_OPENROUTER_ROUTE

    spec = challenge_spec_from_registry(prism)
    assert spec.env["PRISM_LLM_GATEWAY_URL"] == GATEWAY_OPENROUTER_ROUTE


def test_seeded_agent_challenge_record_env_and_reconciler_spec_carry_routing() -> None:
    registry = ChallengeRegistry()
    _seed(registry)

    agent = registry.get("agent-challenge")
    assert agent.env["CHALLENGE_LLM_GATEWAY_BASE_URL"] == GATEWAY_PUBLIC_BASE_URL

    spec = challenge_spec_from_registry(agent)
    assert spec.env["CHALLENGE_LLM_GATEWAY_BASE_URL"] == GATEWAY_PUBLIC_BASE_URL


def test_routing_url_derived_from_gateway_public_base_url() -> None:
    """The routing URL tracks the configured gateway base (not a hardcoded host)."""

    other = SimpleNamespace(
        docker=SimpleNamespace(broker_url="http://base-docker-broker:8082"),
        gateway=SimpleNamespace(public_base_url="http://10.0.0.5:28080"),
        master=SimpleNamespace(registry_url="https://chain.joinbase.ai"),
    )
    prism = cli_module.prism_challenge_create(other)
    assert prism.env["PRISM_LLM_GATEWAY_URL"] == "http://10.0.0.5:28080/llm/openrouter"
    agent_env = cli_module._agent_challenge_own_runner_env(other)
    assert agent_env["CHALLENGE_LLM_GATEWAY_BASE_URL"] == "http://10.0.0.5:28080"
