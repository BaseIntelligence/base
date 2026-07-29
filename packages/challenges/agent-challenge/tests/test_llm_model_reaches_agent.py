"""The LLM model id must reach the agent (VAL-LLM-MODEL).

The packaged agent fails closed when ``LLM_MODEL`` is unset::

    agent run failed: A concrete model id is required: set LLM_MODEL
    (the measured review harness supplies it under .rules).

Three independent layers dropped it, so *every* evaluation scored 0 and every
``result.json`` carried ``"model_name": null``:

1. the runner never set ``LLM_MODEL`` -- the platform only holds
   ``CHALLENGE_LLM_MODEL`` (settings ``llm_model``), which the agent never reads;
2. :func:`sanitize_miner_env_for_job` stripped a miner-supplied ``LLM_MODEL``
   because the name is not token/key shaped;
3. :data:`AGENT_ENV_ALLOWLIST` admitted only ``LLM_COST_LIMIT`` and
   ``OPENROUTER_API_KEY``.

A miner may bring their own key *and* their own model; when they supply neither,
the measured platform model is used.
"""

from __future__ import annotations

from agent_challenge.core.config import settings
from agent_challenge.evaluation.own_runner.isolation import (
    AGENT_ENV_ALLOWLIST,
    filter_agent_env,
)
from agent_challenge.evaluation.runner import _terminal_bench_env
from agent_challenge.submissions.miner_env import (
    sanitize_miner_env_for_job,
    validate_miner_env,
)


def test_agent_env_allowlist_admits_llm_model() -> None:
    """Layer 3: the agent sandbox must be allowed to see the model id."""
    assert "LLM_MODEL" in AGENT_ENV_ALLOWLIST


def test_filter_agent_env_keeps_llm_model() -> None:
    """Layer 3: the final filter must not strip the model id."""
    kept = filter_agent_env({"LLM_MODEL": "x-ai/grok-4.5", "OPENROUTER_API_KEY": "sk-or-x"})
    assert kept["LLM_MODEL"] == "x-ai/grok-4.5"
    assert kept["OPENROUTER_API_KEY"] == "sk-or-x"


def test_runner_injects_platform_llm_model() -> None:
    """Layer 1: with no miner env, the measured platform model is supplied."""
    env = _terminal_bench_env()
    assert env["LLM_MODEL"] == settings.llm_model
    assert env["LLM_MODEL"], "platform model id must not be empty"


def test_miner_may_supply_own_llm_model() -> None:
    """Layer 2: a miner bringing their own key may also choose their model."""
    assert sanitize_miner_env_for_job({"LLM_MODEL": "openai/gpt-5"}) == {
        "LLM_MODEL": "openai/gpt-5"
    }
    assert validate_miner_env({"LLM_MODEL": "openai/gpt-5"}) == {"LLM_MODEL": "openai/gpt-5"}


def test_miner_llm_model_overrides_platform_default() -> None:
    """A miner-supplied model wins over the platform default."""
    env = _terminal_bench_env({"LLM_MODEL": "openai/gpt-5"})
    assert env["LLM_MODEL"] == "openai/gpt-5"


def test_llm_model_value_must_not_be_a_url() -> None:
    """Admitting the name must not open an endpoint-injection hole."""
    assert sanitize_miner_env_for_job({"LLM_MODEL": "http://evil.example/v1"}) == {}


def test_llm_model_is_not_treated_as_a_secret() -> None:
    """The model id is not secret-shaped, so it must not be flagged as one."""
    from agent_challenge.evaluation.own_runner.isolation import disallowed_secret_keys

    assert "LLM_MODEL" not in disallowed_secret_keys({"LLM_MODEL": "x-ai/grok-4.5"})
