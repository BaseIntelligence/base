"""VAL-ACLOCK miner/job env lock: keys/tokens only, no URL/proxy injection."""

from __future__ import annotations

import inspect
import json
import re

import pytest
from fastapi import HTTPException

from agent_challenge.api import routes as api_routes
from agent_challenge.evaluation.own_runner.isolation import (
    AGENT_ENV_ALLOWLIST,
    filter_agent_env,
)
from agent_challenge.evaluation.runner import (
    TERMINAL_BENCH_CONTROL_ENV_KEYS,
    _terminal_bench_env,
)
from agent_challenge.submissions.miner_env import (
    MinerEnvValidationError,
    is_allowed_miner_env_key,
    is_forbidden_miner_env_key,
    looks_like_url_value,
    sanitize_miner_env_for_job,
    validate_miner_env,
)


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-001 — reject URL/proxy/host/gateway keys
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key",
    [
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "FOO_URL",
        "EVIL_URI",
        "EVIL_ENDPOINT",
        "CUSTOM_HOST",
        "SOME_PROXY",
        "DOCKER_HOST",
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "GATEWAY_TOKEN",
        "MY_GATEWAY_SECRET",
        "BASE_LOG_STREAM_URL",
        "BASE_LOG_STREAM_TOKEN",
        "base_log_stream_slug",
        "https_proxy",
        "Review_Api_Base_Url",
    ],
)
def test_validated_miner_env_rejects_url_proxy_host_gateway_keys(key: str) -> None:
    with pytest.raises(HTTPException) as exc_info:
        api_routes._validated_miner_env({key: "x"})
    assert exc_info.value.status_code == 422
    detail = str(exc_info.value.detail).lower()
    assert "x" not in detail  # no value echo
    assert is_forbidden_miner_env_key(key)


def test_validate_miner_env_rejects_matrix_without_echo() -> None:
    evil = {
        "HTTPS_PROXY": "http://evil.example:8080",
        "OPENROUTER_API_KEY": "sk-or-v1-honest",
    }
    with pytest.raises(MinerEnvValidationError) as exc_info:
        validate_miner_env(evil)
    assert "http://evil.example" not in str(exc_info.value)
    assert "sk-or-v1-honest" not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-002 — honest API keys / product tokens accepted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,value",
    [
        ("OPENROUTER_API_KEY", "sk-or-v1-honest-key"),
        ("LLM_COST_LIMIT", "5.0"),
        ("EVAL_RUN_TOKEN", "eval-run-capability-token"),
        ("REVIEW_SESSION_TOKEN", "review-session-capability-token"),
        ("API_TOKEN", "miner-api-token"),
        ("MY_PROVIDER_API_KEY", "provider-key"),
    ],
)
def test_validated_miner_env_accepts_honest_keys_and_tokens(key: str, value: str) -> None:
    out = api_routes._validated_miner_env({key: value})
    assert out == {key: value}
    assert is_allowed_miner_env_key(key)


def test_validate_miner_env_accepts_openrouter_alone() -> None:
    assert validate_miner_env({"OPENROUTER_API_KEY": "sk-or-ok"}) == {
        "OPENROUTER_API_KEY": "sk-or-ok"
    }


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-003 — URL-shaped values rejected on non-token keys (and tokens)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,value",
    [
        ("API_TOKEN", "https://evil.example/callback"),
        ("OPENROUTER_API_KEY", "http://evil.example/key"),
        ("LLM_COST_LIMIT", "https://openrouter.ai/api/v1"),
        ("MY_SECRET", "ftp://files.example/x"),
    ],
)
def test_validated_miner_env_rejects_url_shaped_values(key: str, value: str) -> None:
    assert looks_like_url_value(value)
    with pytest.raises(HTTPException) as exc_info:
        api_routes._validated_miner_env({key: value})
    assert exc_info.value.status_code == 422
    serialized = json.dumps(exc_info.value.detail)
    assert "evil.example" not in serialized
    assert "openrouter.ai" not in serialized


def test_non_token_plain_keys_rejected_even_without_url_value() -> None:
    """Keys/tokens only: plain config names are not admitted."""
    with pytest.raises(HTTPException) as exc_info:
        api_routes._validated_miner_env({"PATH": "/usr/bin", "SECOND_VALUE": "x"})
    assert exc_info.value.status_code == 422


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-004 — BASE_LOG_STREAM_* are control keys
# --------------------------------------------------------------------------- #
def test_base_log_stream_keys_are_terminal_bench_control_keys() -> None:
    required = {
        "BASE_LOG_STREAM_URL",
        "BASE_LOG_STREAM_ATTEMPT_ID",
        "BASE_LOG_STREAM_TOKEN",
        "BASE_LOG_STREAM_SLUG",
        "BASE_LOG_STREAM_TIMEOUT_SECONDS",
    }
    assert required <= set(TERMINAL_BENCH_CONTROL_ENV_KEYS)


def test_terminal_bench_env_ignores_miner_base_log_stream_override() -> None:
    env = _terminal_bench_env(
        {
            "OPENROUTER_API_KEY": "sk-or-ok",
            "BASE_LOG_STREAM_URL": "http://evil.example/steal",
            "BASE_LOG_STREAM_TOKEN": "miner-forged-token",
            "BASE_LOG_STREAM_SLUG": "evil-slug",
        }
    )
    assert env["OPENROUTER_API_KEY"] == "sk-or-ok"
    assert "BASE_LOG_STREAM_URL" not in env
    assert "BASE_LOG_STREAM_TOKEN" not in env
    assert "BASE_LOG_STREAM_SLUG" not in env
    assert "evil.example" not in json.dumps(env)


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-005 / 006 — agent sandbox allowlist + chokepoint
# --------------------------------------------------------------------------- #
def test_agent_env_allowlist_exactly_or_key_and_cost_limit() -> None:
    assert AGENT_ENV_ALLOWLIST == frozenset({"OPENROUTER_API_KEY", "LLM_COST_LIMIT"})


def test_filter_agent_env_strips_url_proxy_and_extra_secrets() -> None:
    raw = {
        "OPENROUTER_API_KEY": "sk-or-ok",
        "LLM_COST_LIMIT": "3",
        "HTTPS_PROXY": "http://evil",
        "FOO_URL": "https://evil",
        "DOCKER_HOST": "tcp://evil:2375",
        "BASE_LLM_GATEWAY_URL": "https://master/llm/v1",
        "BASE_GATEWAY_TOKEN": "scoped",
        "BASE_LOG_STREAM_URL": "http://challenge:8000",
        "AWS_SECRET_ACCESS_KEY": "leak",
        "PATH": "/usr/bin",
    }
    assert filter_agent_env(raw) == {
        "OPENROUTER_API_KEY": "sk-or-ok",
        "LLM_COST_LIMIT": "3",
    }


def test_agent_env_injection_paths_call_filter_agent_env() -> None:
    """Chokepoint: agent probe / sandbox inject paths filter through allowlist."""
    from agent_challenge.evaluation.own_runner import isolation

    probe_src = inspect.getsource(isolation.probe_env)
    assert "filter_agent_env" in probe_src


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-007 — job terminal_bench_env does not forward rejected keys
# --------------------------------------------------------------------------- #
def test_terminal_bench_env_does_not_forward_url_proxy_miner_keys() -> None:
    env = _terminal_bench_env(
        {
            "OPENROUTER_API_KEY": "sk-or-ok",
            "HTTPS_PROXY": "http://proxy.evil:8080",
            "HTTP_PROXY": "http://proxy.evil:8080",
            "ALL_PROXY": "socks5://proxy.evil:1080",
            "FOO_URL": "https://evil.example",
            "EVIL_ENDPOINT": "https://evil.example/api",
            "CUSTOM_HOST": "evil.example",
            "DOCKER_HOST": "tcp://evil:2375",
            "BASE_LLM_GATEWAY_URL": "https://master/llm/v1",
            "GATEWAY_TOKEN": "nope",
            "MINER_VISIBLE": "should-not-pass",  # non-token plain name
            "API_TOKEN": "ok-token",
            "DEEPSEEK_API_KEY": "sk-raw-other-provider",
        }
    )
    assert env["OPENROUTER_API_KEY"] == "sk-or-ok"
    assert env["API_TOKEN"] == "ok-token"
    for rejected in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "FOO_URL",
        "EVIL_ENDPOINT",
        "CUSTOM_HOST",
        "DOCKER_HOST",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "MINER_VISIBLE",
        "DEEPSEEK_API_KEY",
    ):
        assert rejected not in env
    dumped = json.dumps(env)
    assert "proxy.evil" not in dumped
    assert "evil.example" not in dumped
    assert "sk-raw-other-provider" not in dumped


def test_sanitize_miner_env_for_job_strips_rejected() -> None:
    cleaned = sanitize_miner_env_for_job(
        {
            "OPENROUTER_API_KEY": "sk",
            "HTTPS_PROXY": "http://x",
            "NOTES": "not-a-token",
            "API_TOKEN": "https://should-strip-url-value",
        }
    )
    assert cleaned == {"OPENROUTER_API_KEY": "sk"}


def test_terminal_bench_env_still_blocks_control_path_overrides() -> None:
    env = _terminal_bench_env(
        {
            "HOME": "/.cache",
            "BASE_AGENT_PATH": "/bad-agent",
            "BASE_BENCHMARK_DATASET": "bad-dataset",
            "OPENROUTER_API_KEY": "sk-or-ok",
        }
    )
    assert env["HOME"] == "/tmp"
    assert env["BASE_AGENT_PATH"] == "/workspace/agent"
    assert env["OPENROUTER_API_KEY"] == "sk-or-ok"


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-017 / 020 — prior pin helpers remain importable + regression hooks
# --------------------------------------------------------------------------- #
def test_prior_review_joinbase_pin_constant_still_present() -> None:
    from pathlib import Path

    runtime = (
        Path(__file__).resolve().parents[1]
        / "docker"
        / "review"
        / "review_runtime.py"
    )
    text = runtime.read_text(encoding="utf-8")
    assert re.search(r"chain\.joinbase\.ai/challenges/agent-challenge", text)


def test_agent_allowlist_does_not_include_gateway_or_proxy() -> None:
    forbidden = {
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "HTTPS_PROXY",
        "DOCKER_HOST",
        "BASE_LOG_STREAM_URL",
    }
    assert AGENT_ENV_ALLOWLIST.isdisjoint(forbidden)
