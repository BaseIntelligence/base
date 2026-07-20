from __future__ import annotations

import pytest

from agent_challenge.evaluation.runner import validate_terminal_bench_broker_readiness
from agent_challenge.sdk.config import ChallengeSettings


def test_harbor_runner_image_default_is_preserved() -> None:
    challenge_settings = ChallengeSettings()

    assert (
        challenge_settings.harbor_runner_image
        == "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1"
    )
    assert challenge_settings.harbor_forward_env_vars == ()


def test_default_execution_backend_is_own_runner() -> None:
    assert ChallengeSettings().terminal_bench_execution_backend == "own_runner"


def _configure_terminal_bench_broker(
    monkeypatch,
    *,
    execution_backend: str = "own_runner",
    docker_enabled: bool = True,
) -> None:
    settings_paths = (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    )
    for settings_path in settings_paths:
        monkeypatch.setattr(f"{settings_path}.benchmark_backend", "terminal_bench")
        monkeypatch.setattr(f"{settings_path}.terminal_bench_execution_backend", execution_backend)
        monkeypatch.setattr(f"{settings_path}.terminal_bench_task_ids", ("hello-world",))
        monkeypatch.setattr(f"{settings_path}.evaluation_task_count", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_concurrency", 1)
        monkeypatch.setattr(f"{settings_path}.evaluation_timeout_seconds", 120)
        monkeypatch.setattr(f"{settings_path}.docker_enabled", docker_enabled)
        monkeypatch.setattr(f"{settings_path}.docker_backend", "broker")
        monkeypatch.setattr(f"{settings_path}.docker_broker_url", "https://platform-broker.test")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token", "broker-token")
        monkeypatch.setattr(f"{settings_path}.docker_broker_token_file", None)
        monkeypatch.setattr(f"{settings_path}.validator_role", "master")
        monkeypatch.setattr(f"{settings_path}.docker_network", "default")
        monkeypatch.setattr(
            f"{settings_path}.harbor_runner_image",
            "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1",
        )


def test_own_runner_broker_readiness_passes_with_broker_config(monkeypatch) -> None:
    _configure_terminal_bench_broker(monkeypatch, execution_backend="own_runner")

    validate_terminal_bench_broker_readiness()


def test_broker_readiness_requires_broker_url(monkeypatch) -> None:
    _configure_terminal_bench_broker(monkeypatch, execution_backend="own_runner")
    for settings_path in (
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.docker_broker_url", "")

    with pytest.raises(RuntimeError, match="CHALLENGE_DOCKER_BROKER_URL"):
        validate_terminal_bench_broker_readiness()


def test_broker_readiness_requires_docker_enabled(monkeypatch) -> None:
    _configure_terminal_bench_broker(
        monkeypatch, execution_backend="own_runner", docker_enabled=False
    )

    with pytest.raises(RuntimeError, match="CHALLENGE_DOCKER_ENABLED=true"):
        validate_terminal_bench_broker_readiness()


def test_broker_readiness_requires_token_or_token_file(monkeypatch) -> None:
    _configure_terminal_bench_broker(monkeypatch, execution_backend="own_runner")
    for settings_path in (
        "agent_challenge.evaluation.benchmarks.settings",
        "agent_challenge.evaluation.runner.settings",
        "agent_challenge.evaluation.terminal_bench.settings",
    ):
        monkeypatch.setattr(f"{settings_path}.docker_broker_token", None)
        monkeypatch.setattr(f"{settings_path}.docker_broker_token_file", None)

    with pytest.raises(RuntimeError, match="CHALLENGE_DOCKER_BROKER_TOKEN"):
        validate_terminal_bench_broker_readiness()


def _mem_to_bytes(value: str) -> int:
    text = value.strip().lower()
    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "": 1}
    suffix = text[-1] if text and text[-1] in "kmg" else ""
    number = float(text[:-1] if suffix else text)
    return int(number * units[suffix])


def test_terminal_bench_broker_limits_memory_swap_not_below_memory():
    from agent_challenge.evaluation.runner import _terminal_bench_broker_limits

    limits = _terminal_bench_broker_limits()

    assert limits.memory_swap is not None
    assert _mem_to_bytes(limits.memory_swap) >= _mem_to_bytes(limits.memory)
