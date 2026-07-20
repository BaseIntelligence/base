from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_challenge.sdk.config import (
    DEFAULT_LLM_REVIEWER_RETRY_EXCLUDE,
    DEFAULT_LLM_REVIEWER_RETRY_INCLUDE,
    DEFAULT_SECRET_REDACTION,
    MAX_EVALUATION_TASKS_PER_JOB,
    ChallengeSettings,
    effective_evaluation_concurrency,
    effective_evaluation_task_count,
    evaluation_job_lease_seconds,
)

OWNER_HOTKEY = "5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At"


def test_evaluation_job_lease_covers_worst_case_makespan():
    # 20 tasks at concurrency 4 => 5 waves; +1 wave margin => 6 * 3600s = 21600s.
    settings = ChallengeSettings(
        evaluation_task_count=20,
        evaluation_concurrency=4,
        evaluation_timeout_seconds=3600,
    )

    assert evaluation_job_lease_seconds(settings) == 21600


def test_evaluation_job_lease_single_wave_when_concurrency_equals_task_count():
    # 20 tasks at concurrency 20 => 1 wave; +1 wave margin => 2 * 600s = 1200s.
    settings = ChallengeSettings(
        evaluation_task_count=20,
        evaluation_concurrency=20,
        evaluation_timeout_seconds=600,
    )

    assert evaluation_job_lease_seconds(settings) == 1200


def test_normal_validator_defaults():
    settings = ChallengeSettings()

    assert settings.validator_role == "normal"
    assert settings.owner_hotkey == OWNER_HOTKEY
    assert settings.signing_ttl_seconds == 300
    assert settings.zip_max_bytes == 1_048_576
    assert settings.docker_cpus == 4.0
    assert settings.docker_memory == "8g"
    assert settings.docker_memory_swap == "8g"
    assert settings.docker_network == "none"
    assert settings.evaluation_task_count == MAX_EVALUATION_TASKS_PER_JOB
    assert settings.evaluation_concurrency == 4
    assert settings.evaluation_timeout_seconds == 3600
    assert settings.analyzer_timeout_seconds == 3600
    assert settings.analyzer_max_log_bytes == 64_000
    assert settings.analyzer_read_max_bytes == 64_000
    assert settings.analyzer_read_total_budget_bytes == 256_000
    assert settings.analyzer_similarity_high_risk_threshold == 90.0
    assert settings.analyzer_similarity_medium_risk_threshold == 70.0
    assert settings.analyzer_similarity_top_file_pair_limit == 5
    assert settings.analyzer_base_skeleton_manifest is None
    assert settings.submission_rate_limit_window_seconds == 10_800
    assert settings.sse_heartbeat_seconds == 15
    assert settings.langchain_provider is None
    assert settings.langchain_model == "anthropic/claude-opus-4.8"
    assert settings.llm_gateway_base_url is None
    assert settings.llm_gateway_token is None
    assert settings.llm_gateway_token_file is None
    assert settings.llm_reviewer_timeout_seconds == 240
    assert settings.llm_reviewer_max_attempts == 3
    assert settings.llm_reviewer_read_max_bytes == 64_000
    assert settings.llm_reviewer_read_total_budget_bytes == 256_000
    assert settings.llm_reviewer_expected_model == "claude-opus-4-8"
    assert settings.llm_reviewer_prompt_cache_enabled is True
    assert settings.llm_reviewer_max_standby_cycles == 5
    assert settings.llm_reviewer_retry_include == DEFAULT_LLM_REVIEWER_RETRY_INCLUDE
    assert settings.llm_reviewer_retry_exclude == DEFAULT_LLM_REVIEWER_RETRY_EXCLUDE
    # The dead retry knobs are now consumed by the analyzer lifecycle: transient
    # and recoverable tool-miss reasons are retryable by default.
    assert "disallowed_tool" in settings.llm_reviewer_retry_include
    assert "no_submit_after_reads" in settings.llm_reviewer_retry_include
    assert "disallowed_tool" not in settings.llm_reviewer_retry_exclude
    assert settings.benchmark_backend == "swe_forge"
    assert settings.terminal_bench_dataset == "terminal-bench/terminal-bench-2-1"
    assert settings.terminal_bench_label == "terminal-bench@2.1"
    assert settings.terminal_bench_execution_backend == "own_runner"
    assert settings.harbor_forward_env_vars == ()
    assert settings.harbor_n_concurrent == 1
    assert "ghcr.io/baseintelligence/agent-challenge-analyzer:1.0" in settings.docker_allowed_images
    assert (
        "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1"
        in settings.docker_allowed_images
    )
    assert "python:3.12-slim" not in settings.docker_allowed_images


def test_master_validator_override():
    settings = ChallengeSettings(validator_role="master")

    assert settings.validator_role == "master"
    assert settings.owner_hotkey == OWNER_HOTKEY
    assert settings.signing_ttl_seconds == 300
    assert settings.zip_max_bytes == 1_048_576


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CHALLENGE_VALIDATOR_ROLE", "master")
    monkeypatch.setenv("CHALLENGE_OWNER_HOTKEY", "owner-test-hotkey")
    monkeypatch.setenv("CHALLENGE_SIGNING_TTL_SECONDS", "120")
    monkeypatch.setenv("CHALLENGE_ZIP_MAX_BYTES", "2048")
    monkeypatch.setenv("CHALLENGE_DOCKER_CPUS", "1.5")
    monkeypatch.setenv("CHALLENGE_DOCKER_MEMORY", "2g")
    monkeypatch.setenv("CHALLENGE_EVALUATION_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS", "600")
    monkeypatch.setenv("CHALLENGE_SSE_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("CHALLENGE_LANGCHAIN_PROVIDER", "anthropic")
    monkeypatch.setenv("CHALLENGE_LANGCHAIN_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("CHALLENGE_LLM_GATEWAY_BASE_URL", "http://master:19080")
    monkeypatch.setenv("CHALLENGE_LLM_REVIEWER_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("CHALLENGE_TERMINAL_BENCH_DATASET", "custom/terminal-bench")
    monkeypatch.setenv("CHALLENGE_TERMINAL_BENCH_LABEL", "custom-terminal-bench")
    monkeypatch.setenv("CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND", "own_runner")

    settings = ChallengeSettings()

    assert settings.validator_role == "master"
    assert settings.owner_hotkey == "owner-test-hotkey"
    assert settings.signing_ttl_seconds == 120
    assert settings.zip_max_bytes == 2048
    assert settings.docker_cpus == 1.5
    assert settings.docker_memory == "2g"
    assert settings.evaluation_timeout_seconds == 60
    assert settings.submission_rate_limit_window_seconds == 600
    assert settings.sse_heartbeat_seconds == 30
    assert settings.langchain_provider == "anthropic"
    assert settings.langchain_model == "claude-3-5-sonnet-latest"
    assert settings.llm_gateway_base_url == "http://master:19080"
    assert settings.llm_reviewer_timeout_seconds == 45
    assert settings.terminal_bench_dataset == "custom/terminal-bench"
    assert settings.terminal_bench_label == "custom-terminal-bench"
    assert settings.terminal_bench_execution_backend == "own_runner"


@pytest.mark.parametrize("window_seconds", [1, 60, 300, 600, 10_800])
def test_env_submission_rate_limit_window_seconds_values(monkeypatch, window_seconds: int):
    """VAL-E2E-008: Settings loads CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS."""
    monkeypatch.setenv(
        "CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS",
        str(window_seconds),
    )
    settings = ChallengeSettings()
    assert settings.submission_rate_limit_window_seconds == window_seconds


def test_env_submission_rate_limit_window_zero_loads_as_zero_not_disable(monkeypatch):
    """VAL-E2E-010: Settings accepts 0; create/enforce still floors to 1s."""
    monkeypatch.setenv("CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS", "0")
    settings = ChallengeSettings()
    assert settings.submission_rate_limit_window_seconds == 0


def test_evaluation_limits_reject_values_above_twenty():
    with pytest.raises(ValidationError) as task_count_error:
        ChallengeSettings(evaluation_task_count=MAX_EVALUATION_TASKS_PER_JOB + 1)

    assert "evaluation_task_count" in str(task_count_error.value)
    assert f"at most {MAX_EVALUATION_TASKS_PER_JOB}" in str(task_count_error.value)

    with pytest.raises(ValidationError) as concurrency_error:
        ChallengeSettings(evaluation_concurrency=MAX_EVALUATION_TASKS_PER_JOB + 1)

    assert "evaluation_concurrency" in str(concurrency_error.value)
    assert f"at most {MAX_EVALUATION_TASKS_PER_JOB}" in str(concurrency_error.value)


def test_evaluation_concurrency_default_decoupled_from_task_count():
    settings = ChallengeSettings()

    assert settings.evaluation_task_count == MAX_EVALUATION_TASKS_PER_JOB
    assert settings.evaluation_concurrency == 4
    assert effective_evaluation_concurrency(settings.evaluation_concurrency) == 4


def test_evaluation_limit_helpers_cap_monkeypatched_settings():
    assert effective_evaluation_task_count(MAX_EVALUATION_TASKS_PER_JOB + 16) == 30
    assert effective_evaluation_task_count(-1) == 0
    assert effective_evaluation_concurrency(MAX_EVALUATION_TASKS_PER_JOB + 16) == 30
    assert effective_evaluation_concurrency(0) == 1


def test_gateway_token_file_loads_and_safe_dump_redacts_secret(tmp_path):
    token_file = tmp_path / "gateway-token"
    token_file.write_text("file-backed-gateway-token\n", encoding="utf-8")

    settings = ChallengeSettings(
        llm_gateway_token_file=str(token_file),
        shared_token="dummy-shared-token",
        docker_broker_token="dummy-broker-token",
        database_url="sqlite+aiosqlite:////tmp/config-test.sqlite3",
    )
    safe = settings.safe_model_dump()

    assert settings.llm_gateway_token == "file-backed-gateway-token"
    assert safe["llm_gateway_token"] == DEFAULT_SECRET_REDACTION
    assert safe["shared_token"] == DEFAULT_SECRET_REDACTION
    assert safe["docker_broker_token"] == DEFAULT_SECRET_REDACTION
    assert safe["database_url"] == DEFAULT_SECRET_REDACTION
    assert "file-backed-gateway-token" not in str(safe)
    assert "dummy-shared-token" not in str(safe)
    assert "dummy-broker-token" not in str(safe)
    assert "sqlite+aiosqlite:////tmp/config-test.sqlite3" not in str(safe)
    assert "file-backed-gateway-token" not in repr(settings)


def test_database_url_file_overrides_sqlite_default(tmp_path):
    url_file = tmp_path / "database_url"
    url_file.write_text(
        "postgresql+asyncpg://challenge:secret@db-host:5432/challenge\n",
        encoding="utf-8",
    )

    settings = ChallengeSettings(database_url_file=str(url_file))

    assert settings.database_url == "postgresql+asyncpg://challenge:secret@db-host:5432/challenge"
    safe = settings.safe_model_dump()
    assert safe["database_url"] == DEFAULT_SECRET_REDACTION
    assert "secret@db-host" not in str(safe)
    assert "secret@db-host" not in repr(settings)


def test_database_url_defaults_to_sqlite_when_no_file(monkeypatch):
    monkeypatch.delenv("CHALLENGE_DATABASE_URL", raising=False)
    monkeypatch.delenv("CHALLENGE_DATABASE_URL_FILE", raising=False)

    settings = ChallengeSettings()

    assert settings.database_url == "sqlite+aiosqlite:////data/agent-challenge.sqlite3"
    assert settings.database_url_file is None


def test_database_url_file_via_env_overrides_default(monkeypatch, tmp_path):
    url_file = tmp_path / "database_url"
    url_file.write_text(
        "postgresql+asyncpg://challenge:envpw@pg:5432/challenge\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHALLENGE_DATABASE_URL_FILE", str(url_file))

    settings = ChallengeSettings()

    assert settings.database_url == "postgresql+asyncpg://challenge:envpw@pg:5432/challenge"


def test_gateway_token_env_takes_precedence_over_token_file(monkeypatch, tmp_path):
    token_file = tmp_path / "gateway-token"
    token_file.write_text("file-backed-gateway-token\n", encoding="utf-8")
    monkeypatch.setenv("CHALLENGE_LLM_GATEWAY_TOKEN", "env-gateway-token")
    monkeypatch.setenv("CHALLENGE_LLM_GATEWAY_TOKEN_FILE", str(token_file))

    settings = ChallengeSettings()

    assert settings.llm_gateway_token == "env-gateway-token"
    assert settings.safe_model_dump()["llm_gateway_token"] == DEFAULT_SECRET_REDACTION


def test_safe_model_dump_keeps_unset_secrets_empty(monkeypatch):
    monkeypatch.delenv("CHALLENGE_SHARED_TOKEN", raising=False)

    safe = ChallengeSettings().safe_model_dump()

    assert safe["llm_gateway_token"] is None
    assert safe["shared_token"] is None
    assert safe["docker_broker_token"] is None
    assert safe["database_url"] == DEFAULT_SECRET_REDACTION


def test_safe_model_dump_keeps_terminal_bench_non_secret_fields_visible():
    settings = ChallengeSettings(
        terminal_bench_execution_backend="own_runner",
    )

    safe = settings.safe_model_dump()

    assert safe["terminal_bench_execution_backend"] == "own_runner"
    assert (
        safe["harbor_runner_image"] == "ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1"
    )


def test_retry_policy_can_be_overridden_with_init_values():
    settings = ChallengeSettings(
        llm_reviewer_retry_include=("provider_timeout",),
        llm_reviewer_retry_exclude=("unsafe_path", "disallowed_tool"),
    )

    assert settings.llm_reviewer_retry_include == ("provider_timeout",)
    assert settings.llm_reviewer_retry_exclude == ("unsafe_path", "disallowed_tool")


def test_invalid_validator_role_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChallengeSettings(validator_role="worker")

    assert "validator_role" in str(exc_info.value)


def test_terminal_bench_2_0_dataset_rejected():
    for dataset in (
        "terminal-bench@2.0",
        "terminal-bench/terminal-bench-2-0",
        "terminal-bench/terminal-bench-2.0",
    ):
        with pytest.raises(ValidationError) as exc_info:
            ChallengeSettings(terminal_bench_dataset=dataset)

        message = str(exc_info.value)
        assert "terminal_bench_dataset" in message
        assert "Terminal-Bench 2.1" in message

    with pytest.raises(ValidationError) as exc_info:
        ChallengeSettings(terminal_bench_label="terminal-bench@2.0")

    message = str(exc_info.value)
    assert "terminal_bench_label" in message
    assert "Terminal-Bench 2.1" in message


def test_own_runner_backend_loads_without_daytona_or_harbor_env(monkeypatch):
    for name in (
        "CHALLENGE_DAYTONA_API_KEY",
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "CHALLENGE_HARBOR_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND", "own_runner")
    monkeypatch.setenv("CHALLENGE_DOCKER_BACKEND", "broker")
    monkeypatch.setenv("CHALLENGE_DOCKER_BROKER_URL", "https://docker-broker.example.invalid")
    monkeypatch.setenv(
        "CHALLENGE_DOCKER_BROKER_TOKEN_FILE",
        "/run/secrets/base/docker_broker_token",
    )

    settings = ChallengeSettings()

    assert settings.terminal_bench_execution_backend == "own_runner"


def test_invalid_terminal_bench_execution_backend_rejected():
    with pytest.raises(ValidationError) as exc_info:
        ChallengeSettings(terminal_bench_execution_backend="daytona")

    message = str(exc_info.value)
    assert "terminal_bench_execution_backend" in message
    assert "own_runner" in message


def test_own_runner_execution_backend_selectable():
    settings = ChallengeSettings(terminal_bench_execution_backend="own_runner")

    assert settings.terminal_bench_execution_backend == "own_runner"


def test_own_runner_remains_default_execution_backend():
    settings = ChallengeSettings()

    assert settings.terminal_bench_execution_backend == "own_runner"


# ---------------------------------------------------------------------------
# Replay-audit tier-rate ordering (cross-field, VAL-SCORE-025): the high-trust
# attested rate must be STRICTLY below the low-trust unverified rate for non-zero
# rates (higher trust => strictly lower audit rate). A rate of 0 (disabled tier)
# is allowed on either side.
# ---------------------------------------------------------------------------


def test_default_replay_audit_rates_satisfy_tier_ordering():
    settings = ChallengeSettings()

    assert 0.0 < settings.replay_audit_attested_rate < settings.replay_audit_unverified_rate


def test_replay_audit_accepts_strictly_increasing_nonzero_rates():
    settings = ChallengeSettings(
        replay_audit_attested_rate=0.01,
        replay_audit_unverified_rate=0.5,
    )

    assert settings.replay_audit_attested_rate == 0.01
    assert settings.replay_audit_unverified_rate == 0.5


def test_replay_audit_rejects_attested_rate_above_unverified_both_nonzero():
    with pytest.raises(ValidationError) as exc_info:
        ChallengeSettings(
            replay_audit_attested_rate=0.10,
            replay_audit_unverified_rate=0.05,
        )

    message = str(exc_info.value)
    assert "replay_audit_attested_rate" in message
    assert "replay_audit_unverified_rate" in message


def test_replay_audit_rejects_equal_nonzero_rates():
    # Strict ordering: equal non-zero rates violate "higher trust => strictly
    # lower rate".
    with pytest.raises(ValidationError):
        ChallengeSettings(
            replay_audit_attested_rate=0.05,
            replay_audit_unverified_rate=0.05,
        )


def test_replay_audit_allows_zero_attested_rate_disabled_tier():
    settings = ChallengeSettings(
        replay_audit_attested_rate=0.0,
        replay_audit_unverified_rate=0.10,
    )

    assert settings.replay_audit_attested_rate == 0.0
    assert settings.replay_audit_unverified_rate == 0.10


def test_replay_audit_allows_zero_unverified_rate_disabled_tier():
    # The unverified tier disabled (0) is allowed even though the attested rate is
    # non-zero and would otherwise have to sit strictly below it.
    settings = ChallengeSettings(
        replay_audit_attested_rate=0.10,
        replay_audit_unverified_rate=0.0,
    )

    assert settings.replay_audit_attested_rate == 0.10
    assert settings.replay_audit_unverified_rate == 0.0


def test_replay_audit_allows_both_rates_zero():
    settings = ChallengeSettings(
        replay_audit_attested_rate=0.0,
        replay_audit_unverified_rate=0.0,
    )

    assert settings.replay_audit_attested_rate == 0.0
    assert settings.replay_audit_unverified_rate == 0.0
