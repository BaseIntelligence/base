from __future__ import annotations

from pathlib import Path

import yaml

from prism_challenge.config import PrismSettings
from prism_challenge.proof import PROVIDER_ENV_KEYS


def test_base_challenge_env_aliases_are_loaded(monkeypatch):
    # PRISM_* aliases take precedence over CHALLENGE_*; clear ambient PRISM docker env so the
    # CHALLENGE_* wiring under test actually wins (CI also starts with an empty prism docker env).
    for name in (
        "PRISM_DOCKER_BACKEND",
        "PRISM_DOCKER_ENABLED",
        "PRISM_DOCKER_BROKER_URL",
        "PRISM_DOCKER_BROKER_TOKEN",
        "PRISM_DOCKER_BROKER_TOKEN_FILE",
        "PRISM_DATABASE_URL",
        "PRISM_SHARED_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHALLENGE_DATABASE_URL", "sqlite+aiosqlite:////data/challenge.sqlite3")
    monkeypatch.setenv("CHALLENGE_SHARED_TOKEN_FILE", "/run/secrets/base/challenge_token")
    monkeypatch.setenv("CHALLENGE_DOCKER_ENABLED", "true")
    monkeypatch.setenv("CHALLENGE_DOCKER_BACKEND", "broker")
    monkeypatch.setenv("CHALLENGE_DOCKER_BROKER_URL", "http://base-docker-broker:8082")
    monkeypatch.setenv("CHALLENGE_DOCKER_BROKER_TOKEN_FILE", "/run/secrets/base/challenge_token")

    settings = PrismSettings()

    assert settings.database_url == "sqlite+aiosqlite:////data/challenge.sqlite3"
    assert settings.shared_token_file == "/run/secrets/base/challenge_token"
    assert settings.docker_enabled is True
    assert settings.docker_backend == "broker"
    assert settings.docker_broker_url == "http://base-docker-broker:8082"
    assert str(settings.docker_broker_token_file) == "/run/secrets/base/challenge_token"


def test_docker_backend_default_is_broker_safe_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("PRISM_DOCKER_BACKEND", raising=False)
    monkeypatch.delenv("CHALLENGE_DOCKER_BACKEND", raising=False)
    monkeypatch.delenv("PRISM_DOCKER_BROKER_TOKEN", raising=False)
    monkeypatch.delenv("PRISM_DOCKER_BROKER_TOKEN_FILE", raising=False)
    monkeypatch.delenv("CHALLENGE_DOCKER_BROKER_TOKEN", raising=False)
    monkeypatch.delenv("CHALLENGE_DOCKER_BROKER_TOKEN_FILE", raising=False)

    # Bare construction must succeed for pytest collection / packaging imports that do not
    # inject live broker secrets: default token *path* satisfies executor validation without
    # requiring the file (or a live token) to exist on the host.
    settings = PrismSettings()

    assert settings.docker_backend == "broker"
    assert settings.docker_backend != "cli"
    assert settings.docker_broker_token_file == "/run/secrets/base/challenge_token"


def test_docker_backend_explicit_env_overrides_default(monkeypatch) -> None:
    for env_name in ("CHALLENGE_DOCKER_BACKEND", "PRISM_DOCKER_BACKEND"):
        monkeypatch.delenv("PRISM_DOCKER_BACKEND", raising=False)
        monkeypatch.delenv("CHALLENGE_DOCKER_BACKEND", raising=False)
        # Supported executor backends only (ChallengeSettings rejects unknown values).
        for explicit in ("cli", "broker"):
            monkeypatch.setenv(env_name, explicit)
            assert PrismSettings().docker_backend == explicit
            monkeypatch.delenv(env_name, raising=False)

    assert PrismSettings(docker_backend="cli").docker_backend == "cli"


def test_settings_still_accept_field_names() -> None:
    settings = PrismSettings(
        database_url="sqlite+aiosqlite:////tmp/prism.sqlite3",
        shared_token="secret",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://broker",
    )

    assert settings.database_url == "sqlite+aiosqlite:////tmp/prism.sqlite3"
    assert settings.shared_token == "secret"
    assert settings.docker_enabled is True
    assert settings.docker_backend == "broker"
    assert settings.docker_broker_url == "http://broker"


def test_base_eval_artifact_root_prefers_data_tmp_when_writable(
    monkeypatch, tmp_path: Path
) -> None:
    """Compose path wins when /data/tmp is writable; otherwise temp/tmp fallback."""
    from prism_challenge import config as config_mod

    monkeypatch.delenv("PRISM_BASE_EVAL_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("CHALLENGE_BASE_EVAL_ARTIFACT_ROOT", raising=False)

    # Simulate a writable compose volume under /data/tmp.
    data_tmp = tmp_path / "data" / "tmp"
    data_tmp.mkdir(parents=True)
    fake_data_root = data_tmp / "prism-eval-artifacts"
    monkeypatch.setattr(config_mod, "_DATA_TMP_ARTIFACT_ROOT", fake_data_root)

    settings = PrismSettings()
    assert settings.base_eval_artifact_root == fake_data_root


def test_base_eval_artifact_root_falls_back_when_data_unwritable(monkeypatch) -> None:
    """CI hosts without /data must not PermissionError on the unwritable compose path."""
    from prism_challenge import config as config_mod

    monkeypatch.delenv("PRISM_BASE_EVAL_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("CHALLENGE_BASE_EVAL_ARTIFACT_ROOT", raising=False)
    # Point preferred compose root at a path under / that is never creatable as non-root.
    monkeypatch.setattr(
        config_mod, "_DATA_TMP_ARTIFACT_ROOT", Path("/proc/prism-no-such-artifacts")
    )
    settings = PrismSettings()
    root = settings.base_eval_artifact_root
    assert root.name == "prism-eval-artifacts"
    # Must be a creatable location (tmp or system temp), never the unwritable probe path.
    assert root != Path("/proc/prism-no-such-artifacts")
    root.mkdir(parents=True, exist_ok=True)
    assert root.is_dir()


def test_base_eval_artifact_root_env_override(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "custom-artifacts"
    monkeypatch.setenv("PRISM_BASE_EVAL_ARTIFACT_ROOT", str(target))
    settings = PrismSettings()
    assert settings.base_eval_artifact_root == target


def test_proof_runtime_environment_is_not_treated_as_settings(monkeypatch) -> None:
    for index, name in enumerate(PROVIDER_ENV_KEYS):
        monkeypatch.setenv(name, f"runtime-metadata-{index}")

    settings = PrismSettings()

    assert settings.prism_role == "challenge"


def test_secret_file_helpers(tmp_path) -> None:
    shared = tmp_path / "shared-token"
    shared.write_text("shared\n", encoding="utf-8")

    settings = PrismSettings(
        database_url="postgresql://db/prism",
        database_path=tmp_path / "fallback.sqlite3",
        shared_token_file=str(shared),
    )

    assert settings.internal_token() == "shared"
    assert settings.resolved_database_path == tmp_path / "fallback.sqlite3"
    # LLM gateway helpers were removed with deterministic admission; residual attrs must stay gone.
    assert not hasattr(settings, "llm_gateway_token_value")
    assert not hasattr(settings, "llm_gateway_token")


def test_internal_token_requires_secret() -> None:
    # ChallengeSettings forbids constructing without a token *or* secret path; pass a
    # non-existent path so construction succeeds, then assert a missing file fails closed.
    settings = PrismSettings(shared_token_file="/tmp/prism-missing-shared-token")

    try:
        settings.internal_token()
    except RuntimeError as exc:
        assert "PRISM_SHARED_TOKEN" in str(exc)
    else:
        raise AssertionError("internal_token should require a configured secret")


def test_max_code_bytes_holds_five_mib_zip_base64() -> None:
    # 5 MiB raw zip -> base64 length 6,990,508; cap must comfortably exceed it.
    raw_five_mib = 5 * 1024 * 1024
    base64_len = 4 * ((raw_five_mib + 2) // 3)

    settings = PrismSettings()

    assert settings.max_code_bytes == 7_500_000
    assert settings.max_code_bytes > base64_len


def test_example_config_parses_with_nas_defaults() -> None:
    payload = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    settings = PrismSettings(**payload)

    assert settings.slug == "prism"
    assert settings.max_code_bytes == 7_500_000
    assert settings.execution_backend == "base_gpu"
    assert settings.public_submissions_enabled is True
    assert settings.arch_weight == 0.7
    assert settings.recipe_weight == 0.3
    assert settings.base_eval_max_gpu_count == 8
    assert settings.base_eval_gpu_count == 1
    assert settings.docker_enabled is False
    assert settings.docker_backend == "cli"
    assert settings.shared_token is None
    assert settings.docker_broker_token is None
    assert not hasattr(settings, "llm_gateway_token")
    assert not hasattr(settings, "openrouter_api_key")
    assert "shared_token" not in payload
    assert "openrouter_api_key" not in payload
    assert "docker_broker_token" not in payload
