"""Shared challenge-side SDK configuration helpers."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DockerExecutorSettings(BaseSettings):
    """Environment-backed Docker executor settings for challenges."""

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="ignore")

    docker_enabled: bool = False
    docker_bin: str = "docker"
    docker_network: str = "none"
    docker_cpus: float = 2.0
    docker_memory: str = "4g"
    docker_memory_swap: str | None = "4g"
    docker_pids_limit: int = 512
    docker_read_only: bool = True
    docker_user: str | None = None
    docker_allowed_images: tuple[str, ...] = ()
    docker_backend: str = "cli"
    docker_broker_url: str | None = None
    docker_broker_token: str | None = None
    docker_broker_token_file: str | None = None


class ChallengeSettings(DockerExecutorSettings):
    """Canonical settings shared by independently packaged challenges."""

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="forbid")

    slug: str = "challenge"
    name: str = "Challenge"
    version: str = "0.1.0"
    api_version: str = "1.0"
    sdk_version: str = "1.0.0"
    database_url: str = "sqlite+aiosqlite:////data/challenge.sqlite3"
    shared_token: str | None = Field(default=None, repr=False)
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
    )
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


__all__ = ["ChallengeSettings", "DockerExecutorSettings"]
