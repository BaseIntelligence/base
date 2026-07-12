"""Shared challenge-side SDK configuration helpers."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .roles import (
    CAPABILITY_REGISTRY_VERSION,
    Role,
    capabilities_for_role,
)
from .version import API_VERSION, SDK_CONTRACT_VERSION


class DockerExecutorSettings(BaseSettings):
    """Environment-backed Docker executor settings for challenges."""

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="forbid")

    def __init__(self, **values: Any) -> None:
        known = {
            f"CHALLENGE_{field_name.upper()}" for field_name in type(self).model_fields
        }
        known.add("CHALLENGE_ENV_FILE")
        unknown = sorted(
            name
            for name in os.environ
            if name.startswith("CHALLENGE_") and name not in known
        )
        if unknown:
            raise ValueError(f"Unknown challenge configuration key: {unknown[0]}")
        super().__init__(**values)

    docker_enabled: bool = False
    docker_bin: str = "docker"
    docker_network: str = "none"
    docker_cpus: float = Field(default=2.0, gt=0)
    docker_memory: str = "4g"
    docker_memory_swap: str | None = "4g"
    docker_pids_limit: int = Field(default=512, ge=1)
    docker_read_only: bool = True
    docker_user: str | None = None
    docker_allowed_images: tuple[str, ...] = ()
    docker_backend: str = "cli"
    docker_broker_url: str | None = None
    docker_broker_token: str | None = None
    docker_broker_token_file: str | None = None

    @model_validator(mode="after")
    def validate_executor_backend(self) -> DockerExecutorSettings:
        if self.docker_backend not in {"cli", "broker"}:
            raise ValueError(f"unsupported executor backend: {self.docker_backend!r}")
        if self.docker_backend == "broker" and not (
            self.docker_broker_token or self.docker_broker_token_file
        ):
            raise ValueError("broker executor requires a token or secret file")
        return self


class ChallengeSettings(DockerExecutorSettings):
    """Canonical settings shared by independently packaged challenges."""

    model_config = SettingsConfigDict(env_prefix="CHALLENGE_", extra="forbid")

    slug: str = "challenge"
    name: str = "Challenge"
    version: str = "0.1.0"
    api_version: str = API_VERSION
    sdk_version: str = SDK_CONTRACT_VERSION
    sdk_compatibility_range: str = f"^{SDK_CONTRACT_VERSION.split('.')[0]}.0.0"
    api_compatibility_range: str = f"^{API_VERSION.split('.')[0]}.0"
    role: Literal["challenge"] = Role.CHALLENGE.value
    capabilities: tuple[str, ...] = Field(
        default_factory=lambda: (
            "challenge.scoring",
            "challenge.ordinary_proof",
            "challenge.state",
        )
    )
    tee_verification_enabled: bool = False
    raw_weight_push_enabled: bool = False
    capability_registry_version: str = CAPABILITY_REGISTRY_VERSION
    database_url: str = "sqlite+aiosqlite:////data/challenge.sqlite3"
    shared_token: str | None = Field(default=None, repr=False)
    shared_token_file: str | None = Field(
        default="/run/secrets/base/challenge_token",
        repr=False,
    )
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("capabilities")
    @classmethod
    def validate_capabilities(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        expected = {
            "challenge.scoring",
            "challenge.ordinary_proof",
            "challenge.state",
        }
        if "challenge.raw_weight_push" in value:
            expected.add("challenge.raw_weight_push")
        if "challenge.tee_verification" in value:
            expected.add("challenge.tee_verification")
        if set(value) != expected:
            raise ValueError(
                "capabilities must be the server-derived challenge registry set"
            )
        return value

    @model_validator(mode="after")
    def validate_compatibility(self) -> ChallengeSettings:
        if self.api_version != API_VERSION:
            raise ValueError(
                "Incompatible API version: "
                f"expected {API_VERSION!r}, actual {self.api_version!r}"
            )
        if self.sdk_version != SDK_CONTRACT_VERSION:
            raise ValueError(
                "Incompatible SDK version: "
                f"expected {SDK_CONTRACT_VERSION!r}, actual {self.sdk_version!r}"
            )
        if self.capability_registry_version != CAPABILITY_REGISTRY_VERSION:
            raise ValueError(
                "Incompatible capability registry: "
                f"expected {CAPABILITY_REGISTRY_VERSION!r}, "
                f"actual {self.capability_registry_version!r}"
            )
        expected = set(
            capabilities_for_role(
                Role.CHALLENGE,
                tee_verification=self.tee_verification_enabled,
            )
        )
        if not self.raw_weight_push_enabled:
            expected.discard("challenge.raw_weight_push")
        if set(self.capabilities) != expected or len(self.capabilities) != len(
            expected
        ):
            raise ValueError(
                "capabilities do not match the server-derived challenge role"
            )
        if not self.shared_token and not self.shared_token_file:
            raise ValueError("challenge authentication requires a token or secret file")
        return self


__all__ = ["ChallengeSettings", "DockerExecutorSettings"]
