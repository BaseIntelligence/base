from __future__ import annotations

import re
from typing import Any

PRODUCTION_ENVIRONMENTS = {"prod", "production", "staging"}
POSTGRES_SCHEMES = ("postgres://", "postgresql://", "postgresql+asyncpg://")
_SEMVER_TAG_RE = re.compile(r"^v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_BROAD_IMAGE_PREFIXES = {
    "*",
    "docker.io/",
    "ghcr.io/",
    "gcr.io/",
    "quay.io/",
    "registry.k8s.io/",
    "baseintelligence/",
    "python:",
    "python",
    "localhost/",
    "127.0.0.1/",
    "host.docker.internal/",
}


class ProductionPolicyError(ValueError):
    """Raised when production policy is violated."""


def is_production_environment(environment: str | None) -> bool:
    return (environment or "").strip().lower() in PRODUCTION_ENVIRONMENTS


def production_policy_enabled(*, environment: str | None = None) -> bool:
    return is_production_environment(environment)


def production_policy_enabled_for_settings(settings: Any) -> bool:
    return production_policy_enabled(environment=getattr(settings, "environment", None))


def validate_database_url(database_url: str, *, production: bool) -> None:
    if not production:
        return
    if not database_url or database_url.startswith("sqlite"):
        raise ProductionPolicyError(
            "production requires an external PostgreSQL database URL"
        )
    if not database_url.startswith(POSTGRES_SCHEMES):
        raise ProductionPolicyError("production database URL must use PostgreSQL")


def validate_allowed_image_prefixes(prefixes: list[str], *, production: bool) -> None:
    if not production:
        return
    for prefix in prefixes:
        normalized = prefix.strip()
        if not normalized or normalized in _BROAD_IMAGE_PREFIXES or "*" in normalized:
            raise ProductionPolicyError(
                f"production image allowlist prefix is too broad: {prefix!r}"
            )
        registry, slash, remainder = normalized.partition("/")
        if not slash or "." not in registry or not remainder.strip("/"):
            raise ProductionPolicyError(
                "production image allowlist prefix must include registry "
                f"and namespace: {prefix!r}"
            )


def validate_image_reference(image: str, *, production: bool) -> None:
    if not production:
        return
    reference, separator, digest = image.partition("@")
    slash_index = reference.rfind("/")
    colon_index = reference.rfind(":")
    has_tag = colon_index > slash_index
    tag = reference[colon_index + 1 :] if has_tag else ""
    if not has_tag:
        raise ProductionPolicyError("production image references must include a tag")
    if not separator or not digest:
        raise ProductionPolicyError("production image references must include a digest")
    if tag != "latest" and not _SEMVER_TAG_RE.match(tag):
        raise ProductionPolicyError("production image tags must be semver or latest")
    if not _SHA256_DIGEST_RE.match(digest):
        raise ProductionPolicyError("production image digest must be sha256")


def validate_tls_enabled(
    *, verify_tls: bool | None, production: bool, subject: str
) -> None:
    if production and verify_tls is False:
        raise ProductionPolicyError(
            f"{subject} must keep verify_tls=true in production"
        )


def validate_settings_policy(settings: Any) -> None:
    production = production_policy_enabled_for_settings(settings)
    validate_database_url(settings.database.url, production=production)
    validate_allowed_image_prefixes(
        list(settings.docker.broker_allowed_images), production=production
    )
