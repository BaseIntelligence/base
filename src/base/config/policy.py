from __future__ import annotations

import re
import stat
from pathlib import Path
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


def assert_protected_secret_file(
    file_path: str | Path,
    *,
    name: str = "secret",
    max_mode: int = 0o600,
    require_exists: bool = True,
) -> str | None:
    """Validate a secret path. Names diagnostics only; never returns?hermetic leak.

    When ``require_exists`` is false (Settings load of a container-path YAML on
    the host), only check mode/emptiness if the file is present. Runtime startup
    must pass require_exists=True to fail closed on absents.
    """

    path = Path(file_path)
    if not path.is_file():
        if require_exists:
            raise ProductionPolicyError(f"required secret file missing: {name}")
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & ~max_mode:
        raise ProductionPolicyError(
            f"secret file {name!r} has too-permissive mode {oct(mode)}; "
            f"require {oct(max_mode)} or stricter"
        )
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ProductionPolicyError(f"secret file unreadable: {name}") from exc
    if not content:
        raise ProductionPolicyError(f"required secret file empty: {name}")
    parent = path.parent
    if parent.exists() and parent.is_dir():
        parent_mode = stat.S_IMODE(parent.stat().st_mode)
        if parent_mode & 0o077:
            raise ProductionPolicyError(
                f"production secret directory mode too loose: {oct(parent_mode)}"
            )
    return content


def validate_secret_configuration(settings: Any, *, production: bool) -> None:
    """Fail closed when production secrets are inline, missing path, or loose.

    VAL-COMPOSE-051/055: production must not start with inline admin tokens or
    without ``admin_token_file``. When the host path is present, mode and
    emptiness are enforced. Missing container paths are deferred to runtime
    startup (``assert_protected_secret_file(..., require_exists=True)``).
    """

    if not production:
        return
    security = getattr(settings, "security", None)
    if security is None:
        raise ProductionPolicyError("production requires security settings")
    inline = getattr(security, "admin_token", None)
    file_path = getattr(security, "admin_token_file", None)
    inline_set = bool(inline is not None and str(inline).strip())
    if inline_set and file_path:
        # Dual sources are fail-closed even when the file path is present:
        # production must use the secret file exclusively (VAL-COMPOSE-051/055).
        raise ProductionPolicyError(
            "production rejects inline admin_token when admin_token_file is set; "
            "use file-backed admin_token only"
        )
    if inline_set:
        raise ProductionPolicyError(
            "production rejects inline admin_token; use admin_token_file"
        )
    if not file_path:
        raise ProductionPolicyError("production requires security.admin_token_file")
    # Settings models validate offline (host paths to container mounts may not
    # exist). Enforce mode/empty when present; runtime requires existence.
    assert_protected_secret_file(file_path, name="admin_token", require_exists=False)


def validate_settings_policy(settings: Any) -> None:
    production = production_policy_enabled_for_settings(settings)
    validate_database_url(settings.database.url, production=production)
    validate_allowed_image_prefixes(
        list(settings.docker.broker_allowed_images), production=production
    )
    validate_secret_configuration(settings, production=production)
