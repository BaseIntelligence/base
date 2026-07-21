"""Isolated canonical-image and compose tooling for attested review.

The review application is deliberately not a variation of the eval application.
It has one service, a digest-pinned image, the read-only dstack quote socket,
and exactly the encrypted secret names required by the reviewer
(``OPENROUTER_API_KEY``, ``REVIEW_API_BASE_URL``, ``REVIEW_SESSION_TOKEN``).
It has no Docker socket, golden data, task cache, eval nonce, score, gateway,
or key-release capability.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_challenge.canonical.measurement import compose_hash, normalize_app_compose

APP_COMPOSE_MANIFEST_VERSION = 2
APP_COMPOSE_RUNNER = "docker-compose"
DEFAULT_REVIEW_APP_IDENTITY = "agent-challenge-review-v1"
REVIEWER_SERVICE = "reviewer"
DSTACK_QUOTE_SOCKET_PATH = "/var/run/dstack.sock"
# Exactly the non-empty encrypted secret names measured into compose_hash.
# REVIEW_API_BASE_URL remains listed so compose_hash identity is stable, but
# encrypt/deploy + measured runtime force the joinbase pin (anti-cheat): miners
# cannot change callback authority via this slot in production.
REVIEW_ALLOWED_ENVS = (
    "OPENROUTER_API_KEY",
    "REVIEW_API_BASE_URL",
    "REVIEW_SESSION_TOKEN",
)
# Exact service inventory only. Extra Docker/swarm capability keys (privileged,
# devices, network, namespaces, secrets, mounts, ports, etc.) reject.
REVIEWER_SERVICE_KEYS = frozenset({"image", "restart", "environment", "volumes"})

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DOCKERFILE = REPO_ROOT / "docker" / "review" / "Dockerfile"
REVIEW_REQUIREMENTS = REPO_ROOT / "docker" / "review" / "requirements.txt"
EVAL_DOCKERFILE = REPO_ROOT / "docker" / "canonical" / "Dockerfile"
# Phala Cloud injects this fixed boot helper into every app-compose document.  The
# measured compose_hash includes it, so the local generator must emit the same
# bytes or provision/create identity checks fail closed (VAL-REVIEW-053/054).
PHALA_PRE_LAUNCH_SCRIPT_PATH = REPO_ROOT / "docker" / "review" / "phala_pre_launch.sh"
PHALA_DEFAULT_FEATURES: tuple[str, ...] = ("kms", "tproxy-net")

_DIGEST_PIN_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_APP_ID_RE = re.compile(r"^[!-~]{1,128}$")


class ReviewComposeError(ValueError):
    """A review image or compose document violates the isolated contract."""


@dataclass(frozen=True)
class ReviewBuildDefinition:
    """The separate build inputs that define the review application image."""

    dockerfile: Path
    requirements: Path


def assert_review_image_digest_pinned(image_ref: str) -> str:
    """Return a review image ref only when it carries an immutable SHA-256 pin."""

    if not isinstance(image_ref, str) or not _DIGEST_PIN_RE.fullmatch(image_ref):
        raise ReviewComposeError("review image must be repo@sha256:<64 lowercase hex>")
    return image_ref


def review_build_definition() -> ReviewBuildDefinition:
    """Return the separately checked-in, reproducible review build inputs."""

    return ReviewBuildDefinition(dockerfile=REVIEW_DOCKERFILE, requirements=REVIEW_REQUIREMENTS)


def eval_build_definition() -> ReviewBuildDefinition:
    """Return the eval Dockerfile solely for explicit separation checks."""

    return ReviewBuildDefinition(
        dockerfile=EVAL_DOCKERFILE,
        requirements=REPO_ROOT / "docker" / "canonical" / "requirements.txt",
    )


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        import json

        return json.dumps(value, ensure_ascii=False)
    raise ReviewComposeError(f"unsupported review compose scalar {type(value).__name__}")


def _emit_yaml(value: Any, indent: int = 0) -> list[str]:
    import json

    padding = "  " * indent
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key in sorted(value, key=str):
            item = value[key]
            header = f"{padding}{json.dumps(str(key), ensure_ascii=False)}:"
            if isinstance(item, Mapping):
                if item:
                    lines.append(header)
                    lines.extend(_emit_yaml(item, indent + 1))
                else:
                    lines.append(f"{header} {{}}")
            elif isinstance(item, (list, tuple)):
                if item:
                    lines.append(header)
                    lines.extend(_emit_yaml(list(item), indent + 1))
                else:
                    lines.append(f"{header} []")
            else:
                lines.append(f"{header} {_yaml_scalar(item)}")
        return lines
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, (Mapping, list, tuple)) for item in value):
            raise ReviewComposeError("review compose does not support nested list values")
        return [f"{padding}- {_yaml_scalar(item)}" for item in value]
    raise ReviewComposeError("review compose root must be a mapping")


def _render_review_service(service: Mapping[str, Any]) -> str:
    return "\n".join(_emit_yaml({"services": {REVIEWER_SERVICE: service}})) + "\n"


def _phala_pre_launch_script() -> str:
    """Return the fixed Phala Cloud pre-launch helper measured into compose_hash."""

    if not PHALA_PRE_LAUNCH_SCRIPT_PATH.is_file():
        raise ReviewComposeError("Phala pre-launch script is missing from the review image package")
    text = PHALA_PRE_LAUNCH_SCRIPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("#!/bin/bash") or "Phala Cloud Pre-Launch Script" not in text:
        raise ReviewComposeError("Phala pre-launch script is not the expected vendor helper")
    # Phala stores the helper without a trailing newline in app-compose JSON.
    return text.rstrip("\n")


def generate_review_app_compose(
    *,
    review_image: str,
    app_identity: str = DEFAULT_REVIEW_APP_IDENTITY,
) -> dict[str, Any]:
    """Generate the only supported review application compose document.

    The document matches Phala Cloud's rewritten AppCompose exactly (including
    the fixed pre-launch helper and default features factors) so
    ``compose_hash`` from ``POST /cvms/provision`` equals the offline hash.
    """

    image = assert_review_image_digest_pinned(review_image)
    if not isinstance(app_identity, str) or not _APP_ID_RE.fullmatch(app_identity):
        raise ReviewComposeError("review app identity must be a visible ASCII identifier")

    service = {
        "image": image,
        "restart": "no",
        "environment": list(REVIEW_ALLOWED_ENVS),
        "volumes": [f"{DSTACK_QUOTE_SOCKET_PATH}:{DSTACK_QUOTE_SOCKET_PATH}:ro"],
    }
    compose = {
        "manifest_version": APP_COMPOSE_MANIFEST_VERSION,
        "name": app_identity,
        "runner": APP_COMPOSE_RUNNER,
        "docker_compose_file": _render_review_service(service),
        "kms_enabled": True,
        "gateway_enabled": False,
        # Phala still emits the deprecated alias alongside gateway_enabled; both
        # appear in the measured compose_hash returned by provision.
        "tproxy_enabled": True,
        "local_key_provider_enabled": False,
        # Public container logs are required for residual diagnosis when the
        # reviewer exits non-zero before /report (Phala hides endpoints under
        # public_logs=false). Reviewer still must not print secret values.
        "public_logs": True,
        "public_sysinfo": False,
        "public_tcbinfo": True,
        "no_instance_id": False,
        "secure_time": False,
        "storage_fs": "zfs",
        "features": list(PHALA_DEFAULT_FEATURES),
        "allowed_envs": list(REVIEW_ALLOWED_ENVS),
        "pre_launch_script": _phala_pre_launch_script(),
    }
    validate_review_app_compose(compose)
    return compose


def validate_review_app_compose(compose: Mapping[str, Any]) -> None:
    """Reject every compose shape that expands review authority."""

    expected = {
        "manifest_version",
        "name",
        "runner",
        "docker_compose_file",
        "kms_enabled",
        "gateway_enabled",
        "tproxy_enabled",
        "local_key_provider_enabled",
        "public_logs",
        "public_sysinfo",
        "public_tcbinfo",
        "no_instance_id",
        "secure_time",
        "storage_fs",
        "features",
        "allowed_envs",
        "pre_launch_script",
    }
    if set(compose) != expected:
        raise ReviewComposeError("review compose keys must be schema-closed")
    if compose["manifest_version"] != APP_COMPOSE_MANIFEST_VERSION:
        raise ReviewComposeError("unsupported review compose manifest version")
    if compose["runner"] != APP_COMPOSE_RUNNER:
        raise ReviewComposeError("review compose must use docker-compose")
    if not isinstance(compose["name"], str) or not _APP_ID_RE.fullmatch(compose["name"]):
        raise ReviewComposeError("review compose identity is invalid")
    if compose["allowed_envs"] != list(REVIEW_ALLOWED_ENVS):
        raise ReviewComposeError("review compose allowed_envs must be exact")
    if compose.get("features") != list(PHALA_DEFAULT_FEATURES):
        raise ReviewComposeError("review compose features must be the Phala defaults")
    if compose.get("storage_fs") != "zfs":
        raise ReviewComposeError("review compose storage_fs must be zfs")
    if not isinstance(compose.get("pre_launch_script"), str) or not compose["pre_launch_script"]:
        raise ReviewComposeError("review compose requires the Phala pre-launch helper")
    if (
        compose["kms_enabled"] is not True
        or compose["gateway_enabled"] is not False
        or compose["tproxy_enabled"] is not True
        or compose["local_key_provider_enabled"] is not False
        or compose["public_logs"] is not True
        or compose["public_sysinfo"] is not False
        or compose["public_tcbinfo"] is not True
        or compose["no_instance_id"] is not False
        or compose["secure_time"] is not False
    ):
        raise ReviewComposeError("review compose capability flags are invalid")

    try:
        import yaml

        services = yaml.safe_load(compose["docker_compose_file"])["services"]
        service = services[REVIEWER_SERVICE]
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ReviewComposeError("review docker compose service is invalid") from exc
    if set(services) != {REVIEWER_SERVICE}:
        raise ReviewComposeError("review compose must contain exactly one reviewer service")
    if not isinstance(service, Mapping) or set(service) != REVIEWER_SERVICE_KEYS:
        raise ReviewComposeError(
            "reviewer service keys must be schema-closed "
            f"(allowed: {sorted(REVIEWER_SERVICE_KEYS)})"
        )
    assert_review_image_digest_pinned(service.get("image"))
    if service.get("restart") != "no":
        raise ReviewComposeError("reviewer restart policy must be no")
    if service.get("environment") != list(REVIEW_ALLOWED_ENVS):
        raise ReviewComposeError("reviewer environment must contain only encrypted secret names")
    if service.get("volumes") != [f"{DSTACK_QUOTE_SOCKET_PATH}:{DSTACK_QUOTE_SOCKET_PATH}:ro"]:
        raise ReviewComposeError("reviewer may mount only the read-only dstack quote socket")
    # Schema-closed keys already reject privileged/device/network/namespace/secret
    # capability fields.  The residual scan only covers env/mount contents.
    forbidden = (
        "docker.sock",
        "golden",
        "task-cache",
        "key-release",
        "base_llm_gateway",
        "base_gateway",
        "eval_run",
        "weight",
    )
    inventory = str(service).lower()
    if any(item in inventory for item in forbidden):
        raise ReviewComposeError("review compose has an unauthorized capability")


def render_review_app_compose(compose: Mapping[str, Any]) -> str:
    """Serialize exact bytes measured into the review CVM's compose hash."""

    validate_review_app_compose(compose)
    return normalize_app_compose(compose)


def review_app_compose_hash(compose: Mapping[str, Any]) -> str:
    """Return the deterministic canonical compose hash for a review app."""

    validate_review_app_compose(compose)
    return compose_hash(compose)


__all__ = [
    "DEFAULT_REVIEW_APP_IDENTITY",
    "DSTACK_QUOTE_SOCKET_PATH",
    "PHALA_DEFAULT_FEATURES",
    "PHALA_PRE_LAUNCH_SCRIPT_PATH",
    "REVIEW_ALLOWED_ENVS",
    "REVIEWER_SERVICE",
    "REVIEWER_SERVICE_KEYS",
    "ReviewBuildDefinition",
    "ReviewComposeError",
    "assert_review_image_digest_pinned",
    "eval_build_definition",
    "generate_review_app_compose",
    "render_review_app_compose",
    "review_app_compose_hash",
    "review_build_definition",
    "validate_review_app_compose",
]
