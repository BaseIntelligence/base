"""Generate the Phala ``app-compose`` the miner deploys (architecture §4 C2).

The miner self-deploys a Phala TDX CPU CVM running the canonical eval image. The
CVM is described by an ``app-compose.json`` document that embeds a docker-compose
file plus dstack deployment flags. This module generates that document such that:

* **Only the orchestrator service is declared** — the ~89 Terminal-Bench task
  images are NOT static compose services. They are pinned by digest via the
  golden manifest (mounted read-only) and launched dynamically as siblings on
  the guest Docker socket (DooD) at runtime (VAL-ORCH-032).
* **No secrets are embedded** — the compose carries no gateway token, no
  miner-env values, no provider ``*_API_KEY``, and no Phala API key. Secrets are
  supplied at deploy time via dstack ``encrypted_env`` for the names listed in
  ``allowed_envs`` (VAL-ORCH-033).
* **Generation is deterministic** — the same inputs always produce byte-identical
  output, so the SHA-256 compose-hash is stable and matches the value dstack
  measures into RTMR3 on deploy (VAL-ORCH-034).

**Critical byte-for-byte contract (library/measurement-tooling.md):** the bytes
actually written to ``app-compose.json`` and deployed MUST equal
:func:`agent_challenge.canonical.measurement.normalize_app_compose` of the
generated document verbatim. :func:`render_app_compose` is the ONLY serializer a
deployer should use — never a separate ``json.dumps``/pretty-print — otherwise
the live compose-hash will not equal the offline
:func:`agent_challenge.canonical.measurement.compose_hash` and the pinned
allowlist match (M6 verification) fails.

Import-light (stdlib + the measurement helper) so it loads in the lean image.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_challenge.canonical.live_registry import LIVE_REGISTRY_ENV
from agent_challenge.canonical.measurement import compose_hash, normalize_app_compose
from agent_challenge.evaluation.own_runner.dood import (
    DOCKER_SOCKET_PATH,
    DSTACK_SOCKET_PATH,
)
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_TLS_CA_ENV,
    KEY_RELEASE_TLS_CERT_ENV,
    KEY_RELEASE_TLS_KEY_ENV,
    KEY_RELEASE_URL_ENV,
)

#: Production raw RA-TLS endpoint env names (mirrored from the validator listener
#: without importing the server module, which must stay out of the lean image).
RA_TLS_HOST_ENV = "KEY_RELEASE_RA_TLS_HOST"
RA_TLS_PORT_ENV = "KEY_RELEASE_RA_TLS_PORT"
DEFAULT_KEY_RELEASE_RA_TLS_PORT = 8701

#: Default in-CVM paths for raw RA-TLS client mTLS material.
DEFAULT_KEY_RELEASE_TLS_CERT_PATH = "/run/secrets/ra_tls/client.crt"
DEFAULT_KEY_RELEASE_TLS_KEY_PATH = "/run/secrets/ra_tls/client.key"
DEFAULT_KEY_RELEASE_TLS_CA_PATH = "/run/secrets/ra_tls/ca.crt"

#: dstack app-compose runner + manifest version for a docker-compose app.
APP_COMPOSE_MANIFEST_VERSION = 2
APP_COMPOSE_RUNNER = "docker-compose"

#: Default canonical app name.
DEFAULT_APP_NAME = "agent-challenge-canonical"

#: Repository root (``src/agent_challenge/canonical/compose.py`` → repo).
REPO_ROOT = Path(__file__).resolve().parents[3]
#: Phala Cloud injects this fixed boot helper into every app-compose document.
#: The measured compose_hash includes it, so the local generator must emit the
#: same bytes or ``POST /cvms/provision`` identity checks fail closed.  Shared
#: with the review compose path (same vendor helper file).
PHALA_PRE_LAUNCH_SCRIPT_PATH = REPO_ROOT / "docker" / "review" / "phala_pre_launch.sh"
#: Default Phala Cloud ``features`` factor measured into compose_hash.
PHALA_DEFAULT_FEATURES: tuple[str, ...] = ("kms", "tproxy-net")
#: Top-level keys of the provision-compatible (Phala-envelope) eval app-compose.
#: Review uses the same envelope factors with a disjoint service inventory.
PHALA_APP_COMPOSE_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {
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
)

#: In-CVM paths for the orchestrator job dir, task cache, and golden manifest.
DEFAULT_JOB_DIR = "/opt/agent-challenge/job"
DEFAULT_CACHE_ROOT = "/opt/agent-challenge/task-cache"
DEFAULT_GOLDEN_DIR = "/opt/agent-challenge/golden"
DEFAULT_DIGEST_MANIFEST = "/opt/agent-challenge/golden/dataset-digest.json"

# Guest sockets bind-mounted into the orchestrator (DooD + attestation) are
# single-sourced from :mod:`agent_challenge.evaluation.own_runner.dood` (the DooD
# launch-policy reference) so the compose mounts and the socket-exposure guard can
# never diverge: DOCKER_SOCKET_PATH / DSTACK_SOCKET_PATH are imported above.

#: Orchestrator service name in the generated compose.
ORCHESTRATOR_SERVICE = "orchestrator"

#: Env var NAMES injected at deploy via dstack ``encrypted_env`` (values NEVER in
#: the compose bytes). These are the per-run Phala binding inputs only.
#: VAL-ACAT-013: Base LLM gateway names (``BASE_GATEWAY_TOKEN``,
#: ``BASE_LLM_GATEWAY_URL``, …) are intentionally **absent**. Production key-release
#: identity is bound into static compose env as ``KEY_RELEASE_RA_TLS_HOST`` /
#: ``KEY_RELEASE_RA_TLS_PORT`` plus the required client mTLS path names (no HTTP
#: URL fallback in the measured app).
DEFAULT_ALLOWED_ENVS: tuple[str, ...] = (
    "CHALLENGE_PHALA_AGENT_HASH",
    "CHALLENGE_PHALA_ATTESTATION_ENABLED",
    "CHALLENGE_PHALA_CANONICAL_MEASUREMENT",
    "CHALLENGE_PHALA_EVAL_PLAN",
    "CHALLENGE_PHALA_KEY_RELEASE_URL",
    "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM",
    "CHALLENGE_PHALA_RA_TLS_SERVER_CA_FILE",
    "CHALLENGE_PHALA_RTMR3",
    "CHALLENGE_PHALA_VALIDATOR_NONCE",
    # Residual ORCH public_logs probes (VAL-ORCH-009/010/014/022): opt-in only,
    # secret-free marker emission for non-dev (no SSH) live residual scrape.
    "CHALLENGE_RESIDUAL_ORCH_PROBES",
    KEY_RELEASE_TLS_CERT_ENV,
    KEY_RELEASE_TLS_KEY_ENV,
    KEY_RELEASE_TLS_CA_ENV,
    "LLM_COST_LIMIT",
    "EVAL_RUN_TOKEN",
    # Measured OpenRouter (eval agent inside measured CVM only when product allows).
    # Never Base gateway; keys stay miner/session encrypted_env on attested guests.
    "OPENROUTER_API_KEY",
)

_DIGEST_PIN_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
_DIGEST_REF_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class ComposeGenerationError(ValueError):
    """A compose could not be generated deterministically / safely."""


def _parse_raw_key_release_endpoint(endpoint: str) -> tuple[str, int] | None:
    """Return ``(host, port)`` for production raw RA-TLS authority inputs.

    Accepts ``host:8701``, ``ratls://host:port``, ``tls://host:port``, or
    ``tcp://host:port``. HTTP(S) URLs and non-raw authorities return ``None`` so
    legacy offline helpers that still pass an HTTP key-release URL keep working
    under the flag-off / non-production path.
    """

    value = endpoint.strip()
    if not value:
        return None
    scheme = ""
    authority = value
    if "://" in value:
        scheme, authority = value.split("://", 1)
        scheme = scheme.lower()
        if scheme in {"http", "https"}:
            return None
        if scheme not in {"ratls", "tls", "tcp"}:
            return None
    if "/" in authority:
        authority = authority.split("/", 1)[0]
    if ":" not in authority:
        return None
    host, port_text = authority.rsplit(":", 1)
    try:
        port = int(port_text)
    except ValueError:
        return None
    host = host.strip().strip("[]")
    if not host or not 1 <= port <= 65535:
        return None
    # Authority without an explicit raw scheme counts as production only when
    # targeting the enumarated RA-TLS listener port.
    if not scheme and port != DEFAULT_KEY_RELEASE_RA_TLS_PORT:
        return None
    return host, port


def assert_digest_pinned(image_ref: str, *, what: str = "image") -> str:
    """Require ``image_ref`` to be pinned by an immutable ``@sha256:`` digest.

    A floating tag (``:latest``) would break reproducibility of the compose-hash
    and the canonical measurement, so it is rejected fail-closed.
    """

    if not isinstance(image_ref, str) or not _DIGEST_PIN_RE.search(image_ref):
        raise ComposeGenerationError(
            f"{what} must be digest-pinned (repo@sha256:<64hex>), got {image_ref!r}"
        )
    return image_ref


def golden_task_image_digests(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return ``task_id -> sha256:<digest>`` for the golden manifest's task images.

    Terminal-Bench task images are pinned by digest here (never a mutable tag) and
    are launched dynamically via DooD at runtime, so they never appear as static
    compose services (VAL-ORCH-032).
    """

    tasks = manifest.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ComposeGenerationError("golden manifest has no 'tasks' mapping")
    pins: dict[str, str] = {}
    for task_id, entry in tasks.items():
        if not isinstance(entry, Mapping):
            raise ComposeGenerationError(f"golden manifest task {task_id!r} is not a mapping")
        ref = entry.get("harbor_registry_ref") or entry.get("content_digest_sha256")
        if not isinstance(ref, str) or not _DIGEST_REF_RE.match(ref):
            raise ComposeGenerationError(
                f"golden manifest task {task_id!r} is not digest-pinned: {ref!r}"
            )
        pins[str(task_id)] = ref if ref.startswith("sha256:") else f"sha256:{ref}"
    return pins


def load_golden_manifest(path: Path | str) -> dict[str, Any]:
    """Load a golden ``dataset-digest.json`` manifest from disk."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ComposeGenerationError("golden manifest is not a JSON object")
    return data


# --------------------------------------------------------------------------- #
# Deterministic YAML emitter (stdlib only; sorted keys; JSON-quoted scalars)
# --------------------------------------------------------------------------- #
def _yaml_scalar(value: Any) -> str:
    """Render a scalar as a valid, unambiguous YAML token.

    Strings are emitted as JSON double-quoted literals (a valid YAML flow scalar),
    which sidesteps YAML's quoting rules for values like ``"8700:8700"`` or an
    ``@sha256:`` image ref. ``bool``/``int``/``None`` map to YAML ``true``/``false``,
    the integer literal, and ``null``.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ComposeGenerationError(f"unsupported compose scalar type: {type(value).__name__}")


def _emit_yaml(value: Any, indent: int = 0) -> list[str]:
    """Emit ``value`` as deterministic block YAML (mapping keys sorted).

    Supports nested mappings and lists of scalars — the shape a docker-compose
    document needs. Lists of mappings are intentionally unsupported (not needed)
    so the emitter stays small and correct.
    """

    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = value[key]
            key_token = f"{pad}{json.dumps(str(key), ensure_ascii=False)}:"
            if isinstance(child, Mapping):
                if child:
                    lines.append(key_token)
                    lines.extend(_emit_yaml(child, indent + 1))
                else:
                    lines.append(f"{key_token} {{}}")
            elif isinstance(child, (list, tuple)):
                if child:
                    lines.append(key_token)
                    lines.extend(_emit_yaml(list(child), indent + 1))
                else:
                    lines.append(f"{key_token} []")
            else:
                lines.append(f"{key_token} {_yaml_scalar(child)}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                raise ComposeGenerationError("nested list items are not supported in compose YAML")
            lines.append(f"{pad}- {_yaml_scalar(item)}")
    else:  # pragma: no cover - top-level is always a mapping
        raise ComposeGenerationError("compose YAML root must be a mapping")
    return lines


def _render_docker_compose_yaml(services: Mapping[str, Any]) -> str:
    """Render the docker-compose ``services`` block as deterministic block YAML."""

    document = {"services": services}
    return "\n".join(_emit_yaml(document)) + "\n"


def phala_pre_launch_script() -> str:
    """Return the fixed Phala Cloud pre-launch helper measured into compose_hash.

    Phala stores the helper without a trailing newline in app-compose JSON.  The
    offline generator must produce the same text as ``POST /cvms/provision``
    rewrites for reverse-matched identity (review and eval).
    """

    if not PHALA_PRE_LAUNCH_SCRIPT_PATH.is_file():
        raise ComposeGenerationError(
            f"Phala pre-launch script is missing from the checkout ({PHALA_PRE_LAUNCH_SCRIPT_PATH})"
        )
    text = PHALA_PRE_LAUNCH_SCRIPT_PATH.read_text(encoding="utf-8")
    if not text.startswith("#!/bin/bash") or "Phala Cloud Pre-Launch Script" not in text:
        raise ComposeGenerationError("Phala pre-launch script is not the expected vendor helper")
    return text.rstrip("\n")


# --------------------------------------------------------------------------- #
# Compose generation
# --------------------------------------------------------------------------- #
def build_orchestrator_service(
    *,
    orchestrator_image: str,
    command: Sequence[str],
    static_env: Mapping[str, str],
    passthrough_env: Sequence[str],
    golden_dir: str,
    cache_root: str,
) -> dict[str, Any]:
    """Build the single orchestrator compose service (no per-task services).

    Mounts the guest Docker + dstack sockets (DooD + attestation) and the golden
    manifest + task cache read-only; it is NOT privileged and starts no inner
    dockerd. ``static_env`` are non-secret ``NAME=value`` config entries;
    ``passthrough_env`` are secret/per-run NAMES injected at deploy via
    ``encrypted_env`` (name-only, so no value is ever written here).
    """

    environment = sorted(
        [f"{name}={value}" for name, value in static_env.items()]
        + [str(name) for name in passthrough_env if name not in static_env]
    )
    # Guest Docker + dstack sockets only. Golden and task-cache material live in
    # the measured canonical image at golden_dir/cache_root; bind-mounting empty
    # guest host paths over those directories would hide the image assets and break
    # live DooD/key-release evaluation.
    volumes = sorted(
        [
            f"{DOCKER_SOCKET_PATH}:{DOCKER_SOCKET_PATH}",
            f"{DSTACK_SOCKET_PATH}:{DSTACK_SOCKET_PATH}",
        ]
    )
    return {
        "image": assert_digest_pinned(orchestrator_image, what="orchestrator image"),
        "restart": "no",
        "command": list(command),
        "environment": environment,
        "volumes": volumes,
    }


def generate_app_compose(
    *,
    orchestrator_image: str,
    name: str = DEFAULT_APP_NAME,
    command: Sequence[str] | None = None,
    allowed_envs: Sequence[str] = DEFAULT_ALLOWED_ENVS,
    key_release_url: str | None = None,
    attestation_enabled: bool = True,
    job_dir: str = DEFAULT_JOB_DIR,
    cache_root: str = DEFAULT_CACHE_ROOT,
    golden_dir: str = DEFAULT_GOLDEN_DIR,
    digest_manifest_path: str = DEFAULT_DIGEST_MANIFEST,
    live_registry_manifest_path: str | None = None,
    kms_enabled: bool = True,
    public_logs: bool = True,
    public_sysinfo: bool = True,
) -> dict[str, Any]:
    """Generate the deterministic Phala ``app-compose`` document (architecture §4 C2).

    The document declares only the orchestrator service; task images are pinned by
    digest via the golden manifest and launched dynamically. No secret VALUE is
    ever embedded — the gateway token and per-run binding inputs are injected at
    deploy via ``encrypted_env`` for the ``allowed_envs`` NAMES. Serialize the
    result with :func:`render_app_compose` (never a separate ``json.dumps``) so the
    deployed bytes hash to :func:`compose_hash` of this document.
    """

    if command is None:
        command = (
            "run",
            "--job-dir",
            job_dir,
            "--cache-root",
            cache_root,
            "--digest-manifest",
            digest_manifest_path,
        )

    # Non-secret static configuration (never a credential): DooD target + the
    # in-CVM cache/manifest paths the orchestrator reads.
    static_env = {
        "DOCKER_HOST": f"unix://{DOCKER_SOCKET_PATH}",
        "CHALLENGE_OWN_RUNNER_CACHE_ROOT": cache_root,
        "CHALLENGE_OWN_RUNNER_DIGEST_MANIFEST": digest_manifest_path,
    }
    # Production raw RA-TLS endpoint is host + port 8701 (no HTTP URL). Bake the
    # parsed operator endpoint and fixed client credential path names into the
    # measured compose so the in-CVM client has no HTTP fallback path. Legacy
    # callers can still pass a non-raw URL for offline tests; that path keeps
    # the older KEY_RELEASE_URL long enough for flag-off compatibility only.
    if key_release_url and str(key_release_url).strip():
        endpoint = str(key_release_url).strip()
        host_port = _parse_raw_key_release_endpoint(endpoint)
        if host_port is not None:
            host, port = host_port
            static_env[RA_TLS_HOST_ENV] = host
            static_env[RA_TLS_PORT_ENV] = str(port)
            static_env[KEY_RELEASE_TLS_CERT_ENV] = DEFAULT_KEY_RELEASE_TLS_CERT_PATH
            static_env[KEY_RELEASE_TLS_KEY_ENV] = DEFAULT_KEY_RELEASE_TLS_KEY_PATH
            static_env[KEY_RELEASE_TLS_CA_ENV] = DEFAULT_KEY_RELEASE_TLS_CA_PATH
        else:
            static_env[KEY_RELEASE_URL_ENV] = endpoint

    # Optional live-subset task-image resolution: point the in-CVM DooD builder at
    # the live-registry side manifest (mounted read-only in the golden dir). Only
    # added when a path is supplied, so the DEFAULT compose bytes / compose-hash
    # are byte-identical (offline / flag-off resolution unchanged).
    if live_registry_manifest_path and str(live_registry_manifest_path).strip():
        static_env[LIVE_REGISTRY_ENV] = str(live_registry_manifest_path).strip()

    service = build_orchestrator_service(
        orchestrator_image=orchestrator_image,
        command=command,
        static_env=static_env,
        passthrough_env=sorted(set(allowed_envs)),
        golden_dir=golden_dir,
        cache_root=cache_root,
    )
    docker_compose_file = _render_docker_compose_yaml({ORCHESTRATOR_SERVICE: service})

    # Phala Cloud rewrites missing envelope factors into the measured AppCompose
    # (pre_launch_script, features, tproxy/public_tcbinfo/storage_fs, secure_time).
    # Emit them offline so local compose_hash equals POST /cvms/provision's hash
    # (parity with the review reverse-match path). Eval never imports review.compose
    # so docker socket / golden mounts remain eval-only while sharing the vendor
    # helper and default features list.
    compose: dict[str, Any] = {
        "manifest_version": APP_COMPOSE_MANIFEST_VERSION,
        "name": name,
        "runner": APP_COMPOSE_RUNNER,
        "docker_compose_file": docker_compose_file,
        "kms_enabled": kms_enabled,
        "gateway_enabled": False,
        # Phala still emits the deprecated alias alongside gateway_enabled; both
        # appear in the measured compose_hash returned by provision.
        "tproxy_enabled": True,
        "local_key_provider_enabled": False,
        "public_logs": public_logs,
        "public_sysinfo": public_sysinfo,
        "public_tcbinfo": True,
        "no_instance_id": False,
        "secure_time": False,
        "storage_fs": "zfs",
        "features": list(PHALA_DEFAULT_FEATURES),
        "allowed_envs": sorted(set(allowed_envs)),
        "pre_launch_script": phala_pre_launch_script(),
    }
    return compose


def render_app_compose(compose: Mapping[str, Any]) -> str:
    """The exact ``app-compose.json`` text to deploy (== normalize_app_compose).

    This is the ONLY serializer a deployer may use for the app-compose file: it is
    byte-for-byte :func:`normalize_app_compose`, so the deployed file hashes to
    :func:`compose_hash` of ``compose`` and therefore to the live CVM
    ``compose_hash`` / RTMR3 ``compose-hash`` event.
    """

    return normalize_app_compose(compose)


def render_app_compose_bytes(compose: Mapping[str, Any]) -> bytes:
    """The exact ``app-compose.json`` bytes to deploy (UTF-8 of :func:`render_app_compose`)."""

    return render_app_compose(compose).encode("utf-8")


def write_app_compose(path: Path | str, compose: Mapping[str, Any]) -> str:
    """Write the deployable ``app-compose.json`` bytes to ``path`` and return them."""

    text = render_app_compose(compose)
    Path(path).write_text(text, encoding="utf-8")
    return text


def app_compose_hash(compose: Mapping[str, Any]) -> str:
    """SHA-256 (hex) of the deployable app-compose bytes (== measurement.compose_hash)."""

    return compose_hash(compose)


__all__ = [
    "APP_COMPOSE_MANIFEST_VERSION",
    "APP_COMPOSE_RUNNER",
    "DEFAULT_ALLOWED_ENVS",
    "DEFAULT_APP_NAME",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_DIGEST_MANIFEST",
    "DEFAULT_GOLDEN_DIR",
    "DEFAULT_JOB_DIR",
    "DEFAULT_KEY_RELEASE_RA_TLS_PORT",
    "DEFAULT_KEY_RELEASE_TLS_CA_PATH",
    "DEFAULT_KEY_RELEASE_TLS_CERT_PATH",
    "DEFAULT_KEY_RELEASE_TLS_KEY_PATH",
    "ORCHESTRATOR_SERVICE",
    "PHALA_APP_COMPOSE_ENVELOPE_KEYS",
    "PHALA_DEFAULT_FEATURES",
    "PHALA_PRE_LAUNCH_SCRIPT_PATH",
    "RA_TLS_HOST_ENV",
    "RA_TLS_PORT_ENV",
    "REPO_ROOT",
    "ComposeGenerationError",
    "app_compose_hash",
    "assert_digest_pinned",
    "build_orchestrator_service",
    "generate_app_compose",
    "golden_task_image_digests",
    "load_golden_manifest",
    "phala_pre_launch_script",
    "render_app_compose",
    "render_app_compose_bytes",
    "write_app_compose",
]
