"""Docker Compose challenge orchestrator (Compose-only target path).

Replaces Swarm service create/update/remove for single-host master deployments.
All mutations are project-scoped: every command is ``docker compose -p <project>
-f <file> ...`` and every container mutation is rejected unless
``com.docker.compose.project`` matches the configured project name.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from base.master.docker_orchestrator import (
    DEFAULT_API_VERSION,
    ChallengeRuntime,
    ChallengeSpec,
    DockerOrchestrationError,
)
from base.supervisor.image_ref import extract_digest, parse_image_reference

logger = logging.getLogger(__name__)

_PINNED_IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_SAFE_SLUG_RE = re.compile(r"[^a-zA-Z0-9_.-]+")
_SERVICE_NAME_RE = re.compile(r"^challenge-[a-z0-9][a-z0-9_.-]*$")


#: Install-time compose interpolation keys sealed into
#: ``/run/base/compose/.env`` by ``install-master.sh``. Dynamic
#: ``docker compose up`` must carry these so the multi-service base file can
#: interpolate without host shell exports (VAL-COMPOSE-008/025).
_SEALED_COMPOSE_ENV_KEYS: frozenset[str] = frozenset(
    {
        "COMPOSE_PROJECT_NAME",
        "BASE_MASTER_IMAGE_REPOSITORY",
        "BASE_MASTER_IMAGE_DIGEST",
        # Historical dual-run pins (optional; not required after embed drop).
        "PRISM_IMAGE_REPOSITORY",
        "PRISM_IMAGE_DIGEST",
        "POSTGRES_IMAGE_REPOSITORY",
        "POSTGRES_IMAGE_DIGEST",
        "BASE_MASTER_CONFIG",
        "BASE_ADMIN_TOKEN_FILE",
        "BASE_POSTGRES_PASSWORD_FILE",
        "PRISM_SHARED_TOKEN_FILE",
        "BASE_MASTER_HOST_PORT",
        "BASE_DOCKER_GID",
        "BASE_COMPOSE_FILE",
        "BASE_POSTGRES_DB",
        "BASE_POSTGRES_USER",
        "BASE_MASTER_REGISTRY_RECONCILE_INTERVAL_SECONDS",
        "BASE_MASTER_CHALLENGE_WATCHER_INTERVAL_SECONDS",
    }
)


def resolve_compose_env_file(
    compose_file: Path,
    env_file: str | Path | None = None,
) -> Path | None:
    """Return the sealed compose env file if present, else None.

    Prefer an explicit path, then ``<compose_file_dir>/.env`` (the install
    mount target), then ``BASE_COMPOSE_ENV_FILE`` / ``COMPOSE_ENV_FILE``.
    """

    candidates: list[Path] = []
    if env_file is not None and str(env_file).strip():
        candidates.append(Path(env_file))
    candidates.append(Path(compose_file).parent / ".env")
    for key in ("BASE_COMPOSE_ENV_FILE", "COMPOSE_ENV_FILE"):
        raw = os.environ.get(key)
        if raw:
            candidates.append(Path(raw))
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def load_compose_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a sealed compose env file (no export shell)."""

    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DockerOrchestrationError(
            f"cannot read compose env file {path}: {exc}"
        ) from exc
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


@dataclass(frozen=True)
class ComposeRunner:
    """Thin wrapper around ``docker compose`` CLI with fixed project boundary."""

    project_name: str
    compose_file: Path
    docker_bin: str = "docker"
    work_dir: Path | None = None
    timeout_seconds: float = 300.0
    env_file: Path | None = None

    def __post_init__(self) -> None:
        if not self.project_name or not self.project_name.strip():
            raise DockerOrchestrationError("compose project name is required")
        if any(c.isspace() for c in self.project_name):
            raise DockerOrchestrationError(
                "compose project name must not contain spaces"
            )
        # Lazy file existence check: construction must not crash the process if
        # the compose artifact is not yet mounted; mutations validate on use.

    def _require_compose_file(self) -> Path:
        path = Path(self.compose_file)
        if not path.is_file():
            raise DockerOrchestrationError(f"compose file not found: {path}")
        return path

    def _resolved_env_file(self) -> Path | None:
        return resolve_compose_env_file(Path(self.compose_file), self.env_file)

    def _compose_base_cmd(self, compose_file: Path) -> list[str]:
        cmd = [
            self.docker_bin,
            "compose",
            "-p",
            self.project_name,
        ]
        env_file = self._resolved_env_file()
        if env_file is not None:
            cmd.extend(["--env-file", str(env_file)])
        cmd.extend(["-f", str(compose_file)])
        return cmd

    def _merged_env(self, env: Mapping[str, str] | None = None) -> dict[str, str]:
        """Process env for compose, with sealed install pins layered in."""

        merged = dict(os.environ)
        sealed = self._resolved_env_file()
        if sealed is not None:
            for key, value in load_compose_env_file(sealed).items():
                # Process-env wins over sealed file so pods can override pins at
                # runtime for advanced operators; install pins fill gaps.
                merged.setdefault(key, value)
        if env:
            merged.update(env)
        # Force project identity even if COMPOSE_PROJECT_NAME differs in the env.
        merged["COMPOSE_PROJECT_NAME"] = self.project_name
        return merged

    def run(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a project-scoped compose command."""

        compose_file = self._require_compose_file()
        cmd = [*self._compose_base_cmd(compose_file), *args]
        merged = self._merged_env(env)
        logger.debug("compose runner: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout if timeout is not None else self.timeout_seconds,
            cwd=str(self.work_dir) if self.work_dir is not None else None,
            env=merged,
        )
        if check and completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise DockerOrchestrationError(
                f"compose command failed ({completed.returncode}): "
                f"{' '.join(cmd[4:])}: {stderr[:500]}"
            )
        return completed


@dataclass
class ComposeChallengeOrchestrator:
    """Long-lived challenge services via Docker Compose (no Swarm calls)."""

    project_name: str
    compose_file: str | Path
    docker_bin: str = "docker"
    override_dir: str | Path = "/var/lib/base/compose-overrides"
    env_file: str | Path | None = None
    request_timeout_seconds: float = 5.0
    health_retries: int = 12
    health_retry_delay_seconds: float = 2.0
    pull_timeout_seconds: float = 300.0
    recreate_timeout_seconds: float = 300.0
    command_timeout_seconds: float = 120.0
    _runtime: dict[str, ChallengeRuntime] = field(default_factory=dict, init=False)
    _runner: ComposeRunner | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.compose_file = Path(self.compose_file)
        self.override_dir = Path(self.override_dir)
        compose_path = self.compose_path
        override_dir = self.override_dir_path
        resolved_env = resolve_compose_env_file(compose_path, self.env_file)
        self.env_file = resolved_env
        try:
            override_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Host path may be read-only until volume is ready; mutations recreate.
            logger.warning(
                "compose orchestrator: cannot create override dir %s (%s)",
                override_dir,
                exc,
            )
        self._runner = ComposeRunner(
            project_name=self.project_name,
            compose_file=compose_path,
            docker_bin=self.docker_bin,
            work_dir=compose_path.parent if compose_path.exists() else None,
            timeout_seconds=self.command_timeout_seconds,
            env_file=resolved_env,
        )

    @property
    def compose_path(self) -> Path:
        """Normalized Compose file path (fields accept ``str | Path``)."""

        return Path(self.compose_file)

    @property
    def override_dir_path(self) -> Path:
        """Normalized override directory (fields accept ``str | Path``)."""

        return Path(self.override_dir)

    @property
    def runner(self) -> ComposeRunner:
        assert self._runner is not None
        return self._runner

    @property
    def runtime(self) -> dict[str, ChallengeRuntime]:
        return dict(self._runtime)

    def service_name(self, slug: str) -> str:
        safe = _SAFE_SLUG_RE.sub("-", slug.strip()).strip("-.").lower()
        if not safe:
            raise DockerOrchestrationError("Challenge slug cannot be empty")
        name = f"challenge-{safe}"
        if not _SERVICE_NAME_RE.match(name):
            raise DockerOrchestrationError(f"Invalid challenge service name: {name}")
        return name

    def start_challenge(
        self, spec: ChallengeSpec, *, recreate: bool = False
    ) -> ChallengeRuntime:
        """Ensure the challenge Compose service is up on an immutable image."""

        self._require_pinned_image(spec.image)
        service = self.service_name(spec.slug)
        self._write_service_override(service, spec)
        if recreate:
            self._compose_up(service, force_recreate=True, pull=True)
        else:
            current = self.service_image(spec.slug)
            if current is not None and extract_digest(current) == extract_digest(
                spec.image
            ):
                # Already serving desired digest: adopt without recreate.
                health, version = self.wait_until_ready(spec)
                runtime = self._runtime_from(spec, health, version)
                self._runtime[spec.slug] = runtime
                logger.info(
                    "compose orchestrator: adopted %s already-current image=%s",
                    service,
                    current,
                )
                return runtime
            self._compose_up(service, force_recreate=False, pull=True)

        health, version = self.wait_until_ready(spec)
        runtime = self._runtime_from(spec, health, version)
        self._runtime[spec.slug] = runtime
        return runtime

    def stop_challenge(self, slug: str, *, remove: bool = False) -> None:
        """Stop (and optionally remove) a project-scoped challenge service.

        Static compose-topology services (``base.compose.lifecycle=static``)
        are never torn down by reconcile orphan cleanup so the packaged
        ``challenge-prism`` install survives registry gaps.
        """

        service = self.service_name(slug)
        self._assert_service_in_project(service)
        if self._is_static_lifecycle(service):
            logger.info(
                "compose orchestrator: refusing to stop static lifecycle "
                "service %s (slug=%s)",
                service,
                slug,
            )
            return
        del remove  # reconciler always fully tears down managed services
        # Project-scoped tear-down: stop + remove container, keep named volumes
        # for reactivation (VAL-COMPOSE-027/029).
        self.runner.run(["rm", "-sf", service], check=False)
        # Also remove any project-labeled leftovers that compose rm missed
        # (prior-generation orphans without a live compose service).
        self._remove_project_labeled_challenge_containers(slug)
        self._runtime.pop(slug, None)
        # Drop generated override so a later reactivation regenerates intentionally.
        override = self._override_path(service)
        if override.is_file():
            try:
                override.unlink()
            except OSError:
                logger.warning(
                    "compose orchestrator: could not remove override %s", override
                )

    def _remove_project_labeled_challenge_containers(self, slug: str) -> None:
        """Force-remove running containers for slug labeled against this project."""

        completed = subprocess.run(
            [
                self.docker_bin,
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={self.project_name}",
                "--filter",
                f"label=base.challenge.slug={slug}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )
        if completed.returncode != 0:
            return
        ids = [
            line.strip()
            for line in (completed.stdout or "").splitlines()
            if line.strip()
        ]
        if not ids:
            return
        subprocess.run(
            [self.docker_bin, "rm", "-f", *ids],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )

    def restart_challenge(self, spec: ChallengeSpec) -> ChallengeRuntime:
        """Recreate the challenge service on the desired immutable image."""

        return self.start_challenge(spec, recreate=True)

    def pull_image(self, image: str) -> None:
        """Pull an already-pinned image reference (no mutable tags)."""

        self._require_pinned_image(image)
        completed = subprocess.run(
            [self.docker_bin, "image", "pull", image],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.pull_timeout_seconds,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise DockerOrchestrationError(
                f"image pull failed for pinned ref: {stderr[:400]}"
            )

    def pull_challenge(self, spec: ChallengeSpec) -> None:
        self.pull_image(spec.image)

    def service_image(self, slug: str) -> str | None:
        """Return the configured image for the challenge service, or None if absent."""

        service = self.service_name(slug)
        try:
            inspect = self._inspect_service_container(service)
        except DockerOrchestrationError as exc:
            message = str(exc).lower()
            if "not found" in message or "absent" in message or "no such" in message:
                return None
            raise
        if inspect is None:
            return None
        config = inspect.get("Config") or {}
        image = config.get("Image") or inspect.get("Image")
        return str(image) if image else None

    def list_running_challenge_slugs(self) -> frozenset[str]:
        """Discover challenge services inside THIS compose project only.

        Combines ``docker compose ps`` with a project-label docker inspect so
        cross-restart orphans created by a prior generation without a live
        compose service entry (VAL-COMPOSE-028) are still stopped.
        """

        completed = self.runner.run(
            ["ps", "--format", "json", "--status", "running"],
            check=False,
        )
        if completed.returncode != 0:
            raise DockerOrchestrationError(
                f"compose ps failed: {(completed.stderr or '').strip()[:300]}"
            )
        slugs: set[str] = set()
        for entry in _parse_compose_ps_json(completed.stdout or ""):
            service = str(entry.get("Service") or entry.get("Name") or "")
            if not service.startswith("challenge-"):
                continue
            project = str(
                entry.get("Project")
                or (entry.get("Labels") or {}).get("com.docker.compose.project")
                or ""
            )
            if project and project != self.project_name:
                logger.warning(
                    "compose orchestrator: ignoring foreign project "
                    "service %s (project=%s)",
                    service,
                    project,
                )
                continue
            labels = entry.get("Labels") or {}
            if isinstance(labels, str):
                labels = _parse_label_string(labels)
            # Static topology services (packaged challenge-prism) are reported
            # for adoption when ACTIVE, but stop_challenge refuses to remove
            # them so accidental registry absence never uninstalls them.
            slug = labels.get("base.challenge.slug") or service.removeprefix(
                "challenge-"
            )
            if slug:
                slugs.add(str(slug))
        # Supplement with label discovery for project-scoped orphans that
        # compose ps may not surface (prior-generation docker run leftovers).
        try:
            labeled = self._list_project_labeled_challenge_slugs()
        except Exception:
            logger.exception("compose orchestrator: project-label orphan scan failed")
            labeled = frozenset()
        return frozenset(slugs | set(labeled))

    def _list_project_labeled_challenge_slugs(self) -> frozenset[str]:
        """Slugs for running containers labeled with this compose project."""

        completed = subprocess.run(
            [
                self.docker_bin,
                "ps",
                "--filter",
                f"label=com.docker.compose.project={self.project_name}",
                "--filter",
                "label=base.component=challenge",
                "--format",
                "{{.ID}}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )
        if completed.returncode != 0:
            return frozenset()
        slugs: set[str] = set()
        for line in (completed.stdout or "").splitlines():
            cid = line.strip()
            if not cid:
                continue
            inspect = subprocess.run(
                [
                    self.docker_bin,
                    "inspect",
                    "--format",
                    '{{ index .Config.Labels "base.challenge.slug" }}'
                    '|{{ index .Config.Labels "com.docker.compose.service" }}',
                    cid,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout_seconds,
            )
            if inspect.returncode != 0:
                continue
            raw = (inspect.stdout or "").strip()
            if "|" not in raw:
                continue
            slug_label, service = raw.split("|", 1)
            slug = (slug_label or "").strip()
            if not slug and service.startswith("challenge-"):
                slug = service.removeprefix("challenge-")
            if slug:
                slugs.add(slug)
        return frozenset(slugs)

    def wait_until_ready(
        self, spec: ChallengeSpec
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Wait for challenge /health and /version contracts."""

        last_error: Exception | None = None
        for _attempt in range(self.health_retries):
            try:
                health = self._get_json(f"{spec.internal_base_url}/health")
                self._validate_health(spec, health)
                version = self._get_json(f"{spec.internal_base_url}/version")
                self._validate_version(spec, version)
                return health, version
            except Exception as exc:  # readiness retry path
                last_error = exc
                time.sleep(self.health_retry_delay_seconds)
        raise DockerOrchestrationError(
            f"Challenge {spec.slug!r} failed health/version checks"
        ) from last_error

    def belongs_to_project(self, container_id: str) -> bool:
        """True when the container is labeled with this compose project."""

        completed = subprocess.run(
            [
                self.docker_bin,
                "inspect",
                "--format",
                '{{ index .Config.Labels "com.docker.compose.project" }}',
                container_id,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )
        if completed.returncode != 0:
            return False
        return (completed.stdout or "").strip() == self.project_name

    def _compose_up(self, service: str, *, force_recreate: bool, pull: bool) -> None:
        """Bring up one challenge service under the project boundary.

        Static topology services (declared in the base master compose file)
        use base + image-pin override with the sealed install env file so
        shared service pins (postgres/master digests, password paths) still
        interpolate. Fully managed dynamic overrides are self-contained: they
        declare image, networks (by absolute project name), volumes, and labels
        and do not re-require host install vars for `compose up`.
        """

        override = self._override_path(service)
        static = self._service_defined_in_base_compose(service)
        if static or not override.is_file():
            # Base file path requires sealed install pins.
            cmd = self.runner._compose_base_cmd(self.compose_path)
            if override.is_file():
                cmd.extend(["-f", str(override)])
        else:
            # Dynamic managed service: self-contained override only.
            # Still attach --env-file when present so project identity / future
            # pins remain consistent; overrides themselves do not interpolate
            # install-time secrets.
            cmd = [
                self.docker_bin,
                "compose",
                "-p",
                self.project_name,
            ]
            env_file = self.runner._resolved_env_file()
            if env_file is not None:
                cmd.extend(["--env-file", str(env_file)])
            cmd.extend(["-f", str(override)])
        cmd.append("up")
        # Never pass --remove-orphans: a self-contained managed override only
        # defines the challenge service, so Compose would treat master/postgres/
        # prism as orphans and SIGKILL them (exit 137). Orphan challenge cleanup
        # is owned by the registry reconciler (stop_challenge), not compose up.
        cmd.extend(["-d", "--no-deps"])
        if force_recreate:
            cmd.append("--force-recreate")
        if pull:
            cmd.extend(["--pull", "missing"])
        cmd.append(service)
        env = self.runner._merged_env()
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.recreate_timeout_seconds,
            cwd=str(self.compose_path.parent),
            env=env,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise DockerOrchestrationError(
                f"compose up failed for {service}: {stderr[:500]}"
            )

    def _write_service_override(self, service: str, spec: ChallengeSpec) -> Path:
        """Write a per-challenge Compose fragment under ``override_dir``.

        Services already declared in the base master compose file (for example
        packaged ``challenge-prism``) receive an image-pin override only so a
        watcher digest roll does not reinvent topology. Dynamic registry
        challenges write a full long-lived service definition: image, ``app``
        network, isolated data volume, labels, and restart policy. Compose
        merges base + override under the project boundary.
        """

        image = spec.image
        self._require_pinned_image(image)
        path = self._override_path(service)
        path.parent.mkdir(parents=True, exist_ok=True)
        slug = service.removeprefix("challenge-")
        if self._service_defined_in_base_compose(service):
            content = (
                f"# generated by base compose orchestrator — do not hand-edit\n"
                f"services:\n"
                f"  {service}:\n"
                f"    image: {image}\n"
                f"    labels:\n"
                f"      base.component: challenge\n"
                f"      base.challenge.slug: {slug}\n"
                f"      base.managed_by: master-watcher\n"
            )
            path.write_text(content, encoding="utf-8")
            path.chmod(0o600)
            return path

        volume_key = f"{service.replace('.', '_')}_data"
        volume_name = f"{self.project_name}_{volume_key}"
        # Attach to the already-created master app network by absolute name so
        # this override is self-contained (no COMPOSE_PROJECT_NAME / image pin
        # interpolation required for `docker compose -f override up`).
        app_network_name = f"{self.project_name}_app"
        env = dict(spec.env or {})
        # Combined Prism-style defaults when operators register a new challenge
        # without fleshing out env: keep the service startable and private.
        env.setdefault("PRISM_COMBINED_MODE", "true")
        env.setdefault("PRISM_DOCKER_ENABLED", "false")
        env.setdefault("PRISM_WORKER_PLANE__ENABLED", "false")
        env.setdefault("BASE_CHALLENGE_SLUG", slug)
        env.setdefault(
            "PRISM_DATABASE_URL", "sqlite+aiosqlite:////data/challenge.sqlite3"
        )
        env.setdefault(
            "CHALLENGE_DATABASE_URL", "sqlite+aiosqlite:////data/challenge.sqlite3"
        )
        env_lines = [
            f"      {key}: {self._yaml_quote(str(value))}"
            for key, value in sorted(env.items())
        ]
        env_block = "\n".join(env_lines) if env_lines else "      {}"
        content = (
            f"# generated by base compose orchestrator — do not hand-edit\n"
            f"# Self-contained managed challenge: no host install env required.\n"
            f"name: {self.project_name}\n"
            f"services:\n"
            f"  {service}:\n"
            f"    image: {image}\n"
            f"    restart: unless-stopped\n"
            f"    init: true\n"
            f"    networks:\n"
            f"      - app\n"
            f"    volumes:\n"
            f"      - type: volume\n"
            f"        source: {volume_key}\n"
            f"        target: /data\n"
            f"    environment:\n"
            f"{env_block}\n"
            f"    labels:\n"
            f"      base.component: challenge\n"
            f"      base.challenge.slug: {slug}\n"
            f"      base.managed_by: master-watcher\n"
            f"      base.compose.lifecycle: managed\n"
            f"      com.docker.compose.project: {self.project_name}\n"
            f"networks:\n"
            f"  app:\n"
            f"    name: {app_network_name}\n"
            f"    external: true\n"
            f"volumes:\n"
            f"  {volume_key}:\n"
            f"    name: {volume_name}\n"
            f"    labels:\n"
            f"      base.volume.kind: challenge-state\n"
            f"      base.challenge.slug: {slug}\n"
        )
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return path

    def _service_defined_in_base_compose(self, service: str) -> bool:
        """True when the base compose file already declares ``service``."""

        compose_file = Path(self.compose_file)
        if not compose_file.is_file():
            return False
        try:
            text = compose_file.read_text(encoding="utf-8")
        except OSError:
            return False
        # Lightweight marker: avoid a full compose dependency for this check.
        # Matches top-level YAML keys such as ``  challenge-prism:``.
        return bool(re.search(rf"(?m)^[ \t]{{2}}{re.escape(service)}:\s*$", text))

    @staticmethod
    def _yaml_quote(value: str) -> str:
        """Quote a scalar safely for generated YAML without embedding secrets."""

        if value == "":
            return '""'
        if any(
            ch in value for ch in (":", "#", "{", "}", "[", "]", ",", '"', "'", "\n")
        ):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return value

    def _override_path(self, service: str) -> Path:
        return self.override_dir_path / f"{service}.override.yml"

    def _is_static_lifecycle(self, service: str) -> bool:
        inspect = self._inspect_service_container(service)
        if inspect is None:
            return False
        labels = (inspect.get("Config") or {}).get("Labels") or {}
        return labels.get("base.compose.lifecycle") == "static"

    def _assert_service_in_project(self, service: str) -> None:
        inspect = self._inspect_service_container(service)
        if inspect is None:
            return
        labels = (inspect.get("Config") or {}).get("Labels") or {}
        project = labels.get("com.docker.compose.project")
        if project is not None and project != self.project_name:
            raise DockerOrchestrationError(
                f"refusing to mutate container outside project "
                f"{self.project_name!r} (found project={project!r})"
            )

    def _inspect_service_container(self, service: str) -> dict[str, Any] | None:
        completed = self.runner.run(
            ["ps", "-a", "--format", "json", service],
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").lower()
            if "no such service" in stderr or "not found" in stderr:
                return None
            raise DockerOrchestrationError(
                f"compose ps failed for {service}: {(completed.stderr or '')[:300]}"
            )
        entries = _parse_compose_ps_json(completed.stdout or "")
        if not entries:
            return None
        entry = entries[0]
        container_id = entry.get("ID") or entry.get("Container") or entry.get("Name")
        if not container_id:
            return None
        inspect = subprocess.run(
            [self.docker_bin, "inspect", str(container_id)],
            capture_output=True,
            text=True,
            check=False,
            timeout=self.command_timeout_seconds,
        )
        if inspect.returncode != 0:
            message = (inspect.stderr or "").strip().lower()
            if "no such object" in message or "not found" in message:
                return None
            raise DockerOrchestrationError(
                f"docker inspect transient failure for {service}: {message[:200]}"
            )
        data = json.loads(inspect.stdout or "[]")
        if not data:
            return None
        labels = (data[0].get("Config") or {}).get("Labels") or {}
        project = labels.get("com.docker.compose.project")
        if project is not None and project != self.project_name:
            raise DockerOrchestrationError(
                f"container for {service} belongs to foreign project {project!r}"
            )
        return data[0]

    def _runtime_from(
        self,
        spec: ChallengeSpec,
        health: dict[str, Any],
        version: dict[str, Any],
    ) -> ChallengeRuntime:
        inspect = self._inspect_service_container(self.service_name(spec.slug))
        container_id = (inspect or {}).get("Id") or ""
        return ChallengeRuntime(
            slug=spec.slug,
            image=spec.image,
            container_id=container_id,
            container_name=spec.container_name,
            internal_base_url=spec.internal_base_url,
            sqlite_volume_name=spec.sqlite_volume_name,
            health=health,
            version=version,
        )

    def _require_pinned_image(self, image: str) -> None:
        if not image or not _PINNED_IMAGE_RE.match(image):
            raise DockerOrchestrationError(
                "watcher refuses non-immutable image reference "
                f"(expected repository@sha256:<64 hex>, got {image!r})"
            )
        # Reject "repo:tag@sha256:" only if tag is empty — parse validates structure.
        parsed = parse_image_reference(image)
        if parsed.digest is None:
            raise DockerOrchestrationError(
                f"watcher refuses unpinned image reference: {image!r}"
            )

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                payload = response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise DockerOrchestrationError(f"HTTP check failed for {url}") from exc
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DockerOrchestrationError(f"Invalid JSON response from {url}") from exc
        if not isinstance(decoded, dict):
            raise DockerOrchestrationError(f"Expected JSON object from {url}")
        return decoded

    def _validate_health(self, spec: ChallengeSpec, health: dict[str, Any]) -> None:
        if (
            health.get("status") not in {"ok", "degraded"}
            or health.get("ready", True) is not True
        ):
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} health is not ready"
            )
        response_slug = health.get("slug")
        if response_slug is not None and response_slug != spec.slug:
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} returned mismatched health slug"
            )

    def _validate_version(self, spec: ChallengeSpec, version: dict[str, Any]) -> None:
        api_version = version.get("api_version")
        if api_version is not None and str(api_version) != (
            spec.expected_api_version or DEFAULT_API_VERSION
        ):
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} version/capability mismatch: "
                f"api_version={api_version!r}"
            )
        challenge_id = version.get("challenge") or version.get("slug")
        if challenge_id is not None and str(challenge_id) != spec.slug:
            raise DockerOrchestrationError(
                f"Challenge {spec.slug!r} version identity mismatch"
            )
        if spec.required_capabilities:
            caps = version.get("capabilities") or version.get("capability") or []
            if isinstance(caps, str):
                caps = [caps]
            missing = [
                cap for cap in spec.required_capabilities if cap not in set(caps)
            ]
            if missing:
                raise DockerOrchestrationError(
                    f"Challenge {spec.slug!r} missing capabilities {missing}"
                )


def _parse_compose_ps_json(stdout: str) -> list[dict[str, Any]]:
    """Parse ``docker compose ps --format json`` (array or NDJSON)."""

    text = (stdout or "").strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        entries: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                entries.append(item)
        return entries
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    if isinstance(loaded, dict):
        return [loaded]
    return []


def _parse_label_string(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        labels[key.strip()] = value.strip()
    return labels


__all__ = [
    "ComposeChallengeOrchestrator",
    "ComposeRunner",
    "load_compose_env_file",
    "resolve_compose_env_file",
    "_SEALED_COMPOSE_ENV_KEYS",
]
