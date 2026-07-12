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


@dataclass(frozen=True)
class ComposeRunner:
    """Thin wrapper around ``docker compose`` CLI with fixed project boundary."""

    project_name: str
    compose_file: Path
    docker_bin: str = "docker"
    work_dir: Path | None = None
    timeout_seconds: float = 300.0

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
        cmd = [
            self.docker_bin,
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(compose_file),
            *args,
        ]
        merged = dict(os.environ)
        if env:
            merged.update(env)
        # Force project identity even if COMPOSE_PROJECT_NAME differs in the env.
        merged["COMPOSE_PROJECT_NAME"] = self.project_name
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
        try:
            self.override_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Host path may be read-only until volume is ready; mutations recreate.
            logger.warning(
                "compose orchestrator: cannot create override dir %s (%s)",
                self.override_dir,
                exc,
            )
        self._runner = ComposeRunner(
            project_name=self.project_name,
            compose_file=self.compose_file,
            docker_bin=self.docker_bin,
            work_dir=self.compose_file.parent if self.compose_file.exists() else None,
            timeout_seconds=self.command_timeout_seconds,
        )

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
        self._write_service_override(service, spec.image)
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
        """Stop (and optionally remove) a project-scoped challenge service."""

        service = self.service_name(slug)
        self._assert_service_in_project(service)
        args = ["stop", service]
        if remove:
            args = ["rm", "-sf", service]
        self.runner.run(args, check=False)
        self._runtime.pop(slug, None)

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
        """Discover challenge services inside THIS compose project only."""

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
            # Static lifecycle services are adopted but not auto-stopped by
            # orphan cleanup unless the registry marks them non-ACTIVE and the
            # operator later opts managed lifecycle labels in.
            if labels.get("base.compose.lifecycle") == "static":
                slug = labels.get("base.challenge.slug") or service.removeprefix(
                    "challenge-"
                )
                slugs.add(str(slug))
                continue
            slug = labels.get("base.challenge.slug") or service.removeprefix(
                "challenge-"
            )
            if slug:
                slugs.add(str(slug))
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
        override = self._override_path(service)
        # Multi-file compose (base + image pin override) uses docker_bin directly
        # because ComposeRunner pins a single primary file only.
        cmd = [
            self.docker_bin,
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(self.compose_file),
        ]
        if override.is_file():
            cmd.extend(["-f", str(override)])
        cmd.append("up")
        cmd.extend(["-d", "--no-deps", "--remove-orphans"])
        if force_recreate:
            cmd.append("--force-recreate")
        if pull:
            cmd.extend(["--pull", "missing"])
        cmd.append(service)
        env = dict(os.environ)
        env["COMPOSE_PROJECT_NAME"] = self.project_name
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.recreate_timeout_seconds,
            cwd=str(self.compose_file.parent),
            env=env,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise DockerOrchestrationError(
                f"compose up failed for {service}: {stderr[:500]}"
            )

    def _write_service_override(self, service: str, image: str) -> Path:
        self._require_pinned_image(image)
        path = self._override_path(service)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal compose override: pin only this service image, no secrets.
        content = (
            f"# generated by base master watcher — do not hand-edit\n"
            f"services:\n"
            f"  {service}:\n"
            f"    image: {image}\n"
            f"    labels:\n"
            f"      base.component: challenge\n"
            f"      base.challenge.slug: {service.removeprefix('challenge-')}\n"
            f"      base.managed_by: master-watcher\n"
            f"      base.compose.lifecycle: managed\n"
        )
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        return path

    def _override_path(self, service: str) -> Path:
        return self.override_dir / f"{service}.override.yml"

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


__all__ = ["ComposeChallengeOrchestrator", "ComposeRunner"]
