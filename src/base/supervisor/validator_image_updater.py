"""Host-side Compose validator runtime image reconciler (digest-tracked :latest).

Independent validator installs run as agent-only Compose projects and must not
mount ``docker.sock`` into the long-lived agent. Auto-update therefore runs on
the **host** (systemd timer → this module / helper script):

1. Resolve ``ghcr.io/baseintelligence/base-validator-runtime:latest`` to a
   ``sha256:<64-hex>`` digest (via
   :func:`base.supervisor.image_ref.resolve_remote_digest`).
2. Compare to ``BASE_VALIDATOR_IMAGE_DIGEST`` in the project ``.env`` (or the
   running container image when the env is incomplete).
3. On change: record last-known-good, atomically rewrite ``.env`` to a pure
   REPO + DIGEST pin (never bare ``:latest`` as the runtime selector), then
   ``docker compose pull`` + ``up -d --force-recreate --no-deps validator``.
4. On failure: restore the LKG pin and compose-up again; exponential backoff;
   once exhausted for a digest, skip until a *new* remote digest appears.

Mirrors :mod:`base.supervisor.image_updater` durability (hold, LKG, RetryPolicy,
exhausted-skip) without any Swarm ``docker service`` calls.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from base.supervisor.image_ref import (
    ImageReference,
    extract_digest,
    parse_image_reference,
    resolve_remote_digest,
)
from base.supervisor.retry import JitterSource, RetryPolicy, RetryState

logger = logging.getLogger(__name__)

DEFAULT_TRACK_IMAGE = "ghcr.io/baseintelligence/base-validator-runtime:latest"
DEFAULT_SERVICE_NAME = "validator"
DEFAULT_COMPOSE_FILE_NAME = "docker-compose.validator.yml"
DEFAULT_ENV_FILE_NAME = ".env"
DEFAULT_STATE_FILE_NAME = "image_update_state.json"
DEFAULT_COMMAND_TIMEOUT_SECONDS = 300.0
DEFAULT_VERIFY_TIMEOUT_SECONDS = 120.0
DEFAULT_VERIFY_POLL_SECONDS = 2.0

_PINNED_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DigestResolver = Callable[[ImageReference], str]
CommandRunner = Callable[[Sequence[str], float], "CommandResult"]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
WallClock = Callable[[], float]


class UpdaterPhase(StrEnum):
    """Durable rollout phases for a single Compose validator project."""

    IDLE = "idle"
    RESOLVING = "resolving"
    PULLING = "pulling"
    RECREATING = "recreating"
    VERIFYING = "verifying"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    BACKOFF = "backoff"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class CommandResult:
    """Minimal process result for injectable compose/docker runners."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class ValidatorImageUpdateState:
    """Durable per-project image-update bookkeeping (JSON on disk)."""

    desired_digest: str | None = None
    current_digest: str | None = None
    rollback_digest: str | None = None
    phase: UpdaterPhase = UpdaterPhase.IDLE
    attempts: int = 0
    next_eligible_at: float | None = None  # wall-clock unix seconds
    hold: bool = False
    last_error: str | None = None
    alerted: bool = False
    track_image: str = DEFAULT_TRACK_IMAGE

    def to_dict(self) -> dict[str, Any]:
        return {
            "desired_digest": self.desired_digest,
            "current_digest": self.current_digest,
            "rollback_digest": self.rollback_digest,
            "phase": str(self.phase),
            "attempts": self.attempts,
            "next_eligible_at": self.next_eligible_at,
            "hold": self.hold,
            "last_error": self.last_error,
            "alerted": self.alerted,
            "track_image": self.track_image,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ValidatorImageUpdateState:
        phase_raw = raw.get("phase", UpdaterPhase.IDLE)
        try:
            phase = UpdaterPhase(str(phase_raw))
        except ValueError:
            phase = UpdaterPhase.IDLE
        attempts = raw.get("attempts", 0)
        try:
            attempts_i = int(attempts) if attempts is not None else 0
        except (TypeError, ValueError):
            attempts_i = 0
        next_at = raw.get("next_eligible_at")
        next_f: float | None
        try:
            next_f = float(next_at) if next_at is not None else None
        except (TypeError, ValueError):
            next_f = None
        return cls(
            desired_digest=_as_optional_str(raw.get("desired_digest")),
            current_digest=_as_optional_str(raw.get("current_digest")),
            rollback_digest=_as_optional_str(raw.get("rollback_digest")),
            phase=phase,
            attempts=max(0, attempts_i),
            next_eligible_at=next_f,
            hold=bool(raw.get("hold", False)),
            last_error=_as_optional_str(raw.get("last_error")),
            alerted=bool(raw.get("alerted", False)),
            track_image=_as_optional_str(raw.get("track_image")) or DEFAULT_TRACK_IMAGE,
        )


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_digest(value: str | None) -> str | None:
    """Return canonical ``sha256:<64-hex>`` or None.

    Accepts bare 64-hex (as written in Compose ``.env``) or full digests.
    """
    if not value:
        return None
    text = value.strip().lower()
    if text.startswith("sha256:"):
        return text if _PINNED_DIGEST_RE.match(text) else None
    if _HEX64_RE.match(text):
        return f"sha256:{text}"
    return extract_digest(text)


def bare_digest_hex(digest: str) -> str:
    """Strip the ``sha256:`` prefix for Compose ``BASE_VALIDATOR_IMAGE_DIGEST``."""
    normalized = normalize_digest(digest)
    if normalized is None:
        raise ValueError(f"invalid digest {digest!r}")
    return normalized.removeprefix("sha256:")


def has_explicit_tag(image: str) -> bool:
    """True when the raw image string carries an explicit tag before any digest."""
    name, _, _ = image.partition("@")
    return ":" in name.rsplit("/", 1)[-1]


def repository_from_track_image(track_image: str) -> str:
    """Return ``registry/repository`` (no tag, no digest) for env rewrites."""
    ref = parse_image_reference(track_image)
    return f"{ref.registry}/{ref.repository}"


def pinned_runtime_image(repository: str, digest: str) -> str:
    """Production pin: ``repository@sha256:<hex>`` (never bare ``:latest``)."""
    repo = repository.strip()
    if not repo:
        raise ValueError("repository must be non-empty")
    if "@" in repo or ":" in repo.rsplit("/", 1)[-1]:
        # Strip tag/digest may already have been applied; keep path pure.
        repo = repository_from_track_image(
            repo if ":" in repo or "@" in repo else f"{repo}:latest"
        )
    norm = normalize_digest(digest)
    if norm is None:
        raise ValueError(f"digest must be sha256:<64-hex>, got {digest!r}")
    return f"{repo}@{norm}"


def assert_runtime_pin_policy(image: str) -> str:
    """Reject bare mutable tags as runtime selectors; return canonical pin."""
    text = image.strip()
    if not text:
        raise ValueError("runtime image must be non-empty")
    digest = extract_digest(text)
    if digest is None or not _PINNED_DIGEST_RE.match(digest):
        raise ValueError(
            f"runtime image must be repository@sha256:<64-hex>, got {image!r} "
            "(bare :latest is never a compose runtime selector)"
        )
    # Reject pure tag-only selectors (no @digest).
    if "@" not in text:
        raise ValueError(
            f"runtime image must include @sha256 digest pin, got {image!r}"
        )
    name, _, _ = text.partition("@")
    # Optional ``repo:tag@sha256`` is allowed when present with digest.
    return pinned_runtime_image(
        repository_from_track_image(name if ":" in name else f"{name}:latest")
        if ":" in name
        else name,
        digest,
    )


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser (no export, no multiline)."""
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if _ENV_KEY_RE.match(key):
            result[key] = value
    return result


def read_dotenv(path: Path) -> dict[str, str]:
    return parse_dotenv(path.read_text(encoding="utf-8"))


def write_dotenv_atomic(
    path: Path,
    *,
    updates: Mapping[str, str],
    remove_keys: Sequence[str] = (),
) -> dict[str, str]:
    """Rewrite dotenv keys atomically (tmp + fsync + replace); mode 0600.

    Preserves unknown keys and blank/comment lines where possible by rebuild:
    known keys from ``updates`` replace first-seen values; remaining update keys
    are appended. ``remove_keys`` drops matching keys.
    """
    remove = {k for k in remove_keys}
    existing: dict[str, str] = {}
    if path.is_file():
        existing = read_dotenv(path)

    merged = dict(existing)
    for key in remove:
        merged.pop(key, None)
    for key, value in updates.items():
        if not _ENV_KEY_RE.match(key):
            raise ValueError(f"invalid env key {key!r}")
        merged[key] = value

    # Stable preferred key order for Core validator env.
    preferred = [
        "COMPOSE_PROJECT_NAME",
        "BASE_VALIDATOR_IMAGE_REPOSITORY",
        "BASE_VALIDATOR_IMAGE_DIGEST",
        "BASE_VALIDATOR_CONFIG",
        "BASE_VALIDATOR_PROTOCOL_IDENTITY",
        "BASE_VALIDATOR_BROKER_TOKEN",
        "BASE_VALIDATOR_TRACK_IMAGE",
        "BASE_VALIDATOR_IMAGE_UPDATE_HOLD",
    ]
    lines: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if key in merged:
            lines.append(f"{key}={merged[key]}")
            seen.add(key)
    for key in sorted(k for k in merged if k not in seen):
        lines.append(f"{key}={merged[key]}")
    body = "\n".join(lines) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib_suppress(OSError):
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        raise
    return merged


def contextlib_suppress(*exceptions: type[BaseException]):
    """Tiny local suppress helper (avoids importing contextlib at module top noise)."""
    from contextlib import suppress

    return suppress(*exceptions)


def load_state(path: Path) -> ValidatorImageUpdateState:
    if not path.is_file():
        return ValidatorImageUpdateState()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("validator-image-updater: unreadable state at %s", path)
        return ValidatorImageUpdateState()
    if not isinstance(raw, dict):
        return ValidatorImageUpdateState()
    return ValidatorImageUpdateState.from_dict(raw)


def save_state(path: Path, state: ValidatorImageUpdateState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    except Exception:
        with contextlib_suppress(OSError):
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        raise


def current_env_digest(env: Mapping[str, str]) -> str | None:
    return normalize_digest(env.get("BASE_VALIDATOR_IMAGE_DIGEST"))


def current_env_repository(env: Mapping[str, str]) -> str | None:
    repo = env.get("BASE_VALIDATOR_IMAGE_REPOSITORY", "").strip()
    return repo or None


def subprocess_command_runner(
    argv: Sequence[str], timeout_seconds: float
) -> CommandResult:
    # Compose prefers process env over --env-file; scrub IMAGE pin vars so a
    # host/systemd EnvironmentFile cannot freeze auto-update to a stale digest.
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "BASE_VALIDATOR_IMAGE_REPOSITORY",
            "BASE_VALIDATOR_IMAGE_DIGEST",
        }
    }
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=clean_env,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            returncode=124,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"timeout after {timeout_seconds}s: {exc}",
        )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


@dataclass
class ComposeValidatorImageUpdater:
    """One-shot reconciler for a single independent validator Compose project."""

    project_name: str
    compose_file: Path
    env_file: Path
    state_path: Path
    track_image: str = DEFAULT_TRACK_IMAGE
    service_name: str = DEFAULT_SERVICE_NAME
    hold: bool = False
    dry_run: bool = False
    docker_bin: str = "docker"
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS
    verify_timeout_seconds: float = DEFAULT_VERIFY_TIMEOUT_SECONDS
    verify_poll_seconds: float = DEFAULT_VERIFY_POLL_SECONDS
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    resolver: DigestResolver = field(default=resolve_remote_digest)
    runner: CommandRunner = field(default=subprocess_command_runner)
    # Monotonic clock for injected retry eligibility DAMAGES tests; we also keep
    # wall-clock for durable next_eligible_at (restart-safe).
    clock: Clock = field(default=time.monotonic)
    wall_clock: WallClock = field(default=time.time)
    sleep: Sleeper = field(default=time.sleep)
    jitter_source: JitterSource = field(default=random.random)
    # In-memory RetryState rehydrated from durable attempts / next_eligible_at.
    _retry_state: RetryState = field(default_factory=RetryState, init=False, repr=False)

    def run_once(self) -> str:
        """Execute one reconcile tick; return a short outcome token."""
        state = load_state(self.state_path)
        state.track_image = self.track_image
        if self.hold or state.hold or self._hold_from_env():
            state.phase = UpdaterPhase.IDLE
            logger.info(
                "validator-image-updater: skipped-held project %r", self.project_name
            )
            save_state(self.state_path, state)
            return "skipped-held"

        now_mono = self.clock()
        now_wall = self.wall_clock()
        self._rehydrate_retry(state, now_mono=now_mono, now_wall=now_wall)

        if not self._retry_state.is_eligible(now_mono):
            logger.debug(
                "validator-image-updater: project %r backing off (attempts=%d)",
                self.project_name,
                self._retry_state.attempts,
            )
            state.phase = UpdaterPhase.BACKOFF
            save_state(self.state_path, state)
            return "backoff"

        if not has_explicit_tag(self.track_image):
            logger.error(
                "validator-image-updater: rejecting untagged track image %r "
                "(production pin policy requires an explicit tag)",
                self.track_image,
            )
            state.phase = UpdaterPhase.IDLE
            state.last_error = "untagged track image"
            save_state(self.state_path, state)
            return "reject-untagged"

        state.phase = UpdaterPhase.RESOLVING
        save_state(self.state_path, state)

        reference = parse_image_reference(self.track_image)
        try:
            desired = self.resolver(reference)
        except Exception as exc:
            logger.warning(
                "validator-image-updater: digest resolution failed for %s: %s",
                reference.tagged,
                exc,
            )
            state.phase = UpdaterPhase.IDLE
            state.last_error = f"resolve failed: {exc}"
            save_state(self.state_path, state)
            return "resolve-failed"

        desired_norm = normalize_digest(desired)
        if desired_norm is None or not _PINNED_DIGEST_RE.match(desired_norm):
            logger.error(
                "validator-image-updater: non-sha256 digest %r for %s; refusing",
                desired,
                reference.tagged,
            )
            state.phase = UpdaterPhase.IDLE
            state.last_error = "invalid digest"
            save_state(self.state_path, state)
            return "invalid-digest"

        env = self._load_env()
        env_digest = current_env_digest(env)
        running_digest = self._inspect_running_digest()
        # No-op only when both the env pin and the running container (when
        # inspectable) already match the desired digest. A rewritten .env with
        # a lagging container must still force recreate.
        both_current = env_digest == desired_norm and (
            running_digest is None or running_digest == desired_norm
        )
        current = running_digest or env_digest

        # New desired digest ends a prior failure episode.
        if state.desired_digest != desired_norm:
            state.desired_digest = desired_norm
            self._retry_state.record_success()
            state.attempts = 0
            state.next_eligible_at = None
            state.alerted = False
            state.last_error = None

        if both_current:
            self._retry_state.record_success()
            state.current_digest = desired_norm
            state.phase = UpdaterPhase.IDLE
            state.attempts = 0
            state.next_eligible_at = None
            state.last_error = None
            state.alerted = False
            save_state(self.state_path, state)
            logger.debug(
                "validator-image-updater: project %r already at %s; no-op",
                self.project_name,
                desired_norm,
            )
            return "noop"

        if self._retry_state.is_exhausted(self.retry_policy):
            state.phase = UpdaterPhase.EXHAUSTED
            if not state.alerted:
                state.alerted = True
                logger.error(
                    "validator-image-updater: retries exhausted for project %r "
                    "desired %s; skipping until a new digest",
                    self.project_name,
                    desired_norm,
                )
            save_state(self.state_path, state)
            return "exhausted"

        if not current:
            logger.warning(
                "validator-image-updater: no current digest for project %r; "
                "treating apply as first pin",
                self.project_name,
            )

        # LKG before mutate.
        state.rollback_digest = current
        state.desired_digest = desired_norm
        repository = current_env_repository(env) or repository_from_track_image(
            self.track_image
        )
        # Never leave bare :latest as runtime.
        try:
            pin = pinned_runtime_image(repository, desired_norm)
            assert_runtime_pin_policy(pin)
        except ValueError as exc:
            state.phase = UpdaterPhase.IDLE
            state.last_error = str(exc)
            save_state(self.state_path, state)
            return "pin-policy-reject"

        if self.dry_run:
            logger.info(
                "validator-image-updater: dry-run project %r would "
                "update %s -> %s (%s)",
                self.project_name,
                current,
                desired_norm,
                pin,
            )
            state.phase = UpdaterPhase.IDLE
            save_state(self.state_path, state)
            return "dry-run"

        try:
            write_dotenv_atomic(
                self.env_file,
                updates={
                    "COMPOSE_PROJECT_NAME": self.project_name,
                    "BASE_VALIDATOR_IMAGE_REPOSITORY": repository,
                    "BASE_VALIDATOR_IMAGE_DIGEST": bare_digest_hex(desired_norm),
                    "BASE_VALIDATOR_TRACK_IMAGE": self.track_image,
                },
            )
        except OSError as exc:
            return self._fail(
                state,
                now_mono=now_mono,
                now_wall=now_wall,
                reason=f"env write failed: {exc}",
                rollback=False,
            )

        state.phase = UpdaterPhase.PULLING
        save_state(self.state_path, state)
        if not self._compose(["pull", self.service_name]):
            return self._fail(
                state,
                now_mono=now_mono,
                now_wall=now_wall,
                reason="compose pull failed",
                rollback=True,
                repository=repository,
            )

        state.phase = UpdaterPhase.RECREATING
        save_state(self.state_path, state)
        if not self._compose(
            [
                "up",
                "-d",
                "--force-recreate",
                "--no-deps",
                self.service_name,
            ]
        ):
            return self._fail(
                state,
                now_mono=now_mono,
                now_wall=now_wall,
                reason="compose recreate failed",
                rollback=True,
                repository=repository,
            )

        state.phase = UpdaterPhase.VERIFYING
        save_state(self.state_path, state)
        if not self._verify_running(desired_norm):
            return self._fail(
                state,
                now_mono=now_mono,
                now_wall=now_wall,
                reason="post-recreate verify failed",
                rollback=True,
                repository=repository,
            )

        self._retry_state.record_success()
        state.current_digest = desired_norm
        state.phase = UpdaterPhase.IDLE
        state.attempts = 0
        state.next_eligible_at = None
        state.last_error = None
        state.alerted = False
        save_state(self.state_path, state)
        logger.info(
            "validator-image-updater: project %r updated to %s",
            self.project_name,
            pin,
        )
        return "updated"

    # ------------------------------------------------------------------ helpers

    def _hold_from_env(self) -> bool:
        if not self.env_file.is_file():
            return False
        try:
            env = read_dotenv(self.env_file)
        except OSError:
            return False
        flag = env.get("BASE_VALIDATOR_IMAGE_UPDATE_HOLD", "").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    def _load_env(self) -> dict[str, str]:
        if not self.env_file.is_file():
            return {}
        try:
            return read_dotenv(self.env_file)
        except OSError:
            return {}

    def _compose(self, args: Sequence[str]) -> bool:
        argv = [
            self.docker_bin,
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(self.compose_file),
            "--env-file",
            str(self.env_file),
            *args,
        ]
        logger.info(
            "validator-image-updater: running %s",
            " ".join(shlex.quote(a) for a in argv),
        )
        result = self.runner(argv, self.command_timeout_seconds)
        if result.returncode != 0:
            logger.error(
                "validator-image-updater: command failed rc=%d stderr=%s",
                result.returncode,
                (result.stderr or "").strip()[:500],
            )
            return False
        return True

    def _inspect_running_digest(self) -> str | None:
        # Prefer compose ps image, fall back to docker inspect of container name.
        argv = [
            self.docker_bin,
            "compose",
            "-p",
            self.project_name,
            "-f",
            str(self.compose_file),
            "--env-file",
            str(self.env_file),
            "images",
            "--format",
            "{{.Repository}} {{.ID}}",
            self.service_name,
        ]
        result = self.runner(argv, min(self.command_timeout_seconds, 60.0))
        if result.returncode == 0:
            dig = extract_digest(result.stdout)
            if dig:
                return dig.lower()
        # docker inspect common container name pattern.
        for name in (
            f"{self.project_name}-{self.service_name}-1",
            f"{self.project_name}_{self.service_name}_1",
        ):
            inspect = self.runner(
                [
                    self.docker_bin,
                    "inspect",
                    "--format",
                    "{{.Config.Image}}",
                    name,
                ],
                min(self.command_timeout_seconds, 30.0),
            )
            if inspect.returncode == 0:
                dig = extract_digest(inspect.stdout)
                if dig:
                    return dig.lower()
        return None

    def _verify_running(self, desired_digest: str) -> bool:
        deadline = self.clock() + self.verify_timeout_seconds
        while True:
            running = self._is_service_running()
            current = self._inspect_running_digest() or current_env_digest(
                self._load_env()
            )
            if running and current == desired_digest:
                return True
            if self.clock() >= deadline:
                logger.error(
                    "validator-image-updater: verify timeout for project %r "
                    "(running=%s current=%s desired=%s)",
                    self.project_name,
                    running,
                    current,
                    desired_digest,
                )
                return False
            self.sleep(self.verify_poll_seconds)

    def _is_service_running(self) -> bool:
        for name in (
            f"{self.project_name}-{self.service_name}-1",
            f"{self.project_name}_{self.service_name}_1",
        ):
            result = self.runner(
                [
                    self.docker_bin,
                    "inspect",
                    "--format",
                    "{{.State.Running}}",
                    name,
                ],
                min(self.command_timeout_seconds, 30.0),
            )
            if result.returncode == 0 and result.stdout.strip().lower() == "true":
                return True
        # compose ps fallback
        result = self.runner(
            [
                self.docker_bin,
                "compose",
                "-p",
                self.project_name,
                "-f",
                str(self.compose_file),
                "--env-file",
                str(self.env_file),
                "ps",
                "--status",
                "running",
                "--services",
            ],
            min(self.command_timeout_seconds, 60.0),
        )
        if result.returncode == 0 and self.service_name in result.stdout.split():
            return True
        return False

    def _fail(
        self,
        state: ValidatorImageUpdateState,
        *,
        now_mono: float,
        now_wall: float,
        reason: str,
        rollback: bool,
        repository: str | None = None,
    ) -> str:
        logger.error(
            "validator-image-updater: project %r failed (%s)",
            self.project_name,
            reason,
        )
        if rollback and state.rollback_digest:
            state.phase = UpdaterPhase.ROLLING_BACK
            save_state(self.state_path, state)
            repo = repository or current_env_repository(self._load_env())
            if repo is None:
                repo = repository_from_track_image(self.track_image)
            try:
                write_dotenv_atomic(
                    self.env_file,
                    updates={
                        "COMPOSE_PROJECT_NAME": self.project_name,
                        "BASE_VALIDATOR_IMAGE_REPOSITORY": repo,
                        "BASE_VALIDATOR_IMAGE_DIGEST": bare_digest_hex(
                            state.rollback_digest
                        ),
                    },
                )
                self._compose(
                    [
                        "up",
                        "-d",
                        "--force-recreate",
                        "--no-deps",
                        self.service_name,
                    ]
                )
            except Exception:
                logger.exception(
                    "validator-image-updater: rollback apply raised for %r",
                    self.project_name,
                )

        self._retry_state.record_failure(
            now_mono,
            self.retry_policy,
            error=reason,
            jitter_source=self.jitter_source,
        )
        state.attempts = self._retry_state.attempts
        # Persist wall-clock next-eligible for restart safety.
        delay = max(0.0, self._retry_state.next_eligible_monotonic - now_mono)
        state.next_eligible_at = now_wall + delay
        state.last_error = reason
        state.phase = (
            UpdaterPhase.EXHAUSTED
            if self._retry_state.is_exhausted(self.retry_policy)
            else UpdaterPhase.BACKOFF
        )
        if state.phase == UpdaterPhase.EXHAUSTED and not state.alerted:
            state.alerted = True
            logger.error(
                "validator-image-updater: exhaustion alert for project %r after %d "
                "failures (last_error=%s)",
                self.project_name,
                state.attempts,
                reason,
            )
        save_state(self.state_path, state)
        return "failed"

    def _rehydrate_retry(
        self,
        state: ValidatorImageUpdateState,
        *,
        now_mono: float,
        now_wall: float,
    ) -> None:
        self._retry_state = RetryState(
            attempts=state.attempts,
            next_eligible_monotonic=0.0,
            last_error=state.last_error,
        )
        if state.next_eligible_at is not None:
            remaining = max(0.0, float(state.next_eligible_at) - now_wall)
            self._retry_state.next_eligible_monotonic = now_mono + remaining


def build_updater_from_env(
    *,
    env: Mapping[str, str] | None = None,
    resolver: DigestResolver | None = None,
    runner: CommandRunner | None = None,
    dry_run: bool | None = None,
) -> ComposeValidatorImageUpdater:
    """Construct an updater from host environment / dotenv-style variables."""
    environ = dict(os.environ if env is None else env)
    project = environ.get("COMPOSE_PROJECT_NAME", "").strip()
    if not project:
        raise ValueError("COMPOSE_PROJECT_NAME is required")

    artifacts = environ.get("BASE_VALIDATOR_ARTIFACTS_DIR", "").strip()
    compose = environ.get("BASE_VALIDATOR_COMPOSE_FILE", "").strip()
    env_file = environ.get("BASE_VALIDATOR_ENV_FILE", "").strip()
    state_file = environ.get("BASE_VALIDATOR_IMAGE_UPDATE_STATE", "").strip()
    track = environ.get("BASE_VALIDATOR_TRACK_IMAGE", "").strip() or DEFAULT_TRACK_IMAGE
    hold_flag = environ.get("BASE_VALIDATOR_IMAGE_UPDATE_HOLD", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dry = dry_run
    if dry is None:
        dry = environ.get(
            "BASE_VALIDATOR_IMAGE_UPDATE_DRY_RUN", ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if artifacts:
        artifacts_path = Path(artifacts)
        if not compose:
            compose = str(artifacts_path / DEFAULT_COMPOSE_FILE_NAME)
        if not env_file:
            env_file = str(artifacts_path / DEFAULT_ENV_FILE_NAME)
        if not state_file:
            state_file = str(artifacts_path / DEFAULT_STATE_FILE_NAME)
    if not compose or not env_file:
        raise ValueError(
            "BASE_VALIDATOR_COMPOSE_FILE and BASE_VALIDATOR_ENV_FILE "
            "(or BASE_VALIDATOR_ARTIFACTS_DIR) are required"
        )
    if not state_file:
        state_file = str(Path(env_file).resolve().parent / DEFAULT_STATE_FILE_NAME)

    return ComposeValidatorImageUpdater(
        project_name=project,
        compose_file=Path(compose),
        env_file=Path(env_file),
        state_path=Path(state_file),
        track_image=track,
        hold=hold_flag,
        dry_run=bool(dry),
        resolver=resolver if resolver is not None else resolve_remote_digest,
        runner=runner if runner is not None else subprocess_command_runner,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m base.supervisor.validator_image_updater [once]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Host-side Compose validator image reconciler (digest pins only)."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="once",
        choices=("once",),
        help="Run one reconcile tick (default: once)",
    )
    parser.add_argument(
        "--project-name", default=os.environ.get("COMPOSE_PROJECT_NAME")
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.environ.get("BASE_VALIDATOR_ARTIFACTS_DIR"),
    )
    parser.add_argument(
        "--compose-file",
        default=os.environ.get("BASE_VALIDATOR_COMPOSE_FILE"),
    )
    parser.add_argument(
        "--env-file",
        default=os.environ.get("BASE_VALIDATOR_ENV_FILE"),
    )
    parser.add_argument(
        "--state-file",
        default=os.environ.get("BASE_VALIDATOR_IMAGE_UPDATE_STATE"),
    )
    parser.add_argument(
        "--track-image",
        default=os.environ.get("BASE_VALIDATOR_TRACK_IMAGE", DEFAULT_TRACK_IMAGE),
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        default=False,
        help="Freeze auto-update for this tick",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Resolve digests and report without rewriting .env or compose",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_map = dict(os.environ)
    if args.project_name:
        env_map["COMPOSE_PROJECT_NAME"] = args.project_name
    if args.artifacts_dir:
        env_map["BASE_VALIDATOR_ARTIFACTS_DIR"] = args.artifacts_dir
    if args.compose_file:
        env_map["BASE_VALIDATOR_COMPOSE_FILE"] = args.compose_file
    if args.env_file:
        env_map["BASE_VALIDATOR_ENV_FILE"] = args.env_file
    if args.state_file:
        env_map["BASE_VALIDATOR_IMAGE_UPDATE_STATE"] = args.state_file
    if args.track_image:
        env_map["BASE_VALIDATOR_TRACK_IMAGE"] = args.track_image
    if args.hold:
        env_map["BASE_VALIDATOR_IMAGE_UPDATE_HOLD"] = "1"
    if args.dry_run:
        env_map["BASE_VALIDATOR_IMAGE_UPDATE_DRY_RUN"] = "1"

    try:
        updater = build_updater_from_env(env=env_map)
    except ValueError as exc:
        logger.error("validator-image-updater: config error: %s", exc)
        return 2

    outcome = updater.run_once()
    logger.info(
        "validator-image-updater: outcome=%s at %s",
        outcome,
        datetime.now(tz=UTC).isoformat(),
    )
    # Soft success for operational outcomes other than hard config failures.
    return 0


__all__ = [
    "DEFAULT_TRACK_IMAGE",
    "ComposeValidatorImageUpdater",
    "CommandResult",
    "UpdaterPhase",
    "ValidatorImageUpdateState",
    "assert_runtime_pin_policy",
    "bare_digest_hex",
    "build_updater_from_env",
    "current_env_digest",
    "has_explicit_tag",
    "load_state",
    "main",
    "normalize_digest",
    "parse_dotenv",
    "pinned_runtime_image",
    "read_dotenv",
    "repository_from_track_image",
    "save_state",
    "write_dotenv_atomic",
]


if __name__ == "__main__":
    raise SystemExit(main())
