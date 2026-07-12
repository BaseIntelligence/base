"""Master-resident Compose challenge watcher (digest-pinned, durable intent).

Runs inside ``base-master-validator`` as an in-process background lifespan.
It is **not** Swarm/Watchtower/systemd: every mutation goes through a
:class:`~base.master.compose_backend.ComposeChallengeOrchestrator` bound to one
Compose project.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from base.challenge_sdk.roles import Capability, Role, activate_role, role_contract
from base.config.settings import Settings
from base.master.docker_orchestrator import DockerOrchestrationError
from base.schemas.challenge import ChallengeStatus, ChallengeUpdate
from base.supervisor.alerts import ALERT_CHALLENGE_IMAGE_UPDATE_FAILED, build_alert_hook
from base.supervisor.image_ref import (
    ImageReference,
    extract_digest,
    parse_image_reference,
    resolve_remote_digest,
)
from base.supervisor.retry import JitterSource, RetryPolicy, RetryState
from base.supervisor.weight_submit import AlertEmitter, WeightsAlert

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

DEFAULT_WATCHER_INTERVAL_SECONDS = 60.0
DEFAULT_STATE_PATH = Path("/var/lib/base/challenge_watcher_state.json")
_PINNED_DIGEST_RE = __import__("re").compile(r"^sha256:[0-9a-f]{64}$")

DigestResolver = Callable[[ImageReference], str]
RegistryFactory = Callable[[], Any]
ControllerFactory = Callable[[Any], Any]
StateStoreFactory = Callable[[], "WatcherStateStore"]


class WatcherPhase(StrEnum):
    """Durable rollout phases per challenge slug."""

    IDLE = "idle"
    RESOLVING = "resolving"
    PULLING = "pulling"
    RECREATING = "recreating"
    VERIFYING = "verifying"
    COMMITTING = "committing"
    ROLLING_BACK = "rolling_back"
    BACKOFF = "backoff"
    EXHAUSTED = "exhausted"


# Phases that mean a rollout was interrupted and must re-verify on resume even
# when running/desired digests already match (VAL-CROSS-071 mid-rollout recovery).
_MID_ROLLOUT_PHASES = frozenset(
    {
        WatcherPhase.RESOLVING,
        WatcherPhase.PULLING,
        WatcherPhase.RECREATING,
        WatcherPhase.VERIFYING,
        WatcherPhase.COMMITTING,
        WatcherPhase.ROLLING_BACK,
    }
)


@dataclass
class ChallengeWatcherRecord:
    """One challenge's durable watcher intent + outcome bookkeeping.

    Backoff eligibility is always driven by wall-clock fields
    (``next_eligible_at`` / ``last_failure_at``). ``next_eligible_monotonic`` is
    process-local only after rehydration and is never treated as durable across
    restarts (VAL-COMPOSE-039 / VAL-CROSS-071).
    """

    slug: str
    desired_digest: str | None = None
    current_digest: str | None = None
    rollback_digest: str | None = None
    desired_image: str | None = None
    rollback_image: str | None = None
    phase: WatcherPhase = WatcherPhase.IDLE
    attempts: int = 0
    next_eligible_monotonic: float = 0.0
    next_eligible_at: float | None = None
    last_failure_at: float | None = None
    last_error: str | None = None
    last_result: str | None = None
    last_health_ok: bool | None = None
    last_version_ok: bool | None = None
    updated_at: str | None = None
    alerted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "desired_digest": self.desired_digest,
            "current_digest": self.current_digest,
            "rollback_digest": self.rollback_digest,
            "desired_image": self.desired_image,
            "rollback_image": self.rollback_image,
            "phase": self.phase.value,
            "attempts": self.attempts,
            # Do not persist process-local mono timestamps as authoritative.
            "next_eligible_at": self.next_eligible_at,
            "last_failure_at": self.last_failure_at,
            "last_error": self.last_error,
            "last_result": self.last_result,
            "last_health_ok": self.last_health_ok,
            "last_version_ok": self.last_version_ok,
            "updated_at": self.updated_at,
            "alerted": self.alerted,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ChallengeWatcherRecord:
        phase_raw = str(raw.get("phase") or WatcherPhase.IDLE.value)
        try:
            phase = WatcherPhase(phase_raw)
        except ValueError:
            phase = WatcherPhase.IDLE
        return cls(
            slug=str(raw["slug"]),
            desired_digest=_opt_str(raw.get("desired_digest")),
            current_digest=_opt_str(raw.get("current_digest")),
            rollback_digest=_opt_str(raw.get("rollback_digest")),
            desired_image=_opt_str(raw.get("desired_image")),
            rollback_image=_opt_str(raw.get("rollback_image")),
            phase=phase,
            attempts=int(raw.get("attempts") or 0),
            # Intentionally leave next_eligible_monotonic at 0 until rehydrate.
            next_eligible_monotonic=0.0,
            next_eligible_at=_opt_float(raw.get("next_eligible_at")),
            last_failure_at=_opt_float(raw.get("last_failure_at")),
            last_error=_opt_str(raw.get("last_error")),
            last_result=_opt_str(raw.get("last_result")),
            last_health_ok=_opt_bool(raw.get("last_health_ok")),
            last_version_ok=_opt_bool(raw.get("last_version_ok")),
            updated_at=_opt_str(raw.get("updated_at")),
            alerted=bool(raw.get("alerted") or False),
        )

    def as_retry_state(self) -> RetryState:
        return RetryState(
            attempts=self.attempts,
            next_eligible_monotonic=self.next_eligible_monotonic,
            last_error=self.last_error,
        )

    def apply_retry_state(
        self,
        state: RetryState,
        *,
        process_now: float | None = None,
        wall_now: float | None = None,
    ) -> None:
        self.attempts = state.attempts
        self.next_eligible_monotonic = state.next_eligible_monotonic
        self.last_error = state.last_error
        if state.attempts == 0 and state.next_eligible_monotonic == 0.0:
            self.next_eligible_at = None
            self.last_failure_at = None
            return
        if process_now is not None and wall_now is not None:
            remaining = max(0.0, state.next_eligible_monotonic - process_now)
            self.last_failure_at = wall_now
            self.next_eligible_at = wall_now + remaining


class WatcherStateStore:
    """Atomic JSON state for watcher intent (restart-safe)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, ChallengeWatcherRecord]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception(
                "challenge-watcher: failed to load state from %s", self.path
            )
            return {}
        challenges = raw.get("challenges") if isinstance(raw, dict) else None
        if not isinstance(challenges, dict):
            return {}
        loaded: dict[str, ChallengeWatcherRecord] = {}
        for slug, body in challenges.items():
            if not isinstance(body, dict):
                continue
            record = ChallengeWatcherRecord.from_dict({"slug": slug, **body})
            loaded[slug] = record
        return loaded

    def save(self, records: Mapping[str, ChallengeWatcherRecord]) -> None:
        payload = {
            "version": 1,
            "updated_at": datetime.now(UTC).isoformat(),
            "challenges": {
                slug: record.to_dict() for slug, record in sorted(records.items())
            },
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp.chmod(0o600)
        os.replace(tmp, self.path)


class ChallengeWatcher:
    """Digest-compare → pull → recreate → health/version → rollback loop."""

    def __init__(
        self,
        *,
        registry_factory: RegistryFactory,
        controller_factory: ControllerFactory,
        resolver: DigestResolver,
        state_store: WatcherStateStore,
        project_name: str,
        retry_policy: RetryPolicy | None = None,
        alert_emit: AlertEmitter | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        jitter_source: JitterSource = random.random,
        allow_mutable_tracking: bool = True,
    ) -> None:
        self._registry_factory = registry_factory
        self._controller_factory = controller_factory
        self._resolver = resolver
        self._state_store = state_store
        self._project_name = project_name
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._alert_emit = alert_emit
        self._clock = clock
        self._wall_clock = wall_clock
        self._jitter_source = jitter_source
        self._allow_mutable_tracking = allow_mutable_tracking
        self._records: dict[str, ChallengeWatcherRecord] = self._state_store.load()
        self._rehydrate_backoff_eligibility()
        # Per-challenge serialization locks (at most one rollout mutates a slug).
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    def _rehydrate_backoff_eligibility(self) -> None:
        """Rebuild process-local mono deadlines from durable wall-clock state.

        Persisted ``next_eligible_monotonic`` values are never trusted after a
        process restart: monotonic clocks reset and would stick forever. Prefer
        ``next_eligible_at``; otherwise recompute from ``last_failure_at`` +
        current attempt delay (VAL-COMPOSE-039 / VAL-CROSS-071).
        """

        mono_now = self._clock()
        wall_now = self._wall_clock()
        for record in self._records.values():
            if record.next_eligible_at is not None:
                remaining = max(0.0, float(record.next_eligible_at) - wall_now)
                record.next_eligible_monotonic = mono_now + remaining
                continue
            if record.last_failure_at is not None and record.attempts > 0:
                delay = self._retry_policy.compute_delay(
                    record.attempts,
                    jitter=1.0 if not self._retry_policy.jitter else 0.5,
                )
                eligible_at = float(record.last_failure_at) + delay
                record.next_eligible_at = eligible_at
                remaining = max(0.0, eligible_at - wall_now)
                record.next_eligible_monotonic = mono_now + remaining
                continue
            # No durable wall source: treat as immediately eligible.
            record.next_eligible_monotonic = 0.0

    def _lock_for(self, slug: str) -> asyncio.Lock:
        lock = self._locks.get(slug)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[slug] = lock
        return lock

    def _record(self, slug: str) -> ChallengeWatcherRecord:
        record = self._records.get(slug)
        if record is None:
            record = ChallengeWatcherRecord(slug=slug)
            self._records[slug] = record
        return record

    def _persist(self) -> None:
        for record in self._records.values():
            record.updated_at = datetime.now(UTC).isoformat()
        self._state_store.save(self._records)

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_WATCHER)
    async def run_once(self) -> dict[str, str]:
        """One watcher tick; never raises across challenges."""

        actions: dict[str, str] = {}
        try:
            registry = self._registry_factory()
            controller = self._controller_factory(registry)
            records = await _registry_list(registry)
        except Exception:
            logger.exception("challenge-watcher: registry/controller bootstrap failed")
            return actions

        for record in records:
            slug = getattr(record, "slug", None)
            if not slug:
                continue
            try:
                action = await self._refresh_challenge(registry, controller, record)
                actions[slug] = action
            except Exception:
                logger.exception(
                    "challenge-watcher: refresh failed for %r; continuing", slug
                )
                actions[slug] = "error"
        return actions

    async def _refresh_challenge(
        self, registry: Any, controller: Any, registry_record: Any
    ) -> str:
        slug = registry_record.slug
        status = getattr(registry_record, "status", None)
        if status in {ChallengeStatus.DRAFT, ChallengeStatus.DISABLED}:
            return "skipped-not-active"
        if status == ChallengeStatus.INACTIVE:
            return "skipped-inactive"

        async with self._lock_for(slug):
            return await self._refresh_locked(registry, controller, registry_record)

    async def _refresh_locked(
        self, registry: Any, controller: Any, registry_record: Any
    ) -> str:
        slug = registry_record.slug
        image = getattr(registry_record, "image", "") or ""
        watcher = self._record(slug)
        now = self._clock()

        # Resolve desired immutable digest.
        try:
            desired_image, desired_digest = self._resolve_desired(image)
        except DockerOrchestrationError as exc:
            watcher.phase = WatcherPhase.IDLE
            watcher.last_error = str(exc)
            watcher.last_result = "skipped-untracked-or-invalid"
            self._persist()
            logger.info(
                "challenge-watcher: %s action=skipped-untracked-or-invalid (%s)",
                slug,
                exc,
            )
            return "skipped-untracked-or-invalid"
        except Exception as exc:
            watcher.phase = WatcherPhase.BACKOFF
            watcher.last_error = f"resolve-failed: {exc}"
            state = watcher.as_retry_state()
            state.record_failure(
                now,
                self._retry_policy,
                error=str(exc),
                jitter_source=self._jitter_source,
            )
            watcher.apply_retry_state(
                state, process_now=now, wall_now=self._wall_clock()
            )
            self._persist()
            logger.warning(
                "challenge-watcher: %s resolution failure (%s); non-disruptive",
                slug,
                exc,
            )
            return "skipped-resolve-error"

        # New desired digest resets backoff / exhaustion (VAL-COMPOSE-038).
        if watcher.desired_digest != desired_digest:
            watcher.desired_digest = desired_digest
            watcher.desired_image = desired_image
            state = watcher.as_retry_state()
            state.record_success()
            watcher.apply_retry_state(
                state, process_now=now, wall_now=self._wall_clock()
            )
            watcher.alerted = False
            watcher.phase = WatcherPhase.IDLE
            watcher.last_error = None

        # Running digest is source of truth, never the desired DB record alone.
        try:
            running_image = await _running_image(controller, slug)
        except DockerOrchestrationError as exc:
            logger.warning(
                "challenge-watcher: %s inspect failed (%s); skip this tick",
                slug,
                exc,
            )
            watcher.last_result = "skipped-inspect-error"
            self._persist()
            return "skipped-inspect-error"

        running_digest = extract_digest(running_image) if running_image else None
        if running_digest is not None:
            watcher.current_digest = running_digest

        if running_digest == desired_digest:
            # Mid-rollout resume: digests may already match after a crash mid
            # recreate/verify; never short-circuit past re-verification
            # (VAL-CROSS-071).
            if watcher.phase in _MID_ROLLOUT_PHASES:
                return await self._resume_mid_rollout(
                    controller, watcher, slug, desired_digest, desired_image
                )
            state = watcher.as_retry_state()
            state.record_success()
            watcher.apply_retry_state(
                state, process_now=now, wall_now=self._wall_clock()
            )
            watcher.phase = WatcherPhase.IDLE
            watcher.last_result = "already-current"
            watcher.last_error = None
            watcher.last_health_ok = True
            watcher.last_version_ok = True
            self._persist()
            logger.info(
                "challenge-watcher: %s desired=%s action=already-current",
                slug,
                desired_digest,
            )
            return "already-current"

        # Backoff / exhaustion gate for this desired digest.
        state = watcher.as_retry_state()
        if not state.is_eligible(now):
            watcher.phase = WatcherPhase.BACKOFF
            watcher.last_result = "skipped-backoff"
            self._persist()
            logger.debug(
                "challenge-watcher: %s backoff (attempts=%s)", slug, state.attempts
            )
            return "skipped-backoff"
        if state.is_exhausted(self._retry_policy):
            watcher.phase = WatcherPhase.EXHAUSTED
            watcher.last_result = "skipped-exhausted"
            if not watcher.alerted:
                watcher.alerted = True
                self._emit_alert(slug, image, state)
            self._persist()
            return "skipped-exhausted"

        # Capture rollback target BEFORE any mutation.
        rollback_image = running_image
        rollback_digest = running_digest
        watcher.rollback_image = rollback_image
        watcher.rollback_digest = rollback_digest
        watcher.desired_image = desired_image
        watcher.desired_digest = desired_digest

        # Update registry desired image to the fully pinned form (DB intent).
        if desired_image != image:
            try:
                await _registry_update(
                    registry, slug, ChallengeUpdate(image=desired_image)
                )
            except Exception:
                logger.exception(
                    "challenge-watcher: %s failed to update registry image to %s",
                    slug,
                    desired_image,
                )

        # Pull (non-disruptive on failure — VAL-COMPOSE-036).
        watcher.phase = WatcherPhase.PULLING
        self._persist()
        try:
            if hasattr(controller, "pull"):
                await controller.pull(slug)
            else:
                pull_image = getattr(
                    getattr(controller, "orchestrator", None), "pull_image", None
                )
                if callable(pull_image):
                    pull_image(desired_image)
        except Exception as exc:
            watcher.phase = WatcherPhase.BACKOFF
            watcher.last_result = "pull-failed"
            watcher.last_error = str(exc)
            state.record_failure(
                now,
                self._retry_policy,
                error=str(exc),
                jitter_source=self._jitter_source,
            )
            watcher.apply_retry_state(
                state, process_now=now, wall_now=self._wall_clock()
            )
            if state.is_exhausted(self._retry_policy) and not watcher.alerted:
                watcher.alerted = True
                self._emit_alert(slug, image, state)
            self._persist()
            logger.error(
                "challenge-watcher: %s pull failed (%s); leaving running container",
                slug,
                exc,
            )
            return "pull-failed"

        # Targeted recreate + readiness (health + version).
        watcher.phase = WatcherPhase.RECREATING
        self._persist()
        try:
            watcher.phase = WatcherPhase.VERIFYING
            self._persist()
            await controller.restart(slug)
        except Exception as exc:
            return await self._rollback(
                controller,
                watcher,
                slug,
                rollback_image,
                rollback_digest,
                reason=str(exc),
                failure_kind="health-or-version-failed",
            )

        # Commit success.
        watcher.phase = WatcherPhase.COMMITTING
        watcher.current_digest = desired_digest
        watcher.rollback_digest = desired_digest
        watcher.rollback_image = desired_image
        watcher.last_result = "rolled"
        watcher.last_error = None
        watcher.last_health_ok = True
        watcher.last_version_ok = True
        state.record_success()
        watcher.apply_retry_state(state, process_now=now, wall_now=self._wall_clock())
        watcher.phase = WatcherPhase.IDLE
        watcher.alerted = False
        self._persist()
        logger.info(
            "challenge-watcher: %s desired=%s action=rolled",
            slug,
            desired_digest,
        )
        return "rolled"

    async def _resume_mid_rollout(
        self,
        controller: Any,
        watcher: ChallengeWatcherRecord,
        slug: str,
        desired_digest: str,
        desired_image: str,
    ) -> str:
        """Finish or roll back a mid-rollout resume when digests already match.

        Digest equality alone must not short-circuit past readiness when the
        durable phase shows we crashed mid-rollout (VAL-CROSS-071).
        """

        rollback_image = watcher.rollback_image
        rollback_digest = watcher.rollback_digest
        watcher.phase = WatcherPhase.VERIFYING
        self._persist()
        try:
            await _verify_ready(controller, slug)
        except Exception as exc:
            return await self._rollback(
                controller,
                watcher,
                slug,
                rollback_image,
                rollback_digest,
                reason=str(exc),
                failure_kind="health-or-version-failed",
            )

        now = self._clock()
        state = watcher.as_retry_state()
        watcher.phase = WatcherPhase.COMMITTING
        watcher.current_digest = desired_digest
        watcher.rollback_digest = desired_digest
        watcher.rollback_image = desired_image
        watcher.last_result = "resumed-verified"
        watcher.last_error = None
        watcher.last_health_ok = True
        watcher.last_version_ok = True
        state.record_success()
        watcher.apply_retry_state(state, process_now=now, wall_now=self._wall_clock())
        watcher.phase = WatcherPhase.IDLE
        watcher.alerted = False
        self._persist()
        logger.info(
            "challenge-watcher: %s desired=%s action=resumed-verified",
            slug,
            desired_digest,
        )
        return "resumed-verified"

    async def _rollback(
        self,
        controller: Any,
        watcher: ChallengeWatcherRecord,
        slug: str,
        rollback_image: str | None,
        rollback_digest: str | None,
        *,
        reason: str,
        failure_kind: str,
    ) -> str:
        watcher.phase = WatcherPhase.ROLLING_BACK
        watcher.last_error = reason
        self._persist()
        logger.error(
            "challenge-watcher: %s rollout failed (%s); rolling back",
            slug,
            reason,
        )
        if rollback_image and hasattr(controller, "rollback"):
            try:
                await controller.rollback(slug, rollback_image)
                watcher.current_digest = rollback_digest
                watcher.last_result = "rolled-back"
                watcher.last_health_ok = True
            except Exception:
                logger.exception(
                    "challenge-watcher: %s rollback to %s failed",
                    slug,
                    rollback_image,
                )
                watcher.last_result = "rollback-failed"
        else:
            watcher.last_result = "rollback-unavailable"
            logger.warning(
                "challenge-watcher: %s no rollback target/seam for failure", slug
            )

        now = self._clock()
        state = watcher.as_retry_state()
        state.record_failure(
            now, self._retry_policy, error=reason, jitter_source=self._jitter_source
        )
        watcher.apply_retry_state(state, process_now=now, wall_now=self._wall_clock())
        if state.is_exhausted(self._retry_policy):
            watcher.phase = WatcherPhase.EXHAUSTED
            if not watcher.alerted:
                watcher.alerted = True
                self._emit_alert(slug, watcher.desired_image or "", state)
        else:
            watcher.phase = WatcherPhase.BACKOFF
        if "version" in reason.lower() or "capability" in reason.lower():
            watcher.last_version_ok = False
        else:
            watcher.last_health_ok = False
        self._persist()
        return failure_kind

    def _resolve_desired(self, image: str) -> tuple[str, str]:
        """Return ``(pinned_image, digest)`` for the tracking policy.

        When mutable tracking is enabled, re-resolve the image's tracking tag
        even if the current registry record is already ``tag@sha256:...``. This
        matches the Swarm challenge-image-updater: a new remote digest becomes
        desired regardless of a previously-failed pin, and resets backoff when
        the desired digest changes (VAL-COMPOSE-038).
        """

        if not image:
            raise DockerOrchestrationError("empty challenge image")
        parsed = parse_image_reference(image)

        if self._allow_mutable_tracking:
            # Always re-resolve the tag endpoint (ignore currently stored digest).
            tracking = ImageReference(
                registry=parsed.registry,
                repository=parsed.repository,
                tag=parsed.tag,
                digest=None,
            )
            digest = self._resolver(tracking)
            if not digest or not _PINNED_DIGEST_RE.match(digest.lower()):
                raise DockerOrchestrationError(
                    f"resolver returned non-sha256 digest {digest!r}"
                )
            digest = digest.lower()
            pinned = f"{tracking.tagged}@{digest}"
            return pinned, digest

        # Strict pin mode: image must already include a digest; no remote resolve.
        if parsed.digest and _PINNED_DIGEST_RE.match(parsed.digest.lower()):
            digest = parsed.digest.lower()
            if (
                image.count("@") == 1
                and ":" not in image.split("@", 1)[0].rsplit("/", 1)[-1]
            ):
                repo_part, _, _ = image.partition("@")
                pinned = f"{repo_part}@{digest}"
            else:
                pinned = f"{parsed.tagged}@{digest}"
            return pinned, digest
        raise DockerOrchestrationError(
            "mutable tag tracking disabled; image must already be digest-pinned"
        )

    def _emit_alert(self, slug: str, image: str, state: RetryState) -> None:
        if self._alert_emit is None:
            return
        message = (
            f"challenge watcher for {slug!r} exhausted retries "
            f"({state.attempts}); paused until a new digest"
        )
        try:
            self._alert_emit(
                WeightsAlert(
                    kind=ALERT_CHALLENGE_IMAGE_UPDATE_FAILED,
                    message=message,
                    details={
                        "slug": slug,
                        "image": image,
                        "attempts": state.attempts,
                        "last_error": state.last_error,
                        "project": self._project_name,
                    },
                )
            )
        except Exception:
            logger.exception("challenge-watcher: alert hook raised for %r", slug)


async def _registry_list(registry: Any) -> list[Any]:
    listed = registry.list()
    if asyncio.iscoroutine(listed) or hasattr(listed, "__await__"):
        listed = await listed  # type: ignore[misc]
    return list(listed)


async def _registry_update(registry: Any, slug: str, update: ChallengeUpdate) -> None:
    result = registry.update(slug, update)
    if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        await result  # type: ignore[misc]


async def _running_image(controller: Any, slug: str) -> str | None:
    accessor = getattr(controller, "running_image", None)
    if not callable(accessor):
        return None
    result = accessor(slug)
    if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        return await result  # type: ignore[misc]
    return result


async def _verify_ready(controller: Any, slug: str) -> None:
    """Re-probe challenge readiness without force-recreating when possible.

    Prefer an explicit ``verify`` seam (mid-rollout resume). Otherwise fall back
    to ``restart`` so health/version still gate completion, then fail closed when
    neither seam exists.
    """

    verify = getattr(controller, "verify", None)
    if callable(verify):
        result = verify(slug)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        return
    restart = getattr(controller, "restart", None)
    if callable(restart):
        result = restart(slug)
        if asyncio.iscoroutine(result) or hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        return
    raise DockerOrchestrationError(
        f"controller cannot verify readiness for challenge {slug!r}"
    )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _opt_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _opt_float(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_challenge_watcher(
    settings: Settings,
    *,
    registry_factory: RegistryFactory | None = None,
    controller_factory: ControllerFactory | None = None,
    resolver: DigestResolver | None = None,
    state_path: Path | str | None = None,
    retry_policy: RetryPolicy | None = None,
    alert_emit: AlertEmitter | None = None,
    project_name: str | None = None,
) -> ChallengeWatcher:
    """Construct a production/default ChallengeWatcher."""

    def default_registry_factory() -> Any:
        from base.cli_app.main import _master_registry

        return _master_registry(settings)

    def default_controller_factory(registry: Any) -> Any:
        from base.cli_app.main import DockerRuntimeController, _challenge_orchestrator

        return DockerRuntimeController(registry, _challenge_orchestrator(settings))

    resolved_project = (
        project_name
        or getattr(settings.docker, "compose_project_name", None)
        or os.environ.get("COMPOSE_PROJECT_NAME")
        or "base-mission-master"
    )
    path = Path(
        state_path
        if state_path is not None
        else getattr(
            settings.master,
            "challenge_watcher_state_path",
            str(DEFAULT_STATE_PATH),
        )
    )
    policy = retry_policy
    if policy is None:
        policy = RetryPolicy(
            max_attempts=max(
                1,
                int(getattr(settings.supervisor, "image_update_max_attempts", 5) or 5),
            )
            if hasattr(settings, "supervisor")
            else 5,
            base_delay=float(
                getattr(settings.supervisor, "image_update_backoff_base_seconds", 60.0)
                if hasattr(settings, "supervisor")
                else 60.0
            ),
            max_delay=float(
                getattr(settings.supervisor, "image_update_backoff_max_seconds", 1800.0)
                if hasattr(settings, "supervisor")
                else 1800.0
            ),
        )
    return ChallengeWatcher(
        registry_factory=(
            registry_factory
            if registry_factory is not None
            else default_registry_factory
        ),
        controller_factory=(
            controller_factory
            if controller_factory is not None
            else default_controller_factory
        ),
        resolver=resolver if resolver is not None else resolve_remote_digest,
        state_store=WatcherStateStore(path),
        project_name=str(resolved_project),
        retry_policy=policy,
        alert_emit=alert_emit if alert_emit is not None else build_alert_hook(settings),
    )


async def run_challenge_watcher_loop(
    watcher: ChallengeWatcher,
    *,
    interval_seconds: float,
    shutdown_event: asyncio.Event,
) -> None:
    """Run watcher ticks until shutdown; log failures without aborting."""

    logger.info(
        "challenge-watcher: starting in-process loop interval_seconds=%s "
        "enabled=true (not a separate Compose service)",
        interval_seconds,
    )
    while not shutdown_event.is_set():
        try:
            with activate_role(Role.MASTER):
                await watcher.run_once()
        except Exception:
            logger.exception("challenge-watcher: tick failed; will retry next interval")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
        except TimeoutError:
            continue


def build_challenge_watcher_lifespan(
    settings: Settings | None,
    interval_seconds: float | None,
    *,
    registry_factory: RegistryFactory | None = None,
    controller_factory: ControllerFactory | None = None,
    resolver: DigestResolver | None = None,
    state_path: Path | str | None = None,
    retry_policy: RetryPolicy | None = None,
    alert_emit: AlertEmitter | None = None,
    project_name: str | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]] | None:
    """FastAPI lifespan for the Compose challenge watcher (in-process)."""

    if settings is None or interval_seconds is None or interval_seconds <= 0:
        return None

    watcher = build_challenge_watcher(
        settings,
        registry_factory=registry_factory,
        controller_factory=controller_factory,
        resolver=resolver,
        state_path=state_path,
        retry_policy=retry_policy,
        alert_emit=alert_emit,
        project_name=project_name,
    )
    loop_interval = interval_seconds

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            run_challenge_watcher_loop(
                watcher,
                interval_seconds=loop_interval,
                shutdown_event=shutdown,
            )
        )
        try:
            yield
        finally:
            shutdown.set()
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    return lifespan


__all__ = [
    "ChallengeWatcher",
    "ChallengeWatcherRecord",
    "DEFAULT_STATE_PATH",
    "DEFAULT_WATCHER_INTERVAL_SECONDS",
    "WatcherPhase",
    "WatcherStateStore",
    "build_challenge_watcher",
    "build_challenge_watcher_lifespan",
    "run_challenge_watcher_loop",
]
