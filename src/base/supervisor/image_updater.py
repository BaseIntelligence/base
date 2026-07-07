"""Supervisor image-updater — master Swarm service digest pinning.

A :class:`ScheduledTask` that resolves the public GHCR tag digest and pins
the first-party master Swarm services (proxy/broker/...) to
``tag@sha256:<digest>`` only when the digest changes. It reuses the
registry-only digest-resolution core (:func:`resolve_remote_digest` /
:func:`parse_image_reference` / :func:`extract_digest` from
:mod:`base.supervisor.image_ref`) and rolls the first-party
Swarm services via ``docker service update --image tag@sha256:<digest>``
through the existing :class:`SwarmCommandRunner` seam.

Idempotency is RESTART-SAFE by design: instead of remembering the
last-applied digest in process memory, each tick inspects the service's
currently-pinned image (``docker service inspect --format
'{{.Spec.TaskTemplate.ContainerSpec.Image}}'``) and compares digests — a
supervisor restart can therefore never re-issue an update for an
already-current service.

Rollouts are CONVERGENCE-VERIFIED and RETRY-WITH-ROLLBACK (durable auto-update):
before updating, the service's current (pre-update) digest is recorded as the
last-known-good (persisted to a small JSON under the release root so it survives
a supervisor restart); the update is issued WITHOUT ``--detach`` and its
``UpdateStatus.State`` is polled until ``completed`` (success), ``paused`` /
``rolled_back`` / a bounded timeout (failure). On failure the service is rolled
back (re-pinned to the last-known-good digest, or ``docker service update
--rollback`` when none is recorded) and a per-target :class:`RetryState`
schedules an exponential backoff between ticks; once the failure budget is
exhausted the target is skipped (an ``image_update_failed`` alert is emitted
once) until a NEW desired digest appears, which resets the state.

Operator freeze: a target with ``hold`` set (globally via
``supervisor.image_update_hold`` or per-target) is skipped entirely (logged
``skipped-held``) and is never rolled or rolled back — an opt-in way for
operators to pin a known-good digest and stop a bad rollout.

Production pin policy (README "Deployment Policy") is enforced on the way
out: targets without an explicit tag are rejected, and an update is only
ever emitted with a full ``tag@sha256:<64-hex>`` reference — a resolver
that fails or returns anything but a sha256 digest yields a logged no-op,
never an un-pinned ``service update``.

Health-gate note: this job talks to dockerd and the public GHCR registry,
NOT the broker, so the shared :class:`BrokerHealthGate` is accepted for
recipe parity but deliberately not consulted.

Swarm service naming: the defaults below name the installer-created
master-side services (``base-master-proxy`` + ``base-docker-broker``, both
tracking the master image, per deploy/swarm/install-swarm.sh); production
overrides these via ``build_scheduled_tasks`` (same names), so they are a
test/fallback default that nevertheless points only at services the
installer actually creates. A service that does not exist on the daemon is
a logged skip, so partial deployments are safe.
The single-port consolidation removed the separate ``base-admin``
service (the admin/registry surface is served by the proxy), so it is no
longer a rollout target; ``base-config-sync`` is likewise not a Swarm
service (under the supervisor it is a periodic task), so it is not a target.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from base.config.settings import Settings
from base.master.swarm_backend import SwarmCliRunner, SwarmCommandRunner
from base.supervisor.alerts import ALERT_IMAGE_UPDATE_FAILED
from base.supervisor.health import BrokerHealthGate
from base.supervisor.image_ref import (
    ImageReference,
    extract_digest,
    parse_image_reference,
    resolve_remote_digest,
)
from base.supervisor.retry import JitterSource, RetryPolicy, RetryState
from base.supervisor.scheduler import ScheduledTask
from base.supervisor.self_update import DEFAULT_RELEASE_ROOT
from base.supervisor.weight_submit import AlertEmitter, WeightsAlert

logger = logging.getLogger(__name__)

# One-minute image-updater cadence.
IMAGE_UPDATER_INTERVAL_SECONDS = 60.0
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
DEFAULT_MASTER_IMAGE = "ghcr.io/baseintelligence/base-master:latest"

# Convergence poll: after issuing a (non-detached) service update, poll the
# service's UpdateStatus.State until it reaches a terminal state or the timeout
# elapses. A short interval keeps a fast roll responsive; the timeout bounds a
# stuck rollout so a hung update becomes a rollback+retry instead of hanging the
# task thread.
CONVERGENCE_POLL_INTERVAL_SECONDS = 2.0
CONVERGENCE_TIMEOUT_SECONDS = 120.0

_UPDATE_STATE_COMPLETED = "completed"
_UPDATE_STATE_TIMEOUT = "timeout"
#: Terminal ``UpdateStatus.State`` values that end the convergence poll. Only
#: ``completed`` is a success; ``paused``/``rolled_back`` mean the rollout failed
#: Swarm's own health check and must be rolled back + retried.
_TERMINAL_UPDATE_STATES = frozenset({_UPDATE_STATE_COMPLETED, "paused", "rolled_back"})

#: Last-known-good digests persisted next to the self-update release root so a
#: rollback target survives a supervisor restart (see module docstring).
DEFAULT_LAST_KNOWN_GOOD_PATH = (
    DEFAULT_RELEASE_ROOT / "image_update_last_known_good.json"
)

_PINNED_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

DigestResolver = Callable[[ImageReference], str]


@dataclass(frozen=True)
class ImageUpdateTarget:
    """One Swarm service tracking a mutable (tagged, un-pinned) image.

    ``hold`` freezes auto-update for this service (skipped, never rolled or
    rolled back); it is an opt-in operator freeze, default OFF.
    """

    service: str
    image: str
    hold: bool = False


DEFAULT_FIRST_PARTY_TARGETS: tuple[ImageUpdateTarget, ...] = (
    ImageUpdateTarget(service="base-master-proxy", image=DEFAULT_MASTER_IMAGE),
    ImageUpdateTarget(service="base-docker-broker", image=DEFAULT_MASTER_IMAGE),
)


def resolve_image_update_targets(settings: Settings) -> tuple[ImageUpdateTarget, ...]:
    """Resolve the effective image-updater targets from supervisor settings.

    Settings-driven (G-A5): ``supervisor.image_updater_targets`` overrides the
    target list when set; when unset (``None``) the built-in
    :data:`DEFAULT_FIRST_PARTY_TARGETS` (the two master services) are used, so a
    deploy that never configures targets keeps its prior behaviour (back-compat).

    When ``supervisor.validator_agent_target_enabled`` is True a validator-agent
    target tracking the mutable validator runtime image is appended (skipped if
    the same service name is already present), so a validator NODE running the
    agent as a swarm service auto-rolls on a new digest.
    """
    sup = settings.supervisor
    if sup.image_updater_targets is not None:
        targets = tuple(
            ImageUpdateTarget(service=t.service, image=t.image, hold=t.hold)
            for t in sup.image_updater_targets
        )
    else:
        targets = DEFAULT_FIRST_PARTY_TARGETS
    if sup.validator_agent_target_enabled and not any(
        t.service == sup.validator_agent_service for t in targets
    ):
        targets = (
            *targets,
            ImageUpdateTarget(
                service=sup.validator_agent_service,
                image=sup.validator_agent_image,
            ),
        )
    return targets


def _has_explicit_tag(image: str) -> bool:
    """True when the raw image string carries an explicit tag.

    :func:`parse_image_reference` silently defaults a missing tag to
    ``latest``; production policy rejects untagged images, so the check
    must look at the raw string before parsing.
    """
    name, _, _ = image.partition("@")
    return ":" in name.rsplit("/", 1)[-1]


class SwarmImageUpdater:
    """Digest-compare-and-update loop body for first-party Swarm services.

    Each rollout is convergence-verified and retry-with-rollback; per-target
    :class:`RetryState` implements exponential backoff between the fixed ticks
    and an exhaustion alert. ``state_path`` persists last-known-good digests for
    restart-durable rollback (``None`` keeps them in memory only, e.g. in tests).
    """

    def __init__(
        self,
        targets: Sequence[ImageUpdateTarget],
        *,
        runner: SwarmCommandRunner,
        resolver: DigestResolver,
        docker_bin: str = "docker",
        command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        retry_policy: RetryPolicy | None = None,
        alert_emit: AlertEmitter | None = None,
        global_hold: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        convergence_poll_interval_seconds: float = CONVERGENCE_POLL_INTERVAL_SECONDS,
        convergence_timeout_seconds: float = CONVERGENCE_TIMEOUT_SECONDS,
        state_path: Path | None = None,
        jitter_source: JitterSource = random.random,
    ) -> None:
        self._targets = tuple(targets)
        self._runner = runner
        self._resolver = resolver
        self._docker_bin = docker_bin
        self._command_timeout_seconds = command_timeout_seconds
        self._retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self._alert_emit = alert_emit
        self._global_hold = global_hold
        self._clock = clock
        self._sleep = sleep
        self._convergence_poll_interval_seconds = convergence_poll_interval_seconds
        self._convergence_timeout_seconds = convergence_timeout_seconds
        self._jitter_source = jitter_source
        self._state_path = state_path
        self._retry_state: dict[str, RetryState] = {}
        # Desired digest last pursued per service; a change ends a failure episode
        # (resets retry state + re-arms the exhaustion alert).
        self._desired_digest: dict[str, str] = {}
        self._alerted: set[str] = set()
        self._last_known_good: dict[str, str] = self._load_last_known_good()

    @property
    def targets(self) -> tuple[ImageUpdateTarget, ...]:
        """The Swarm service targets this updater manages (stable public seam).

        Exposed so callers/tests can assert the wired targets without reaching
        into the private ``_targets`` field (see :func:`image_updater_from_task`).
        """
        return self._targets

    def run_once(self) -> None:
        """One tick: refresh every target; per-target failures are isolated."""
        for target in self._targets:
            try:
                self._refresh_target(target, self._clock())
            except Exception:
                logger.exception(
                    "image-updater: refresh failed for service %r (image %r); "
                    "continuing with remaining targets",
                    target.service,
                    target.image,
                )

    def _refresh_target(self, target: ImageUpdateTarget, now: float) -> bool:
        state = self._retry_state.setdefault(target.service, RetryState())

        # Operator freeze (E): a held target is never rolled or rolled back.
        if self._global_hold or target.hold:
            logger.info(
                "image-updater: skipped-held service %r (auto-update frozen)",
                target.service,
            )
            return False

        if not _has_explicit_tag(target.image):
            logger.error(
                "image-updater: rejecting untagged image %r for service %r "
                "(production pin policy requires an explicit tag)",
                target.image,
                target.service,
            )
            return False

        # Backoff gate: skip a target still serving its retry backoff so the
        # fixed 60s ticks do not hammer a failing rollout.
        if not state.is_eligible(now):
            logger.debug(
                "image-updater: service %r backing off after %d failure(s); "
                "skipping until eligible",
                target.service,
                state.attempts,
            )
            return False

        reference = parse_image_reference(target.image)
        try:
            digest = self._resolver(reference)
        except Exception:
            logger.warning(
                "image-updater: digest resolution failed for %s (service %r); "
                "skipping this tick",
                reference.tagged,
                target.service,
                exc_info=True,
            )
            return False
        if not digest or not _PINNED_DIGEST_RE.match(digest):
            logger.error(
                "image-updater: resolver returned non-sha256 digest %r for %s; "
                "refusing un-pinned update (production pin policy)",
                digest,
                reference.tagged,
            )
            return False

        current_image = self._current_service_image(target.service)
        if current_image is None:
            return False
        current_digest = extract_digest(current_image)

        # A new desired digest ends any prior failure episode: reset the retry
        # budget and re-arm the alert so a fresh rollout is attempted even after
        # a previous digest exhausted its retries.
        if self._desired_digest.get(target.service) != digest:
            self._desired_digest[target.service] = digest
            state.record_success()
            self._alerted.discard(target.service)

        if current_digest == digest:
            state.record_success()
            self._alerted.discard(target.service)
            logger.debug(
                "image-updater: service %r already at %s; no-op",
                target.service,
                digest,
            )
            return False

        if state.is_exhausted(self._retry_policy):
            # Budget spent for this digest: stop hammering (the alert already
            # fired) and wait for a new digest, handled above.
            logger.debug(
                "image-updater: service %r retries exhausted for %s; skipping",
                target.service,
                digest,
            )
            return False

        # Record the CURRENT (pre-update) digest as last-known-good BEFORE the
        # update so a rollback has a concrete revert target across a restart.
        self._record_last_known_good(target.service, current_digest)

        pinned = reference.pinned(digest)
        if not self._issue_update(target.service, pinned):
            return self._handle_failure(
                target, reference, state, now, "docker service update returned non-zero"
            )

        converged = self._await_convergence(target.service)
        if converged == _UPDATE_STATE_COMPLETED:
            state.record_success()
            self._alerted.discard(target.service)
            logger.info(
                "image-updater: updated service %r to %s (converged)",
                target.service,
                pinned,
            )
            return True

        return self._handle_failure(
            target, reference, state, now, f"convergence state {converged!r}"
        )

    def _handle_failure(
        self,
        target: ImageUpdateTarget,
        reference: ImageReference,
        state: RetryState,
        now: float,
        reason: str,
    ) -> bool:
        """Roll back, count the failure + schedule backoff, alert on exhaustion."""
        logger.error(
            "image-updater: rollout of service %r failed (%s); rolling back",
            target.service,
            reason,
        )
        self._rollback(target, reference)
        state.record_failure(
            now, self._retry_policy, error=reason, jitter_source=self._jitter_source
        )
        if (
            state.is_exhausted(self._retry_policy)
            and target.service not in self._alerted
        ):
            self._alerted.add(target.service)
            self._emit_alert(target, state)
        return False

    def _issue_update(self, service: str, pinned: str) -> bool:
        result = self._runner.run(
            [
                self._docker_bin,
                "service",
                "update",
                "--image",
                pinned,
                service,
            ],
            timeout_seconds=self._command_timeout_seconds,
        )
        if result.returncode != 0:
            logger.error(
                "image-updater: docker service update failed for %r (rc=%d): %s",
                service,
                result.returncode,
                result.stderr.strip(),
            )
            return False
        return True

    def _await_convergence(self, service: str) -> str:
        """Poll ``UpdateStatus.State`` until terminal or the timeout elapses."""
        deadline = self._clock() + self._convergence_timeout_seconds
        while True:
            state = self._update_status_state(service)
            if state in _TERMINAL_UPDATE_STATES:
                return state
            if self._clock() >= deadline:
                logger.warning(
                    "image-updater: service %r did not converge within %.0fs "
                    "(last state %r)",
                    service,
                    self._convergence_timeout_seconds,
                    state,
                )
                return _UPDATE_STATE_TIMEOUT
            self._sleep(self._convergence_poll_interval_seconds)

    def _update_status_state(self, service: str) -> str:
        result = self._runner.run(
            [
                self._docker_bin,
                "service",
                "inspect",
                "--format",
                "{{.UpdateStatus.State}}",
                service,
            ],
            timeout_seconds=self._command_timeout_seconds,
        )
        if result.returncode != 0:
            logger.warning(
                "image-updater: cannot read update status for %r (rc=%d): %s",
                service,
                result.returncode,
                result.stderr.strip(),
            )
            return ""
        return result.stdout.strip()

    def _rollback(self, target: ImageUpdateTarget, reference: ImageReference) -> None:
        """Revert a failed rollout to the recorded last-known-good digest.

        Prefer re-pinning the persisted pre-update digest (restart-durable and
        deterministic); fall back to Swarm's ``--rollback`` (previous spec) only
        when no last-known-good digest was recorded (e.g. the service was running
        an un-pinned tag).
        """
        last_good = self._last_known_good.get(target.service)
        if last_good:
            pinned = reference.pinned(last_good)
            logger.warning(
                "image-updater: rolling back service %r to last-known-good %s",
                target.service,
                last_good,
            )
            argv = [
                self._docker_bin,
                "service",
                "update",
                "--image",
                pinned,
                target.service,
            ]
        else:
            logger.warning(
                "image-updater: rolling back service %r via --rollback "
                "(no recorded last-known-good digest)",
                target.service,
            )
            argv = [self._docker_bin, "service", "update", "--rollback", target.service]
        result = self._runner.run(argv, timeout_seconds=self._command_timeout_seconds)
        if result.returncode != 0:
            logger.error(
                "image-updater: rollback of service %r failed (rc=%d): %s",
                target.service,
                result.returncode,
                result.stderr.strip(),
            )

    def _emit_alert(self, target: ImageUpdateTarget, state: RetryState) -> None:
        if self._alert_emit is None:
            return
        message = (
            f"image auto-update for service {target.service!r} failed "
            f"{state.attempts} times; rollout is paused until a new digest"
        )
        try:
            self._alert_emit(
                WeightsAlert(
                    kind=ALERT_IMAGE_UPDATE_FAILED,
                    message=message,
                    details={
                        "service": target.service,
                        "image": target.image,
                        "attempts": state.attempts,
                        "last_error": state.last_error,
                    },
                )
            )
        except Exception:
            logger.exception(
                "image-updater: alert hook raised for service %r", target.service
            )

    def _current_service_image(self, service: str) -> str | None:
        result = self._runner.run(
            [
                self._docker_bin,
                "service",
                "inspect",
                "--format",
                "{{.Spec.TaskTemplate.ContainerSpec.Image}}",
                service,
            ],
            timeout_seconds=self._command_timeout_seconds,
        )
        if result.returncode != 0:
            logger.warning(
                "image-updater: cannot inspect service %r (rc=%d): %s; skipping",
                service,
                result.returncode,
                result.stderr.strip(),
            )
            return None
        return result.stdout.strip()

    def _record_last_known_good(self, service: str, digest: str | None) -> None:
        if not digest:
            return
        if self._last_known_good.get(service) == digest:
            return
        self._last_known_good[service] = digest
        self._persist_last_known_good()

    def _load_last_known_good(self) -> dict[str, str]:
        if self._state_path is None:
            return {}
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): v for k, v in raw.items() if isinstance(v, str)}

    def _persist_last_known_good(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(self._last_known_good, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._state_path)
        except OSError:
            logger.warning(
                "image-updater: could not persist last-known-good to %s "
                "(rollback stays in-memory only this run)",
                self._state_path,
                exc_info=True,
            )


def build_image_updater_task(
    settings: Settings,
    *,
    health_gate: BrokerHealthGate | None = None,
    targets: Sequence[ImageUpdateTarget] | None = None,
    resolver: DigestResolver | None = None,
    runner: SwarmCommandRunner | None = None,
    docker_bin: str = "docker",
    interval_seconds: float = IMAGE_UPDATER_INTERVAL_SECONDS,
    alert_emit: AlertEmitter | None = None,
    state_path: Path | None = DEFAULT_LAST_KNOWN_GOOD_PATH,
) -> ScheduledTask:
    """Build the first-party image-updater :class:`ScheduledTask`.

    ``health_gate`` follows the Task-16 builder recipe but is not consulted (the
    job depends on dockerd + GHCR, not the broker — see module docstring).
    ``settings`` drives the retry/backoff policy and the global hold. ``resolver``
    defaults to the REUSED :func:`resolve_remote_digest`; ``runner`` defaults to
    the existing :class:`SwarmCliRunner` subprocess seam. ``alert_emit`` is the
    Task-16 alert seam, fired once when a target exhausts its retry budget.
    ``state_path`` persists last-known-good digests for restart-durable rollback.
    """
    del health_gate  # recipe parity; not broker-dependent.
    sup = settings.supervisor
    retry_policy = RetryPolicy(
        max_attempts=sup.image_update_max_attempts,
        base_delay=sup.image_update_backoff_base_seconds,
        max_delay=sup.image_update_backoff_max_seconds,
    )
    updater = SwarmImageUpdater(
        targets if targets is not None else DEFAULT_FIRST_PARTY_TARGETS,
        runner=runner if runner is not None else SwarmCliRunner(),
        resolver=resolver if resolver is not None else resolve_remote_digest,
        docker_bin=docker_bin,
        retry_policy=retry_policy,
        alert_emit=alert_emit,
        global_hold=sup.image_update_hold,
        state_path=state_path,
    )
    return ScheduledTask(
        name="image-updater",
        interval_seconds=interval_seconds,
        run=updater.run_once,
    )


def image_updater_from_task(task: ScheduledTask) -> SwarmImageUpdater:
    """Return the :class:`SwarmImageUpdater` backing an image-updater task.

    A stable, typed public seam so callers can inspect the updater's public
    surface (e.g. :attr:`SwarmImageUpdater.targets`) after it has been wired
    into a :class:`ScheduledTask` by :func:`build_image_updater_task` /
    :func:`base.supervisor.tasks.build_scheduled_tasks`, instead of reaching
    into the bound ``run`` method's ``__self__`` and its private fields.
    """
    updater = getattr(task.run, "__self__", None)
    if not isinstance(updater, SwarmImageUpdater):
        raise TypeError(f"task {task.name!r} is not backed by a SwarmImageUpdater")
    return updater
