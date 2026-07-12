from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import typer

from base.bittensor.factory import (
    create_bittensor_runtime,
    create_bittensor_submit_runtime,
    create_validator_keypair,
    create_worker_keypair,
    create_worker_miner_keypair,
)
from base.bittensor.identity_cache import (
    IDENTITY_DISPLAY_NAME_KEY,
    IDENTITY_LOGO_URL_KEY,
    ValidatorIdentityResolver,
)
from base.bittensor.metagraph_cache import MetagraphCache
from base.bittensor.validator_loop import run_epoch_loop
from base.compute import (
    InstanceSpec,
    LiumClient,
    TargonClient,
)
from base.compute.worker_deployment import WORKER_TEMPLATE_NAME
from base.config import load_settings
from base.config.policy import production_policy_enabled_for_settings
from base.config.settings import Settings
from base.db.session import create_engine, create_session_factory
from base.master.agent_challenge_compat import (
    AGENT_CHALLENGE_INCOMPATIBLE_CODE,
)
from base.master.app_proxy import create_proxy_app
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.assignment_coordination import (
    AssignmentCoordinationService,
)
from base.master.challenge_client import ChallengeClient
from base.master.challenge_work_source import (
    HttpChallengeFoldTrigger,
    HttpChallengeResultForwarder,
    HttpChallengeWorkSource,
)
from base.master.docker_broker import create_docker_broker_app
from base.master.docker_orchestrator import (
    DEFAULT_SECRET_MOUNT_DIR,
    ChallengeResources,
    ChallengeSpec,
    combined_mode_env_from_metadata,
    port_from_internal_base_url,
)
from base.master.health import migration_head, postgres_readiness_probe
from base.master.orchestration import (
    MasterChallengeReconciler,
    MasterOrchestrationDriver,
)
from base.master.raw_weight_ingress import (
    ChallengeCredentialStore,
    RawWeightIngressService,
)
from base.master.registry import (
    ChallengeNotFoundError,
    DatabaseChallengeRegistry,
)
from base.master.service import (
    MasterWeightService,
    active_challenge_inputs,
)
from base.master.submission_observation import ValidatorSubmissionObservationService
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import WorkerReconciliationService
from base.master.worker_unit_status import WorkerUnitStatusService
from base.observability.logging import configure_logging
from base.observability.otel import init_otel
from base.observability.sentry import init_sentry
from base.schemas.challenge import (
    ChallengeCreate,
    ChallengeStatus,
    ChallengeUpdate,
)
from base.schemas.weights import FinalWeights, MasterWeightsResponse
from base.security.admin_auth import read_secret
from base.security.miner_auth import SqlAlchemyMinerNonceStore
from base.security.validator_auth import (
    MetagraphValidatorEligibility,
    SqlAlchemyValidatorNonceStore,
    ValidatorSignedRequestVerifier,
)
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    RegisteredWorkerEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerSignedRequestVerifier,
)
from base.template_engine import (
    ChallengeTemplateContext,
    render_challenge_template,
)
from base.validator.agent import (
    BrokerAssignmentExecutor,
    BrokerConfig,
    ChallengeDispatchExecutor,
    CoordinationClient,
    KeypairRequestSigner,
    ValidatorAgent,
)
from base.validator.normal_runner import NormalValidatorRunner
from base.validator.registry_client import RegistryClient
from base.validator.status import validator_runtime_status
from base.validator.weight_submitter import ValidatorWeightSubmitter
from base.validator.weights_client import WeightsClient
from base.worker import (
    WorkerAgent,
    WorkerBinding,
    WorkerCoordinationClient,
    WorkerProofExecutor,
    WorkerProvenance,
    build_signed_binding,
    normalize_provider,
    plan_provider_deployment,
    require_provider_api_key,
)
from base.worker.deploy import (
    LOCAL_PROVIDER,
    MissingProviderKeyError,
    NoOfferWithinBudgetError,
    UnsupportedProviderError,
    WorkerDeployError,
    build_worker_pod_env,
    require_worker_image,
)

app = typer.Typer(help="BASE multi-challenge subnet CLI")
master_app = typer.Typer(help="Run master components")
master_challenges_app = typer.Typer(help="Manage master challenge records")
validator_app = typer.Typer(help="Run normal validator components")
challenge_app = typer.Typer(help="Manage and scaffold challenges")
db_app = typer.Typer(help="Database helpers")
registry_app = typer.Typer(help="Registry helpers")
worker_app = typer.Typer(
    help=(
        "HISTORICAL/NON-TARGET: Swarm node join/label helpers. "
        "Not part of the supported Compose operator path; prefer "
        "deploy/compose/install-master.sh."
    ),
)
worker_plane_app = typer.Typer(help="Deploy and manage miner-funded GPU worker agents")
master_app.add_typer(master_challenges_app, name="challenges")
master_app.add_typer(worker_app, name="worker")
app.add_typer(master_app, name="master")
app.add_typer(validator_app, name="validator")
app.add_typer(challenge_app, name="challenge")
app.add_typer(db_app, name="db")
app.add_typer(registry_app, name="registry")
app.add_typer(worker_plane_app, name="worker")
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Historical isolated Swarm eval overlay name (``base_jobs_internal``). Kept as
#: a local string so the default master CLI import graph does **not** pull in
#: :mod:`base.master.swarm_backend` (VAL-CROSS-065 Compose-only operator surface).
DEFAULT_JOB_NETWORK = "base_jobs_internal"


def _resolved_log_level(settings: Settings) -> int:
    """Resolve ``observability.log_level`` to a ``logging`` level int.

    Case-insensitive; an unrecognized name falls back to ``INFO`` so a
    misconfigured level never crashes an entrypoint.
    """
    return getattr(logging, settings.observability.log_level.upper(), logging.INFO)


def _configure_observability(settings: Settings) -> None:
    """Configure logging then wire Sentry + OTEL from settings-derived args.

    Every CLI entrypoint that sets up logging calls this so observability is
    initialized uniformly. ``init_sentry``/``init_otel`` are no-op safe when
    unconfigured (no DSN / no OTLP endpoint), so a default deploy stays inert.
    """
    configure_logging(
        settings.observability.log_json, level=_resolved_log_level(settings)
    )
    init_sentry(settings.observability.sentry_dsn, environment=settings.environment)
    init_otel(
        settings.observability.otel_service_name,
        settings.observability.otel_endpoint,
    )


def _admin_token(config: Path) -> str:
    settings = load_settings(config)
    return read_secret(
        settings.security.admin_token,
        settings.security.admin_token_file,
    )


def _admin_post(
    config: Path,
    path: str,
    payload: dict[str, object] | None = None,
) -> None:
    _admin_request(config, "POST", path, payload)


def _admin_request(
    config: Path,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> None:
    settings = load_settings(config)
    token = _admin_token(config)
    url = f"{settings.master.registry_url.rstrip('/')}{path}"
    headers = {"X-Admin-Token": token} if token else {}
    with httpx.Client(timeout=30.0) as client:
        response = client.request(method, url, json=payload, headers=headers)
        response.raise_for_status()
        if response.text:
            typer.echo(response.text)


class DockerRuntimeController:
    def __init__(
        self,
        registry: Any,
        orchestrator: Any,
    ) -> None:
        self.registry = registry
        self.orchestrator = orchestrator

    async def _spec(self, slug: str) -> ChallengeSpec:
        record = await _resolve(self.registry.get(slug))
        get_broker_token = getattr(self.registry, "get_broker_token", None)
        broker_token = get_broker_token(slug) if callable(get_broker_token) else None
        # Record-declared secret names beyond the per-slug registry tokens
        # (e.g. agent-challenge's submission_env_encryption_key) have no value
        # source on the master; they are provisioned out-of-band and carried
        # as external references so the Swarm backend mounts the pre-created
        # docker secrets without ever handling the values.
        external_secrets = tuple(
            name
            for name in (getattr(record, "secrets", []) or [])
            if name not in ("challenge_token", "docker_broker_token")
        )
        # Combined mode: the single service runs the image default CMD with the
        # worker loop in-process (no separate ``-worker`` service), enabled by the
        # per-metadata opt-in env var.
        metadata = getattr(record, "metadata", {}) or {}
        env = dict(record.env)
        combined_env = combined_mode_env_from_metadata(metadata)
        if combined_env is not None:
            env.setdefault(combined_env, "true")
        return ChallengeSpec(
            slug=record.slug,
            image=record.image,
            version=record.version,
            challenge_token=self.registry.get_token(slug),
            docker_broker_token=broker_token,
            env=env,
            external_secrets=external_secrets,
            resources=ChallengeResources.from_mapping(record.resources),
            required_capabilities=tuple(record.required_capabilities),
            port=port_from_internal_base_url(
                getattr(record, "internal_base_url", None)
            ),
            workload_class="service",
        )

    async def pull(self, slug: str):
        spec = await self._spec(slug)
        if hasattr(self.orchestrator, "pull_challenge"):
            self.orchestrator.pull_challenge(spec)
        else:
            self.orchestrator.pull_image(spec.image)
        return {
            "slug": slug,
            "operation": "pull",
            "status": "ok",
            "detail": spec.image,
        }

    async def restart(self, slug: str):
        runtime = self.orchestrator.restart_challenge(await self._spec(slug))
        return {
            "slug": slug,
            "operation": "restart",
            "status": "ok",
            "detail": runtime.container_name,
        }

    async def verify(self, slug: str):
        """Re-probe /health+/version without force-recreating the service.

        Used by the challenge watcher when resuming a mid-rollout durable phase
        with digests already matching (VAL-CROSS-071). Prefer orchestrator
        ``wait_until_ready``; fall back to a full restart path only when that
        seam is absent.
        """
        spec = await self._spec(slug)
        waiter = getattr(self.orchestrator, "wait_until_ready", None)
        if callable(waiter):
            health, version = waiter(spec)
            return {
                "slug": slug,
                "operation": "verify",
                "status": "ok",
                "detail": {
                    "health": health,
                    "version": version,
                },
            }
        runtime = self.orchestrator.restart_challenge(spec)
        return {
            "slug": slug,
            "operation": "verify",
            "status": "ok",
            "detail": runtime.container_name,
        }

    async def rollback(self, slug: str, image: str):
        """Roll the challenge service BACK to a specific (previous) image.

        Used by the challenge-image-updater when a roll to the desired digest
        comes up UNHEALTHY: the service is reverted to the digest it was running
        before the roll. Reuses ``restart_challenge`` with the record's image
        overridden so the same update + readiness path applies to the revert.
        """
        spec = replace(await self._spec(slug), image=image)
        runtime = self.orchestrator.restart_challenge(spec)
        return {
            "slug": slug,
            "operation": "rollback",
            "status": "ok",
            "detail": runtime.container_name,
        }

    async def running_image(self, slug: str) -> str | None:
        """Return the challenge service's ACTUALLY-running image ref, or None.

        Delegates to the orchestrator's ``service_image`` accessor when
        available so the challenge-image-updater can gate a roll on the running
        service digest (not the DB record). A backend without that seam returns
        None, and the updater then degrades to record-change gating. None means
        the service is absent (→ converge/create); a transient inspect failure
        propagates as an exception so the updater skips the roll this tick
        (retried next tick) instead of spuriously redeploying an already-current
        service.
        """
        accessor = getattr(self.orchestrator, "service_image", None)
        if not callable(accessor):
            return None
        return accessor(slug)

    async def status(self, slug: str):
        runtime = self.orchestrator.runtime.get(slug)
        return {
            "slug": slug,
            "operation": "status",
            "status": "running" if runtime else "unknown",
            "detail": runtime.container_name if runtime else None,
        }


async def _resolve(value):
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value


def _run_startup_migrations(settings) -> None:
    from base.db.migrations import upgrade

    upgrade(PROJECT_ROOT / "alembic.ini", database_url=settings.database.url)


def _master_session_factory(settings):
    engine = create_engine(settings.database.url)
    return create_session_factory(engine)


def _master_registry(settings, session_factory=None) -> DatabaseChallengeRegistry:
    return DatabaseChallengeRegistry(
        session_factory or _master_session_factory(settings),
        secret_dir=settings.docker.secret_dir,
        network=settings.network.name,
        master_uid=settings.network.master_uid,
        production_policy=production_policy_enabled_for_settings(settings),
    )


def _master_compute_metagraph_cache(settings) -> MetagraphCache:
    return create_bittensor_runtime(settings).metagraph_cache


def _master_weight_service(
    settings,
    metagraph_cache: MetagraphCache | None = None,
    session_factory: Any = None,
) -> MasterWeightService:
    return MasterWeightService(
        metagraph_cache=metagraph_cache or _master_compute_metagraph_cache(settings),
        challenge_client=ChallengeClient(
            timeout_seconds=settings.master.challenge_timeout_seconds,
            retries=settings.master.challenge_retries,
        ),
        session_factory=session_factory,
    )


def _validator_coordination_service(
    settings: Any, session_factory: Any
) -> ValidatorCoordinationService:
    return ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=settings.master.validator_heartbeat_interval_seconds,
        heartbeat_timeout_seconds=settings.master.validator_heartbeat_timeout_seconds,
    )


def _validator_signed_request_verifier(
    settings: Any,
    session_factory: Any,
    metagraph_cache: MetagraphCache,
) -> ValidatorSignedRequestVerifier:
    return ValidatorSignedRequestVerifier(
        nonce_store=SqlAlchemyValidatorNonceStore(
            session_factory,
            ttl_seconds=settings.master.validator_nonce_ttl_seconds,
        ),
        eligibility=MetagraphValidatorEligibility(metagraph_cache),
        ttl_seconds=settings.master.validator_signature_ttl_seconds,
    )


def _worker_coordination_service(
    settings: Any, session_factory: Any, metagraph_cache: MetagraphCache
) -> WorkerCoordinationService:
    return WorkerCoordinationService(
        session_factory,
        miner_membership=MetagraphMinerMembership(metagraph_cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(
            session_factory,
            ttl_seconds=settings.compute.worker_nonce_ttl_seconds,
        ),
        heartbeat_ttl_seconds=settings.compute.worker_heartbeat_ttl_seconds,
    )


def _worker_signed_request_verifier(
    settings: Any,
    session_factory: Any,
    metagraph_cache: MetagraphCache,
) -> WorkerSignedRequestVerifier:
    return WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(
            session_factory,
            ttl_seconds=settings.compute.worker_nonce_ttl_seconds,
        ),
        eligibility=CoordinationReadEligibility(session_factory, metagraph_cache),
        ttl_seconds=settings.compute.worker_signature_ttl_seconds,
    )


def _worker_assignment_service(
    settings: Any,
    session_factory: Any,
    worker_service: WorkerCoordinationService,
) -> WorkerAssignmentService:
    return WorkerAssignmentService(
        session_factory,
        worker_service=worker_service,
        lease_seconds=settings.master.assignment_lease_seconds,
    )


def _worker_assignment_verifier(
    settings: Any,
    session_factory: Any,
) -> WorkerSignedRequestVerifier:
    """Verifier for worker pull/result: WORKER identity only, no permit.

    Distinct from the fleet-read verifier (which also admits validators): the
    assignment surface requires a ``worker_registrations`` row so an unregistered
    or validator-only key can never pull/post work (VAL-AGENT-018).
    """

    return WorkerSignedRequestVerifier(
        nonce_store=SqlAlchemyWorkerNonceStore(
            session_factory,
            ttl_seconds=settings.compute.worker_nonce_ttl_seconds,
        ),
        eligibility=RegisteredWorkerEligibility(session_factory),
        ttl_seconds=settings.compute.worker_signature_ttl_seconds,
    )


def _assignment_coordination_service(
    settings: Any,
    session_factory: Any,
) -> AssignmentCoordinationService:
    """Build the pull/progress/result coordination service."""

    return AssignmentCoordinationService(
        session_factory,
        lease_seconds=settings.master.assignment_lease_seconds,
    )


def _master_orchestration_driver(
    settings: Any,
    session_factory: Any,
    registry: Any,
    validator_service: ValidatorCoordinationService,
    *,
    worker_service: WorkerCoordinationService | None = None,
    worker_assignment_service: WorkerAssignmentService | None = None,
) -> MasterOrchestrationDriver:
    """Build the live master orchestration driver (architecture.md sec 4).

    Bridges each challenge's HTTP-exposed pending work units into
    ``work_assignments``, runs the balanced assignment + full reassignment pass,
    and folds retry-exhausted units back into their EvaluationJob via the
    challenge fold route.

    When the worker plane is enabled (``compute.worker_plane_enabled``, with both
    worker services wired), gpu units are routed AWAY from validators (the
    validator ``AssignmentService`` skips them via ``worker_plane_capabilities``)
    and materialized as replicas by a :class:`WorkerAssignmentEngine` run each
    pass. With the flag OFF the engine is ``None`` and gpu units route to
    validators byte-identically to legacy.
    """

    worker_plane_on = (
        settings.compute.worker_plane_enabled
        and worker_service is not None
        and worker_assignment_service is not None
    )
    worker_plane_capabilities = (
        frozenset({CAPABILITY_GPU}) if worker_plane_on else frozenset()
    )
    assignment_service = AssignmentService(
        session_factory, worker_plane_capabilities=worker_plane_capabilities
    )
    worker_engine: WorkerAssignmentEngine | None = None
    worker_reconciler: WorkerReconciliationService | None = None
    if worker_plane_on:
        assert worker_service is not None
        assert worker_assignment_service is not None
        worker_engine = WorkerAssignmentEngine(
            session_factory,
            assignment_service=worker_assignment_service,
            worker_service=worker_service,
            replication_factor=settings.compute.replication_factor,
        )
        worker_reconciler = WorkerReconciliationService(
            session_factory,
            result_forwarder=HttpChallengeResultForwarder(
                registry,
                timeout_seconds=settings.master.challenge_timeout_seconds,
                retries=settings.master.challenge_retries,
            ),
        )
    return MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=validator_service,
        work_source=HttpChallengeWorkSource(
            registry,
            timeout_seconds=settings.master.challenge_timeout_seconds,
            retries=settings.master.challenge_retries,
        ),
        fold_trigger=HttpChallengeFoldTrigger(
            registry,
            timeout_seconds=settings.master.challenge_timeout_seconds,
            retries=settings.master.challenge_retries,
        ),
        worker_assignment_engine=worker_engine,
        worker_reconciler=worker_reconciler,
        seed=settings.master.orchestration_seed,
    )


def _challenge_orchestrator(settings):
    """Return the challenge service orchestrator for the configured backend.

    Compose is the mission target path (single-host master). Swarm remains only
    as an explicit override for legacy host tooling — never selected by the
    Compose installer.
    """
    import os

    backend = str(
        getattr(settings.docker, "orchestration_backend", "compose") or "compose"
    )
    if backend == "swarm":
        from base.master.swarm_backend import SwarmChallengeOrchestrator

        return SwarmChallengeOrchestrator(
            network_name=settings.docker.network_name,
            internal_network=settings.docker.internal_network,
            docker_broker_url=settings.docker.broker_url,
            challenge_placement_constraint=settings.docker.challenge_placement_constraint,
            # Multi-home the agent-challenge service onto the isolated eval overlay
            # (base_jobs_internal) so its DooD eval job can reach the API for log
            # streaming by name; the job network is base_jobs_internal (see
            # AGENT_CHALLENGE_JOB_NETWORK).
            job_network_slugs=frozenset({AGENT_CHALLENGE_SLUG}),
        )

    from base.master.compose_backend import ComposeChallengeOrchestrator

    project = (
        getattr(settings.docker, "compose_project_name", None)
        or os.environ.get("COMPOSE_PROJECT_NAME")
        or "base-mission-master"
    )
    return ComposeChallengeOrchestrator(
        project_name=str(project),
        compose_file=getattr(
            settings.docker, "compose_file", "/run/base/compose/docker-compose.yml"
        ),
        override_dir=getattr(
            settings.docker, "compose_override_dir", "/var/lib/base/compose-overrides"
        ),
        docker_bin="docker",
    )


def _resolve_master_weight_epoch(
    settings: Any,
    *,
    epoch: int | None = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    """Resolve the integer epoch identity for a durable seal tick.

    Preference order: explicit CLI/operator override, then a deterministic
    wall-clock bucket derived from ``master.epoch_interval_seconds``. Challenge
    raw-weight pushes and the master seal share this identity so snapshots land
    in the same epoch key (VAL-CROSS-067).
    """

    if epoch is not None:
        return int(epoch)
    raw_interval = getattr(settings.master, "epoch_interval_seconds", 360) or 360
    interval = max(1, int(raw_interval))
    return int(now_fn().timestamp()) // interval


async def _run_master_weight_epoch(
    service: MasterWeightService,
    registry: Any,
    *,
    epoch: int,
    netuid: int,
    chain_endpoint: str = "",
) -> FinalWeights:
    """Run one master weight epoch with explicit seal identity.

    When the service is wired with a DB ``session_factory``, this seals and
    publishes from durable ``raw_weight_snapshots`` only. Callers must provide
    concrete ``epoch`` and ``netuid`` so production never falls back to
    intermediate ``get_weights`` (VAL-CROSS-067).
    """

    from base.challenge_sdk.roles import Role, activate_role

    challenges, tokens = await active_challenge_inputs(registry)
    with activate_role(Role.MASTER):
        return await service.run_epoch(
            challenges,
            tokens,
            epoch=int(epoch),
            netuid=int(netuid),
            chain_endpoint=chain_endpoint or "",
        )


async def _run_master_weight_epoch_response(
    service: MasterWeightService,
    registry: Any,
    *,
    netuid: int,
    chain_endpoint: str,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> MasterWeightsResponse:
    from base.challenge_sdk.roles import Role, activate_role

    challenges, tokens = await active_challenge_inputs(registry)
    with activate_role(Role.MASTER):
        return await service.compute_latest_response(
            challenges,
            tokens,
            netuid=netuid,
            chain_endpoint=chain_endpoint,
            now_fn=now_fn,
        )


PRISM_SLUG = "prism"
AGENT_CHALLENGE_SLUG = "agent-challenge"
AGENT_CHALLENGE_TERMINAL_BENCH_RUNNER_IMAGE = (
    "ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner:latest"
)
AGENT_CHALLENGE_SUBMISSION_ENV_SECRET = "submission_env_encryption_key"
AGENT_CHALLENGE_SUBMISSION_ENV_KEY_FILE = (
    f"{DEFAULT_SECRET_MOUNT_DIR}/{AGENT_CHALLENGE_SUBMISSION_ENV_SECRET}"
)
#: own_runner reads the task cache + frozen digest manifest from these in-job
#: paths; the broker bind-mounts them read-only (``broker_eval_readonly_mounts``)
#: from a host path or named volume provisioned out-of-band by
#: ``deploy/swarm/acquire-agent-challenge-cache.sh``.
AGENT_CHALLENGE_TASK_CACHE_DIR = "/opt/agent-challenge/task-cache"
AGENT_CHALLENGE_GOLDEN_DIR = "/opt/agent-challenge/golden"
#: Challenge API service DNS on the challenges overlay (matches
#: ``default_internal_base_url("agent-challenge")``); own_runner jobs POST
#: real-time trial logs here.
AGENT_CHALLENGE_INTERNAL_BASE_URL = "http://challenge-agent-challenge:8000"
#: Overlay the DooD eval JOB attaches to (``CHALLENGE_DOCKER_BROKER_NETWORK``).
#: This is the dedicated ISOLATED eval overlay (``base_jobs_internal``: created
#: ``--internal`` so no internet egress, and NOT hosting postgres). Historical
#: Swarm tooling uses the same name; Compose target path does not create it.
#: See AGENTS.md "Eval job network isolation (base_jobs_internal)".
AGENT_CHALLENGE_JOB_NETWORK = DEFAULT_JOB_NETWORK
PRISM_IMAGE = "ghcr.io/baseintelligence/prism:latest"
PRISM_EVALUATOR_IMAGE = "ghcr.io/baseintelligence/prism-evaluator:latest"
PRISM_VERSION = "0.1.0"
PRISM_EMISSION_PERCENT = Decimal("30")
AGENT_CHALLENGE_EMISSION_PERCENT = Decimal("15")
DEFAULT_BASE_BROKER_URL = "http://base-docker-broker:8082"
#: Master LLM gateway overlay service name + default published port (byte-matches
#: install-swarm.sh ``base-master-proxy`` / ``MASTER_PROXY_PORT``). The master's
#: OWN agent-challenge eval JOB + analyzer run on the ``--internal``
#: ``base_jobs_internal`` overlay (NO egress), so they reach the gateway by this
#: INTERNAL service name, NOT the gateway PUBLIC IP (which is unreachable there).
MASTER_PROXY_SERVICE_NAME = "base-master-proxy"
MASTER_PROXY_SERVICE_PORT = 19080


def _settings_docker_broker_url(settings: Any | None) -> str:
    docker_settings = getattr(settings, "docker", None)
    broker_url = getattr(docker_settings, "broker_url", None)
    return str(broker_url or DEFAULT_BASE_BROKER_URL)


def _parse_eval_readonly_mounts(values: list[str]) -> tuple[tuple[str, str], ...]:
    """Parse ``source:target`` mount specs into ``(source, target)`` tuples.

    ``source`` is an absolute host path or a Docker named volume; ``target`` is
    the absolute container mount path (split on the final ``:`` so neither side
    may itself contain a colon). Malformed entries are skipped.
    """
    parsed: list[tuple[str, str]] = []
    for raw in values:
        source, sep, target = raw.rpartition(":")
        if not sep or not source or not target.startswith("/"):
            continue
        parsed.append((source, target))
    return tuple(parsed)


#: Miner-visible mount target for the locked FineWeb-Edu TRAIN split; the
#: secret val/test splits are NEVER mounted into the eval container. The prism
#: evaluator's ``ctx.data_dir`` resolves to this read-only path.
PRISM_FINEWEB_EDU_TRAIN_DIR = "/data/fineweb-edu/train"
#: Mount target for the offline reference tokenizers (gpt2 tiktoken cache +
#: non-gated llama sentencepiece ``.model``).
PRISM_REFERENCE_TOKENIZER_DIR = "/opt/prism/reference-tokenizers"
#: Docker named volumes staged READ-ONLY on the GPU node (out-of-band, NOT
#: in-band tar) by the data-staging deploy feature. Only the train volume is
#: bound into the miner container; the held-out splits live in separate volumes
#: the eval container never mounts.
PRISM_FINEWEB_EDU_TRAIN_VOLUME = "prism_fineweb_edu_train"
PRISM_REFERENCE_TOKENIZER_VOLUME = "prism_reference_tokenizers"
#: Built-in prism locked-data read-only mounts: train split + reference
#: tokenizers, applied unless ``broker_eval_readonly_mounts_by_slug`` overrides
#: the prism slug. Keeps the broker wiring live before deploy config exists.
DEFAULT_PRISM_EVAL_READONLY_MOUNTS: tuple[tuple[str, str], ...] = (
    (PRISM_FINEWEB_EDU_TRAIN_VOLUME, PRISM_FINEWEB_EDU_TRAIN_DIR),
    (PRISM_REFERENCE_TOKENIZER_VOLUME, PRISM_REFERENCE_TOKENIZER_DIR),
)


def _eval_readonly_mounts_by_slug(
    configured: Mapping[str, list[str]] | None,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Resolve the per-slug read-only eval mounts for the broker.

    The prism slug receives :data:`DEFAULT_PRISM_EVAL_READONLY_MOUNTS` so the
    locked train split + reference tokenizers bind-mount READ-ONLY into the
    eval container out of the box. Any slug present in ``configured`` (from
    ``docker.broker_eval_readonly_mounts_by_slug``) overrides that default with
    its parsed ``source:target`` specs.
    """
    resolved: dict[str, tuple[tuple[str, str], ...]] = {
        PRISM_SLUG: DEFAULT_PRISM_EVAL_READONLY_MOUNTS,
    }
    for slug, specs in (configured or {}).items():
        resolved[slug] = _parse_eval_readonly_mounts(specs)
    return resolved


def _egress_locked_slugs(configured: list[str] | None) -> frozenset[str]:
    """Resolve the egress-locked eval slugs for the broker.

    The prism slug is locked by default so its untrusted eval job is pinned to
    the internal (no external route) overlay out of the box; any slug present
    in ``docker.broker_egress_locked_slugs`` is added to that allowlist.
    """
    return frozenset({PRISM_SLUG, *(configured or ())})


def _prism_image_for_settings(image: str, settings: Any | None) -> str:
    if settings is None or not production_policy_enabled_for_settings(settings):
        return image
    from base.supervisor.image_ref import (
        parse_image_reference,
        resolve_remote_digest,
    )

    reference = parse_image_reference(image)
    if reference.immutable:
        return image
    return reference.pinned(resolve_remote_digest(reference))


def _agent_challenge_own_runner_env(settings: Any | None) -> dict[str, str]:
    """Env retained only for historical call sites; agent-challenge is blocked.

    After LLM-gateway removal, Base refuses to activate/seed/reconcile current
    agent-challenge images with ``AGENT_CHALLENGE_INCOMPATIBLE_NO_LLM_GATEWAY``.
    This helper intentionally omits every gateway URL/token secret.
    """
    broker_url = _settings_docker_broker_url(settings)
    docker_broker_token_file = f"{DEFAULT_SECRET_MOUNT_DIR}/docker_broker_token"
    return {
        "CHALLENGE_BENCHMARK_BACKEND": "terminal_bench",
        "CHALLENGE_DOCKER_ENABLED": "true",
        "CHALLENGE_DOCKER_BACKEND": "broker",
        "CHALLENGE_DOCKER_BROKER_URL": broker_url,
        "CHALLENGE_DOCKER_BROKER_TOKEN_FILE": docker_broker_token_file,
        "CHALLENGE_DOCKER_BROKER_NETWORK": AGENT_CHALLENGE_JOB_NETWORK,
        "CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND": "own_runner",
        "CHALLENGE_HARBOR_RUNNER_IMAGE": AGENT_CHALLENGE_TERMINAL_BENCH_RUNNER_IMAGE,
        "CHALLENGE_OWN_RUNNER_CACHE_ROOT": AGENT_CHALLENGE_TASK_CACHE_DIR,
        "CHALLENGE_OWN_RUNNER_DIGEST_MANIFEST": (
            f"{AGENT_CHALLENGE_GOLDEN_DIR}/dataset-digest.json"
        ),
        "CHALLENGE_TERMINAL_BENCH_LOG_STREAM_URL": AGENT_CHALLENGE_INTERNAL_BASE_URL,
        "CHALLENGE_SUBMISSION_ENV_ENCRYPTION_KEY_FILE": (
            AGENT_CHALLENGE_SUBMISSION_ENV_KEY_FILE
        ),
        "CHALLENGE_EVALUATION_CONCURRENCY": "13",
    }


def _agent_challenge_secret_names(existing: list[str] | None = None) -> list[str]:
    names = [
        "challenge_token",
        "docker_broker_token",
        AGENT_CHALLENGE_SUBMISSION_ENV_SECRET,
    ]
    for name in existing or []:
        if name not in names:
            names.append(name)
    return names


def prism_challenge_create(settings: Any | None = None) -> ChallengeCreate:
    challenge_token_file = f"{DEFAULT_SECRET_MOUNT_DIR}/challenge_token"
    docker_broker_token_file = f"{DEFAULT_SECRET_MOUNT_DIR}/docker_broker_token"
    broker_url = _settings_docker_broker_url(settings)
    prism_image = _prism_image_for_settings(PRISM_IMAGE, settings)
    evaluator_image = _prism_image_for_settings(PRISM_EVALUATOR_IMAGE, settings)
    return ChallengeCreate(
        slug=PRISM_SLUG,
        name="PRISM",
        image=prism_image,
        version=PRISM_VERSION,
        emission_percent=PRISM_EMISSION_PERCENT,
        status=ChallengeStatus.ACTIVE,
        description="PRISM architecture and training reward challenge.",
        internal_base_url="http://challenge-prism:8080",
        required_capabilities=["get_weights", "proxy_routes"],
        resources={
            "cpu": "2",
            "memory": "8g",
        },
        volumes={"data": "/data"},
        env={
            "PRISM_SHARED_TOKEN_FILE": challenge_token_file,
            "CHALLENGE_SHARED_TOKEN_FILE": challenge_token_file,
            "PRISM_DOCKER_ENABLED": "true",
            "PRISM_DOCKER_BACKEND": "broker",
            "CHALLENGE_DOCKER_BACKEND": "broker",
            "PRISM_DOCKER_BROKER_URL": broker_url,
            "CHALLENGE_DOCKER_BROKER_URL": broker_url,
            "PRISM_DOCKER_BROKER_TOKEN_FILE": docker_broker_token_file,
            "CHALLENGE_DOCKER_BROKER_TOKEN_FILE": docker_broker_token_file,
            "PRISM_BASE_EVAL_IMAGE": evaluator_image,
        },
        secrets=["challenge_token", "docker_broker_token"],
        metadata={
            "repository_url": "https://github.com/BaseIntelligence/prism",
            "category": "Agentic (Multi-step)",
            "benchmark_label": "PRISM architecture and training reward boards",
            "evaluation_timeout_seconds": 900,
            "submission_format": "zip",
            # Task 24: PRISM metadata DB is SQLite on the challenge's named
            # LOCAL docker volume (base_prism_sqlite -> /data), WAL mode,
            # single writer (replicas=1). The retired managed Postgres is
            # archived to disk by scripts/archive_prism_postgres.sh (never
            # imported). workload_class is declarative here; the scheduling
            # authority remains ChallengeSpec.workload_class / Swarm Spec.Mode.
            "runtime_database": "challenge-local-sqlite",
            "runtime_database_url": "sqlite+aiosqlite:////data/challenge.sqlite3",
            "runtime_database_journal_mode": "wal",
            "workload_class": "service",
            "base_eval_image": evaluator_image,
            "base_eval_gpu_count": "1",
            "base_eval_max_gpu_count": "8",
            # Combined mode: the single ``challenge-prism`` service runs the API
            # AND the eval-drain worker loop in-process when this env var is set.
            "combined_mode_env": "PRISM_COMBINED_MODE",
        },
    )


def _prism_challenge_update(settings: Any | None = None) -> ChallengeUpdate:
    payload = prism_challenge_create(settings)
    data = payload.model_dump(exclude={"slug"})
    return ChallengeUpdate(**data)


async def seed_prism_challenges(
    registry: Any, settings: Any | None = None
) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        await _resolve(registry.get(PRISM_SLUG))
    except (ChallengeNotFoundError, KeyError):
        await _resolve(registry.create(prism_challenge_create(settings)))
        result[PRISM_SLUG] = "created"
    else:
        await _resolve(registry.update(PRISM_SLUG, _prism_challenge_update(settings)))
        result[PRISM_SLUG] = "updated"

    try:
        await _resolve(registry.get(AGENT_CHALLENGE_SLUG))
    except (ChallengeNotFoundError, KeyError):
        result[AGENT_CHALLENGE_SLUG] = "missing"
    else:
        # Existing registry rows remain visible but must not be re-activated or
        # upgraded through the removed LLM gateway contract.
        result[AGENT_CHALLENGE_SLUG] = AGENT_CHALLENGE_INCOMPATIBLE_CODE
    return result


@master_app.command("proxy")
def master_proxy(config: Path = typer.Option(Path("config/master.example.yaml"))):
    settings = load_settings(config)
    _configure_observability(settings)
    import uvicorn

    _run_startup_migrations(settings)
    engine = create_engine(settings.database.url)
    session_factory = create_session_factory(engine)
    database_probe = postgres_readiness_probe(
        session_factory,
        expected_migration_revision=migration_head(
            str(PROJECT_ROOT / "alembic.ini"),
            settings.database.url,
        ),
    )
    registry = _master_registry(settings, session_factory)
    runtime = create_bittensor_runtime(settings)
    # bittensor init resets the root logger to WARNING ("Enabling default logging
    # (Warning level)"), swallowing this app's INFO records for the rest of the
    # process. Re-assert our logging config AFTER the runtime is built (mirrors
    # deploy/swarm/submitter/run_submitter.py); basicConfig(force=True) clears
    # bittensor's handler and restores our handler with no duplicates.
    configure_logging(
        settings.observability.log_json, level=_resolved_log_level(settings)
    )
    nonce_store = SqlAlchemyMinerNonceStore(
        session_factory,
        ttl_seconds=settings.master.upload_nonce_ttl_seconds,
    )
    # Single public API: the proxy app also serves the admin/registry router, so
    # build the orchestrator + runtime controller + weight service (reusing the
    # already-built metagraph cache) and the admin token provider here. The
    # separate ``master run`` admin server is retired.
    orchestrator = _challenge_orchestrator(settings)
    runtime_controller = DockerRuntimeController(registry, orchestrator)
    weight_service = _master_weight_service(
        settings,
        metagraph_cache=runtime.metagraph_cache,
        session_factory=session_factory,
    )
    raw_weight_ingress_service = RawWeightIngressService(
        session_factory,
        credential_store=ChallengeCredentialStore(registry),
    )
    submission_observation_service = ValidatorSubmissionObservationService(
        session_factory
    )
    # Coordination plane: hotkey-signed register/heartbeat/pull/progress/result
    # routes, the token-gated GET /v1/validators read view, the in-app
    # crash-detection loop, and durable raw-weight push ingress.
    validator_service = _validator_coordination_service(settings, session_factory)
    validator_verifier = _validator_signed_request_verifier(
        settings, session_factory, runtime.metagraph_cache
    )
    # Miner-funded GPU worker plane (architecture.md sec 3.3), gated behind
    # compute.worker_plane_enabled: OFF (default) leaves these None so the worker
    # coordination surface is never mounted and legacy behavior is unchanged.
    worker_service: WorkerCoordinationService | None = None
    worker_verifier: WorkerSignedRequestVerifier | None = None
    worker_assignment_service: WorkerAssignmentService | None = None
    worker_assignment_verifier: WorkerSignedRequestVerifier | None = None
    worker_unit_status_service: WorkerUnitStatusService | None = None
    if settings.compute.worker_plane_enabled:
        worker_service = _worker_coordination_service(
            settings, session_factory, runtime.metagraph_cache
        )
        worker_verifier = _worker_signed_request_verifier(
            settings, session_factory, runtime.metagraph_cache
        )
        worker_assignment_service = _worker_assignment_service(
            settings, session_factory, worker_service
        )
        worker_assignment_verifier = _worker_assignment_verifier(
            settings, session_factory
        )
        worker_unit_status_service = WorkerUnitStatusService(session_factory)
    assignment_service = _assignment_coordination_service(settings, session_factory)
    # Live autonomy: the orchestration driver bridges challenge pending work into
    # work_assignments, runs balanced assignment + the full reassignment pass,
    # and folds retry-exhausted units, all on a Settings-driven interval.
    orchestration_driver = _master_orchestration_driver(
        settings,
        session_factory,
        registry,
        validator_service,
        worker_service=worker_service,
        worker_assignment_service=worker_assignment_service,
    )
    # Registry-driven challenge deploy (architecture.md sec 4 + sec 9.2): the
    # master reconcile loop turns every ACTIVE registry challenge into a running
    # service (idempotent, reusing the orchestrator's existing service) and tears
    # down services for challenges no longer ACTIVE, so a newly-registered
    # challenge auto-deploys with no static per-challenge step.
    registry_reconciler = MasterChallengeReconciler(
        registry=registry, orchestrator=orchestrator
    )
    proxy = create_proxy_app(
        registry=registry,
        metagraph_cache=runtime.metagraph_cache,
        nonce_store=nonce_store,
        netuid=settings.network.netuid,
        upload_signature_ttl_seconds=settings.master.upload_signature_ttl_seconds,
        upload_nonce_ttl_seconds=settings.master.upload_nonce_ttl_seconds,
        upload_max_body_bytes=settings.master.upload_max_body_bytes,
        upload_require_registered_hotkey=settings.master.upload_require_registered_hotkey,
        extra_registered_hotkeys=settings.master.upload_extra_registered_hotkeys,
        runtime_controller=runtime_controller,
        weight_service=weight_service,
        chain_endpoint=settings.network.chain_endpoint or "",
        admin_token_provider=lambda: read_secret(
            settings.security.admin_token,
            settings.security.admin_token_file,
        ),
        enforce_production_policy=production_policy_enabled_for_settings(settings),
        validator_service=validator_service,
        validator_verifier=validator_verifier,
        validator_health_interval_seconds=(
            settings.master.validator_health_interval_seconds
        ),
        worker_service=worker_service,
        worker_verifier=worker_verifier,
        worker_health_interval_seconds=(
            settings.compute.worker_health_interval_seconds
        ),
        worker_assignment_service=worker_assignment_service,
        worker_assignment_verifier=worker_assignment_verifier,
        worker_unit_status_service=worker_unit_status_service,
        assignment_coordination_service=assignment_service,
        raw_weight_ingress_service=raw_weight_ingress_service,
        submission_observation_service=submission_observation_service,
        orchestration_driver=orchestration_driver,
        orchestration_interval_seconds=(settings.master.orchestration_interval_seconds),
        registry_reconciler=registry_reconciler,
        registry_reconcile_interval_seconds=(
            settings.master.registry_reconcile_interval_seconds
        ),
        # Challenge-image auto-roll (architecture.md sec 9.1) runs INSIDE the
        # proxy (which reaches the overlay registry DB + docker socket), not the
        # host supervisor. The default factories build the same registry +
        # DockerRuntimeController the CLI uses; <=0 disables the loop.
        challenge_image_updater_settings=settings,
        challenge_image_update_interval_seconds=(
            settings.master.challenge_image_update_interval_seconds
        ),
        # Compose certificate: digest-pinned challenge watcher (mission target).
        # Independent of the Swarm GHCR tag tracker above; <=0 disables.
        challenge_watcher_settings=settings,
        challenge_watcher_interval_seconds=(
            settings.master.challenge_watcher_interval_seconds
        ),
        identity_resolver=ValidatorIdentityResolver(cache=runtime.identity_cache),
        readiness_probes=(database_probe,),
    )
    endpoint = f"{settings.master.proxy_host}:{settings.master.proxy_port}"
    typer.echo(f"Starting proxy API on {endpoint}")
    uvicorn.run(proxy, host=settings.master.proxy_host, port=settings.master.proxy_port)


@master_app.command("broker")
def master_broker(config: Path = typer.Option(Path("config/master.example.yaml"))):
    settings = load_settings(config)
    _configure_observability(settings)
    import uvicorn

    _run_startup_migrations(settings)
    registry = _master_registry(settings)
    from base.master.swarm_backend import (
        SwarmBrokerConfig,
        SwarmBrokerService,
    )

    docker_service = SwarmBrokerService(
        SwarmBrokerConfig(
            workspace_dir=Path(settings.docker.broker_workspace_dir),
            allowed_images=tuple(settings.docker.broker_allowed_images),
            log_limit_bytes=settings.docker.broker_log_limit_bytes,
            max_concurrent_global=settings.docker.broker_max_concurrent_global,
            node_role=settings.docker.broker_node_role,
            privileged_escape_slugs=(
                frozenset(settings.docker.broker_privileged_slugs)
                if settings.docker.allow_privileged
                else frozenset()
            ),
            allow_privileged_escape=(
                settings.docker.allow_privileged
                and settings.docker.broker_allow_privileged_escape
            ),
            cpu_job_constraint=settings.docker.cpu_job_constraint,
            gpu_job_constraint=settings.docker.gpu_job_constraint,
            docker_socket_slugs=frozenset(settings.docker.broker_docker_socket_slugs),
            docker_socket_path=settings.docker.broker_docker_socket_path,
            eval_readonly_mounts=_parse_eval_readonly_mounts(
                settings.docker.broker_eval_readonly_mounts
            ),
            eval_readonly_mounts_by_slug=_eval_readonly_mounts_by_slug(
                settings.docker.broker_eval_readonly_mounts_by_slug
            ),
            egress_locked_slugs=_egress_locked_slugs(
                settings.docker.broker_egress_locked_slugs
            ),
        )
    )
    broker = create_docker_broker_app(registry=registry, service=docker_service)
    endpoint = f"{settings.docker.broker_host}:{settings.docker.broker_port}"
    typer.echo(f"Starting Docker broker API on {endpoint}")
    uvicorn.run(
        broker, host=settings.docker.broker_host, port=settings.docker.broker_port
    )


@master_app.command("supervisor")
def master_supervisor(config: Path = typer.Option(Path("config/master.example.yaml"))):
    """Unsupported legacy host supervisor (historical Swarm control plane).

    Compose is the only supported shipping runtime. This command remains only
    for frozen multi-host tooling and is not part of the operator install path.
    Prefer ``deploy/compose/install-master.sh``.
    """
    settings = load_settings(config)
    _configure_observability(settings)
    from base.supervisor import build_supervisor

    typer.echo(
        "WARNING: base master supervisor is a historical/non-target surface. "
        "Compose installers (deploy/compose/install-master.sh) are the supported "
        "operator entrypoint.",
        err=True,
    )
    supervisor = build_supervisor(settings)
    typer.echo(
        f"Starting platform supervisor with {len(supervisor.tasks)} scheduled task(s)"
    )
    raise typer.Exit(code=supervisor.run())


def _docker_cli(args: list[str]) -> None:
    import subprocess

    completed = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        typer.echo(completed.stdout.rstrip())
    if completed.returncode != 0:
        if completed.stderr:
            typer.echo(completed.stderr.rstrip(), err=True)
        raise typer.Exit(code=completed.returncode)


def _warn_master_worker_historical() -> None:
    """Surface that ``base master worker`` is non-target on Compose installs."""
    typer.echo(
        "WARNING: base master worker is a historical/non-target Swarm surface. "
        "Compose installers (deploy/compose/install-master.sh) are the supported "
        "operator entrypoint.",
        err=True,
    )


@worker_app.command("token")
def worker_token(
    role: str = typer.Option("worker", "--role", help="worker or manager"),
    rotate: bool = typer.Option(False, "--rotate", help="Rotate the join token first."),
):
    """Historical: print the ``docker swarm join`` command (non-target)."""
    _warn_master_worker_historical()
    if role not in {"worker", "manager"}:
        raise typer.BadParameter("role must be 'worker' or 'manager'")
    args = ["swarm", "join-token"]
    if rotate:
        args.append("--rotate")
    args.append(role)
    _docker_cli(args)


@worker_app.command("list")
def worker_list():
    """Historical: list Swarm nodes (``docker node ls``; non-target)."""
    _warn_master_worker_historical()
    _docker_cli(["node", "ls"])


@worker_app.command("label")
def worker_label(
    node: str,
    workload: str = typer.Option(..., "--workload", help="cpu or gpu"),
):
    """Historical: label a Swarm node (non-target)."""
    _warn_master_worker_historical()
    if workload not in {"cpu", "gpu"}:
        raise typer.BadParameter("workload must be 'cpu' or 'gpu'")
    _docker_cli(["node", "update", "--label-add", f"base.workload={workload}", node])


@worker_app.command("drain")
def worker_drain(
    node: str,
    active: bool = typer.Option(
        False, "--active", help="Restore availability=active instead of draining."
    ),
):
    """Historical: drain or reactivate a Swarm node (non-target)."""
    _warn_master_worker_historical()
    availability = "active" if active else "drain"
    _docker_cli(["node", "update", "--availability", availability, node])


@worker_app.command("rm")
def worker_rm(
    node: str,
    force: bool = typer.Option(False, "--force"),
):
    """Historical: remove a Swarm node (non-target)."""
    _warn_master_worker_historical()
    args = ["node", "rm"]
    if force:
        args.append("--force")
    args.append(node)
    _docker_cli(args)


@worker_app.command("inspect")
def worker_inspect(node: str):
    """Historical: inspect a Swarm node (non-target)."""
    _warn_master_worker_historical()
    _docker_cli(["node", "inspect", node])


@master_app.command("refresh-challenge-images")
def master_refresh_challenge_images(
    config: Path = typer.Option(Path("config/master.example.yaml")),
    tag: str = typer.Option("latest", "--tag"),
):
    settings = load_settings(config)
    registry = _master_registry(settings)
    controller = DockerRuntimeController(registry, _challenge_orchestrator(settings))

    def mutable_base(image: str) -> str | None:
        from base.supervisor.image_ref import parse_image_reference

        parsed = parse_image_reference(image)
        if parsed.registry != "ghcr.io":
            return None
        if parsed.tag.startswith("sha-"):
            return None
        return f"{parsed.registry}/{parsed.repository}:{tag}"

    async def refresh() -> None:
        from base.schemas.challenge import ChallengeStatus, ChallengeUpdate
        from base.supervisor.image_ref import (
            parse_image_reference,
            resolve_remote_digest,
        )

        for record in await registry.list():
            if record.status in {ChallengeStatus.DRAFT, ChallengeStatus.DISABLED}:
                continue
            base = mutable_base(record.image)
            if base is None:
                typer.echo(f"{record.slug}: skipped {record.image}")
                continue
            digest = resolve_remote_digest(parse_image_reference(base))
            desired = f"{base}@{digest}"
            changed = desired != record.image
            if changed:
                await registry.update(record.slug, ChallengeUpdate(image=desired))
                typer.echo(f"{record.slug}: updated {desired}")
            else:
                typer.echo(f"{record.slug}: already-current {desired}")
            if record.status == ChallengeStatus.ACTIVE and changed:
                result = await controller.restart(record.slug)
                typer.echo(f"{record.slug}: restarted {result['status']}")

    asyncio.run(refresh())


@master_challenges_app.command("seed-prism")
def master_challenges_seed_prism(
    config: Path = typer.Option(Path("config/master.example.yaml")),
):
    settings = load_settings(config)
    registry = _master_registry(settings)

    async def seed() -> None:
        result = await seed_prism_challenges(registry, settings)
        typer.echo(f"prism: {result[PRISM_SLUG]} emission={PRISM_EMISSION_PERCENT}")
        typer.echo(
            "agent-challenge: "
            f"{result[AGENT_CHALLENGE_SLUG]} "
            f"emission={AGENT_CHALLENGE_EMISSION_PERCENT}"
        )

    asyncio.run(seed())


@master_app.command("weights")
def master_weights(
    config: Path = typer.Option(Path("config/master.example.yaml")),
    once: bool = typer.Option(False, "--once/--loop"),
    epoch: int | None = typer.Option(
        None,
        "--epoch",
        help="Optional explicit epoch identity for durable seal; defaults to "
        "wall-clock bucket from master.epoch_interval_seconds.",
    ),
):
    settings = load_settings(config)
    _configure_observability(settings)
    _run_startup_migrations(settings)
    engine = create_engine(settings.database.url)
    session_factory = create_session_factory(engine)
    registry = _master_registry(settings, session_factory)
    runtime = create_bittensor_runtime(settings)
    # Re-assert logging AFTER bittensor init (which resets root logging to
    # WARNING) so this loop's INFO records are not swallowed; see master_proxy.
    configure_logging(
        settings.observability.log_json, level=_resolved_log_level(settings)
    )
    service = _master_weight_service(
        settings,
        metagraph_cache=runtime.metagraph_cache,
        session_factory=session_factory,
    )
    resolved_epoch = _resolve_master_weight_epoch(settings, epoch=epoch)
    netuid = int(settings.network.netuid)
    chain_endpoint = str(settings.network.chain_endpoint or "")

    async def epoch_tick() -> None:
        final = await _run_master_weight_epoch(
            service,
            registry,
            epoch=resolved_epoch
            if once
            else _resolve_master_weight_epoch(settings, epoch=epoch),
            netuid=netuid,
            chain_endpoint=chain_endpoint,
        )
        typer.echo(f"computed {len(final.uids)} weights")

    if once:
        asyncio.run(epoch_tick())
        return
    asyncio.run(run_epoch_loop(settings.master.epoch_interval_seconds, epoch_tick))


async def _run_validator_runtime(
    runner: NormalValidatorRunner,
    weights_interval_seconds: int,
) -> None:
    async def submit_weights() -> None:
        await runner.submit_latest_weights()

    await asyncio.gather(
        runner.run_forever(),
        run_epoch_loop(weights_interval_seconds, submit_weights),
    )


@contextlib.asynccontextmanager
async def _validator_status_client(
    base_url: str,
    timeout_seconds: float,
):
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        yield client


async def _read_validator_runtime_status(settings: Any):
    from base.challenge_sdk.schemas import HealthResponse, VersionResponse

    agent_settings = settings.validator.agent
    master_url = agent_settings.master_url or settings.validator.resolved_weights_url
    master_health: HealthResponse | None = None
    master_version: VersionResponse | None = None
    try:
        async with _validator_status_client(
            master_url,
            agent_settings.request_timeout_seconds,
        ) as client:
            health_response = await client.get("/health")
            health_response.raise_for_status()
            master_health = HealthResponse.model_validate(health_response.json())
            version_response = await client.get("/version")
            version_response.raise_for_status()
            master_version = VersionResponse.model_validate(version_response.json())
    except Exception:
        pass
    return validator_runtime_status(
        master_health=master_health,
        master_version=master_version,
        submission_enabled=settings.validator.submit_on_chain_enabled,
    )


@validator_app.command("status")
def validator_status(
    config: Path = typer.Option(Path("config/validator.example.yaml")),
) -> None:
    """Report validator identity, capabilities, and master-aware readiness."""

    settings = load_settings(config)
    runtime_status = asyncio.run(_read_validator_runtime_status(settings))
    typer.echo(runtime_status.model_dump_json())
    if not runtime_status.health.ready:
        raise typer.Exit(code=1)


@validator_app.command("run")
def validator_run(config: Path = typer.Option(Path("config/validator.example.yaml"))):
    """Legacy registry-sync runner WITHOUT on-chain weight submission.

    The sole gated ``set_weights`` path is ``base validator agent`` via
    :class:`ValidatorWeightSubmitter`. This legacy entry point no longer
    constructs a ``WeightSetter`` / calls ``set_weights`` so operators cannot
    bypass ledger, identity, provenance, or observation wiring.
    """

    settings = load_settings(config)
    _configure_observability(settings)
    # Registry sync only — do not build a submit runtime or inject WeightSetter.
    configure_logging(
        settings.observability.log_json, level=_resolved_log_level(settings)
    )
    runner = NormalValidatorRunner(
        registry_client=RegistryClient(settings.validator.registry_url),
        orchestrator=_challenge_orchestrator(settings),
        retry_seconds=settings.validator.registry_retry_seconds,
        weights_client=None,
        weight_setter=None,
        netuid=settings.network.netuid,
        weights_freshness_seconds=settings.validator.weights_freshness_seconds,
        allow_weight_submission=False,
    )
    asyncio.run(
        _run_validator_runtime(runner, settings.validator.weights_interval_seconds)
    )


def _require_validator_master_url(settings: Any) -> str:
    """Resolve the explicit master coordination URL for a validator process.

    VAL-SDK-086: missing/invalid master URL must fail closed without inventing a
    localhost default or silently treating the validator as its own master.
    ``registry_url`` / ``weights_url`` alone are not enough for coordination
    installer surfaces (Compose, CLI agent).
    """

    agent_cfg = settings.validator.agent
    master_url = getattr(agent_cfg, "master_url", None)
    if not master_url or not str(master_url).strip():
        raise typer.BadParameter(
            "validator.agent.master_url is required (absolute http/https URL to "
            "the master coordination plane); refusing to default to localhost"
        )
    url = str(master_url).strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        raise typer.BadParameter(
            "validator.agent.master_url must be an absolute http:// or https:// URL"
        )
    return url


def _require_validator_protocol_identity(settings: Any) -> Any:
    """Load and validate the protocol signing identity (hotkey keypair).

    The protocol identity is required in every mode for registration/heartbeat.
    A Bittensor submission wallet is only constructed when submission is
    explicitly enabled (see weight submitter wiring).
    """

    try:
        keypair = create_validator_keypair(settings)
    except Exception as exc:  # noqa: BLE001 - surface operator-facing failure
        raise typer.BadParameter(
            "validator protocol signing identity is missing or unreadable "
            "(network.wallet_name / wallet_hotkey / wallet_path); "
            "refusing to start with anonymous identity"
        ) from exc
    ss58 = getattr(keypair, "ss58_address", None)
    if not ss58:
        raise typer.BadParameter(
            "validator protocol signing identity did not yield a hotkey address"
        )
    return keypair


def _build_coordination_client(settings: Any) -> CoordinationClient:
    """Build a signed coordination client from validator settings (testable)."""

    agent_cfg = settings.validator.agent
    master_url = _require_validator_master_url(settings)
    signer = KeypairRequestSigner(_require_validator_protocol_identity(settings))
    return CoordinationClient(
        master_url,
        signer,
        timeout_seconds=agent_cfg.request_timeout_seconds,
    )


def _build_validator_agent(settings: Any) -> ValidatorAgent:
    """Wire the decentralized validator agent from settings (testable)."""

    agent_cfg = settings.validator.agent
    client = _build_coordination_client(settings)
    broker = BrokerConfig(
        broker_url=agent_cfg.broker_url or settings.docker.broker_url,
        broker_token=agent_cfg.broker_token,
        broker_token_file=agent_cfg.broker_token_file,
        allowed_images=tuple(
            [*settings.docker.broker_allowed_images, *agent_cfg.allowed_images]
        ),
    )
    executor = ChallengeDispatchExecutor(
        generic=BrokerAssignmentExecutor(
            run_timeout_seconds=agent_cfg.run_timeout_seconds
        )
    )
    identity_meta: dict[str, Any] = {}
    if agent_cfg.display_name is not None:
        identity_meta[IDENTITY_DISPLAY_NAME_KEY] = agent_cfg.display_name
    if agent_cfg.logo_url is not None:
        identity_meta[IDENTITY_LOGO_URL_KEY] = agent_cfg.logo_url
    last_seen_meta_factory: Callable[[], Mapping[str, Any]] | None = (
        (lambda: dict(identity_meta)) if identity_meta else None
    )
    return ValidatorAgent(
        client=client,
        executor=executor,
        broker=broker,
        capabilities=list(agent_cfg.capabilities),
        version=agent_cfg.version,
        heartbeat_interval_seconds=agent_cfg.heartbeat_interval_seconds,
        poll_interval_seconds=agent_cfg.poll_interval_seconds,
        last_seen_meta_factory=last_seen_meta_factory,
    )


def _build_validator_weight_submitter(settings: Any) -> ValidatorWeightSubmitter:
    """Wire this validator's OWN on-chain weight submitter (architecture.md 9.3).

    The submitter fetches the master-aggregated vector from ``/v1/weights/latest``
    and commits it under THIS node's hotkey (its own ``WeightSetter``), gated by
    ``validator.submit_on_chain_enabled`` (default off). It never aggregates its
    own vector. The ``WeightSetter`` is built lazily so a gate-off validator never
    constructs a live ``Subtensor``. Durable ledger state lives under
    ``validator.submission_state_dir`` (validator Compose volume).

    Production wiring (sole gated path):
    * ``require_provenance=True`` (forced)
    * fail-closed identity when enabled and expected hotkey cannot be bound
    * optional non-authoritative observation reporter to master
    * UNKNOWN outcomes held/reconciled across restart (no blind multi-submit)
    """

    weights_client = WeightsClient(
        settings.validator.resolved_weights_url,
        timeout_seconds=settings.validator.weights_timeout_seconds,
        retries=settings.validator.weights_retries,
    )
    expected_hotkey: str | None = None
    identity_unbound = False
    if settings.validator.submit_on_chain_enabled:
        # Public identity fingerprint only when submission is explicitly enabled.
        # Disabled mode must not construct a submission wallet/Subtensor.
        try:
            expected_hotkey = str(create_validator_keypair(settings).ss58_address)
        except Exception:
            # Fail closed: leave expected_hotkey unset so the submitter rejects
            # identity before set_weights rather than submitting anonymously.
            expected_hotkey = None
            identity_unbound = True
            logging.getLogger(__name__).error(
                "validator submit enabled but cannot bind expected hotkey; "
                "identity will fail closed on every tick"
            )
        if not expected_hotkey:
            identity_unbound = True

    async def _report_observation(payload: dict[str, Any]) -> Any:
        return await weights_client.report_submission_observation(payload)

    submitter = ValidatorWeightSubmitter(
        submit_enabled=settings.validator.submit_on_chain_enabled,
        netuid=settings.network.netuid,
        weights_client=weights_client,
        weight_setter_factory=lambda: (
            create_bittensor_submit_runtime(settings).weight_setter
        ),
        weights_freshness_seconds=settings.validator.weights_freshness_seconds,
        expected_hotkey=expected_hotkey,
        expected_chain_endpoint=settings.network.chain_endpoint or "",
        state_dir=settings.validator.submission_state_dir,
        max_attempts=settings.validator.submission_max_attempts,
        backoff_base_seconds=settings.validator.submission_backoff_base_seconds,
        backoff_max_seconds=settings.validator.submission_backoff_max_seconds,
        require_provenance=True,
        observation_reporter=_report_observation,
    )
    # Surface the fail-closed binding explicitly for tests and operators.
    submitter._identity_unbound = identity_unbound  # type: ignore[attr-defined]
    return submitter


async def _run_validator_agent_runtime(
    agent: ValidatorAgent,
    submitter: ValidatorWeightSubmitter,
    weights_interval_seconds: int,
) -> None:
    """Run the agent loop and this node's OWN weight-submit loop concurrently.

    The submit loop runs alongside the agent so every validator node that runs
    ``base validator agent`` also submits its OWN on-chain weights (a no-op while
    the gate is off). Submit failures never crash the agent (``run_epoch_loop``
    swallows tick errors); when the agent loop exits the submit loop is cancelled.
    """

    async def submit_weights() -> None:
        from base.challenge_sdk.roles import Role, activate_role

        with activate_role(Role.VALIDATOR):
            await submitter.run_once()

    submit_task = asyncio.create_task(
        run_epoch_loop(weights_interval_seconds, submit_weights)
    )
    try:
        await agent.run_forever()
    finally:
        submit_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await submit_task


@validator_app.command("agent")
def validator_agent(
    config: Path = typer.Option(Path("config/validator.example.yaml")),
):
    """Run the decentralized validator agent (own-broker executor + submitter).

    Hotkey-registers + heartbeats with the master, pulls assignments, executes
    them on the validator's OWN broker + Docker, posts results, and routes all
    LLM calls through the master gateway (no provider key on the validator). In
    the same runtime it runs THIS node's OWN on-chain weight submitter, which
    fetches the master-aggregated vector and commits it under this validator's
    hotkey (a no-op while ``validator.submit_on_chain_enabled`` is off).
    """

    settings = load_settings(config)
    _configure_observability(settings)
    # Fail closed before constructing loops when master URL/identity are absent.
    _require_validator_master_url(settings)
    _require_validator_protocol_identity(settings)
    agent = _build_validator_agent(settings)
    submitter = _build_validator_weight_submitter(settings)
    typer.echo(f"Starting validator agent for hotkey {agent.hotkey}")
    asyncio.run(
        _run_validator_agent_runtime(
            agent, submitter, settings.validator.weights_interval_seconds
        )
    )


@validator_app.command("subscribe")
def validator_subscribe(
    config: Path = typer.Option(Path("config/validator.example.yaml")),
    challenges: str = typer.Option(
        "",
        "--challenges",
        help=(
            "Comma-separated challenge slugs to validate (e.g. "
            "'prism,agent-challenge'). Pass an empty value to clear the "
            "subscription (validate ALL challenges)."
        ),
    ),
):
    """Set this validator's challenge subscriptions (sr25519-signed).

    The master only assigns subscribed challenges to a validator with a
    non-empty subscription; an empty subscription validates ALL challenges.
    """

    settings = load_settings(config)
    _configure_observability(settings)
    slugs = [slug.strip() for slug in challenges.split(",") if slug.strip()]
    client = _build_coordination_client(settings)
    response = asyncio.run(client.subscribe(slugs))
    if response.subscriptions:
        typer.echo(
            "Subscribed validator "
            f"{client.hotkey} to: {', '.join(response.subscriptions)}"
        )
    else:
        typer.echo(
            f"Cleared subscriptions for validator {client.hotkey} "
            "(validating ALL challenges)"
        )


# -- worker plane (base worker ...) ------------------------------------------
#
# Top-level ``base worker`` app (DISTINCT from the legacy ``base master worker``
# Swarm-node group). Deploys a miner-funded GPU worker agent locally or onto a
# rented provider instance, runs the agent loop, and renders fleet status. The
# provider API key comes from the MINER's env and is NEVER sent to the master.


def _worker_master_url(settings: Any) -> str:
    url = settings.worker.agent.master_url
    if not url:
        raise typer.BadParameter(
            "worker.agent.master_url must be set to reach the master coordination plane"
        )
    return str(url)


def _build_worker_binding(settings: Any, worker_pubkey: str) -> WorkerBinding:
    """Resolve the miner-signed enrollment binding for the worker.

    Prefers a pre-signed binding (miner_hotkey + signature + nonce) so a pod that
    never holds the miner key can enroll; otherwise signs a fresh binding with the
    configured miner keypair at deploy time.
    """

    identity = settings.worker.identity
    if identity.binding_signature and identity.miner_hotkey and identity.binding_nonce:
        return WorkerBinding(
            miner_hotkey=str(identity.miner_hotkey),
            signature=str(identity.binding_signature),
            nonce=str(identity.binding_nonce),
        )
    miner_keypair = create_worker_miner_keypair(settings)
    if miner_keypair is None:
        raise typer.BadParameter(
            "worker deploy needs a miner key (worker.identity.miner_key_uri / "
            "miner_key_mnemonic / miner_wallet_*) or a pre-signed binding "
            "(worker.identity.miner_hotkey + binding_signature + binding_nonce)"
        )
    return build_signed_binding(
        worker_pubkey=worker_pubkey,
        miner_signer=KeypairRequestSigner(miner_keypair),
    )


def _build_worker_agent(
    settings: Any,
    *,
    provider: str,
    provider_instance_ref: str | None = None,
) -> WorkerAgent:
    """Wire the miner-funded worker agent from settings (testable).

    The agent signs coordination requests + ExecutionProofs with the WORKER
    keypair, executes gpu units on its OWN broker via the shared executor seam,
    and enrolls under a miner-signed binding. No provider API key is involved.
    """

    agent_cfg = settings.worker.agent
    worker_keypair = create_worker_keypair(settings)
    signer = KeypairRequestSigner(worker_keypair)
    binding = _build_worker_binding(settings, signer.hotkey)
    broker = BrokerConfig(
        broker_url=agent_cfg.broker_url or settings.docker.broker_url,
        broker_token=agent_cfg.broker_token,
        broker_token_file=agent_cfg.broker_token_file,
        allowed_images=tuple(
            [*settings.docker.broker_allowed_images, *agent_cfg.allowed_images]
        ),
    )
    executor = WorkerProofExecutor(
        ChallengeDispatchExecutor(
            generic=BrokerAssignmentExecutor(
                run_timeout_seconds=agent_cfg.run_timeout_seconds
            )
        ),
        signer=signer,
        provenance=WorkerProvenance(
            provider_name=provider, miner_hotkey=binding.miner_hotkey
        ),
    )
    master_url = _worker_master_url(settings)
    client = WorkerCoordinationClient(
        master_url, signer, timeout_seconds=agent_cfg.request_timeout_seconds
    )
    return WorkerAgent(
        client=client,
        executor=executor,
        broker=broker,
        binding=binding,
        provider=provider,
        provider_instance_ref=(
            provider_instance_ref or settings.worker.deploy.provider_instance_ref
        ),
        capabilities=list(agent_cfg.capabilities),
        heartbeat_interval_seconds=agent_cfg.heartbeat_interval_seconds,
        poll_interval_seconds=agent_cfg.poll_interval_seconds,
    )


def _spawn_worker_agent_process(config: Path) -> Any:
    """Launch a detached ``base worker agent`` process (monkeypatchable seam)."""

    import subprocess
    import sys

    return subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "base.cli_app.main",
            "worker",
            "agent",
            "--config",
            str(config),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def _wait_worker_active(
    settings: Any,
    signer: KeypairRequestSigner,
    worker_pubkey: str,
    *,
    ready_timeout: float,
    poll_interval: float = 3.0,
) -> Any:
    """Poll ``GET /v1/workers`` until ``worker_pubkey`` is active or time out."""

    import time as _time

    client = WorkerCoordinationClient(
        _worker_master_url(settings),
        signer,
        timeout_seconds=settings.worker.agent.request_timeout_seconds,
    )
    deadline = _time.monotonic() + ready_timeout
    last_error: Exception | None = None
    while _time.monotonic() < deadline:
        try:
            for worker in await client.list_workers():
                if worker.worker_pubkey == worker_pubkey and worker.status == "active":
                    return worker
        except Exception as exc:  # noqa: BLE001 - retried until the deadline
            last_error = exc
        await asyncio.sleep(poll_interval)
    detail = f": {last_error}" if last_error is not None else ""
    raise WorkerDeployError(
        f"worker {worker_pubkey} did not reach active within "
        f"{ready_timeout:.0f}s{detail}"
    )


def _run_worker_local_deploy(settings: Any, config: Path) -> None:
    worker_keypair = create_worker_keypair(settings)
    worker_pubkey = str(worker_keypair.ss58_address)
    signer = KeypairRequestSigner(worker_keypair)
    process = _spawn_worker_agent_process(config)
    typer.echo(f"Started worker agent process pid={process.pid} (provider=local)")
    ready_timeout = settings.worker.deploy.ready_timeout_seconds
    try:
        worker = asyncio.run(
            _wait_worker_active(
                settings, signer, worker_pubkey, ready_timeout=ready_timeout
            )
        )
    except BaseException as exc:
        with contextlib.suppress(Exception):
            process.terminate()
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Worker {worker.worker_id} active "
        f"(pubkey={worker.worker_pubkey}, owner={worker.miner_hotkey}, "
        f"provider={worker.provider})"
    )


def _worker_ssh_public_keys(deploy_cfg: Any) -> tuple[str, ...]:
    if deploy_cfg.ssh_public_key:
        return (str(deploy_cfg.ssh_public_key).strip(),)
    if deploy_cfg.ssh_public_key_file:
        text = Path(deploy_cfg.ssh_public_key_file).read_text(encoding="utf-8").strip()
        return (text,) if text else ()
    return ()


def _worker_instance_spec(
    settings: Any,
    *,
    provider: str,
    offer: Any,
    env: Mapping[str, str],
    max_price: float | None,
    image: str,
    image_digest: str,
) -> InstanceSpec:
    deploy_cfg = settings.worker.deploy
    return InstanceSpec(
        name=f"prism-worker-{uuid.uuid4().hex[:12]}",
        template_ref=deploy_cfg.template_name or WORKER_TEMPLATE_NAME,
        image=image,
        image_digest=image_digest,
        env=dict(env),
        ports=(22,),
        ssh_public_keys=_worker_ssh_public_keys(deploy_cfg),
        ssh_key_name=deploy_cfg.ssh_key_name,
        startup_commands=deploy_cfg.startup_commands,
        max_lifetime_hours=deploy_cfg.max_lifetime_hours,
        max_price_per_hour=(
            max_price if max_price is not None else float(offer.price_per_hour)
        ),
        gpu_count=deploy_cfg.gpu_count,
    )


async def _run_worker_provider_deploy_async(
    settings: Any,
    *,
    provider: str,
    api_key: str,
    max_price: float | None,
) -> None:
    deploy_cfg = settings.worker.deploy
    # Fail fast (before any provider network call) when the deploy has no explicit,
    # publicly-pullable, digest-pinned worker image: the deploy never silently pins
    # a private-namespace placeholder that fails Lium pod creation.
    image, image_digest = require_worker_image(
        image=deploy_cfg.image,
        image_digest=deploy_cfg.image_digest,
        provider=provider,
    )
    resolved_max_price = (
        max_price if max_price is not None else deploy_cfg.max_price_per_hour
    )
    client: Any = LiumClient(api_key) if provider == "lium" else TargonClient(api_key)
    # Planning is offer selection only: no rent/deploy call is issued here, and an
    # all-over-cap situation raises before anything is provisioned.
    offer = await plan_provider_deployment(
        client, gpu_count=deploy_cfg.gpu_count, max_price=resolved_max_price
    )
    typer.echo(
        f"Selected {provider} offer {offer.id} "
        f"({offer.gpu_type} x{offer.gpu_count}) @ {offer.price_per_hour}/GPU/hr"
    )
    worker_keypair = create_worker_keypair(settings)
    signer = KeypairRequestSigner(worker_keypair)
    binding = _build_worker_binding(settings, signer.hotkey)
    pod_env = build_worker_pod_env(
        master_url=_worker_master_url(settings),
        provider=provider,
        binding=binding,
        worker_key_uri=settings.worker.identity.key_uri,
        worker_key_mnemonic=settings.worker.identity.key_mnemonic,
        broker_url=settings.worker.agent.broker_url,
    )
    spec = _worker_instance_spec(
        settings,
        provider=provider,
        offer=offer,
        env=pod_env,
        max_price=resolved_max_price,
        image=image,
        image_digest=image_digest,
    )
    if provider == "lium":
        instance = await client.provision(spec, offer=offer)
    else:
        instance = await client.provision(spec)
    typer.echo(
        f"Provisioned {provider} instance {instance.id} (status={instance.status}); "
        "the worker enrolls with the master on boot"
    )


@worker_plane_app.command("agent")
def worker_agent(
    config: Path = typer.Option(Path("config/worker.example.yaml")),
):
    """Run the miner-funded GPU worker agent loop.

    Registers with the master under a miner-signed binding, heartbeats to stay
    active, pulls gpu work units, executes them on the worker's OWN broker, and
    posts ExecutionProof-carrying results. Authenticates as the worker keypair and
    never holds a provider API key.
    """

    settings = load_settings(config)
    _configure_observability(settings)
    agent = _build_worker_agent(settings, provider=settings.worker.deploy.provider)
    typer.echo(f"Starting worker agent for pubkey {agent.worker_pubkey}")
    asyncio.run(agent.run_forever())


@worker_plane_app.command("deploy")
def worker_deploy(
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Where to run the worker agent: lium | targon | local.",
    ),
    max_price: float | None = typer.Option(
        None,
        "--max-price",
        help="Max price per GPU/hour bounding provider offer selection.",
    ),
    config: Path = typer.Option(Path("config/worker.example.yaml")),
):
    """Deploy a worker agent locally or onto a rented provider instance.

    ``--provider local`` starts an agent against the local master. ``--provider
    lium|targon`` requires the provider API key env (``LIUM_API_KEY`` /
    ``TARGON_API_KEY``), selects an offer within ``--max-price`` (preferring an
    exact GPU-count executor), and provisions the worker image. The provider key
    is used only to authenticate provider calls and is NEVER sent to the master.
    """

    try:
        normalized = normalize_provider(provider)
    except UnsupportedProviderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    # Enforce the provider key BEFORE loading config or touching the network so a
    # missing key is an actionable refusal with no side effects (VAL-AGENT-010).
    api_key: str | None = None
    if normalized != LOCAL_PROVIDER:
        try:
            api_key = require_provider_api_key(normalized)
        except MissingProviderKeyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc

    settings = load_settings(config)
    _configure_observability(settings)

    if normalized == LOCAL_PROVIDER:
        _run_worker_local_deploy(settings, config)
        return

    assert api_key is not None
    try:
        asyncio.run(
            _run_worker_provider_deploy_async(
                settings, provider=normalized, api_key=api_key, max_price=max_price
            )
        )
    except NoOfferWithinBudgetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except WorkerDeployError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@worker_plane_app.command("status")
def worker_status(
    config: Path = typer.Option(Path("config/worker.example.yaml")),
    hotkey: str | None = typer.Option(
        None,
        "--hotkey",
        help="Show only the ACTIVE workers of this miner hotkey.",
    ),
):
    """Render the worker fleet from the master's ``GET /v1/workers``.

    Reads the fleet as the worker keypair (a registered worker is an eligible
    coordination reader) and prints each worker's status, owner, provider,
    last-seen timestamp, and attributed fault count (so the CLI fleet view agrees
    with the ``GET /v1/workers`` JSON, VAL-CROSS-009).
    """

    settings = load_settings(config)
    _configure_observability(settings)
    signer = KeypairRequestSigner(create_worker_keypair(settings))
    client = WorkerCoordinationClient(
        _worker_master_url(settings),
        signer,
        timeout_seconds=settings.worker.agent.request_timeout_seconds,
    )
    workers = asyncio.run(client.list_workers(hotkey=hotkey))
    if not workers:
        typer.echo("No workers registered.")
        return
    typer.echo(
        f"{'WORKER_ID':<20} {'OWNER':<20} {'PROVIDER':<10} {'STATUS':<8} "
        f"{'FAULTS':<6} LAST_SEEN"
    )
    for worker in workers:
        last_seen = (
            worker.last_heartbeat_at.isoformat()
            if worker.last_heartbeat_at is not None
            else "-"
        )
        fault_count = len(getattr(worker, "faults", None) or [])
        typer.echo(
            f"{worker.worker_id:<20} {worker.miner_hotkey:<20} "
            f"{worker.provider:<10} {worker.status:<8} {fault_count:<6} {last_seen}"
        )


@challenge_app.command("create")
def challenge_create(
    slug: str,
    out: Path = typer.Option(..., "--out", help="Destination challenge repo path."),
    name: str | None = None,
    image: str | None = None,
    version: str = "0.1.0",
    overwrite: bool = False,
):
    context = ChallengeTemplateContext.from_slug(
        slug, name=name, ghcr_image=image, challenge_version=version
    )
    written = render_challenge_template(out, context, overwrite=overwrite)
    typer.echo(f"Created challenge template at {out} ({len(written)} files)")


@challenge_app.command("register")
def challenge_register(
    slug: str,
    image: str,
    emission: float,
    name: str | None = None,
    config: Path = typer.Option(Path("config/master.example.yaml")),
):
    _admin_post(
        config,
        "/v1/admin/challenges",
        {
            "slug": slug,
            "name": name or slug,
            "image": image,
            "version": image.rsplit(":", 1)[-1] if ":" in image else "latest",
            "emission_percent": emission,
        },
    )


@challenge_app.command("activate")
def challenge_activate(
    slug: str, config: Path = typer.Option(Path("config/master.example.yaml"))
):
    _admin_post(config, f"/v1/admin/challenges/{slug}/activate")


@challenge_app.command("deactivate")
def challenge_deactivate(
    slug: str, config: Path = typer.Option(Path("config/master.example.yaml"))
):
    _admin_post(config, f"/v1/admin/challenges/{slug}/deactivate")


@challenge_app.command("pull")
def challenge_pull(
    slug: str, config: Path = typer.Option(Path("config/master.example.yaml"))
):
    _admin_post(config, f"/v1/admin/challenges/{slug}/pull")


@challenge_app.command("restart")
def challenge_restart(
    slug: str, config: Path = typer.Option(Path("config/master.example.yaml"))
):
    _admin_post(config, f"/v1/admin/challenges/{slug}/restart")


@db_app.command("migrate")
def db_migrate(config: Path = typer.Option(Path("config/master.example.yaml"))):
    from base.db.migrations import upgrade

    settings = load_settings(config)
    upgrade(PROJECT_ROOT / "alembic.ini", database_url=settings.database.url)


@db_app.command("revision")
def db_revision(message: str):
    from alembic.config import Config

    from alembic import command

    command.revision(
        Config(str(PROJECT_ROOT / "alembic.ini")),
        message=message,
        autogenerate=True,
    )


@registry_app.command("print")
def registry_print(config: Path = typer.Option(Path("config/validator.example.yaml"))):
    settings = load_settings(config)
    client = RegistryClient(settings.validator.registry_url)
    registry = asyncio.run(client.fetch_registry())
    typer.echo(registry.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
