from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from base.config.policy import validate_settings_policy


class MockMetagraphNode(BaseModel):
    """A single static metagraph entry for the no-chain ``mock_metagraph`` seam.

    Mirrors the on-chain fields the eligibility auth depends on so a configured
    validator hotkey can be made eligible WITHOUT a live Subtensor.
    """

    hotkey: str
    uid: int | None = None
    validator_permit: bool = False
    stake: float = 0.0
    # Optional self-declared subnet identity for the no-chain mock deploy
    # (architecture.md sec 7.2). Seeded UNTRUSTED into the identity cache as the
    # fallback when no on-chain identity is available. Empty => no identity.
    display_name: str | None = None
    logo_url: str | None = None


class NetworkSettings(BaseModel):
    name: str = "base"
    netuid: int = 100
    chain_endpoint: str | None = None
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str | None = None
    master_uid: int = 0
    # Config-driven static metagraph (architecture.md G1). When non-empty the
    # bittensor runtime factory seeds ``MetagraphCache`` from this set and does
    # NOT construct a live Subtensor, so the listed validator hotkeys are
    # eligible with no chain. Empty default = OFF (production-safe, inert): the
    # live-metagraph path is unchanged. Miners stay submit-eligible via
    # ``master.upload_extra_registered_hotkeys`` independent of this set.
    mock_metagraph: list[MockMetagraphNode] = Field(default_factory=list)


class MasterSettings(BaseModel):
    # Public registry/alias default for this control plane.
    # Live known-good Base master network front (2026-07-13): chain.joinbase.ai.
    # Preferred product hostname chain.platform.network is documented for
    # cutover once it fronts Base master (today it serves agent-challenge).
    # Do not silently force the preferred hostname as a runtime default while
    # /health there is not role=master.
    registry_url: str = "https://chain.joinbase.ai"
    # Ignored back-compat: the admin/registry surface is served by the proxy on
    # proxy_port (single public API); there is no separate admin listener.
    admin_host: str = "0.0.0.0"
    admin_port: int = 8080
    proxy_host: str = "0.0.0.0"
    proxy_port: int = 8081
    epoch_interval_seconds: int = 360
    metagraph_cache_ttl_seconds: int = 300
    challenge_timeout_seconds: float = 10.0
    challenge_retries: int = 3
    registry_state_file: str = "/var/lib/base/registry.json"
    upload_signature_ttl_seconds: int = 300
    upload_nonce_ttl_seconds: int = 86_400
    upload_max_body_bytes: int = 7_500_000
    upload_require_registered_hotkey: bool = True
    # ss58 hotkeys accepted without on-chain registration (QA/allowlist; empty in prod)
    upload_extra_registered_hotkeys: list[str] = Field(default_factory=list)
    # Validator coordination plane (architecture.md sec 4). The proxy serves the
    # hotkey-signed register/heartbeat/pull/progress/result routes, returns
    # ``validator_heartbeat_interval_seconds`` to validators, marks a validator
    # offline once its last heartbeat exceeds ``validator_heartbeat_timeout_seconds``,
    # and runs the crash-detection loop every ``validator_health_interval_seconds``.
    validator_heartbeat_interval_seconds: int = 60
    validator_heartbeat_timeout_seconds: int = 180
    validator_health_interval_seconds: float = 60.0
    validator_signature_ttl_seconds: int = 300
    validator_nonce_ttl_seconds: int = 86_400
    assignment_lease_seconds: int = 900
    # Master orchestration driver (architecture.md sec 4): the live loop that
    # bridges challenge pending work units into ``work_assignments``, runs the
    # balanced ``assign_pending`` engine + the full reassignment pass, and folds
    # retry-exhausted units on the challenge side. Runs every
    # ``orchestration_interval_seconds`` (<=0 disables it). ``orchestration_seed``
    # makes the balanced tie-breaking reproducible when set.
    orchestration_interval_seconds: float = 30.0
    orchestration_seed: int | None = None
    # Registry-driven challenge deploy (architecture.md sec 4 + sec 9.2): the
    # master reconcile loop that turns every ACTIVE registry challenge into a
    # running service (idempotent) and tears down services for challenges no
    # longer ACTIVE, so installing ``base`` auto-deploys all ACTIVE challenges
    # and a newly-registered challenge propagates automatically. Runs every
    # ``registry_reconcile_interval_seconds`` (<=0 disables it; default-on).
    registry_reconcile_interval_seconds: float = 60.0
    # Challenge-image auto-roll (architecture.md sec 9.1): the proxy-hosted
    # challenge-image-update loop that per ACTIVE challenge digest-compares the
    # mutable GHCR tag, updates the registry DB record to ``tag@sha256:<digest>``
    # on a change, and rolls the running service. It runs INSIDE the master proxy
    # (which can reach the overlay registry DB + docker socket) rather than the
    # host supervisor. Runs every ``challenge_image_update_interval_seconds``
    # (<=0 disables it; default-on).
    challenge_image_update_interval_seconds: float = 60.0
    # Compose challenge watcher (mission Compose path): digest-pinned pull +
    # targeted recreation with health/version verification, rollback, backoff
    # and durable intent. Runs in-process inside base-master-validator.
    # ``<=0`` disables; default-on when compose backend is selected.
    challenge_watcher_interval_seconds: float = 60.0
    challenge_watcher_state_path: str = "/var/lib/base/challenge_watcher_state.json"


class ValidatorAgentSettings(BaseModel):
    """Decentralized validator executor agent (architecture.md sec 2.2).

    The agent hotkey-registers + heartbeats with the master coordination plane,
    pulls its assignments, executes them on its OWN broker + Docker, and posts
    results. ``heartbeat_interval_seconds`` left unset (``None``) means the agent
    uses the interval the master returns from ``register``.
    """

    #: Base master coordination API URL (register/heartbeat/pull/result).
    #: Required by the agent runtime; installers set this from ``--master-url``.
    #: Distinct from ``registry_url`` / ``weights_url`` aliases: those may share
    #: the same host when master hosts both, but must never silently force a
    #: non-master public hostname. No localhost invent / no role-default.
    master_url: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["cpu"])
    version: str | None = None
    heartbeat_interval_seconds: int | None = None
    poll_interval_seconds: float = 5.0
    request_timeout_seconds: float = 15.0
    #: Validator's OWN Docker broker. Falls back to ``docker.broker_url``.
    broker_url: str | None = None
    broker_token: str | None = None
    broker_token_file: str | None = "/run/secrets/base_broker_token"
    #: Extra allowed eval-image prefixes (added to ``docker.broker_allowed_images``).
    allowed_images: list[str] = Field(default_factory=list)
    run_timeout_seconds: int = 3_600
    # Optional self-declared subnet identity (architecture.md sec 7.2). When set,
    # threaded UNTRUSTED into the agent's ``last_seen_meta`` so the master reads
    # it back as the identity fallback (zero new server schema). Empty => omitted.
    display_name: str | None = None
    logo_url: str | None = None


class ValidatorSettings(BaseModel):
    # Public registry/weights alias default. Live known-good Base master front:
    # https://chain.joinbase.ai (2026-07-13). Preferred product hostname
    # https://chain.platform.network is not a runtime default until cutover.
    registry_url: str = "https://chain.joinbase.ai"
    registry_retry_seconds: int = 15
    weights_url: str | None = None
    weights_interval_seconds: int = 360
    weights_timeout_seconds: float = 15.0
    weights_retries: int = 3
    weights_freshness_seconds: int = 720
    # RUNTIME-OFF gate for the supervisor on-chain weights task (plan Task 8).
    # Defaults False so a deploy NEVER auto-commits weights on-chain; the first
    # on-chain commit is human-gated (plan Task 27) by flipping this flag.
    submit_on_chain_enabled: bool = False
    #: Durable per-validator submission ledger directory (own volume). Composes
    #: mount ``validator-state`` at ``/var/lib/base/state`` by default.
    submission_state_dir: str = "/var/lib/base/state"
    #: Bound on per-vector set_weights attempts before RETRY_EXHAUSTED.
    submission_max_attempts: int = 5
    submission_backoff_base_seconds: float = 1.0
    submission_backoff_max_seconds: float = 300.0
    agent: ValidatorAgentSettings = Field(default_factory=ValidatorAgentSettings)

    @property
    def resolved_weights_url(self) -> str:
        return self.weights_url or self.registry_url


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://base:base@postgres.base.svc.cluster.local/base"


class DockerSettings(BaseModel):
    network_name: str = "base_challenges"
    secret_dir: str = "/var/lib/base/secrets"
    internal_network: bool = True
    #: Challenge orchestration backend for the master proxy.
    #: ``compose`` is the mission target path; ``swarm`` remains for host-side
    #: tooling only and is never the default Compose deployment.
    orchestration_backend: Literal["compose", "swarm"] = "compose"
    #: Compose project boundary the watcher may mutate (must match
    #: ``COMPOSE_PROJECT_NAME`` of the master install).
    compose_project_name: str | None = None
    #: Master Compose file path visible inside the application container.
    compose_file: str = "/run/base/compose/docker-compose.yml"
    #: Directory for per-challenge compose override fragments (image pins).
    compose_override_dir: str = "/var/lib/base/compose-overrides"
    #: Sealed install-time compose env file (image digests + secret paths).
    #: Default sits next to ``compose_file`` at ``/run/base/compose/.env``.
    compose_env_file: str = "/run/base/compose/.env"
    broker_host: str = "0.0.0.0"
    broker_port: int = 8082
    broker_url: str = "http://base-docker-broker:8082"
    broker_workspace_dir: str = "/tmp/base-docker-broker"
    broker_allowed_images: list[str] = Field(
        default_factory=lambda: ["ghcr.io/baseintelligence/"]
    )
    #: Server-wide cap on TOTAL concurrent broker jobs across ALL challenge
    #: slugs, enforced atomically alongside the per-slug cap at
    #: ``/v1/docker/run`` (surfaces as HTTP 429 ``docker_quota_exceeded``).
    #: Defaults to ``13`` — a RAM-derived bound for a ~62 GiB host at a
    #: 4 GB/task budget on historical worker-broker deploys. Compose installs
    #: inherit the same default when a Docker broker is enabled; ``None`` means
    #: UNLIMITED (not recommended for shared hosts).
    broker_max_concurrent_global: int | None = 13
    #: Max bytes of stdout/stderr the broker returns per job before tail-capping
    #: (the last-resort bound; ``DockerExecutor``/``_cap_log`` keep only the
    #: tail beyond it). A generous default so challenges receive effectively the
    #: full job output while staying bounded against OOM.
    broker_log_limit_bytes: int = 5_000_000
    allow_privileged: bool = False
    broker_privileged_slugs: list[str] = Field(default_factory=list)
    broker_node_role: Literal["manager", "worker"] = "manager"
    broker_allow_privileged_escape: bool = False
    #: Challenge slugs whose Swarm jobs are bind-mounted the host Docker
    #: socket (Docker-out-of-Docker) so the job can create sibling task
    #: containers on the worker daemon. Swarm services cannot run
    #: ``--privileged`` (``docker service create`` rejects it), so this is the
    #: supported way to let a broker-created Swarm job spawn containers.
    #: Socket access is root-equivalent on the worker, so the empty default
    #: grants it to no one; gate enforced in ``SwarmBrokerService``.
    broker_docker_socket_slugs: list[str] = Field(default_factory=list)
    broker_docker_socket_path: str = "/var/run/docker.sock"
    #: Read-only mounts injected into the Swarm eval job for the same slugs as
    #: ``broker_docker_socket_slugs`` (e.g. the terminal-bench task cache + the
    #: frozen digest manifest, provisioned out-of-band onto a host path or named
    #: volume). Each entry is ``source:target`` where ``source`` is an absolute
    #: host path or a Docker named volume and ``target`` is the absolute mount
    #: path inside the job. Empty default mounts nothing.
    broker_eval_readonly_mounts: list[str] = Field(default_factory=list)
    #: Per-slug read-only mounts injected into the Swarm eval job, decoupled
    #: from ``broker_docker_socket_slugs``. Maps a challenge slug to a list of
    #: ``source:target`` specs (same format as ``broker_eval_readonly_mounts``).
    #: Used to bind-mount the locked prism FineWeb-Edu train split + reference
    #: tokenizers READ-ONLY into the prism eval container (which must NOT get the
    #: host Docker socket). The prism slug receives a built-in default when
    #: unset; see ``cli_app.main._eval_readonly_mounts_by_slug``.
    broker_eval_readonly_mounts_by_slug: dict[str, list[str]] = Field(
        default_factory=dict
    )
    #: Challenge slugs whose untrusted Swarm eval job is pinned to the internal
    #: (no-egress) overlay regardless of the requested network. The prism slug
    #: is locked by default in ``cli_app.main._egress_locked_slugs``; entries
    #: here are added to that allowlist.
    broker_egress_locked_slugs: list[str] = Field(default_factory=list)
    # Challenge API services run on the manager/host; broker jobs run on
    # workers, steered to CPU- vs GPU-labeled nodes (base.workload).
    challenge_placement_constraint: str | None = "node.role==manager"
    cpu_job_constraint: str | None = "node.labels.base.workload==cpu"
    gpu_job_constraint: str | None = "node.labels.base.workload==gpu"


class SecuritySettings(BaseModel):
    admin_token: str | None = None
    admin_token_file: str | None = None


class ComputeSettings(BaseModel):
    """Miner-funded GPU worker plane (architecture.md sec 3.3).

    ALL worker-plane behavior is gated behind ``worker_plane_enabled`` (env
    ``BASE_COMPUTE__WORKER_PLANE_ENABLED``); OFF (the default) preserves legacy
    behavior byte-for-byte: the worker coordination surface is not mounted and
    gpu units route to validators exactly as today. ``worker_heartbeat_ttl_seconds``
    is the freshness window: an ``active`` worker whose last heartbeat is older
    than the TTL is reported ``stale`` and is not assignable.
    """

    worker_plane_enabled: bool = False
    worker_heartbeat_ttl_seconds: int = 120
    worker_signature_ttl_seconds: int = 300
    worker_nonce_ttl_seconds: int = 86_400
    worker_health_interval_seconds: float = 60.0
    #: Number of DISTINCT-owner workers each gpu work unit is replicated across
    #: when the worker plane is on. Degrades to 1 (with a recorded warning) when
    #: fewer eligible distinct owners exist.
    replication_factor: int = 2


class WorkerAgentSettings(BaseModel):
    """Miner-funded GPU worker agent runtime (architecture.md sec 3.2).

    The agent registers with the master under a miner-signed binding, heartbeats,
    pulls gpu work units, executes them on its OWN local broker, and posts
    ExecutionProof-carrying results. It authenticates as its worker keypair, never
    as a metagraph validator permit, and never holds a provider API key.
    ``master_url`` is required to reach the coordination plane;
    ``heartbeat_interval_seconds`` left unset uses the interval the master
    returns from ``register``.
    """

    master_url: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["gpu"])
    version: str | None = None
    heartbeat_interval_seconds: int | None = None
    poll_interval_seconds: float = 5.0
    request_timeout_seconds: float = 15.0
    #: Worker's OWN local Docker broker. Falls back to ``docker.broker_url``.
    broker_url: str | None = None
    broker_token: str | None = None
    broker_token_file: str | None = "/run/secrets/base_broker_token"
    #: Extra allowed eval-image prefixes (added to ``docker.broker_allowed_images``).
    allowed_images: list[str] = Field(default_factory=list)
    run_timeout_seconds: int = 3_600


class WorkerDeploySettings(BaseModel):
    """Provisioning inputs for ``base worker deploy`` (architecture.md sec 3.2).

    ``provider`` selects the deploy target (``local`` runs the agent on this host;
    ``lium``/``targon`` provision a paid GPU instance running the worker image).
    Image fields default to the published worker image when unset. Cost guardrails
    (``max_price_per_hour``/``max_lifetime_hours``) bound provider provisioning.
    ``startup_commands`` MUST be metachar-free (Lium rejects shell metacharacters
    at rent time).
    """

    provider: Literal["local", "lium", "targon"] = "local"
    provider_instance_ref: str | None = None
    image: str | None = None
    image_digest: str | None = None
    #: Informational-only: a human-readable tag recorded alongside the pin. The
    #: digest-pinned deploy path provisions BY DIGEST (``image`` + ``image_digest``)
    #: and never consumes this value, so it never affects which image bytes run.
    image_tag: str | None = None
    template_name: str | None = None
    gpu_count: int = 1
    max_price_per_hour: float | None = None
    max_lifetime_hours: float = 1.0
    ssh_public_key: str | None = None
    ssh_public_key_file: str | None = None
    ssh_key_name: str | None = None
    startup_commands: str = "tail -f /dev/null"
    #: Seconds ``deploy --provider local`` polls the master for the worker to
    #: reach ``active`` before reporting failure.
    ready_timeout_seconds: float = 60.0


class WorkerIdentitySettings(BaseModel):
    """Worker keypair + miner binding material for the worker agent/CLI.

    The WORKER keypair signs coordination requests + ExecutionProofs; the MINER
    keypair signs the enrollment binding (``worker-binding:{worker_pubkey}:
    {miner_hotkey}:{nonce}``). Each key resolves from an sr25519 dev URI
    (``//Worker``), a mnemonic, or a bittensor wallet, in that order, falling
    back to ``network.wallet`` for the worker key. When the binding is signed
    out-of-band (e.g. a Lium/Targon pod that never holds the miner key) supply the
    pre-signed ``miner_hotkey`` + ``binding_signature`` + ``binding_nonce``
    instead of a miner key.
    """

    key_uri: str | None = None
    key_mnemonic: str | None = None
    wallet_name: str | None = None
    wallet_hotkey: str | None = None
    wallet_path: str | None = None
    miner_hotkey: str | None = None
    miner_key_uri: str | None = None
    miner_key_mnemonic: str | None = None
    miner_wallet_name: str | None = None
    miner_wallet_hotkey: str | None = None
    miner_wallet_path: str | None = None
    binding_signature: str | None = None
    binding_nonce: str | None = None


class WorkerSettings(BaseModel):
    """Top-level ``base worker`` config: agent runtime, deploy, and identity."""

    agent: WorkerAgentSettings = Field(default_factory=WorkerAgentSettings)
    deploy: WorkerDeploySettings = Field(default_factory=WorkerDeploySettings)
    identity: WorkerIdentitySettings = Field(default_factory=WorkerIdentitySettings)


class ObservabilitySettings(BaseModel):
    log_json: bool = True
    #: Application log level applied by ``configure_logging`` (case-insensitive,
    #: e.g. ``DEBUG``/``INFO``/``WARNING``). An unrecognized value falls back to
    #: ``INFO`` rather than raising.
    log_level: str = "INFO"
    sentry_dsn: str | None = None
    otel_service_name: str = "base"
    #: OTLP span exporter target (e.g. an OpenTelemetry collector gRPC endpoint
    #: like ``http://otel-collector:4317``). When set, ``init_otel`` attaches a
    #: ``BatchSpanProcessor(OTLPSpanExporter(...))`` to the tracer provider; when
    #: None the provider has no export path (inert, behaviour-preserving).
    otel_endpoint: str | None = None
    # Task 16: lightweight, config-driven webhook alerting (NO Prometheus/
    # Grafana). All endpoints default to None so a default deploy makes ZERO
    # network calls — the alert hook is a structured-log-only no-op until a
    # webhook URL is set, and the drand/GPU reachability probes are skipped
    # until their health URLs are configured.
    alert_webhook_url: str | None = None
    alert_webhook_timeout_seconds: float = 5.0
    #: drand beacon reachability probe target (e.g. a drand HTTP API health
    #: URL). When set, a supervisor probe fires a ``drand_unreachable`` alert on
    #: failure; when None the probe is skipped.
    drand_health_url: str | None = None
    #: GPU liveness probe target (e.g. the GPU worker's health endpoint). When
    #: set, a supervisor probe fires a ``gpu_down`` alert on failure; when None
    #: the probe is skipped.
    gpu_health_url: str | None = None
    #: Cadence for the drand/GPU reachability probe task.
    health_probe_interval_seconds: float = 60.0


#: Default Swarm service name + mutable image for a validator-agent updater
#: target. The validator agent runs the ``base validator agent`` CMD from the
#: ``base-validator-runtime`` image (docker/Dockerfile.validator-runtime), so a
#: validator NODE running it as a swarm service auto-rolls on a new digest.
DEFAULT_VALIDATOR_AGENT_SERVICE = "base-validator-agent"
DEFAULT_VALIDATOR_RUNTIME_IMAGE = (
    "ghcr.io/baseintelligence/base-validator-runtime:latest"
)


class ImageUpdateTargetSetting(BaseModel):
    """One image-updater target: a Swarm service tracking a mutable image.

    ``image`` must carry an explicit tag (the updater rejects untagged images
    under the production pin policy) and must NOT already be digest-pinned.
    """

    service: str
    image: str
    #: Per-target freeze: when True this service is skipped (never rolled or
    #: rolled back) even while the global hold is off. An opt-in operator
    #: pin/freeze to stop a bad rollout on a single service; default OFF so
    #: auto-update happens by default.
    hold: bool = False


class SupervisorSettings(BaseModel):
    """Control-plane supervisor wiring (image-updater auth + self-update).

    Both seams default OFF/behaviour-preserving: with no registry credentials the
    digest resolver stays anonymous (PUBLIC-package path), and with self-update
    disabled the self-update task is NOT registered (rather than registered-but-
    inert), so there is no silent half-state.
    """

    #: Registry the image-updaters authenticate against for PRIVATE digests.
    registry: str = "ghcr.io"
    #: Explicit registry username (e.g. a GitHub user/org for GHCR). When set
    #: together with a password/password_file it takes precedence over the docker
    #: config.json fallback.
    registry_username: str | None = None
    registry_password: str | None = None
    registry_password_file: str | None = None
    #: Docker ``config.json`` whose ``auths[<registry>].auth`` (base64
    #: ``user:password``) the resolver decodes when no explicit credentials are
    #: given. On the manager this is the file ``docker login ghcr.io`` writes, so
    #: the supervisor resolves PRIVATE ``ghcr.io/baseintelligence/*`` digests with
    #: the same credentials the deploy already provisions. None disables the
    #: fallback (anonymous resolver).
    registry_docker_config_path: str | None = "/root/.docker/config.json"
    #: Host-reachable broker ``/health`` URL for the supervisor's broker-health
    #: probe. The host systemd supervisor runs OUTSIDE the swarm overlay, so it
    #: cannot resolve the overlay service DNS in ``docker.broker_url``
    #: (``http://base-docker-broker:8082``) — the probe would fail forever and
    #: permanently trip the gate (blocking self-update's pre-swap gate). The
    #: broker publishes 8082 in host mode on the manager, so point the probe at
    #: the host-published port instead. ``None`` falls back to
    #: ``docker.broker_url`` (in-overlay callers like the proxy keep the service
    #: name).
    broker_health_url: str | None = None
    #: Master self-update (Task 22). Enable ONLY with a manifest_url wired — the
    #: builder refuses ``self_update_enabled=true`` without one so the task is
    #: never registered-but-inert. Default OFF: the self-update task is not
    #: registered at all (explicit-disable, no silent no-op).
    self_update_enabled: bool = False
    self_update_manifest_url: str | None = None
    #: Self-update timing/retry knobs (Task 22 hardening). Defaults equal the
    #: historical module constants so behaviour is unchanged unless configured.
    #: ``interval``/``min_uptime`` drive the tick cadence and the commit dwell;
    #: ``max_boot_attempts`` bounds the post-swap boot-storm rollback budget;
    #: ``max_swap_attempts`` is how many DISTINCT swap attempts a rolled-back
    #: version gets before it is blacklisted (so a transient boot failure is
    #: retried rather than permanently blacklisting a possibly-good version).
    self_update_interval_seconds: float = 300.0
    self_update_min_uptime_seconds: float = 30.0
    self_update_max_boot_attempts: int = 3
    self_update_max_swap_attempts: int = 3
    #: Image-updater targets (each a Swarm service tracking a mutable tagged
    #: image). ``None`` (the default) means "use the built-in master defaults"
    #: (``base-master-proxy`` + ``base-docker-broker``), preserving prior
    #: behaviour. Set an explicit list to drive the targets from config — e.g. an
    #: empty list on a validator NODE that watches only its agent via
    #: ``validator_agent_target_enabled`` below.
    image_updater_targets: list[ImageUpdateTargetSetting] | None = None
    #: When True, append a validator-agent target tracking the mutable validator
    #: runtime image so a validator agent running as a swarm service auto-rolls on
    #: a new digest. Default OFF (master nodes do not run the agent).
    validator_agent_target_enabled: bool = False
    validator_agent_service: str = DEFAULT_VALIDATOR_AGENT_SERVICE
    validator_agent_image: str = DEFAULT_VALIDATOR_RUNTIME_IMAGE
    #: Durable-retry policy for the master image-updater's convergence-verified
    #: rollout. ``image_update_max_attempts`` bounds how many times a failing
    #: target is retried (with rollback) before the updater stops hammering it
    #: and emits an ``image_update_failed`` alert; a NEW desired digest resets
    #: the budget. The backoff doubles from ``base`` each attempt, capped at
    #: ``max`` — see :class:`base.supervisor.retry.RetryPolicy`.
    image_update_max_attempts: int = 5
    image_update_backoff_base_seconds: float = 60.0
    image_update_backoff_max_seconds: float = 1800.0
    #: Global freeze: when True the image-updater skips EVERY target (logging
    #: ``skipped-held``) and never rolls or rolls back. An opt-in operator freeze
    #: to stop a bad rollout fleet-wide; default OFF so auto-update happens by
    #: default (a held target is also settable per-target via
    #: :attr:`ImageUpdateTargetSetting.hold`).
    image_update_hold: bool = False
    #: Orphan own-runner sandbox sweep (host-level backstop for DooD sandbox
    #: containers leaked when a job is killed externally). Enabled by default.
    orphan_sweep_enabled: bool = True
    #: Age (seconds) beyond which an own-runner sandbox is considered orphaned
    #: and force-removed. MUST exceed the max legit job lease
    #: (evaluation_timeout_seconds + lease ~= 4500s); default 2h.
    orphan_sweep_ttl_seconds: int = 7200
    #: How often the orphan sweep runs.
    orphan_sweep_interval_seconds: float = 300.0


class Settings(BaseModel):
    environment: str = "development"
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    master: MasterSettings = Field(default_factory=MasterSettings)
    validator: ValidatorSettings = Field(default_factory=ValidatorSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)

    @model_validator(mode="after")
    def validate_production_policy(self) -> Settings:
        validate_settings_policy(self)
        return self
