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


class ValidatorAgentSettings(BaseModel):
    """Decentralized validator executor agent (architecture.md sec 2.2).

    The agent hotkey-registers + heartbeats with the master coordination plane,
    pulls its assignments, executes them on its OWN broker + Docker, posts
    results, and routes LLM calls through the master gateway (it never holds a
    provider key). ``heartbeat_interval_seconds`` left unset (``None``) means the
    agent uses the interval the master returns from ``register``.
    """

    #: Master coordination-plane base URL. Falls back to ``registry_url``.
    master_url: str | None = None
    #: Master LLM gateway base URL. Falls back to ``master_url``.
    gateway_url: str | None = None
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
    broker_host: str = "0.0.0.0"
    broker_port: int = 8082
    broker_url: str = "http://base-docker-broker:8082"
    broker_workspace_dir: str = "/tmp/base-docker-broker"
    broker_allowed_images: list[str] = Field(
        default_factory=lambda: ["ghcr.io/baseintelligence/"]
    )
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


class ProviderEntry(BaseModel):
    """One configured LLM provider: its OpenAI-compatible base URL + key.

    The key is injected server-side by the gateway (validators/agents never hold
    it). ``api_key`` is an inline value (dev/tests); ``api_key_file`` points at a
    mounted secret (production, e.g. ``/run/secrets/yunwu_api_key``).
    """

    base_url: str
    api_key: str | None = None
    api_key_file: str | None = None


class SourceRoute(BaseModel):
    """A request ``source`` claim's routing: provider + optional model override."""

    provider: str
    model: str | None = None


def _default_gateway_providers() -> dict[str, ProviderEntry]:
    return {
        "yunwu": ProviderEntry(
            base_url="https://yunwu.ai/v1",
            api_key_file="/run/secrets/yunwu_api_key",
        )
    }


def _default_gateway_sources() -> dict[str, SourceRoute]:
    return {
        "agent": SourceRoute(provider="yunwu", model="claude-opus-4-8"),
        "llm_review": SourceRoute(provider="yunwu", model="claude-opus-4-8"),
    }


class GatewaySettings(BaseModel):
    """Master LLM gateway config (architecture.md sec 5; llm-yunwu-contract).

    Provider-agnostic + config-driven: a provider registry (name -> base URL +
    key) plus a per-``source`` route map selects the provider + model the gateway
    injects. The provider is config-selected: ``mock`` (deterministic, no egress;
    used by tests) or ``real`` (HTTP clients pinned to the configured bases).
    Provider keys are injected server-side; validators/eval runtimes hold only a
    scoped gateway token and point their client base URL at the master gateway.
    The defaults exist only so local/dev + tests work; production values come from
    ``deploy/swarm/master.yaml``. There is NO model-enforcement constant and no
    hardcoded provider base URL used at runtime.
    """

    provider_mode: Literal["mock", "real"] = "mock"
    #: Externally-reachable master gateway root URL advertised to validators in
    #: the pull payload (the LLM route is mounted under ``/llm/v1`` on the proxy).
    #: The master stamps ``BASE_LLM_GATEWAY_URL`` + the scoped token from this
    #: base; falls back to ``master.registry_url``.
    public_base_url: str | None = None
    #: Configured provider registry (name -> base URL + server-side key).
    providers: dict[str, ProviderEntry] = Field(
        default_factory=_default_gateway_providers
    )
    #: Provider used when a token's ``source`` has no configured route.
    default_provider: str = "yunwu"
    #: Model injected when neither the token nor the source route pins one.
    default_model: str = "claude-opus-4-8"
    #: Per-``source`` routing (``agent`` for coded agents, ``llm_review`` for the
    #: central safety gates); both map to yunwu / ``claude-opus-4-8`` by default.
    sources: dict[str, SourceRoute] = Field(default_factory=_default_gateway_sources)
    token_secret: str | None = None
    token_secret_file: str | None = "/run/secrets/gateway_token_secret"
    token_ttl_seconds: int = 3_600
    request_timeout_seconds: float = 30.0


class ObservabilitySettings(BaseModel):
    log_json: bool = True
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
    #: Master self-update (Task 22). Enable ONLY with a manifest_url wired — the
    #: builder refuses ``self_update_enabled=true`` without one so the task is
    #: never registered-but-inert. Default OFF: the self-update task is not
    #: registered at all (explicit-disable, no silent no-op).
    self_update_enabled: bool = False
    self_update_manifest_url: str | None = None
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


class Settings(BaseModel):
    environment: str = "development"
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    master: MasterSettings = Field(default_factory=MasterSettings)
    validator: ValidatorSettings = Field(default_factory=ValidatorSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    docker: DockerSettings = Field(default_factory=DockerSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    gateway: GatewaySettings = Field(default_factory=GatewaySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    supervisor: SupervisorSettings = Field(default_factory=SupervisorSettings)

    @model_validator(mode="after")
    def validate_production_policy(self) -> Settings:
        validate_settings_policy(self)
        return self
