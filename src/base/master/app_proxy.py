"""Public FastAPI proxy app for challenge routes."""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from posixpath import normpath
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from base.bittensor.identity_cache import ValidatorIdentityResolver
from base.bittensor.metagraph_cache import MetagraphCache
from base.challenge_sdk.health import (
    ReadinessProbe,
    evaluate_readiness,
    health_from_checks,
)
from base.challenge_sdk.roles import Role, capabilities_for_role
from base.challenge_sdk.schemas import HealthResponse, VersionResponse
from base.challenge_sdk.version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_ID,
    SDK_CONTRACT_VERSION,
)
from base.config.settings import Settings
from base.master.admin.auth import (
    TokenProvider,
    build_admin_token_dependency,
    load_admin_token_from_environment,
)
from base.master.admin.runtime import RuntimeController
from base.master.app_admin import build_admin_router
from base.master.assignment_coordination import (
    AssignmentCoordinationService,
    build_assignment_coordination_router,
)
from base.master.challenge_dashboard import ChallengeMetricsProvider
from base.master.docker_orchestrator import DockerOrchestrationError
from base.master.orchestration import (
    MasterChallengeReconciler,
    MasterOrchestrationDriver,
    build_master_orchestration_lifespan,
    build_master_registry_reconcile_lifespan,
)
from base.master.raw_weight_ingress import (
    RawWeightIngressService,
    build_raw_weight_ingress_router,
)
from base.master.registry import ChallengeNotFoundError
from base.master.service import MasterWeightService
from base.master.submission_observation import ValidatorSubmissionObservationService
from base.master.validator_coordination import (
    ValidatorCoordinationService,
    build_validator_coordination_router,
    build_validator_health_lifespan,
)
from base.master.worker_assignment import (
    WorkerAssignmentService,
    build_worker_assignment_router,
)
from base.master.worker_coordination import (
    WorkerCoordinationService,
    build_worker_coordination_router,
    build_worker_health_lifespan,
)
from base.master.worker_unit_status import (
    WorkerUnitStatusService,
    build_worker_unit_status_router,
)
from base.schemas.challenge import ChallengeRecord, ChallengeStatus
from base.security.miner_auth import (
    MinerAuthError,
    MinerNonceStore,
    MinerUploadVerifier,
    NonceReplayError,
)
from base.security.validator_auth import (
    ValidatorSignedRequestVerifier,
    build_validator_auth_dependency,
)
from base.security.worker_auth import (
    WorkerSignedRequestVerifier,
    build_internal_bearer_auth,
    build_worker_auth_dependency,
)
from base.supervisor.challenge_image_updater import (
    build_challenge_image_update_lifespan,
)
from base.supervisor.challenge_watcher import (
    build_challenge_watcher_lifespan,
)

DEFAULT_CORS_ALLOWED_ORIGINS = [
    "https://joinbase.ai",
    "https://www.joinbase.ai",
    "http://localhost:3000",
    "http://localhost:3100",
]
# Vercel preview deployments of the ``platform`` frontend project, e.g.
# ``platform-lz6erslee-mathismassimino-6459s-projects.vercel.app``.
CORS_ALLOWED_ORIGIN_REGEX = r"^https://platform-[a-z0-9-]+\.vercel\.app$"

SENSITIVE_REQUEST_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-admin-token",
    "x-base-admin-token",
    "x-base-challenge-token",
    "x-base-internal-token",
    "x-base-shared-token",
    "x-hotkey",
    "x-signature",
    "x-nonce",
    "x-timestamp",
    "x-base-verified-hotkey",
    "x-base-verified-uid",
    "x-base-verified-nonce",
    "x-base-request-hash",
}

MINER_SIGNATURE_HEADERS = {
    "x-hotkey",
    "x-signature",
    "x-nonce",
    "x-timestamp",
}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

BLOCKED_EXACT_PATHS = {"/health", "/version"}
BENCHMARK_EXECUTION_ACTIONS = {"run", "execute", "launch"}
PRISM_EXACT_PUBLIC_PATHS = {
    "/leaderboard",
    "/architectures",
    "/training-variants",
    "/epochs",
    "/epochs/current",
    "/gpu/status",
    "/health/eval-jobs",
}


ClientFactory = Callable[[], AbstractAsyncContextManager[httpx.AsyncClient]]
ChallengeTokenProvider = Callable[[str], str]

#: Challenge slug whose prism<->master bridge shared token additionally
#: authenticates the admission fleet-read ``GET /v1/workers/active`` (in addition
#: to the signed-request path). prism reuses this same token as its
#: admission-query bearer (architecture.md sec 3.5).
WORKER_ADMISSION_BRIDGE_SLUG = "prism"


def is_architecture_report_path(path: str) -> bool:
    """Return whether a path targets the removed architecture-report surface.

    Matches both bare and ``/v1``-prefixed forms used by direct and
    ``/challenges/{slug}/...`` proxy prefixes (VAL-GATE-015).
    """

    normalized = normpath(f"/{path.lstrip('/')}")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    if parts[0] == "v1":
        parts = parts[1:]
    # architectures/{id}/report[/*]
    return len(parts) >= 3 and parts[0] == "architectures" and parts[2] == "report"


def is_blocked_proxy_path(path: str) -> bool:
    """Return whether a public proxy path targets a private challenge route."""

    normalized = normpath(f"/{path.lstrip('/')}")
    return (
        normalized in BLOCKED_EXACT_PATHS
        or normalized == "/internal"
        or normalized.startswith("/internal/")
        or _is_benchmark_execution_path(normalized)
        or is_architecture_report_path(normalized)
    )


def _is_benchmark_execution_path(normalized: str) -> bool:
    if normalized == "/benchmark-executions":
        return True
    if normalized.startswith("/benchmark-executions/"):
        return True
    parts = [part for part in normalized.split("/") if part]
    if not parts or parts[-1] not in BENCHMARK_EXECUTION_ACTIONS:
        return False
    return parts[0] in {"benchmark", "benchmarks"}


def prism_upstream_proxy_path(slug: str, path: str) -> str:
    normalized = normpath(f"/{path.lstrip('/')}")
    if slug != "prism" or normalized == "/.":
        return path
    # Never rewrite the removed architecture-report surface (VAL-GATE-015).
    if is_architecture_report_path(normalized):
        return path
    if normalized.startswith("/v1/"):
        return normalized
    if is_blocked_proxy_path(normalized):
        return path
    if normalized in PRISM_EXACT_PUBLIC_PATHS:
        return f"/v1{normalized}"
    parts = [part for part in normalized.split("/") if part]
    # List/detail/variants remain public; report is excluded above.
    if (
        parts
        and parts[0] == "architectures"
        and not (len(parts) >= 3 and parts[2] == "report")
    ):
        return f"/v1{normalized}"
    if len(parts) == 2 and parts[0] == "submissions":
        return f"/v1{normalized}"
    if len(parts) == 3 and parts[0] == "submissions" and parts[2] == "curve":
        return f"/v1{normalized}"
    return path


def _is_agent_challenge_env_route(slug: str, method: str, path: str) -> bool:
    if slug != "agent-challenge":
        return False

    normalized = normpath(f"/{path.lstrip('/')}")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) == 3 and parts[0] == "submissions" and parts[2] == "env":
        return method.upper() in {"GET", "PUT"}
    if (
        len(parts) == 4
        and parts[0] == "submissions"
        and parts[2] == "env"
        and parts[3] == "confirm-empty"
    ):
        return method.upper() == "POST"
    if len(parts) == 3 and parts[0] == "submissions" and parts[2] == "launch":
        return method.upper() == "POST"
    return False


def _is_agent_challenge_signed_route(slug: str, method: str, path: str) -> bool:
    """Routes where the miner signs the challenge-local path.

    The miner's signature headers (``X-Hotkey``/``X-Signature``/``X-Nonce``/
    ``X-Timestamp``) must survive the generic ``/challenges/{slug}`` passthrough
    so the challenge can verify them. This covers the signed env actions plus the
    JSON base64 submission upload (``POST /submissions``) per the frontend API
    contract.
    """

    if _is_agent_challenge_env_route(slug, method, path):
        return True
    if slug != "agent-challenge":
        return False
    normalized = normpath(f"/{path.lstrip('/')}")
    parts = [part for part in normalized.split("/") if part]
    return len(parts) == 1 and parts[0] == "submissions" and method.upper() == "POST"


def _forward_headers(
    request: Request, *, preserve_miner_signature_headers: bool = False
) -> dict[str, str]:
    """Copy safe request headers for forwarding to a public challenge route."""

    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        preserve_header = (
            preserve_miner_signature_headers and lowered in MINER_SIGNATURE_HEADERS
        )
        if (
            lowered in HOP_BY_HOP_HEADERS
            or (lowered in SENSITIVE_REQUEST_HEADERS and not preserve_header)
            or lowered == "host"
        ):
            continue
        headers[key] = value

    headers["X-Base-Proxy"] = "true"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    """Copy safe upstream response headers back to the caller."""

    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _is_event_stream(response: httpx.Response) -> bool:
    return (
        response.headers.get("content-type", "").lower().startswith("text/event-stream")
    )


def _target_url(base_url: str, path: str, query: str) -> str:
    safe_path = quote(path.lstrip("/"), safe="/")
    url = f"{base_url.rstrip('/')}/{safe_path}"
    if query:
        url = f"{url}?{query}"
    return url


def _challenge_token_provider(registry: Any) -> ChallengeTokenProvider:
    def provider(slug: str) -> str:
        get_token = getattr(registry, "get_token", None)
        if callable(get_token):
            return str(get_token(slug))
        return ""

    return provider


async def _resolve_value(value):  # type: ignore[no-untyped-def]
    if inspect.isawaitable(value):
        return await value
    return value


async def _resolve_challenge(registry: Any, value: str) -> ChallengeRecord:
    try:
        return await _resolve_value(registry.get(value))
    except ChallengeNotFoundError:
        matches = [
            item
            for item in await _resolve_value(registry.list())
            if item.name.lower() == value.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        raise


async def _active_challenge(registry: Any, value: str) -> ChallengeRecord:
    try:
        challenge = await _resolve_challenge(registry, value)
    except ChallengeNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Challenge not found"
        ) from exc
    if challenge.status != ChallengeStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Challenge not found"
        )
    return challenge


@asynccontextmanager
async def _default_client_factory() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        yield client


Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def _combine_lifespans(*lifespans: Lifespan | None) -> Lifespan | None:
    """Compose several FastAPI lifespans into one (entered in order).

    ``None`` entries are ignored. Returns ``None`` when nothing is configured so
    the app keeps its default (no-op) lifespan.
    """

    active = [lifespan for lifespan in lifespans if lifespan is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    @asynccontextmanager
    async def combined(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for lifespan in active:
                await stack.enter_async_context(lifespan(app))
            yield

    return combined


def create_proxy_app(
    *,
    registry: Any,
    client_factory: ClientFactory = _default_client_factory,
    miner_verifier: MinerUploadVerifier | None = None,
    nonce_store: MinerNonceStore | None = None,
    metagraph_cache: MetagraphCache | None = None,
    challenge_token_provider: ChallengeTokenProvider | None = None,
    netuid: int = 100,
    upload_signature_ttl_seconds: int = 300,
    upload_nonce_ttl_seconds: int = 86_400,
    upload_max_body_bytes: int = 2_000_000,
    upload_require_registered_hotkey: bool = True,
    extra_registered_hotkeys: list[str] | None = None,
    runtime_controller: RuntimeController | None = None,
    weight_service: MasterWeightService | None = None,
    metrics_provider: ChallengeMetricsProvider | None = None,
    chain_endpoint: str = "",
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    admin_token_provider: TokenProvider = load_admin_token_from_environment,
    enforce_production_policy: bool = False,
    validator_service: ValidatorCoordinationService | None = None,
    validator_verifier: ValidatorSignedRequestVerifier | None = None,
    validator_health_interval_seconds: float | None = None,
    worker_service: WorkerCoordinationService | None = None,
    worker_verifier: WorkerSignedRequestVerifier | None = None,
    worker_health_interval_seconds: float | None = None,
    worker_assignment_service: WorkerAssignmentService | None = None,
    worker_assignment_verifier: WorkerSignedRequestVerifier | None = None,
    worker_unit_status_service: WorkerUnitStatusService | None = None,
    assignment_coordination_service: AssignmentCoordinationService | None = None,
    raw_weight_ingress_service: RawWeightIngressService | None = None,
    submission_observation_service: (
        ValidatorSubmissionObservationService | None
    ) = None,
    orchestration_driver: MasterOrchestrationDriver | None = None,
    orchestration_interval_seconds: float | None = None,
    registry_reconciler: MasterChallengeReconciler | None = None,
    registry_reconcile_interval_seconds: float | None = None,
    challenge_image_updater_settings: Settings | None = None,
    challenge_image_update_interval_seconds: float | None = None,
    challenge_watcher_settings: Settings | None = None,
    challenge_watcher_interval_seconds: float | None = None,
    identity_resolver: ValidatorIdentityResolver | None = None,
    allowed_cors_origins: list[str] | None = None,
    readiness_probes: Sequence[ReadinessProbe] = (),
) -> FastAPI:
    """Create the public proxy FastAPI app.

    This is the single public API. When ``runtime_controller`` is provided, the
    admin/registry router (``/v1/registry``, ``/v1/weights/latest``,
    ``/v1/challenges/dashboard.svg``, ``/admin`` and the token-gated
    ``/v1/admin/*`` management/runtime-control routes) is included on the same
    app, so everything is served on one port. The admin router's duplicate
    ``GET /health`` is deduped (the proxy's own ``/health`` is kept).
    """

    app = FastAPI(
        title="BASE Challenge Proxy",
        version="1.0",
        lifespan=_combine_lifespans(
            build_validator_health_lifespan(
                validator_service, validator_health_interval_seconds
            ),
            build_worker_health_lifespan(
                worker_service, worker_health_interval_seconds
            ),
            build_master_orchestration_lifespan(
                orchestration_driver, orchestration_interval_seconds
            ),
            build_master_registry_reconcile_lifespan(
                registry_reconciler, registry_reconcile_interval_seconds
            ),
            build_challenge_image_update_lifespan(
                challenge_image_updater_settings,
                challenge_image_update_interval_seconds,
            ),
            build_challenge_watcher_lifespan(
                challenge_watcher_settings,
                challenge_watcher_interval_seconds,
            ),
        ),
    )
    with_role = capabilities_for_role(Role.MASTER)

    @app.middleware("http")
    async def establish_master_role(request: Request, call_next: Any) -> Response:
        from base.challenge_sdk.roles import activate_role

        with activate_role(Role.MASTER, capabilities=with_role):
            return await call_next(request)

    async def current_health() -> HealthResponse:
        checks = await evaluate_readiness(readiness_probes)
        return health_from_checks(
            slug="base-master",
            version=ARTIFACT_VERSION,
            role=Role.MASTER.value,
            capabilities=tuple(capabilities_for_role(Role.MASTER)),
            checks=checks,
        )

    @app.middleware("http")
    async def refuse_mutations_while_unready(
        request: Request, call_next: Any
    ) -> Response:
        discovery_path = request.url.path in {"/health", "/ready", "/version"}
        if request.method not in {"GET", "HEAD", "OPTIONS"} and not discovery_path:
            health = await current_health()
            if not health.ready:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={
                        "detail": {
                            "code": "runtime_not_ready",
                            "detail": (
                                "mandatory runtime dependencies are unavailable"
                            ),
                        }
                    },
                )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            allowed_cors_origins
            if allowed_cors_origins is not None
            else DEFAULT_CORS_ALLOWED_ORIGINS
        ),
        allow_origin_regex=CORS_ALLOWED_ORIGIN_REGEX,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
        max_age=600,
    )
    challenge_registry = registry
    token_provider = challenge_token_provider or _challenge_token_provider(
        challenge_registry
    )
    if miner_verifier is None and nonce_store is None:
        raise ValueError("nonce_store or miner_verifier is required")
    if miner_verifier is None and metagraph_cache is None:
        raise ValueError("metagraph_cache or miner_verifier is required")
    if miner_verifier is None:
        assert nonce_store is not None
        assert metagraph_cache is not None
        verifier = MinerUploadVerifier(
            netuid=netuid,
            nonce_store=nonce_store,
            metagraph_cache=metagraph_cache,
            ttl_seconds=upload_signature_ttl_seconds,
            require_registered_hotkey=upload_require_registered_hotkey,
            extra_registered_hotkeys=set(extra_registered_hotkeys)
            if extra_registered_hotkeys
            else None,
        )
    else:
        verifier = miner_verifier

    @app.api_route(
        "/health",
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    async def health() -> HealthResponse:
        return await current_health()

    @app.api_route(
        "/ready",
        methods=["GET", "HEAD"],
        response_model=HealthResponse,
        include_in_schema=False,
    )
    async def ready() -> HealthResponse | JSONResponse:
        response = await current_health()
        if response.ready:
            return response
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(mode="json"),
        )

    @app.api_route(
        "/version",
        methods=["GET", "HEAD"],
        response_model=VersionResponse,
        include_in_schema=False,
    )
    async def version() -> VersionResponse:
        return VersionResponse(
            distribution_name=DISTRIBUTION_NAME,
            artifact_version=ARTIFACT_VERSION,
            release_id=RELEASE_ID,
            api_version=API_VERSION,
            challenge_slug=None,
            challenge_version=ARTIFACT_VERSION,
            sdk_contract_version=SDK_CONTRACT_VERSION,
            sdk_version=SDK_CONTRACT_VERSION,
            role=Role.MASTER.value,
            capabilities=tuple(capabilities_for_role(Role.MASTER)),
        )

    async def forward_upstream(
        challenge: ChallengeRecord,
        *,
        method: str,
        path: str,
        query: str,
        body: bytes,
        headers: dict[str, str],
    ) -> httpx.Response:
        url = _target_url(challenge.internal_base_url, path, query)
        async with client_factory() as client:
            return await client.request(
                method,
                url,
                content=body,
                headers=headers,
            )

    async def forward_proxy_response(
        challenge: ChallengeRecord,
        *,
        method: str,
        path: str,
        query: str,
        body: bytes,
        headers: dict[str, str],
    ) -> Response:
        url = _target_url(challenge.internal_base_url, path, query)
        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(client_factory())
            upstream = await stack.enter_async_context(
                client.stream(
                    method,
                    url,
                    content=body,
                    headers=headers,
                )
            )
            if _is_event_stream(upstream):
                return StreamingResponse(
                    upstream.aiter_raw(),
                    status_code=upstream.status_code,
                    headers=_response_headers(upstream),
                    media_type=upstream.headers.get("content-type"),
                    background=BackgroundTask(stack.aclose),
                )

            content = await upstream.aread()
        except Exception:
            await stack.aclose()
            raise

        await stack.aclose()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=upstream.headers.get("content-type"),
        )

    async def proxy_request(slug: str, path: str, request: Request) -> Response:
        # Removed LLM architecture-report surface: normal not-found with zero
        # upstream challenge resolution or provider work (VAL-GATE-015).
        if is_architecture_report_path(path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Not Found",
            )
        if is_blocked_proxy_path(path):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Proxy path is not allowed",
            )

        challenge = await _active_challenge(challenge_registry, slug)

        body = await request.body()
        is_agent_challenge_env_route = _is_agent_challenge_env_route(
            slug, request.method, path
        )
        headers = _forward_headers(
            request,
            preserve_miner_signature_headers=_is_agent_challenge_signed_route(
                slug, request.method, path
            ),
        )
        headers["X-Base-Challenge-Slug"] = slug
        try:
            return await forward_proxy_response(
                challenge,
                method=request.method,
                path=prism_upstream_proxy_path(slug, path),
                query=request.url.query,
                body=body,
                headers=headers,
            )
        except (httpx.HTTPError, DockerOrchestrationError) as exc:
            unavailable_status = (
                status.HTTP_503_SERVICE_UNAVAILABLE
                if is_agent_challenge_env_route
                else status.HTTP_502_BAD_GATEWAY
            )
            raise HTTPException(
                status_code=unavailable_status, detail="Challenge unavailable"
            ) from exc

    async def bridge_upload(challenge_name: str, request: Request) -> Response:
        challenge = await _active_challenge(challenge_registry, challenge_name)
        body = await request.body()
        if len(body) > upload_max_body_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Submission too large",
            )
        try:
            identity = await verifier.verify(
                method=request.method,
                path=request.url.path,
                headers=request.headers,
                body=body,
                challenge_slug=challenge.slug,
            )
        except NonceReplayError as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        except MinerAuthError as exc:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc

        token = token_provider(challenge.slug)
        if not token:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "Challenge token is unavailable"
            )
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Base-Challenge-Slug": challenge.slug,
            "X-Base-Verified-Hotkey": identity.hotkey,
            "X-Base-Verified-Nonce": identity.nonce,
            "X-Base-Request-Hash": identity.body_hash,
            "Content-Type": request.headers.get(
                "content-type", "application/octet-stream"
            ),
            "Accept": "application/json",
        }
        if identity.uid is not None:
            headers["X-Base-Verified-Uid"] = str(identity.uid)
        filename = request.headers.get("x-submission-filename")
        if filename:
            headers["X-Submission-Filename"] = filename
        try:
            upstream = await forward_upstream(
                challenge,
                method="POST",
                path="/internal/v1/bridge/submissions",
                query=request.url.query,
                body=body,
                headers=headers,
            )
        except (httpx.HTTPError, DockerOrchestrationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Challenge unavailable"
            ) from exc
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=upstream.headers.get("content-type"),
        )

    async def bridge_status(
        challenge_name: str, submission_id: str, request: Request
    ) -> Response:
        challenge = await _active_challenge(challenge_registry, challenge_name)
        try:
            upstream = await forward_upstream(
                challenge,
                method="GET",
                path=f"/v1/submissions/{submission_id}",
                query=request.url.query,
                body=b"",
                headers=_forward_headers(request),
            )
        except (httpx.HTTPError, DockerOrchestrationError) as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Challenge unavailable"
            ) from exc
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=upstream.headers.get("content-type"),
        )

    @app.post("/v1/challenges/{challenge_name}/submissions")
    async def upload_submission(challenge_name: str, request: Request) -> Response:
        return await bridge_upload(challenge_name, request)

    @app.get("/v1/challenges/{challenge_name}/submissions/{submission_id}")
    async def bridge_submission_status(
        challenge_name: str, submission_id: str, request: Request
    ) -> Response:
        return await bridge_status(challenge_name, submission_id, request)

    @app.api_route(
        "/challenges/{slug}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_root(slug: str, request: Request) -> Response:
        return await proxy_request(slug, "", request)

    @app.api_route(
        "/challenges/{slug}/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_path(slug: str, path: str, request: Request) -> Response:
        return await proxy_request(slug, path, request)

    if runtime_controller is not None:
        app.include_router(
            build_admin_router(
                registry=challenge_registry,
                runtime_controller=runtime_controller,
                metrics_provider=metrics_provider,
                weight_service=weight_service,
                netuid=netuid,
                chain_endpoint=chain_endpoint,
                now_fn=now_fn,
                admin_token_provider=admin_token_provider,
                enforce_production_policy=enforce_production_policy,
                include_health=False,
                validator_service=validator_service,
                identity_resolver=identity_resolver,
                submission_observation_service=submission_observation_service,
                validator_auth_dependency=(
                    build_validator_auth_dependency(validator_verifier)
                    if validator_verifier is not None
                    else None
                ),
            )
        )
        app.state.runtime_controller = runtime_controller

    if validator_service is not None and validator_verifier is not None:
        app.include_router(
            build_validator_coordination_router(
                service=validator_service,
                auth_dependency=build_validator_auth_dependency(validator_verifier),
                admin_dependency=build_admin_token_dependency(admin_token_provider),
                registry=challenge_registry,
            )
        )
        app.state.validator_coordination_service = validator_service

    if worker_service is not None and worker_verifier is not None:

        def _worker_admission_tokens() -> list[str]:
            try:
                token = token_provider(WORKER_ADMISSION_BRIDGE_SLUG)
            except Exception:  # noqa: BLE001 - missing token => no internal bearer
                return []
            return [token] if token else []

        app.include_router(
            build_worker_coordination_router(
                service=worker_service,
                auth_dependency=build_worker_auth_dependency(worker_verifier),
                internal_auth=build_internal_bearer_auth(_worker_admission_tokens),
            )
        )
        app.state.worker_coordination_service = worker_service

    if worker_assignment_service is not None and worker_assignment_verifier is not None:
        app.include_router(
            build_worker_assignment_router(
                service=worker_assignment_service,
                auth_dependency=build_worker_auth_dependency(
                    worker_assignment_verifier
                ),
            )
        )
        app.state.worker_assignment_service = worker_assignment_service

    if worker_unit_status_service is not None and worker_verifier is not None:
        app.include_router(
            build_worker_unit_status_router(
                service=worker_unit_status_service,
                auth_dependency=build_worker_auth_dependency(worker_verifier),
            )
        )
        app.state.worker_unit_status_service = worker_unit_status_service

    if assignment_coordination_service is not None and validator_verifier is not None:
        app.include_router(
            build_assignment_coordination_router(
                service=assignment_coordination_service,
                auth_dependency=build_validator_auth_dependency(validator_verifier),
            )
        )
        app.state.assignment_coordination_service = assignment_coordination_service

    if raw_weight_ingress_service is not None:
        app.include_router(
            build_raw_weight_ingress_router(service=raw_weight_ingress_service)
        )
        app.state.raw_weight_ingress_service = raw_weight_ingress_service

    if orchestration_driver is not None:
        app.state.orchestration_driver = orchestration_driver

    if registry_reconciler is not None:
        app.state.registry_reconciler = registry_reconciler

    app.state.challenge_registry = challenge_registry
    app.state.miner_upload_verifier = verifier
    return app
