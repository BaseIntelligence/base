"""The master LLM gateway: config-driven provider routing + key injection.

The gateway exposes a SINGLE source-driven route ``POST /llm/v1/{path}``
(architecture.md sec 5; ``library/llm-yunwu-contract.md``). It authenticates the
caller with a scoped gateway token, resolves the provider + model from the
token's ``source`` claim via master config (NOT the URL and NOT a hardcoded
constant), OVERWRITES the request-body ``model`` with the resolved model, injects
the provider private key server-side, and forwards to the resolved provider.
Validators/eval runtimes hold NO provider key and send no real model; they point
their client base URL at the gateway and pass a scoped token.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import APIRouter, Request, Response

from base.master.llm_gateway.lifecycle import AssignmentLifecycleResolver
from base.master.llm_gateway.providers import (
    LLMProvider,
    ProviderConfig,
    ProviderRequest,
    ProviderResponse,
    build_providers,
)
from base.master.llm_gateway.redaction import (
    install_secret_redaction,
    redact_in_context,
    redact_secrets,
)
from base.master.llm_gateway.tokens import (
    CENTRAL_GATE_KIND,
    GatewayTokenAuthority,
    GatewayTokenClaims,
    GatewayTokenError,
    GatewayTokenExpired,
    GatewayTokenInvalid,
    GatewayTokenScopeError,
)
from base.master.llm_gateway.usage import (
    NullUsageRecorder,
    UsageRecord,
    UsageRecorder,
    parse_usage,
)

logger = logging.getLogger(__name__)

#: Default routing when a token carries no ``source`` claim (backward compatible:
#: an old assignment token resolves the same provider+model as ``source=agent``).
DEFAULT_SOURCE = "agent"
#: Config defaults; production values come from ``GatewaySettings``.
DEFAULT_PROVIDER = "yunwu"
DEFAULT_MODEL = "claude-opus-4-8"

#: Header carrying the scoped gateway token (preferred over ``Authorization``).
GATEWAY_TOKEN_HEADER = "X-Gateway-Token"
#: Optional cross-check headers declaring the scope a call is attributed to.
GATEWAY_VALIDATOR_HEADER = "X-Gateway-Validator"
GATEWAY_ASSIGNMENT_HEADER = "X-Gateway-Assignment"

#: Controlled, non-leaking detail strings for surfaced upstream failures. Both
#: the upstream-5xx and upstream-exception paths share ``UPSTREAM_ERROR_DETAIL``
#: so a server-side failure looks identical to the caller regardless of how it
#: arose (no upstream body/headers are ever relayed).
UPSTREAM_ERROR_DETAIL = "upstream provider error"
UPSTREAM_RATE_LIMITED_DETAIL = "upstream rate limited"
UPSTREAM_REJECTED_DETAIL = "upstream rejected request"

#: Consumption contract: eval runtimes point ``BASE_LLM_GATEWAY_URL`` at the
#: gateway (``{root}/llm/v1``) and pass a scoped token; the master resolves the
#: provider+model and injects the real provider credential. The raw provider key
#: is never delivered to the caller.
BASE_LLM_GATEWAY_URL_ENV = "BASE_LLM_GATEWAY_URL"
GATEWAY_TOKEN_ENV = "BASE_GATEWAY_TOKEN"

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
#: Response headers never relayed back to the caller (secret / framing).
_STRIPPED_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | {
    "authorization",
    "content-length",
    "content-encoding",
}


@dataclass(frozen=True)
class SourceRoute:
    """A ``source`` claim's resolved provider + optional model override."""

    provider: str
    model: str | None = None


class GatewayError(Exception):
    """A controlled gateway failure that maps to a safe HTTP status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class UnknownProviderError(GatewayError):
    def __init__(self, provider: str) -> None:
        super().__init__(502, "gateway provider not configured")
        self.provider = provider


class GatewayAssignmentInactiveError(GatewayTokenError):
    """Token's assignment is completed/failed/reassigned (maps to HTTP 403)."""


@dataclass(frozen=True)
class AuthenticatedCall:
    """A token-verified gateway call ready to forward.

    ``provider`` + ``model`` are resolved from the token's ``source`` claim via
    config, so the forward path can inject the key + overwrite the body model.
    """

    provider: str
    model: str
    claims: GatewayTokenClaims


class LLMGatewayService:
    """Routes authenticated gateway calls to providers with key injection.

    The provider + model are resolved from the token's ``source`` claim against
    the configured ``sources`` map (falling back to ``default_provider`` /
    ``default_model``); the request-body ``model`` is overwritten with the
    resolved model before forwarding.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, LLMProvider],
        api_keys: Mapping[str, str],
        token_authority: GatewayTokenAuthority,
        sources: Mapping[str, SourceRoute] | None = None,
        default_provider: str = DEFAULT_PROVIDER,
        default_model: str = DEFAULT_MODEL,
        default_source: str = DEFAULT_SOURCE,
        usage_recorder: UsageRecorder | None = None,
        assignment_resolver: AssignmentLifecycleResolver | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._api_keys = dict(api_keys)
        self._token_authority = token_authority
        self._sources = dict(sources or {})
        self._default_provider = default_provider
        self._default_model = default_model
        self._default_source = default_source
        self._usage_recorder: UsageRecorder = usage_recorder or NullUsageRecorder()
        self._assignment_resolver = assignment_resolver
        # Guarantee the injected provider keys are scrubbed from any gateway log.
        install_secret_redaction(self._api_keys.values(), logger=logger)

    @property
    def token_authority(self) -> GatewayTokenAuthority:
        return self._token_authority

    def _redact(self, text: str) -> str:
        return redact_secrets(text, self._api_keys.values())

    def provider(self, name: str) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise UnknownProviderError(name) from exc

    def resolve_route(self, claims: GatewayTokenClaims) -> tuple[str, str]:
        """Resolve ``(provider, model)`` from a verified token's ``source`` claim.

        A missing ``source`` resolves ``default_source`` (``"agent"``); a source
        with no config entry falls back to ``default_provider``; the model is the
        token's own ``model`` claim, else the source route's model, else
        ``default_model``.
        """

        source = claims.source or self._default_source
        route = self._sources.get(source)
        provider_name = route.provider if route is not None else self._default_provider
        model = (
            claims.model
            or (route.model if route is not None else None)
            or self._default_model
        )
        return provider_name, model

    def authenticate(
        self,
        *,
        token: str | None,
        expected_validator: str | None,
        expected_assignment: str | None,
    ) -> AuthenticatedCall:
        """Verify the gateway token and resolve its provider + model.

        Authentication + resolution happen BEFORE any provider call so a rejected
        token never reaches an upstream provider. Raises
        :class:`UnknownProviderError` when the resolved provider is not
        configured (a server-side misconfiguration).
        """

        claims = self._token_authority.verify(
            token,
            expected_validator=expected_validator,
            expected_assignment=expected_assignment,
        )
        provider_name, model = self.resolve_route(claims)
        if provider_name not in self._providers:
            raise UnknownProviderError(provider_name)
        return AuthenticatedCall(provider=provider_name, model=model, claims=claims)

    async def ensure_assignment_active(self, claims: GatewayTokenClaims) -> None:
        """Reject a token whose assignment is no longer active (VAL-LLM-023).

        A no-op when no resolver is configured (the token is then bound only by
        signature, expiry, and scope). Raises before any provider call so a
        terminated/reassigned assignment never reaches an upstream provider.

        A ``central-gate`` token has no live work assignment, so it is treated as
        active by valid signature + unexpired ``exp`` alone (the verification that
        produced ``claims`` already enforced both), bypassing the resolver.
        """

        if claims.kind == CENTRAL_GATE_KIND:
            return
        if self._assignment_resolver is None:
            return
        active = await self._assignment_resolver.is_active(
            validator_hotkey=claims.validator_hotkey,
            assignment_id=claims.assignment_id,
        )
        if not active:
            raise GatewayAssignmentInactiveError("assignment is no longer active")

    async def record_usage(
        self,
        *,
        claims: GatewayTokenClaims,
        provider: str,
        model: str,
        response: ProviderResponse,
    ) -> None:
        """Meter a successful call, keyed by ``(validator, assignment)``.

        Best-effort: a metering failure is logged (redacted) and never breaks
        the proxied response. No secret material is recorded. ``model`` is the
        resolved model the gateway actually forwarded.
        """

        prompt_tokens, completion_tokens, total_tokens = parse_usage(response.body)
        record = UsageRecord(
            validator_hotkey=claims.validator_hotkey,
            assignment_id=claims.assignment_id,
            provider=provider,
            model=model,
            status_code=response.status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        try:
            await self._usage_recorder.record(record)
        except Exception as exc:
            logger.error(
                "llm gateway usage metering failed: %s", self._redact(str(exc))
            )

    def _inject_model(self, body: bytes, model: str) -> bytes:
        """Overwrite (or add) the request-body ``model`` with the resolved model.

        A body that is not a JSON object is forwarded unchanged (the gateway
        cannot safely rewrite a non-JSON payload); OpenAI-compatible chat calls
        always send a JSON object, so the model is overwritten in practice.
        """

        if not body:
            return json.dumps({"model": model}).encode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return body
        if not isinstance(payload, dict):
            return body
        payload["model"] = model
        return json.dumps(payload).encode("utf-8")

    def _inject_headers(
        self, provider: str, caller_headers: Mapping[str, str]
    ) -> dict[str, str]:
        content_type = _header(caller_headers, "content-type") or "application/json"
        return {
            "Authorization": f"Bearer {self._api_keys.get(provider, '')}",
            "Content-Type": content_type,
            "Accept": "application/json",
        }

    async def forward(
        self,
        *,
        provider: str,
        model: str,
        path: str,
        body: bytes,
        caller_headers: Mapping[str, str],
    ) -> ProviderResponse:
        """Overwrite the model, inject the key, and forward to the provider.

        The caller's ``Authorization``/api-key headers are NOT forwarded; only
        the server-injected provider key reaches the upstream.
        """

        impl = self.provider(provider)
        upstream_body = self._inject_model(body, model)
        upstream_headers = self._inject_headers(provider, caller_headers)
        return await impl.forward(
            ProviderRequest(
                method="POST",
                path=path,
                headers=upstream_headers,
                body=upstream_body,
            )
        )

    def issue_token(
        self,
        *,
        validator_hotkey: str,
        assignment_id: str,
        ttl_seconds: int | None = None,
        source: str | None = None,
        model: str | None = None,
    ) -> str:
        return self._token_authority.issue(
            validator_hotkey=validator_hotkey,
            assignment_id=assignment_id,
            ttl_seconds=ttl_seconds,
            source=source,
            model=model,
        )

    def issue_central_gate_token(
        self,
        *,
        principal: str,
        label: str,
        ttl_seconds: int | None = None,
        source: str | None = None,
        model: str | None = None,
    ) -> str:
        return self._token_authority.issue_central_gate(
            principal=principal,
            label=label,
            ttl_seconds=ttl_seconds,
            source=source,
            model=model,
        )


def build_llm_gateway_service(
    *,
    api_keys: Mapping[str, str],
    token_secret: str,
    provider_config: ProviderConfig | None = None,
    sources: Mapping[str, SourceRoute] | None = None,
    default_provider: str = DEFAULT_PROVIDER,
    default_model: str = DEFAULT_MODEL,
    token_ttl_seconds: int = 3_600,
    usage_recorder: UsageRecorder | None = None,
    assignment_resolver: AssignmentLifecycleResolver | None = None,
) -> LLMGatewayService:
    """Construct the gateway service from config (provider mode + provider keys).

    Provider-agnostic: ``api_keys`` maps each configured provider name to its
    server-side key. Fails fast when a non-``mock`` provider is selected but a
    configured provider has no API key, so the gateway never forwards a real
    provider call with an empty ``Authorization`` header.
    """

    config = provider_config or ProviderConfig()
    if config.mode != "mock":
        missing = [name for name in config.providers if not api_keys.get(name)]
        if missing:
            raise ValueError(
                f"llm gateway provider_mode={config.mode!r} requires a configured "
                f"API key for: {', '.join(sorted(missing))}"
            )
    return LLMGatewayService(
        providers=build_providers(config),
        api_keys=dict(api_keys),
        token_authority=GatewayTokenAuthority(
            token_secret, default_ttl_seconds=token_ttl_seconds
        ),
        sources=sources,
        default_provider=default_provider,
        default_model=default_model,
        usage_recorder=usage_recorder,
        assignment_resolver=assignment_resolver,
    )


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    for key, val in headers.items():
        if key.lower() == lowered:
            return val
    return None


def _extract_token(headers: Mapping[str, str]) -> str | None:
    token = _header(headers, GATEWAY_TOKEN_HEADER)
    if token and token.strip():
        return token.strip()
    authorization = _header(headers, "authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip() or None
    return None


def _response_headers(response: ProviderResponse) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _STRIPPED_RESPONSE_HEADERS
    }


def build_llm_gateway_router(*, service: LLMGatewayService) -> APIRouter:
    """Build the LLM gateway router (single source-driven ``/llm/v1`` route)."""

    router = APIRouter()

    async def handle(path: str, request: Request) -> Response:
        token = _extract_token(request.headers)
        # Defensively register the per-request bearer token for redaction across
        # the whole forward/log path: today headers are never logged, but if such
        # logging is ever introduced the scoped token still cannot leak.
        with redact_in_context(token):
            return await _handle_call(path, request, token)

    async def _handle_call(path: str, request: Request, token: str | None) -> Response:
        try:
            call = service.authenticate(
                token=token,
                expected_validator=_header(request.headers, GATEWAY_VALIDATOR_HEADER),
                expected_assignment=_header(request.headers, GATEWAY_ASSIGNMENT_HEADER),
            )
            await service.ensure_assignment_active(call.claims)
        except GatewayAssignmentInactiveError:
            return _error_response(403, "gateway token assignment is not active")
        except GatewayTokenScopeError:
            return _error_response(403, "gateway token scope mismatch")
        except (GatewayTokenExpired, GatewayTokenInvalid):
            return _error_response(401, "invalid gateway token")
        except GatewayTokenError:
            return _error_response(401, "invalid gateway token")
        except UnknownProviderError as exc:
            return _error_response(exc.status_code, exc.detail)

        body = await request.body()
        try:
            upstream = await service.forward(
                provider=call.provider,
                model=call.model,
                path=path,
                body=body,
                caller_headers=request.headers,
            )
        except GatewayError as exc:
            return _error_response(exc.status_code, exc.detail)
        except Exception:
            logger.exception("llm gateway upstream forward failed")
            return _error_response(502, UPSTREAM_ERROR_DETAIL)

        # An upstream error is surfaced as a controlled, non-leaking status; the
        # raw upstream body/headers are never relayed back to the caller.
        if upstream.status_code >= 400:
            return _surface_upstream_error(upstream.status_code)

        await service.record_usage(
            claims=call.claims,
            provider=call.provider,
            model=call.model,
            response=upstream,
        )
        return Response(
            content=upstream.body,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=upstream.media_type,
        )

    @router.post("/llm/v1/{path:path}")
    async def llm_v1(path: str, request: Request) -> Response:
        return await handle(path, request)

    return router


def _error_response(status_code: int, detail: str) -> Response:
    """A safe JSON error body that never carries key/token material."""

    return Response(
        content=json.dumps({"detail": detail}).encode("utf-8"),
        status_code=status_code,
        media_type="application/json",
    )


def _surface_upstream_error(status_code: int) -> Response:
    """Map an upstream error status to a controlled, non-leaking response.

    The upstream body and headers are never relayed; only a generic detail and a
    controlled status are returned. Rate limiting (``429``) is preserved so
    callers can back off. Any other caller-induced upstream ``4xx`` collapses to
    a controlled ``400`` (so a bad request is distinguishable from a server-side
    failure, improving debuggability without leaking secrets). Every ``5xx``
    (and any non-4xx error) collapses to ``502``.
    """

    if status_code == 429:
        return _error_response(429, UPSTREAM_RATE_LIMITED_DETAIL)
    if 400 <= status_code < 500:
        return _error_response(400, UPSTREAM_REJECTED_DETAIL)
    return _error_response(502, UPSTREAM_ERROR_DETAIL)


__all__ = [
    "BASE_LLM_GATEWAY_URL_ENV",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "DEFAULT_SOURCE",
    "GATEWAY_ASSIGNMENT_HEADER",
    "GATEWAY_TOKEN_ENV",
    "GATEWAY_TOKEN_HEADER",
    "GATEWAY_VALIDATOR_HEADER",
    "UPSTREAM_ERROR_DETAIL",
    "UPSTREAM_RATE_LIMITED_DETAIL",
    "UPSTREAM_REJECTED_DETAIL",
    "AuthenticatedCall",
    "GatewayAssignmentInactiveError",
    "GatewayError",
    "LLMGatewayService",
    "SourceRoute",
    "UnknownProviderError",
    "build_llm_gateway_router",
    "build_llm_gateway_service",
]
