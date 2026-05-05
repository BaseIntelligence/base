"""Public FastAPI proxy app for challenge routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from posixpath import normpath
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status

from platform_network.master.registry import ChallengeNotFoundError, ChallengeRegistry
from platform_network.schemas.challenge import ChallengeStatus

SENSITIVE_REQUEST_HEADERS = {
    "authorization",
    "proxy-authorization",
    "x-admin-token",
    "x-platform-admin-token",
    "x-platform-challenge-token",
    "x-platform-internal-token",
    "x-platform-shared-token",
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


ClientFactory = Callable[[], AbstractAsyncContextManager[httpx.AsyncClient]]


def is_blocked_proxy_path(path: str) -> bool:
    """Return whether a public proxy path targets a private challenge route."""

    normalized = normpath(f"/{path.lstrip('/')}")
    return (
        normalized in BLOCKED_EXACT_PATHS
        or normalized == "/internal"
        or normalized.startswith("/internal/")
    )


def _forward_headers(request: Request) -> dict[str, str]:
    """Copy safe request headers for forwarding to a public challenge route."""

    headers: dict[str, str] = {}
    for key, value in request.headers.items():
        lowered = key.lower()
        if (
            lowered in HOP_BY_HOP_HEADERS
            or lowered in SENSITIVE_REQUEST_HEADERS
            or lowered == "host"
        ):
            continue
        headers[key] = value

    headers["X-Platform-Proxy"] = "true"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    """Copy safe upstream response headers back to the caller."""

    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


def _target_url(base_url: str, path: str, query: str) -> str:
    safe_path = quote(path.lstrip("/"), safe="/")
    url = f"{base_url.rstrip('/')}/{safe_path}"
    if query:
        url = f"{url}?{query}"
    return url


@asynccontextmanager
async def _default_client_factory() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        yield client


def create_proxy_app(
    *,
    registry: ChallengeRegistry | None = None,
    client_factory: ClientFactory = _default_client_factory,
) -> FastAPI:
    """Create the public proxy FastAPI app.

    Admin/registry concerns are intentionally not mounted in this app.
    """

    app = FastAPI(title="Platform Network Challenge Proxy", version="1.0")
    challenge_registry = registry or ChallengeRegistry()

    async def proxy_request(slug: str, path: str, request: Request) -> Response:
        if is_blocked_proxy_path(path):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Proxy path is not allowed",
            )

        try:
            challenge = challenge_registry.get(slug)
        except ChallengeNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Challenge not found"
            ) from exc

        if challenge.status != ChallengeStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Challenge not found"
            )

        body = await request.body()
        headers = _forward_headers(request)
        headers["X-Platform-Challenge-Slug"] = slug
        url = _target_url(challenge.internal_base_url, path, request.url.query)

        try:
            async with client_factory() as client:
                upstream = await client.request(
                    request.method,
                    url,
                    content=body,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail="Challenge unavailable"
            ) from exc

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=upstream.headers.get("content-type"),
        )

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

    app.state.challenge_registry = challenge_registry
    return app


app = create_proxy_app()
