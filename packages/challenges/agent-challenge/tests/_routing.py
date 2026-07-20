"""Test helpers for inspecting FastAPI routes regardless of inclusion strategy.

Modern FastAPI defers router inclusion: ``app.routes`` holds ``_IncludedRouter``
placeholders whose concrete ``APIRoute`` objects live on
``placeholder.original_router.routes``. Older FastAPI eagerly flattens routes
directly into ``app.routes``. The helpers below resolve the real ``APIRoute``
objects in either case so route-contract assertions reflect what the app serves.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import BaseRoute


def iter_api_routes(app: FastAPI) -> Iterator[APIRoute]:
    """Yield every ``APIRoute`` served by ``app``, flattening included routers."""

    seen: set[int] = set()

    def _walk(routes: Iterable[BaseRoute]) -> Iterator[APIRoute]:
        for route in routes:
            if isinstance(route, APIRoute):
                if id(route) not in seen:
                    seen.add(id(route))
                    yield route
                continue
            sub = getattr(route, "routes", None)
            if sub is None:
                original = getattr(route, "original_router", None)
                sub = getattr(original, "routes", None)
            if sub:
                yield from _walk(sub)

    yield from _walk(app.routes)


def public_route_paths(app: FastAPI) -> set[str]:
    """Return the set of paths decorated with ``@public_route``."""

    return {
        route.path
        for route in iter_api_routes(app)
        if getattr(route.endpoint, "__base_public_route__", False)
    }
