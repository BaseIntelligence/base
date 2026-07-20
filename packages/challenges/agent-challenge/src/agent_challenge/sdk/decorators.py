"""Route decorators consumed by the BASE proxy discovery layer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

Fn = TypeVar("Fn", bound=Callable[..., object])


def public_route(*, tags: list[str] | None = None) -> Callable[[Fn], Fn]:
    """Mark a FastAPI endpoint as publicly proxied by BASE."""

    def decorate(fn: Fn) -> Fn:
        fn.__base_public_route__ = True  # type: ignore[attr-defined]
        fn.__base_public_route_tags__ = tags or []  # type: ignore[attr-defined]
        return fn

    return decorate
