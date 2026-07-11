"""Inspectable role metadata shared by Base challenge consumers."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def public_route(
    *,
    tags: list[str] | None = None,
    auth_required: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a route as part of a challenge's published HTTP surface."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func.__base_public_route__ = True  # type: ignore[attr-defined]
        func.__base_public_tags__ = tuple(tags or ())  # type: ignore[attr-defined]
        func.__base_public_auth_required__ = auth_required  # type: ignore[attr-defined]
        return func

    return decorator


def is_public_route(func: Callable[..., object]) -> bool:
    return bool(getattr(func, "__base_public_route__", False))


__all__ = ["is_public_route", "public_route"]
