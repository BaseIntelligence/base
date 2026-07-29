"""Public TEE math surface removed (T40). Always reports unavailable."""

from __future__ import annotations

from typing import Any, Mapping


def public_tee_unavailable() -> dict[str, Any]:
    return {
        "available": False,
        "reason": "phala_tee_removed_host_trust_only",
        "tee": None,
        "measurement": None,
        "quote": None,
    }


def public_tee_assignment_qualifies(*args: Any, **kwargs: Any) -> bool:
    return False


def build_public_tee_math(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return public_tee_unavailable()


def build_public_tee_math_from_assignment(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return public_tee_unavailable()


def assert_public_tee_safe(payload: Mapping[str, Any]) -> None:
    if payload.get("available") is True:
        raise ValueError("public TEE math must not claim available after Phala removal")


__all__ = [
    "assert_public_tee_safe",
    "build_public_tee_math",
    "build_public_tee_math_from_assignment",
    "public_tee_assignment_qualifies",
    "public_tee_unavailable",
]
