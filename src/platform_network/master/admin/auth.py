from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

TokenProvider = Callable[[], str | Awaitable[str]]


def load_admin_token_from_environment() -> str:
    token = os.getenv("ADMIN_TOKEN")
    if token:
        return token
    token_file = os.getenv("ADMIN_TOKEN_FILE")
    if token_file:
        with open(token_file, encoding="utf-8") as file:
            return file.read().strip()
    return ""


async def resolve_token(provider: TokenProvider) -> str:
    token = provider()
    if hasattr(token, "__await__"):
        return await token  # type: ignore[misc]
    return token


def constant_time_match(left: str, right: str) -> bool:
    return bool(left and right and hmac.compare_digest(left, right))
