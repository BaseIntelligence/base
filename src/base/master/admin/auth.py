from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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


def build_admin_token_dependency(
    provider: TokenProvider,
) -> Callable[..., Awaitable[None]]:
    """Build a FastAPI dependency enforcing the admin token.

    Accepts the token via the ``X-Admin-Token`` header or a bearer
    ``Authorization`` header and rejects a missing/invalid token with ``401``.
    """

    bearer_scheme = HTTPBearer(auto_error=False)

    async def require_admin(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        expected = await resolve_token(provider)
        provided = x_admin_token or (credentials.credentials if credentials else "")
        if not constant_time_match(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
            )

    return require_admin
