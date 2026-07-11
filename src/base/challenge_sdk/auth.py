"""Fail-closed authentication helpers for challenge-owned internal routes."""

from __future__ import annotations

import secrets
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException, status

from .config import ChallengeSettings


def load_shared_token(settings: ChallengeSettings) -> str | None:
    """Load the configured challenge token without exposing it in diagnostics."""

    if settings.shared_token:
        return settings.shared_token
    if not settings.shared_token_file:
        return None
    token_path = Path(settings.shared_token_file)
    if not token_path.is_file():
        return None
    token = token_path.read_text(encoding="utf-8").strip()
    return token or None


def build_internal_auth_dependency(
    settings: ChallengeSettings,
) -> Callable[..., Coroutine[Any, Any, None]]:
    """Build the canonical bearer-token and challenge-binding dependency."""

    async def verify_internal_auth(
        authorization: str | None = Header(default=None),
        challenge_slug: str | None = Header(
            default=None,
            alias="X-Base-Challenge-Slug",
        ),
    ) -> None:
        expected_token = load_shared_token(settings)
        if expected_token is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Internal challenge token is not configured",
            )
        expected_header = f"Bearer {expected_token}"
        if authorization is None or not secrets.compare_digest(
            authorization,
            expected_header,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid internal authorization",
            )
        if challenge_slug != settings.slug:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid challenge slug header",
            )

    return verify_internal_auth


__all__ = ["build_internal_auth_dependency", "load_shared_token"]
