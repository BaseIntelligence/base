"""Internal BASE authentication helpers."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
from typing import Annotated

from fastapi import Header, HTTPException, status

from .config import ChallengeSettings

#: Domain separation tag so a per-attempt stream token can never collide with
#: any other HMAC derived from the shared internal token.
_ATTEMPT_STREAM_TOKEN_CONTEXT = b"agent-challenge:eval-log-stream:v1"


def load_internal_token(settings: ChallengeSettings) -> str | None:
    """Return the configured internal shared token (inline or file-backed)."""

    return _load_token(settings)


def mint_attempt_stream_token(shared_token: str, attempt_id: int) -> str:
    """Derive a per-attempt log-stream token from the internal shared token.

    The own_runner job runs the miner agent in-process, so it can read its own
    env. We therefore never hand it the raw ``shared_token`` (which authorizes
    every internal route); instead it gets this HMAC, scoped to a single
    ``attempt_id`` and usable only to append log lines to that one attempt.
    """

    message = _ATTEMPT_STREAM_TOKEN_CONTEXT + b":" + str(int(attempt_id)).encode("ascii")
    return hmac.new(shared_token.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_attempt_stream_token(shared_token: str, attempt_id: int, presented: str) -> bool:
    """Constant-time check that ``presented`` is the token for ``attempt_id``."""

    expected = mint_attempt_stream_token(shared_token, attempt_id)
    return hmac.compare_digest(expected, presented)


def build_attempt_stream_auth_dependency(settings: ChallengeSettings):
    """Build a dependency validating a per-attempt log-stream bearer token.

    Unlike :func:`build_internal_auth_dependency` (which requires the full shared
    token), this accepts only the scoped HMAC bound to the path ``attempt_id`` --
    so a leaked job token cannot be replayed against another attempt or any other
    internal route.
    """

    async def authenticate(
        attempt_id: int,
        authorization: Annotated[str | None, Header()] = None,
        x_base_challenge_slug: Annotated[str | None, Header()] = None,
    ) -> None:
        token = _load_token(settings)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="internal token is not configured",
            )
        if x_base_challenge_slug != settings.slug:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid challenge slug",
            )
        prefix = "Bearer "
        presented = (
            authorization[len(prefix) :]
            if authorization and authorization.startswith(prefix)
            else ""
        )
        if not presented or not verify_attempt_stream_token(token, attempt_id, presented):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
            )

    return authenticate


def build_internal_auth_dependency(settings: ChallengeSettings):
    """Build a FastAPI dependency that validates BASE internal calls."""

    async def authenticate(
        authorization: Annotated[str | None, Header()] = None,
        x_base_challenge_slug: Annotated[str | None, Header()] = None,
    ) -> None:
        token = _load_token(settings)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="internal token is not configured",
            )
        if x_base_challenge_slug != settings.slug:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid challenge slug",
            )
        if authorization != f"Bearer {token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
            )

    return authenticate


def _load_token(settings: ChallengeSettings) -> str | None:
    if settings.shared_token:
        return settings.shared_token
    if settings.shared_token_file:
        path = Path(settings.shared_token_file)
        if path.is_file():
            token = path.read_text(encoding="utf-8").strip()
            return token or None
    return None
