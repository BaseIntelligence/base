from __future__ import annotations

from fastapi import Header, HTTPException, status

from base.security.tokens import verify_token


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return ""
    return authorization.split(" ", 1)[1].strip()


def require_challenge_token(token_hash: str):
    async def dependency(authorization: str | None = Header(default=None)) -> None:
        if not verify_token(bearer_token(authorization), token_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
            )

    return dependency
