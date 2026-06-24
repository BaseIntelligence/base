from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_token(token: str, token_hash: str) -> bool:
    return bool(
        token and token_hash and hmac.compare_digest(hash_token(token), token_hash)
    )


def token_hint(token: str) -> str:
    return f"{token[:4]}…{token[-4:]}"
