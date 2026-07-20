from __future__ import annotations

import hmac
import logging
import time
from hashlib import sha256
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from .config import PrismSettings

logger = logging.getLogger(__name__)


def canonical_submission_message(*, hotkey: str, nonce: str, timestamp: str, body: bytes) -> bytes:
    body_hash = sha256(body).hexdigest()
    return f"prism:{hotkey}:{nonce}:{timestamp}:{body_hash}".encode()


def canonical_checkpoint_message(*, hotkey: str, nonce: str, timestamp: str, body: bytes) -> bytes:
    """Canonical bytes a validator signs to upload a checkpoint for HF publication.

    A validator-specific domain prefix keeps a signed checkpoint upload from being replayable as a
    miner submission (and vice versa) even though both reuse the same hotkey signature scheme.
    """
    body_hash = sha256(body).hexdigest()
    return f"prism:checkpoint:{hotkey}:{nonce}:{timestamp}:{body_hash}".encode()


def _decode_signature(signature: str) -> bytes | str:
    value = signature.removeprefix("0x")
    try:
        return bytes.fromhex(value)
    except ValueError:
        return signature


def verify_hotkey_signature(hotkey: str, message: bytes, signature: str) -> bool:
    try:
        # bittensor v11 removed bt.Keypair; resolve via wallet/sp_core (see keypair.py).
        from .keypair import keypair_from_ss58

        keypair = keypair_from_ss58(hotkey)
        return bool(keypair.verify(message, _decode_signature(signature)))
    except Exception as exc:
        # Security: log only exception type + message, never the signature or message bytes.
        logger.debug(
            "hotkey signature verification failed for hotkey=%s: %s: %s",
            hotkey,
            type(exc).__name__,
            exc,
        )
        return False


def verify_dev_signature(secret: str, message: bytes, signature: str) -> bool:
    expected = hmac.new(secret.encode(), message, sha256).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix("sha256="))


async def authenticate_miner(
    request: Request,
    x_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
    x_signature: Annotated[str, Header(min_length=1)],
    x_nonce: Annotated[str, Header(min_length=1, max_length=128)],
    x_timestamp: Annotated[str, Header(min_length=1)],
) -> str:
    app_settings: PrismSettings = request.app.state.settings
    if not app_settings.public_submissions_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "submission route disabled")
    try:
        timestamp = int(x_timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid timestamp") from exc
    if abs(int(time.time()) - timestamp) > app_settings.signature_ttl_seconds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "stale signature")
    if x_hotkey in app_settings.validator_hotkeys:
        logger.warning("rejected self-submission from validator hotkey %s", x_hotkey)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "validator hotkey is not allowed to submit")
    body = await request.body()
    message = canonical_submission_message(
        hotkey=x_hotkey, nonce=x_nonce, timestamp=x_timestamp, body=body
    )
    valid = verify_hotkey_signature(x_hotkey, message, x_signature)
    if not valid and app_settings.allow_insecure_signatures:
        valid = verify_dev_signature(app_settings.internal_token(), message, x_signature)
    if not valid:
        logger.warning("submission signature rejected for hotkey=%s nonce=%s", x_hotkey, x_nonce)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")
    async with request.app.state.database.connect() as conn:
        try:
            await conn.execute(
                "INSERT INTO nonces(hotkey, nonce, created_at) VALUES (?, ?, datetime('now'))",
                (x_hotkey, x_nonce),
            )
        except Exception as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "nonce already used") from exc
    logger.info("submission authenticated hotkey=%s nonce=%s", x_hotkey, x_nonce)
    return x_hotkey


def authenticate_internal(
    request: Request, authorization: Annotated[str | None, Header()] = None
) -> None:
    app_settings: PrismSettings = request.app.state.settings
    expected = f"Bearer {app_settings.internal_token()}"
    if authorization != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")


async def authenticate_validator(
    request: Request,
    x_hotkey: Annotated[str, Header(min_length=1, max_length=128)],
    x_signature: Annotated[str, Header(min_length=1)],
    x_nonce: Annotated[str, Header(min_length=1, max_length=128)],
    x_timestamp: Annotated[str, Header(min_length=1)],
) -> str:
    """Authenticate a validator-signed, permit-gated request (checkpoint upload / HF publish).

    Enforces the SAME canonical hotkey-signature scheme as miner submissions over a
    validator-specific domain, then a metagraph-validator-permit gate: only a hotkey present in the
    configured ``validator_hotkeys`` (the mocked permit set in tests; the on-graph permitted set at
    deploy) may publish. Signature is checked BEFORE the permit so a forged/absent signature is a
    401 while a correctly-signed-but-ineligible hotkey is a 403 (architecture.md sections 4, 11;
    VAL-PRISM-038). A rejected request never reaches the publish path, so no checkpoint_ref is
    recorded on rejection.
    """
    app_settings: PrismSettings = request.app.state.settings
    try:
        timestamp = int(x_timestamp)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid timestamp") from exc
    if abs(int(time.time()) - timestamp) > app_settings.signature_ttl_seconds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "stale signature")
    body = await request.body()
    message = canonical_checkpoint_message(
        hotkey=x_hotkey, nonce=x_nonce, timestamp=x_timestamp, body=body
    )
    valid = verify_hotkey_signature(x_hotkey, message, x_signature)
    if not valid and app_settings.allow_insecure_signatures:
        valid = verify_dev_signature(app_settings.internal_token(), message, x_signature)
    if not valid:
        logger.warning(
            "checkpoint upload signature rejected for hotkey=%s nonce=%s", x_hotkey, x_nonce
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid signature")
    if x_hotkey not in app_settings.validator_hotkeys:
        logger.warning("checkpoint upload rejected for ineligible hotkey %s", x_hotkey)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "hotkey is not an eligible validator")
    async with request.app.state.database.connect() as conn:
        try:
            await conn.execute(
                "INSERT INTO nonces(hotkey, nonce, created_at) VALUES (?, ?, datetime('now'))",
                (x_hotkey, x_nonce),
            )
        except Exception as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, "nonce already used") from exc
    logger.info("checkpoint upload authenticated hotkey=%s nonce=%s", x_hotkey, x_nonce)
    return x_hotkey
