from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Protocol
from urllib.parse import parse_qsl, urlencode

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.db import database
from ..core.models import RequestNonce
from ..sdk.config import ChallengeSettings

HOTKEY_HEADER = "X-Hotkey"
SIGNATURE_HEADER = "X-Signature"
NONCE_HEADER = "X-Nonce"
TIMESTAMP_HEADER = "X-Timestamp"
INVALID_AUTH_DETAIL = "invalid signed request"
REPLAY_AUTH_DETAIL = "replayed request"


class SignatureVerifier(Protocol):
    def __call__(self, hotkey: str, message: str, signature: str) -> bool: ...


class SignatureVerifierUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SignedRequestAuth:
    hotkey: str
    signature: str
    nonce: str
    timestamp: str
    body_sha256: str
    canonical_request: str


DatabaseSession = Annotated[AsyncSession, Depends(database.session_dependency)]
NowProvider = Callable[[], datetime]


def body_sha256(raw_body: bytes) -> str:
    return hashlib.sha256(raw_body).hexdigest()


def sorted_path_with_query(path: str, query_string: str | bytes = "") -> str:
    if isinstance(query_string, bytes):
        query_string = query_string.decode("utf-8")
    if not query_string:
        return path
    pairs = sorted(parse_qsl(query_string, keep_blank_values=True))
    return f"{path}?{urlencode(pairs)}"


def canonical_request_string(
    *,
    method: str,
    path: str,
    query_string: str | bytes,
    timestamp: str,
    nonce: str,
    raw_body: bytes,
) -> str:
    return "\n".join(
        (
            method.upper(),
            sorted_path_with_query(path, query_string),
            timestamp,
            nonce,
            body_sha256(raw_body),
        )
    )


async def authenticate_signed_request(
    *,
    request: Request,
    session: AsyncSession,
    settings: ChallengeSettings,
    hotkey: str | None,
    signature: str | None,
    nonce: str | None,
    timestamp: str | None,
    verifier: SignatureVerifier | None = None,
    now_provider: NowProvider | None = None,
    require_owner: bool = False,
) -> SignedRequestAuth:
    hotkey = _present_header(hotkey)
    signature = _present_header(signature)
    nonce = _present_header(nonce)
    timestamp = _present_header(timestamp)
    _validate_timestamp(timestamp, settings.signing_ttl_seconds, now_provider)
    if require_owner and hotkey != settings.owner_hotkey:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    raw_body = await request.body()
    canonical_request = canonical_request_string(
        method=request.method,
        path=request.url.path,
        query_string=request.url.query,
        timestamp=timestamp,
        nonce=nonce,
        raw_body=raw_body,
    )
    if not await asyncio.to_thread(
        _verify_signature,
        verifier or verify_substrate_signature,
        hotkey,
        canonical_request,
        signature,
    ):
        raise _invalid_auth_error()

    await reserve_nonce(
        session=session,
        hotkey=hotkey,
        nonce=nonce,
        ttl_seconds=settings.signing_ttl_seconds,
        now_provider=now_provider,
    )
    return SignedRequestAuth(
        hotkey=hotkey,
        signature=signature,
        nonce=nonce,
        timestamp=timestamp,
        body_sha256=body_sha256(raw_body),
        canonical_request=canonical_request,
    )


async def reserve_nonce(
    *,
    session: AsyncSession,
    hotkey: str,
    nonce: str,
    ttl_seconds: int,
    now_provider: NowProvider | None = None,
) -> None:
    now = _utc_now(now_provider)
    await session.execute(delete(RequestNonce).where(RequestNonce.expires_at < now))
    session.add(
        RequestNonce(
            hotkey=hotkey,
            nonce=nonce,
            expires_at=now + timedelta(seconds=max(ttl_seconds, 0)),
        )
    )
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=REPLAY_AUTH_DETAIL,
        ) from exc


def build_signed_auth_dependency(
    settings: ChallengeSettings,
    *,
    verifier: SignatureVerifier | None = None,
    now_provider: NowProvider | None = None,
):
    async def authenticate(
        request: Request,
        session: DatabaseSession,
        x_hotkey: Annotated[str | None, Header(alias=HOTKEY_HEADER)] = None,
        x_signature: Annotated[str | None, Header(alias=SIGNATURE_HEADER)] = None,
        x_nonce: Annotated[str | None, Header(alias=NONCE_HEADER)] = None,
        x_timestamp: Annotated[str | None, Header(alias=TIMESTAMP_HEADER)] = None,
    ) -> SignedRequestAuth:
        return await authenticate_signed_request(
            request=request,
            session=session,
            settings=settings,
            hotkey=x_hotkey,
            signature=x_signature,
            nonce=x_nonce,
            timestamp=x_timestamp,
            verifier=verifier,
            now_provider=now_provider,
        )

    return authenticate


def build_owner_signed_auth_dependency(
    settings: ChallengeSettings,
    *,
    verifier: SignatureVerifier | None = None,
    now_provider: NowProvider | None = None,
):
    async def authenticate_owner(
        request: Request,
        session: DatabaseSession,
        x_hotkey: Annotated[str | None, Header(alias=HOTKEY_HEADER)] = None,
        x_signature: Annotated[str | None, Header(alias=SIGNATURE_HEADER)] = None,
        x_nonce: Annotated[str | None, Header(alias=NONCE_HEADER)] = None,
        x_timestamp: Annotated[str | None, Header(alias=TIMESTAMP_HEADER)] = None,
    ) -> SignedRequestAuth:
        return await authenticate_signed_request(
            request=request,
            session=session,
            settings=settings,
            hotkey=x_hotkey,
            signature=x_signature,
            nonce=x_nonce,
            timestamp=x_timestamp,
            verifier=verifier,
            now_provider=now_provider,
            require_owner=True,
        )

    return authenticate_owner


def verify_substrate_signature(hotkey: str, message: str, signature: str) -> bool:
    try:
        import bittensor as bt
    except ImportError as exc:
        raise SignatureVerifierUnavailable("substrate signature verifier is unavailable") from exc

    try:
        return bool(bt.Keypair(ss58_address=hotkey).verify(message, signature))
    except Exception:
        return False


def _verify_signature(
    verifier: SignatureVerifier,
    hotkey: str,
    canonical_request: str,
    signature: str,
) -> bool:
    try:
        return bool(verifier(hotkey, canonical_request, signature))
    except SignatureVerifierUnavailable:
        return False


def _present_header(value: str | None) -> str:
    if value is None or not value.strip():
        raise _invalid_auth_error()
    return value.strip()


def _validate_timestamp(
    timestamp: str,
    ttl_seconds: int,
    now_provider: NowProvider | None,
) -> None:
    try:
        request_time = _parse_timestamp(timestamp)
    except (OSError, OverflowError, ValueError) as exc:
        raise _invalid_auth_error() from exc
    skew = abs((_utc_now(now_provider) - request_time).total_seconds())
    if skew > max(ttl_seconds, 0):
        raise _invalid_auth_error()


def _parse_timestamp(timestamp: str) -> datetime:
    try:
        numeric_timestamp = float(timestamp)
    except ValueError:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        if not math.isfinite(numeric_timestamp):
            raise ValueError("timestamp must be finite")
        parsed = datetime.fromtimestamp(numeric_timestamp, UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utc_now(now_provider: NowProvider | None) -> datetime:
    now = now_provider() if now_provider else datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _invalid_auth_error() -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=INVALID_AUTH_DETAIL)
