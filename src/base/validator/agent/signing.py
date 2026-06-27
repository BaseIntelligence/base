"""Client-side hotkey signing for validator coordination requests.

Produces the ``X-Hotkey``/``X-Signature``/``X-Nonce``/``X-Timestamp`` headers the
master coordination plane verifies. The canonical string is built with the same
:func:`base.security.validator_auth.canonical_validator_request` the server uses,
so the signed bytes are byte-for-byte identical on both sides.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from base.security.validator_auth import (
    HOTKEY_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIMESTAMP_HEADER,
    canonical_validator_request,
)


@runtime_checkable
class RequestSigner(Protocol):
    """Signs canonical request bytes with a validator hotkey."""

    @property
    def hotkey(self) -> str: ...

    def sign(self, message: bytes) -> str: ...


@dataclass(frozen=True)
class KeypairRequestSigner:
    """A :class:`RequestSigner` backed by a substrate (bittensor) keypair."""

    keypair: Any

    @property
    def hotkey(self) -> str:
        return str(self.keypair.ss58_address)

    def sign(self, message: bytes) -> str:
        signature = self.keypair.sign(message)
        if isinstance(signature, bytes | bytearray):
            return "0x" + bytes(signature).hex()
        return str(signature)


def build_signed_headers(
    signer: RequestSigner,
    *,
    method: str,
    path: str,
    body: bytes = b"",
    query_string: str | bytes = "",
    nonce: str | None = None,
    timestamp: int | None = None,
    now_fn: Callable[[], float] = time.time,
) -> dict[str, str]:
    """Build the signed-request headers for a single coordination call."""

    nonce = nonce or uuid.uuid4().hex
    ts = str(int(timestamp if timestamp is not None else now_fn()))
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string=query_string,
        timestamp=ts,
        nonce=nonce,
        body=body,
    )
    return {
        HOTKEY_HEADER: signer.hotkey,
        SIGNATURE_HEADER: signer.sign(canonical.encode()),
        NONCE_HEADER: nonce,
        TIMESTAMP_HEADER: ts,
    }
