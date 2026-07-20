"""Validator-side periodic checkpoint cadence + push client (architecture.md section 7).

During a long re-execution the validator persists a crash-recovery checkpoint on a configurable
cadence (hourly by default) and PUSHES it to the master, which publishes it to HuggingFace. This
module owns the cadence decision and the signed HTTP push; the validator holds NO HuggingFace token
(only the master publishes). The push reuses the hotkey-signature scheme the master verifies
(:func:`prism_challenge.auth.canonical_checkpoint_message`) and sends the EXACT signed body bytes so
the server body-hash matches.
"""

from __future__ import annotations

import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import httpx

from ..auth import canonical_checkpoint_message
from .checkpoints import resolve_checkpoint_artifact_path

DEFAULT_CHECKPOINT_PUSH_PATH = "/internal/v1/checkpoints"


@dataclass(frozen=True)
class CheckpointCadence:
    """A configurable persist/push cadence (architecture.md section 7; hourly by default).

    The cadence is crash-recovery only and never part of the score; a smaller interval (e.g. in
    tests) persists/pushes more frequently. ``due`` is a strict ``>=`` so the first checkpoint at
    ``interval_seconds`` elapsed is taken.
    """

    interval_seconds: float

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("checkpoint cadence interval_seconds must be > 0")

    def due(self, *, elapsed_seconds: float) -> bool:
        """Whether enough time has elapsed since the last checkpoint to persist/push the next."""
        return elapsed_seconds >= self.interval_seconds

    def due_at(self, *, last_checkpoint_at: float, now: float) -> bool:
        """Whether a checkpoint is due given the wall-clock of the last push and now."""
        return self.due(elapsed_seconds=now - last_checkpoint_at)


class CheckpointPushError(RuntimeError):
    """The master rejected (or failed) a checkpoint push."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"checkpoint push failed: HTTP {status_code}: {detail}")
        self.status_code = status_code


@runtime_checkable
class CheckpointSigner(Protocol):
    """Signs the canonical checkpoint-upload message with the validator hotkey."""

    @property
    def hotkey(self) -> str: ...

    def sign(self, message: bytes) -> str: ...


@dataclass(frozen=True)
class DevHmacCheckpointSigner:
    """Dev HMAC signer (only accepted by the master when ``allow_insecure_signatures`` is on)."""

    hotkey: str
    secret: str

    def sign(self, message: bytes) -> str:
        return hmac.new(self.secret.encode(), message, sha256).hexdigest()


class KeypairCheckpointSigner:
    """Production signer wrapping a bittensor ``Keypair`` (sr25519 hotkey signature)."""

    def __init__(self, keypair: Any) -> None:
        self._keypair = keypair

    @property
    def hotkey(self) -> str:
        return str(self._keypair.ss58_address)

    def sign(self, message: bytes) -> str:
        return self._keypair.sign(message).hex()


def read_checkpoint_files(checkpoint_dir: Path, files: tuple[str, ...]) -> dict[str, bytes]:
    """Read the persisted checkpoint ``files`` (path-safe) into a name -> bytes mapping."""
    if not files:
        raise ValueError("checkpoint push must list at least one file")
    contents: dict[str, bytes] = {}
    for name in files:
        source = resolve_checkpoint_artifact_path(checkpoint_dir, name)
        if not source.is_file():
            raise ValueError(f"checkpoint file is missing: {name}")
        contents[name] = source.read_bytes()
    return contents


def _encode_files(files: Mapping[str, bytes]) -> dict[str, str]:
    import base64

    return {name: base64.b64encode(data).decode("ascii") for name, data in files.items()}


class CheckpointPushClient:
    """Async client that signs and pushes a persisted checkpoint to the master publish endpoint."""

    def __init__(
        self,
        base_url: str,
        signer: CheckpointSigner,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
        path: str = DEFAULT_CHECKPOINT_PUSH_PATH,
        now_fn: Any = time.time,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._signer = signer
        self._transport = transport
        self._timeout = timeout_seconds
        self._path = path
        self._now_fn = now_fn

    @property
    def hotkey(self) -> str:
        return self._signer.hotkey

    def _signed_headers(self, body: bytes) -> dict[str, str]:
        nonce = uuid4().hex
        timestamp = str(int(self._now_fn()))
        message = canonical_checkpoint_message(
            hotkey=self._signer.hotkey, nonce=nonce, timestamp=timestamp, body=body
        )
        return {
            "X-Hotkey": self._signer.hotkey,
            "X-Signature": self._signer.sign(message),
            "X-Nonce": nonce,
            "X-Timestamp": timestamp,
            "Content-Type": "application/json",
        }

    async def push(
        self,
        *,
        submission_id: str,
        attempt: int,
        files: Mapping[str, bytes],
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Sign and POST the checkpoint files to the master; return the publish result JSON.

        Sends the EXACT signed body bytes (``content=body``) so the server body-hash matches the
        signature. Non-2xx -> :class:`CheckpointPushError` (carrying the status code), so an
        unsigned/forged/ineligible push surfaces the 401/403 to the caller.
        """
        if not files:
            raise ValueError("checkpoint push must contain at least one file")
        payload: dict[str, Any] = {
            "submission_id": submission_id,
            "attempt": attempt,
            "files": _encode_files(files),
        }
        if revision is not None:
            payload["revision"] = revision
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = self._signed_headers(body)
        async with httpx.AsyncClient(
            base_url=self._base_url, transport=self._transport, timeout=self._timeout
        ) as client:
            response = await client.post(self._path, content=body, headers=headers)
        if response.status_code >= 400:
            raise CheckpointPushError(response.status_code, response.text)
        return dict(response.json())

    async def push_checkpoint_dir(
        self,
        *,
        submission_id: str,
        attempt: int,
        checkpoint_dir: Path,
        files: tuple[str, ...],
        revision: str | None = None,
    ) -> dict[str, Any]:
        """Read a persisted checkpoint dir's ``files`` and push them to the master."""
        contents = read_checkpoint_files(checkpoint_dir, files)
        return await self.push(
            submission_id=submission_id,
            attempt=attempt,
            files=contents,
            revision=revision,
        )
