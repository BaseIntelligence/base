"""HTTP client for the master validator coordination plane.

Wraps the hotkey-signed ``register``/``heartbeat``/``pull``/``progress``/
``result`` routes (architecture.md sec 4). Every request is signed with the
validator's hotkey via :mod:`base.validator.agent.signing`; the exact body bytes
that are signed are the bytes sent on the wire so the server's body hash matches.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from base.schemas.assignment import (
    AssignmentProgressResponse,
    AssignmentPullResponse,
    AssignmentResultResponse,
    AssignmentView,
)
from base.schemas.validator import (
    ValidatorHeartbeatResponse,
    ValidatorRegisterResponse,
    ValidatorSubscriptionResponse,
)
from base.validator.agent.signing import RequestSigner, build_signed_headers


class CoordinationClientError(RuntimeError):
    """A coordination request failed (non-2xx response or transport error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CoordinationClient:
    """Async client for the validator-facing coordination endpoints."""

    def __init__(
        self,
        base_url: str,
        signer: RequestSigner,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 15.0,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._signer = signer
        self._transport = transport
        self._timeout = timeout_seconds
        self._now_fn = now_fn

    @property
    def hotkey(self) -> str:
        return self._signer.hotkey

    async def register(
        self,
        *,
        capabilities: list[str],
        version: str | None,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> ValidatorRegisterResponse:
        payload: dict[str, Any] = {
            "capabilities": list(capabilities),
            "version": version,
        }
        if last_seen_meta is not None:
            payload["last_seen_meta"] = dict(last_seen_meta)
        data = await self._post("/v1/validators/register", payload)
        return ValidatorRegisterResponse.model_validate(data)

    async def heartbeat(
        self,
        *,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> ValidatorHeartbeatResponse:
        payload: dict[str, Any] = {}
        if last_seen_meta is not None:
            payload["last_seen_meta"] = dict(last_seen_meta)
        data = await self._post("/v1/validators/heartbeat", payload)
        return ValidatorHeartbeatResponse.model_validate(data)

    async def subscribe(self, slugs: Sequence[str]) -> ValidatorSubscriptionResponse:
        data = await self._post("/v1/validators/subscriptions", {"slugs": list(slugs)})
        return ValidatorSubscriptionResponse.model_validate(data)

    async def pull(self) -> list[AssignmentView]:
        data = await self._post("/v1/assignments/pull", {})
        return AssignmentPullResponse.model_validate(data).assignments

    async def progress(
        self,
        assignment_id: str,
        *,
        checkpoint_ref: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> AssignmentProgressResponse:
        payload: dict[str, Any] = {}
        if checkpoint_ref is not None:
            payload["checkpoint_ref"] = checkpoint_ref
        if meta is not None:
            payload["meta"] = dict(meta)
        data = await self._post(f"/v1/assignments/{assignment_id}/progress", payload)
        return AssignmentProgressResponse.model_validate(data)

    async def post_result(
        self,
        assignment_id: str,
        *,
        success: bool,
        payload: Mapping[str, Any] | None = None,
        checkpoint_ref: str | None = None,
    ) -> AssignmentResultResponse:
        body: dict[str, Any] = {"success": success, "payload": dict(payload or {})}
        if checkpoint_ref is not None:
            body["checkpoint_ref"] = checkpoint_ref
        data = await self._post(f"/v1/assignments/{assignment_id}/result", body)
        return AssignmentResultResponse.model_validate(data)

    async def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = build_signed_headers(
            self._signer,
            method="POST",
            path=path,
            body=body,
            now_fn=self._now_fn,
        )
        headers["Content-Type"] = "application/json"
        try:
            async with self._build_client() as client:
                response = await client.post(path, content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise CoordinationClientError(
                f"coordination request to {path} failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise CoordinationClientError(
                f"coordination request to {path} returned {response.status_code}",
                status_code=response.status_code,
            )
        return response.json()

    def _build_client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)
