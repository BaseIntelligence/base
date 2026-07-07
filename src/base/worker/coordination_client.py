"""HTTP client for the master worker coordination + assignment plane.

The worker agent authenticates as its WORKER keypair (NOT a metagraph validator
permit, VAL-AGENT-018): registration carries the miner's binding signature in the
body, while heartbeat/pull/result are hotkey-signed requests in the canonical
scheme shared with the validator plane (the signer here is the worker keypair).
Pull/result reuse the assignment schemas so the executor seam is identical to the
validator agent's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from base.schemas.assignment import (
    AssignmentPullResponse,
    AssignmentResultResponse,
    AssignmentView,
)
from base.schemas.worker import WorkerHeartbeatResponse, WorkerRegisterResponse
from base.validator.agent.signing import RequestSigner, build_signed_headers


class WorkerCoordinationClientError(RuntimeError):
    """A worker coordination request failed (non-2xx or transport error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class WorkerCoordinationClient:
    """Async client for the worker-facing register/heartbeat/pull/result routes."""

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
    def worker_pubkey(self) -> str:
        return self._signer.hotkey

    async def register(
        self,
        *,
        miner_hotkey: str,
        binding_signature: str,
        nonce: str,
        provider: str,
        provider_instance_ref: str | None = None,
        capabilities: list[str],
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> WorkerRegisterResponse:
        payload: dict[str, Any] = {
            "worker_pubkey": self._signer.hotkey,
            "miner_hotkey": miner_hotkey,
            "binding_signature": binding_signature,
            "nonce": nonce,
            "provider": provider,
            "provider_instance_ref": provider_instance_ref,
            "capabilities": list(capabilities),
        }
        if last_seen_meta is not None:
            payload["last_seen_meta"] = dict(last_seen_meta)
        data = await self._post("/v1/workers/register", payload, signed=False)
        return WorkerRegisterResponse.model_validate(data)

    async def heartbeat(
        self,
        *,
        worker_id: str,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> WorkerHeartbeatResponse:
        payload: dict[str, Any] = {}
        if last_seen_meta is not None:
            payload["last_seen_meta"] = dict(last_seen_meta)
        data = await self._post(
            f"/v1/workers/{worker_id}/heartbeat", payload, signed=True
        )
        return WorkerHeartbeatResponse.model_validate(data)

    async def pull(self) -> list[AssignmentView]:
        data = await self._post("/v1/workers/assignments/pull", {}, signed=True)
        return AssignmentPullResponse.model_validate(data).assignments

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
        data = await self._post(
            f"/v1/workers/assignments/{assignment_id}/result", body, signed=True
        )
        return AssignmentResultResponse.model_validate(data)

    async def _post(
        self, path: str, payload: Mapping[str, Any], *, signed: bool
    ) -> dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if signed:
            headers.update(
                build_signed_headers(
                    self._signer,
                    method="POST",
                    path=path,
                    body=body,
                    now_fn=self._now_fn,
                )
            )
        try:
            async with self._build_client() as client:
                response = await client.post(path, content=body, headers=headers)
        except httpx.HTTPError as exc:
            raise WorkerCoordinationClientError(
                f"worker request to {path} failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            raise WorkerCoordinationClientError(
                f"worker request to {path} returned {response.status_code}",
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


__all__ = [
    "WorkerCoordinationClient",
    "WorkerCoordinationClientError",
]
