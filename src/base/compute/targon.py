"""Targon compute provider client (architecture.md sec 3.1).

Thin async ``httpx`` wrapper over the Targon REST API (base
``https://api.targon.com/tha/v2``, auth header ``Authorization: Bearer``). It
implements the :class:`~base.compute.provider` contract with the cost guardrails
baked in and two Targon-specific realities surfaced as typed errors:

* Targon exposes NO balance/credits endpoint, so :meth:`TargonClient.balance`
  raises :class:`BalanceUnavailableError` WITHOUT issuing any HTTP request rather
  than returning a fake/silent value.
* A deploy that fails for insufficient credits is surfaced as a distinct typed
  :class:`InsufficientCreditsError` and is NEVER retried (exactly one HTTP call).

The API key lives only in the request header; it is never logged, embedded in an
error message, or exposed via ``repr``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from base.compute.provider import (
    CostGuardrailError,
    Instance,
    InstanceSpec,
    Offer,
    ProviderError,
)

logger = logging.getLogger(__name__)

TARGON_API_BASE_URL = "https://api.targon.com/tha/v2"

_CREDIT_TOKENS = ("insufficient", "credit", "payment required", "out of credit")


class TargonError(ProviderError):
    """A Targon API request failed (non-2xx response or transport error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class InsufficientCreditsError(TargonError):
    """A Targon deploy was rejected for insufficient credits.

    Targon exposes no balance endpoint, so an insufficient-credit deploy failure
    is only observable at deploy time. It is a distinct typed error (identifiable
    via ``isinstance``) that callers must NOT retry.
    """


class BalanceUnavailableError(TargonError):
    """Account balance is not queryable: Targon exposes no balance endpoint."""


class TargonClient:
    """Async client for the Targon GPU compute API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = TARGON_API_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds

    def __repr__(self) -> str:
        return f"TargonClient(base_url={self._base_url!r})"

    # -- offers ---------------------------------------------------------------

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]:
        response = await self._request(
            "GET", "/inventory", params={"type": "rental", "gpu": "true"}
        )
        offers: list[Offer] = []
        for item in _as_list(response.json(), "items"):
            offer = _parse_inventory_offer(item)
            if offer is None:
                continue
            if (
                max_price_per_hour is not None
                and offer.price_per_hour > max_price_per_hour
            ):
                continue
            offers.append(offer)
        return offers

    # -- apps / workloads listing --------------------------------------------

    async def list_apps(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/apps")
        return _as_list(response.json(), "items")

    async def create_app(
        self, name: str, *, project_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if project_id is not None:
            body["project_id"] = project_id
        response = await self._request("POST", "/apps", json_body=body)
        result = response.json()
        return result if isinstance(result, dict) else {}

    async def list_workloads(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/workloads")
        return _as_list(response.json(), "items")

    async def workload_state(self, uid: str) -> dict[str, Any]:
        response = await self._request("GET", f"/workloads/{uid}/state")
        result = response.json()
        return result if isinstance(result, dict) else {}

    async def workload_events(self, uid: str) -> dict[str, Any]:
        response = await self._request("GET", f"/workloads/{uid}/events")
        result = response.json()
        return result if isinstance(result, dict) else {}

    # -- lifecycle ------------------------------------------------------------

    async def provision(self, spec: InstanceSpec) -> Instance:
        lifetime = spec.max_lifetime_hours
        if lifetime is None or lifetime <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_lifetime_hours must be a positive bound (hours)"
            )
        if spec.max_price_per_hour is None or spec.max_price_per_hour <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_price_per_hour must be a positive bound"
            )
        return await self.deploy(_build_workload_payload(spec, lifetime=lifetime))

    async def deploy(self, workload: Mapping[str, Any]) -> Instance:
        response = await self._send(
            "POST", "/workloads/deploy", json_body=dict(workload)
        )
        if response.status_code >= 400:
            if _is_insufficient_credits(response.status_code, response.text):
                raise InsufficientCreditsError(
                    "Targon POST /workloads/deploy rejected for insufficient "
                    f"credits (status {response.status_code})",
                    status_code=response.status_code,
                )
            raise TargonError(
                f"Targon POST /workloads/deploy returned {response.status_code}",
                status_code=response.status_code,
            )
        return _parse_workload_instance(response.json())

    async def status(self, instance_id: str) -> Instance:
        response = await self._request("GET", f"/workloads/{instance_id}")
        return _parse_workload_instance(response.json())

    async def stream_logs(self, instance_id: str) -> AsyncIterator[str]:
        async with self._client() as client:
            async with client.stream(
                "GET", f"/workloads/{instance_id}/logs"
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise TargonError(
                        f"Targon GET /workloads/{instance_id}/logs returned "
                        f"{response.status_code}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    yield line

    async def terminate(self, instance_id: str) -> None:
        response = await self._send("DELETE", f"/workloads/{instance_id}")
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise TargonError(
                f"Targon DELETE /workloads/{instance_id} returned "
                f"{response.status_code}",
                status_code=response.status_code,
            )

    async def verify_terminated(self, instance_id: str) -> bool:
        response = await self._send("GET", f"/workloads/{instance_id}")
        if response.status_code == 404:
            return True
        if response.status_code >= 400:
            raise TargonError(
                f"Targon GET /workloads/{instance_id} returned {response.status_code}",
                status_code=response.status_code,
            )
        instance = _parse_workload_instance(response.json())
        return instance.status.lower() in {"deleted", "terminated", "stopped"}

    async def balance(self) -> float:
        raise BalanceUnavailableError(
            "Targon exposes no balance endpoint; balance is only visible in the "
            "web dashboard"
        )

    # -- internals ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict[str, Any] = {
            "base_url": self._base_url,
            "timeout": self._timeout,
            "headers": self._headers(),
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            async with self._client() as client:
                return await client.request(method, path, json=json_body, params=params)
        except httpx.HTTPError as exc:
            raise TargonError(f"Targon request {method} {path} failed") from exc

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        response = await self._send(method, path, json_body=json_body, params=params)
        if response.status_code >= 400:
            raise TargonError(
                f"Targon {method} {path} returned {response.status_code}",
                status_code=response.status_code,
            )
        return response


def _build_workload_payload(spec: InstanceSpec, *, lifetime: float) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": spec.name, "type": "rental"}
    if spec.template_ref:
        payload["resource_name"] = spec.template_ref
    if spec.image:
        payload["image"] = spec.image
    if spec.image_digest:
        payload["image_digest"] = spec.image_digest
    if spec.gpu_count:
        payload["gpu_count"] = spec.gpu_count
    payload["envs"] = [{"name": k, "value": v} for k, v in spec.env.items()]
    if spec.ports:
        payload["ports"] = [{"port": port} for port in spec.ports]
    if spec.ssh_public_keys:
        payload["ssh_public_keys"] = list(spec.ssh_public_keys)
    payload["termination_hours"] = int(lifetime)
    return payload


def _is_insufficient_credits(status_code: int, body_text: str) -> bool:
    if status_code in (402, 403):
        return True
    lowered = (body_text or "").lower()
    return any(token in lowered for token in _CREDIT_TOKENS)


def _as_list(data: Any, wrapper_key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, Mapping):
        inner = data.get(wrapper_key)
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def _parse_inventory_offer(item: Mapping[str, Any]) -> Offer | None:
    name = item.get("name") or item.get("display_name")
    cost = item.get("cost_per_hour")
    if not name or cost is None:
        return None
    try:
        price = float(cost)
    except (TypeError, ValueError):
        return None
    if _coerce_int(item.get("available")) <= 0:
        return None
    return Offer(
        id=str(name),
        gpu_type=_extract_gpu_type(item),
        gpu_count=_extract_gpu_count(item),
        price_per_hour=price,
        provider="targon",
        raw=item,
    )


def _extract_gpu_type(item: Mapping[str, Any]) -> str:
    spec = item.get("spec")
    if isinstance(spec, Mapping):
        gpu_type = spec.get("gpu_type")
        if gpu_type:
            return str(gpu_type)
    name = item.get("display_name") or item.get("name")
    return str(name) if name else ""


def _extract_gpu_count(item: Mapping[str, Any]) -> int:
    spec = item.get("spec")
    if isinstance(spec, Mapping):
        count = spec.get("gpu_count")
        if count is not None:
            return _coerce_int(count)
    return _coerce_int(item.get("gpu_count"))


def _coerce_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _parse_workload_instance(data: Any) -> Instance:
    if not isinstance(data, Mapping):
        raise TargonError("unexpected workload response shape")
    uid = data.get("uid") or data.get("id") or ""
    status = ""
    state = data.get("state")
    if isinstance(state, Mapping):
        status = str(state.get("status", ""))
    if not status:
        status = str(data.get("status", ""))
    return Instance(id=str(uid), status=status, provider="targon", raw=data)
