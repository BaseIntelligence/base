"""Targon compute provider client (architecture.md sec 3.1).

Thin async ``httpx`` wrapper over the Targon REST API (base
``https://api.targon.com/tha/v2``, auth header ``Authorization: Bearer``). It
implements the :class:`~base.compute.provider` contract with the cost guardrails
baked in and two Targon-specific realities surfaced as typed errors:

* Targon exposes NO balance/credits endpoint, so :meth:`TargonClient.balance`
  raises :class:`BalanceUnavailableError` WITHOUT issuing any HTTP request rather
  than returning a fake/silent value.
* A deploy that fails for insufficient credits is surfaced as a distinct typed
  :class:`InsufficientCreditsError` and is NEVER retried. A deploy is a two-step
  create-then-deploy flow (Targon has no single ``POST /workloads/deploy`` route):
  ``POST /workloads`` (create -> ``uid``) then ``POST /workloads/{uid}/deploy``;
  a credit failure at EITHER step raises and is not retried.

Live: ``POST /tha/v2/workloads`` JSON bodies **reject unknown fields** (e.g.
``termination_hours``, ``gpu_count``, ``ssh_public_keys``, ``image_digest`` cause
``INVALID_JSON_BODY`` / validation 400). Ports must be 1024–65535; ``ssh_keys``
are account key UIDs (``shk-…``) from ``GET /ssh-keys``, not raw public-key
strings. Client-side ``max_lifetime_hours`` remains a provision guardrail only —
auto-termination is caller ``DELETE`` (always-terminate).

The API key lives only in the request header; it is never logged, embedded in an
error message, or exposed via ``repr``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
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
# Live workload name: lowercase alphanumeric + hyphens, max 32 (docs 2026-07).
_WORKLOAD_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?$")
_MIN_PORT = 1024
_MAX_PORT = 65535
# InstanceSpec defaults include host SSH 22; Targon rejects ports below 1024.
# Map conventional SSH to Targon's common DIRECT 2222 when the caller supplied 22.
_SSH_HOST_PORT = 22
_SSH_TARGON_PORT = 2222
_ERROR_BODY_CAP = 400


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
        """List apps.

        Targon has retired ``GET /apps`` on some accounts/regions with HTTP 410
        Gone (workloads remain the inventory surface). Treat 410 as an empty
        list so shared Lium/Targon read-only preflights do not hard-fail forever
        when the retired apps route is gone; non-410 errors still raise.
        """
        response = await self._send("GET", "/apps")
        if response.status_code == 410:
            logger.info("Targon GET /apps returned 410 Gone; treating as empty list")
            return []
        if response.status_code >= 400:
            raise TargonError(
                f"Targon GET /apps returned {response.status_code}",
                status_code=response.status_code,
            )
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

    async def list_ssh_keys(self) -> list[dict[str, Any]]:
        """List account SSH keys (``GET /ssh-keys``).

        Attach keys to a workload via UIDs in ``ssh_keys`` (not raw public-key
        material). ``public_key_raw`` may appear in responses — callers must not
        log full key bodies.
        """
        response = await self._request("GET", "/ssh-keys")
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

    async def provision(
        self,
        spec: InstanceSpec,
        *,
        ssh_key_uids: Sequence[str] | None = None,
        commands: Sequence[str] | None = None,
    ) -> Instance:
        lifetime = spec.max_lifetime_hours
        if lifetime is None or lifetime <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_lifetime_hours must be a positive bound (hours)"
            )
        if lifetime < 1:
            raise CostGuardrailError(
                "InstanceSpec.max_lifetime_hours must be at least 1 hour so "
                "operators have a positive always-terminate window (Targon has "
                "no provider auto-termination field on create; callers must DELETE)"
            )
        if spec.max_price_per_hour is None or spec.max_price_per_hour <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_price_per_hour must be a positive bound"
            )
        payload = _build_workload_payload(
            spec, ssh_key_uids=ssh_key_uids, commands=commands
        )
        return await self.deploy(payload)

    async def deploy(self, workload: Mapping[str, Any]) -> Instance:
        """Deploy a workload via Targon's two-step create-then-deploy flow.

        Targon has NO single ``POST /workloads/deploy`` route (confirmed against
        the live API and the ``targon-sdk``): a workload is first CREATED
        (``POST /workloads`` -> ``uid``) then DEPLOYED
        (``POST /workloads/{uid}/deploy``). Both calls are part of ONE deploy
        attempt -- an insufficient-credit failure at EITHER step is surfaced as a
        typed :class:`InsufficientCreditsError` and is NEVER retried.
        """
        create = await self._send("POST", "/workloads", json_body=dict(workload))
        self._raise_deploy_error(create, "POST /workloads")
        uid = _extract_workload_uid(create.json())
        if not uid:
            raise TargonError("Targon POST /workloads returned no workload uid")
        deployed = await self._send("POST", f"/workloads/{uid}/deploy")
        self._raise_deploy_error(deployed, f"POST /workloads/{uid}/deploy")
        return _parse_workload_instance(deployed.json(), fallback_uid=uid)

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

    def _raise_deploy_error(self, response: httpx.Response, route: str) -> None:
        if response.status_code < 400:
            return
        detail = _safe_error_detail(response)
        if _is_insufficient_credits(response.status_code, response.text):
            raise InsufficientCreditsError(
                f"Targon {route} rejected for insufficient credits "
                f"(status {response.status_code}){detail}",
                status_code=response.status_code,
            )
        raise TargonError(
            f"Targon {route} returned {response.status_code}{detail}",
            status_code=response.status_code,
        )

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


def _build_workload_payload(
    spec: InstanceSpec,
    *,
    ssh_key_uids: Sequence[str] | None = None,
    commands: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a docs-compliant create body (unknown fields are rejected live).

    Emits only fields accepted by ``POST /tha/v2/workloads`` for ``RENTAL``:
    ``type``, ``name``, ``image``, ``resource_name``, optional ``envs``,
    ``ports``, ``commands``, ``args``, ``ssh_keys``, ``volumes``,
    ``registry_auth``, ``experiments``. Does **not** emit client-only
    guardrails (``max_lifetime_hours``), raw SSH material, ``gpu_count``, or
    ``image_digest``.
    """
    name = _normalize_workload_name(spec.name)
    if not spec.template_ref:
        raise CostGuardrailError(
            "InstanceSpec.template_ref is required as Targon resource_name "
            "(inventory shape id, e.g. rtx4090-small)"
        )
    if not spec.image:
        raise CostGuardrailError(
            "InstanceSpec.image is required for Targon RENTAL workloads"
        )
    if spec.image_digest:
        logger.info(
            "Ignoring InstanceSpec.image_digest for Targon create: image_digest "
            "is not a create-body field (pin publicly pullable image refs only)"
        )
    if spec.ssh_public_keys and not ssh_key_uids:
        logger.info(
            "Ignoring InstanceSpec.ssh_public_keys for Targon create: attach "
            "account key UIDs via ssh_key_uids / payload ssh_keys (GET /ssh-keys)"
        )

    payload: dict[str, Any] = {
        "name": name,
        "type": "RENTAL",
        "resource_name": str(spec.template_ref),
        "image": str(spec.image),
    }
    if spec.env:
        payload["envs"] = [
            {"name": str(k), "value": str(v)} for k, v in spec.env.items()
        ]
    ports = _normalize_ports(spec.ports)
    if ports:
        payload["ports"] = ports
    if ssh_key_uids:
        uids = [str(u).strip() for u in ssh_key_uids if str(u).strip()]
        if uids:
            payload["ssh_keys"] = uids
    if commands:
        cmd = [str(c) for c in commands if str(c)]
        if cmd:
            payload["commands"] = cmd
    return payload


def _normalize_workload_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9-]+", "-", (name or "").strip().lower())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if len(cleaned) > 32:
        cleaned = cleaned[:32].rstrip("-")
    if not cleaned or not _WORKLOAD_NAME_RE.match(cleaned):
        raise CostGuardrailError(
            "InstanceSpec.name must normalize to lowercase alphanumeric/hyphen "
            f"max 32 chars for Targon (got {name!r})"
        )
    return cleaned


def _normalize_ports(ports: Sequence[int] | None) -> list[dict[str, Any]]:
    if not ports:
        return []
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in ports:
        try:
            port = int(raw)
        except (TypeError, ValueError):
            continue
        if port == _SSH_HOST_PORT:
            port = _SSH_TARGON_PORT
        if port < _MIN_PORT or port > _MAX_PORT:
            logger.info(
                "Skipping Targon port %s outside %s-%s",
                raw,
                _MIN_PORT,
                _MAX_PORT,
            )
            continue
        if port in seen:
            continue
        seen.add(port)
        entry: dict[str, Any] = {
            "port": port,
            "protocol": "TCP",
            "routing": "DIRECT" if port == _SSH_TARGON_PORT else "PROXIED",
        }
        out.append(entry)
    return out


def _safe_error_detail(response: httpx.Response) -> str:
    text = (response.text or "").strip()
    if not text:
        return ""
    # Prefer structured reason without flooding logs.
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, Mapping):
        reason = data.get("reason") or data.get("error") or data.get("message")
        if reason is not None:
            snippet = str(reason).replace("\n", " ")[:_ERROR_BODY_CAP]
            return f": {snippet}"
    snippet = text.replace("\n", " ")[:_ERROR_BODY_CAP]
    return f": {snippet}"


def _is_insufficient_credits(status_code: int, body_text: str) -> bool:
    # Historical contract: 402/403 on deploy are treated as non-retryable credit
    # refusals (Targon has no balance endpoint; deploy is the only signal).
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
        cost_per_hour = float(cost)
    except (TypeError, ValueError):
        return None
    if _coerce_int(item.get("available")) <= 0:
        return None
    gpu_count = _extract_gpu_count(item)
    # Targon inventory quotes cost_per_hour for the WHOLE shape; Offer.price_per_hour
    # is per-GPU (what the per-GPU max_price cap filters on). Fall back to the
    # whole-shape cost when the count is unknown (avoids ZeroDivisionError).
    price_per_gpu = cost_per_hour / gpu_count if gpu_count >= 1 else cost_per_hour
    return Offer(
        id=str(name),
        gpu_type=_extract_gpu_type(item),
        gpu_count=gpu_count,
        price_per_hour=price_per_gpu,
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
        count = _coerce_int(spec.get("gpu_count"))
        if count >= 1:
            return count
    return _coerce_int(item.get("gpu_count"))


def _coerce_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _parse_workload_instance(data: Any, *, fallback_uid: str | None = None) -> Instance:
    if not isinstance(data, Mapping):
        raise TargonError("unexpected workload response shape")
    uid = data.get("uid") or data.get("id") or fallback_uid or ""
    status = ""
    state = data.get("state")
    if isinstance(state, Mapping):
        status = str(state.get("status", ""))
    if not status:
        status = str(data.get("status", ""))
    return Instance(id=str(uid), status=status, provider="targon", raw=data)


def _extract_workload_uid(data: Any) -> str:
    if isinstance(data, Mapping):
        uid = data.get("uid") or data.get("id")
        if uid:
            return str(uid)
    return ""
