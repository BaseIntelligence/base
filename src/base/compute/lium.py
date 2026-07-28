"""Lium compute provider client (architecture.md sec 3.1).

Thin async ``httpx`` wrapper over the Lium REST API (base ``https://lium.io/api``,
auth header ``X-API-Key``). It implements the :class:`~base.compute.provider`
contract with the cost guardrails baked in:

* :meth:`LiumClient.provision` refuses an unbounded or over-priced spec BEFORE any
  network call and always sends a bounded ``termination_hours``.
* Any failure after the pod is rented terminates + verifies the pod (try/finally),
  so a failed provisioning never leaks a billable pod.
* :meth:`LiumClient.terminate` is idempotent (a ``404`` delete is success) and
  :meth:`LiumClient.verify_terminated` reflects real pod absence via ``GET /pods``.

Prism training lock (``training_gpu_lock=True`` /
:meth:`LiumClient.for_prism_training`):
fail-closed to natively 1-GPU :data:`LIUM_TRAINING_GPU_TYPE` offers only. Live API
evidence (2026-07): POST .../rent with ``gpu_count=1`` on an 8-GPU machine returns
HTTP 400 "Provider doesn't allow GPU splitting." — multi-GPU hosts are refused
before rent rather than rented in full.

The API key lives only in the request header; it is never logged, embedded in an
error message, or exposed via ``repr``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from base.compute.provider import (
    CostGuardrailError,
    Instance,
    InstanceSpec,
    Offer,
    ProviderError,
)
from base.compute.worker_deployment import is_loopback_url

logger = logging.getLogger(__name__)

LIUM_API_BASE_URL = "https://lium.io/api"
_DEFAULT_SSH_KEY_NAME = "prism-mission-worker"

# Canonical display name for the only GPU type Prism training may rent.
LIUM_TRAINING_GPU_TYPE: Final[str] = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition"
)
_TRAINING_GPU_REQUIRED_TOKENS: Final[frozenset[str]] = frozenset(
    {"rtx", "pro", "6000", "blackwell"}
)
_LEADING_GPU_NOISE_TOKENS: Final[frozenset[str]] = frozenset({"nvidia", "gpu"})
_NON_ALNUM_RUN: Final[re.Pattern[str]] = re.compile(r"[-_]+")
_WHITESPACE_RUN: Final[re.Pattern[str]] = re.compile(r"\s+")


def normalize_gpu_type(value: str | None) -> str:
    """Normalize a provider GPU type string for token matching.

    Lowercase; replace runs of ``-``/``_`` with a space; collapse whitespace;
    strip; drop leading tokens in {nvidia, gpu}.
    """
    if value is None:
        return ""
    text = str(value).lower()
    text = _NON_ALNUM_RUN.sub(" ", text)
    text = _WHITESPACE_RUN.sub(" ", text).strip()
    if not text:
        return ""
    tokens = text.split(" ")
    while tokens and tokens[0] in _LEADING_GPU_NOISE_TOKENS:
        tokens = tokens[1:]
    return " ".join(tokens)


def is_allowed_lium_training_gpu(offer: Offer) -> bool:
    """Return True iff ``offer`` is a natively 1-GPU PRO 6000 Blackwell Server.

    Fail-closed: empty/unparseable gpu_type, bare "Blackwell", H100, RTX 5090,
    and any ``gpu_count != 1`` (including 8-GPU PRO 6000 hosts) are rejected.
    """
    if offer.gpu_count != 1:
        return False
    raw = offer.gpu_type
    if raw is None or not str(raw).strip():
        return False
    normalized = normalize_gpu_type(str(raw))
    if not normalized:
        return False
    tokens = set(normalized.split(" "))
    if not _TRAINING_GPU_REQUIRED_TOKENS.issubset(tokens):
        return False
    return "server" in tokens


class LiumError(ProviderError):
    """A Lium API request failed (non-2xx response or transport error)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LiumAuthError(LiumError):
    """Lium rejected the API key (HTTP 401). Never treat as a missing resource."""


class LiumNotFoundError(LiumError):
    """Requested Lium pod or template does not exist (HTTP 404)."""


class LiumRateLimitError(LiumError):
    """Lium rate-limited the client (HTTP 429). Fail closed — do not skip."""


class LiumTemplateDigestMismatchError(LiumError):
    """An existing template name collides with a different docker_image_digest.

    Silent reuse of a name-matched template under a new pin is forbidden: the
    caller must create a distinct template or resolve the collision explicitly.
    """

    def __init__(
        self,
        message: str,
        *,
        template_id: str,
        existing_digest: str | None,
        requested_digest: str | None,
    ) -> None:
        super().__init__(message, status_code=None)
        self.template_id = template_id
        self.existing_digest = existing_digest
        self.requested_digest = requested_digest


@dataclass(frozen=True, slots=True)
class LiumPodRead:
    """Declared pod configuration from ``GET /pods/{id}`` (OpenAPI PodDetailResponse).

    ``docker_image_digest`` and ``template_id`` come from the nested template
    object Lium records at rent time. This is **declared configuration**, not a
    runtime measurement of the running container.
    """

    pod_id: str
    template_id: str | None
    docker_image_digest: str | None
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LiumTemplateRead:
    """Template record from ``GET /templates/{id}`` (OpenAPI Template)."""

    template_id: str
    docker_image_digest: str | None
    name: str | None
    raw: Mapping[str, Any]


class LiumClient:
    """Async client for the Lium GPU rental API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = LIUM_API_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
        training_gpu_lock: bool = False,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport
        self._timeout = timeout_seconds
        self._training_gpu_lock = training_gpu_lock

    @classmethod
    def for_prism_training(
        cls,
        api_key: str,
        *,
        base_url: str = LIUM_API_BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> LiumClient:
        """Build a client locked to 1× RTX PRO 6000 Blackwell Server Edition."""
        return cls(
            api_key,
            base_url=base_url,
            transport=transport,
            timeout_seconds=timeout_seconds,
            training_gpu_lock=True,
        )

    def __repr__(self) -> str:
        return f"LiumClient(base_url={self._base_url!r})"

    # -- offers ---------------------------------------------------------------

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]:
        response = await self._request("GET", "/executors")
        offers: list[Offer] = []
        for item in _as_list(response.json(), "executors"):
            offer = _parse_offer(item)
            if offer is None:
                continue
            if (
                max_price_per_hour is not None
                and offer.price_per_hour > max_price_per_hour
            ):
                continue
            if self._training_gpu_lock and not is_allowed_lium_training_gpu(offer):
                continue
            offers.append(offer)
        return offers

    # -- lifecycle ------------------------------------------------------------

    async def provision(
        self, spec: InstanceSpec, *, offer: Offer | None = None
    ) -> Instance:
        lifetime = spec.max_lifetime_hours
        if lifetime is None or lifetime <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_lifetime_hours must be a positive bound (hours)"
            )
        if lifetime < 1:
            raise CostGuardrailError(
                "InstanceSpec.max_lifetime_hours must be at least 1 hour: Lium "
                "termination_hours has 1-hour granularity, so a sub-hour bound "
                "would truncate to termination_hours=0 and disable auto-termination"
            )
        if spec.max_price_per_hour is None or spec.max_price_per_hour <= 0:
            raise CostGuardrailError(
                "InstanceSpec.max_price_per_hour must be a positive bound"
            )
        if not spec.ssh_public_keys:
            raise LiumError("Lium rent requires at least one SSH public key")
        if self._training_gpu_lock and spec.gpu_count != 1:
            raise CostGuardrailError(
                "Prism training lock requires InstanceSpec.gpu_count == 1 "
                f"(got {spec.gpu_count}); maximum 1 GPU per instance"
            )

        selected = await self._resolve_offer(spec, offer)

        for public_key in spec.ssh_public_keys:
            await self.ensure_ssh_key(
                public_key=public_key,
                name=spec.ssh_key_name or _DEFAULT_SSH_KEY_NAME,
            )
        template_id = await self._resolve_template(spec)

        # Unlocked path: InstanceSpec.gpu_count is a minimum-need filter; Lium
        # returns HTTP 400 "Provider doesn't allow GPU splitting" on partial
        # rents, so rent the full selected offer capacity.
        # Locked Prism training path: only natively 1-GPU allowed offers reach
        # here, so the rent body always requests gpu_count=1.
        rent_body: dict[str, Any] = {
            "pod_name": spec.name,
            "user_public_key": list(spec.ssh_public_keys),
            "termination_hours": int(lifetime),
        }
        if self._training_gpu_lock:
            rent_body["gpu_count"] = 1
        elif selected.gpu_count and selected.gpu_count > 0:
            rent_body["gpu_count"] = selected.gpu_count
        if template_id is not None:
            rent_body["template_id"] = template_id
        if spec.dockerfile_content is not None:
            rent_body["dockerfile_content"] = spec.dockerfile_content

        rent = await self._request(
            "POST", f"/executors/{selected.id}/rent", json_body=rent_body
        )
        # The rent succeeded: a billable pod may now exist. Every subsequent
        # failure -- an unparseable rent body, pod-id resolution, AND the status
        # poll -- must best-effort terminate + verify before re-raising, so
        # cleanup keys strictly off "rent HTTP call succeeded" rather than "pod
        # id resolved". The pod-id extraction stays INSIDE the try so a 2xx rent
        # with a non-JSON body cannot leak a just-rented pod.
        pod_id: str | None = None
        try:
            pod_id = self._extract_pod_id(rent)
            if pod_id is None:
                pod_id = await self._find_pod_id(spec.name)
            if pod_id is None:
                raise LiumError(
                    "could not determine provisioned pod id from rent response"
                )
            return await self.status(pod_id)
        except BaseException:
            await self._cleanup_after_rent(pod_id, spec.name)
            raise

    async def status(self, instance_id: str) -> Instance:
        response = await self._request("GET", f"/pods/{instance_id}")
        return _parse_instance(response.json())

    async def stream_logs(self, instance_id: str) -> AsyncIterator[str]:
        async with self._client() as client:
            async with client.stream("GET", f"/pods/{instance_id}/logs") as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise _lium_http_error(
                        "GET", f"/pods/{instance_id}/logs", response.status_code
                    )
                async for line in response.aiter_lines():
                    yield line

    async def terminate(self, instance_id: str) -> None:
        response = await self._send("DELETE", f"/pods/{instance_id}")
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise _lium_http_error(
                "DELETE", f"/pods/{instance_id}", response.status_code
            )

    async def verify_terminated(self, instance_id: str) -> bool:
        for pod in await self.list_pods():
            if str(pod.get("id")) == str(instance_id):
                return False
        return True

    async def list_pods(self) -> list[dict[str, Any]]:
        response = await self._request("GET", "/pods")
        return _as_list(response.json(), "pods")

    async def get_pod_raw(self, pod_id: str) -> LiumPodRead:
        """Read one pod via OpenAPI ``GET /pods/{id}``.

        Returns the declared template id and ``docker_image_digest`` from the
        nested template object. Raises typed :class:`LiumAuthError` /
        :class:`LiumNotFoundError` / :class:`LiumRateLimitError` on 401/404/429.
        """
        response = await self._request("GET", f"/pods/{pod_id}")
        data = response.json()
        if not isinstance(data, Mapping):
            raise LiumError("unexpected pod response shape")
        template = data.get("template")
        template_id: str | None = None
        digest: str | None = None
        if isinstance(template, Mapping):
            if template.get("id"):
                template_id = str(template["id"])
            digest = _optional_str(template.get("docker_image_digest"))
        if template_id is None and data.get("template_id"):
            template_id = str(data["template_id"])
        if digest is None:
            digest = _optional_str(data.get("docker_image_digest"))
        return LiumPodRead(
            pod_id=str(data.get("id") or pod_id),
            template_id=template_id,
            docker_image_digest=digest,
            raw=data,
        )

    async def get_template_raw(self, template_id: str) -> LiumTemplateRead:
        """Read one template via OpenAPI ``GET /templates/{id}``."""
        response = await self._request("GET", f"/templates/{template_id}")
        data = response.json()
        if not isinstance(data, Mapping):
            raise LiumError("unexpected template response shape")
        tid = data.get("id")
        if not tid:
            raise LiumError("template response missing 'id'")
        return LiumTemplateRead(
            template_id=str(tid),
            docker_image_digest=_optional_str(data.get("docker_image_digest")),
            name=_optional_str(data.get("name")),
            raw=data,
        )

    # -- idempotent deploy helpers -------------------------------------------

    async def ensure_ssh_key(
        self, *, public_key: str, name: str | None = None
    ) -> dict[str, Any]:
        response = await self._request("GET", "/ssh-keys")
        normalized = public_key.strip()
        for key in _as_list(response.json(), "ssh_keys"):
            if str(key.get("public_key", "")).strip() == normalized:
                return key
        body: dict[str, Any] = {"public_key": public_key}
        if name is not None:
            body["name"] = name
        created = await self._request("POST", "/ssh-keys", json_body=body)
        result = created.json()
        return result if isinstance(result, dict) else {}

    async def ensure_template(
        self,
        *,
        name: str,
        docker_image: str,
        docker_image_digest: str | None = None,
        docker_image_tag: str | None = None,
        internal_ports: Sequence[int] = (22,),
        environment: Mapping[str, str] | None = None,
        startup_commands: str | None = None,
        is_private: bool = True,
        container_start_immediately: bool = True,
    ) -> str:
        response = await self._request("GET", "/templates")
        requested_digest = _normalize_digest(docker_image_digest)
        for template in _as_list(response.json(), "templates"):
            if str(template.get("name")) != name or not template.get("id"):
                continue
            existing_id = str(template["id"])
            existing_digest = _normalize_digest(template.get("docker_image_digest"))
            if _digests_allow_reuse(existing_digest, requested_digest):
                return existing_id
            raise LiumTemplateDigestMismatchError(
                f"template {name!r} exists as {existing_id} with "
                f"docker_image_digest={existing_digest!r}, refusing reuse for "
                f"requested digest {requested_digest!r}",
                template_id=existing_id,
                existing_digest=existing_digest,
                requested_digest=requested_digest,
            )
        body: dict[str, Any] = {
            "name": name,
            "docker_image": docker_image,
            "internal_ports": list(internal_ports),
            "is_private": is_private,
            "container_start_immediately": container_start_immediately,
        }
        if docker_image_digest:
            body["docker_image_digest"] = docker_image_digest
        if docker_image_tag:
            body["docker_image_tag"] = docker_image_tag
        if environment:
            # Lium's edge WAF returns 403 "Request blocked" for ANY body carrying a
            # loopback URL (http://127.0.0.1... / http://localhost...), so strip
            # loopback-valued entries before POSTing the template. Such values are
            # redundant anyway: the pod config defaults them to loopback and the
            # agent resolves them at runtime (a reverse SSH tunnel reaches a local
            # master). Non-loopback (real) values are preserved.
            safe_environment = {
                key: value
                for key, value in environment.items()
                if not is_loopback_url(value)
            }
            if safe_environment:
                body["environment"] = safe_environment
        # Lium rejects rents whose template startup_commands contain shell
        # metacharacters ("Malicious startup command detected"), so a caller
        # supplies a single metachar-free keep-alive here (e.g. "tail -f
        # /dev/null"); omitted when None to preserve the image's own entrypoint.
        if startup_commands is not None:
            body["startup_commands"] = startup_commands
        created = await self._request("POST", "/templates", json_body=body)
        return str(created.json().get("id"))

    # -- account / proof inputs ----------------------------------------------

    async def watchtower_digest(self) -> str:
        response = await self._request("GET", "/watchtower/digest")
        data = response.json()
        if isinstance(data, Mapping):
            digest = data.get("digest")
            if digest:
                return str(digest)
        if isinstance(data, str) and data:
            return data
        raise LiumError("watchtower digest response missing 'digest'")

    async def balance(self) -> float:
        response = await self._request("GET", "/users/me")
        data = response.json()
        balance = data.get("balance") if isinstance(data, Mapping) else None
        if balance is None:
            raise LiumError("users/me response missing 'balance'")
        return float(balance)

    # -- internals ------------------------------------------------------------

    async def _resolve_offer(self, spec: InstanceSpec, offer: Offer | None) -> Offer:
        if offer is not None:
            if (
                spec.max_price_per_hour is not None
                and offer.price_per_hour > spec.max_price_per_hour
            ):
                raise CostGuardrailError(
                    f"offer {offer.id} at {offer.price_per_hour}/hr exceeds "
                    f"max_price_per_hour {spec.max_price_per_hour}"
                )
            self._assert_training_gpu_offer(offer)
            return offer
        offers = await self.list_offers(max_price_per_hour=spec.max_price_per_hour)
        if not offers:
            if self._training_gpu_lock:
                raise CostGuardrailError(
                    "no Lium offer available within max_price_per_hour bound for "
                    f"{LIUM_TRAINING_GPU_TYPE} with gpu_count=1; wait for new inventory"
                )
            raise CostGuardrailError(
                "no Lium offer available within max_price_per_hour bound"
            )
        selected = min(offers, key=lambda candidate: candidate.price_per_hour)
        self._assert_training_gpu_offer(selected)
        return selected

    def _assert_training_gpu_offer(self, offer: Offer) -> None:
        if not self._training_gpu_lock:
            return
        if offer.gpu_count != 1:
            raise CostGuardrailError(
                f"offer {offer.id} has gpu_count={offer.gpu_count}; Prism training "
                "lock requires exactly 1 GPU per instance"
            )
        if not is_allowed_lium_training_gpu(offer):
            raise CostGuardrailError(
                f"offer {offer.id} gpu_type={offer.gpu_type!r} is not allowed; "
                f"Prism training lock requires {LIUM_TRAINING_GPU_TYPE} "
                "with gpu_count=1"
            )

    async def _resolve_template(self, spec: InstanceSpec) -> str | None:
        if spec.template_ref is not None:
            return await self.ensure_template(
                name=spec.template_ref,
                docker_image=spec.image or "",
                docker_image_digest=spec.image_digest,
                internal_ports=spec.ports or (22,),
                environment=spec.env,
                startup_commands=spec.startup_commands,
            )
        if spec.dockerfile_content is None:
            raise LiumError("InstanceSpec requires template_ref or dockerfile_content")
        return None

    async def _find_pod_id(self, pod_name: str) -> str | None:
        for pod in await self.list_pods():
            if str(pod.get("pod_name")) == pod_name and pod.get("id"):
                return str(pod["id"])
        return None

    async def _cleanup_after_rent(self, pod_id: str | None, pod_name: str) -> None:
        if pod_id is None:
            pod_id = await self._resolve_pod_id_quietly(pod_name)
        if pod_id is None:
            logger.warning(
                "post-rent cleanup could not resolve a pod id for %r; the rented "
                "pod (if any) will auto-terminate via termination_hours",
                pod_name,
            )
            return
        await self._terminate_and_verify_quietly(pod_id)

    async def _resolve_pod_id_quietly(self, pod_name: str) -> str | None:
        try:
            return await self._find_pod_id(pod_name)
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            logger.warning("post-rent cleanup pod lookup failed for %r", pod_name)
            return None

    async def _terminate_and_verify_quietly(self, instance_id: str) -> None:
        try:
            await self.terminate(instance_id)
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            logger.warning("cleanup terminate failed for pod %s", instance_id)
        try:
            await self.verify_terminated(instance_id)
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            logger.warning("cleanup verify_terminated failed for pod %s", instance_id)

    @staticmethod
    def _extract_pod_id(response: httpx.Response) -> str | None:
        if not response.content:
            return None
        try:
            data = response.json()
        except ValueError as exc:
            raise LiumError(
                "Lium rent returned a 2xx response with an unparseable body"
            ) from exc
        if not isinstance(data, Mapping):
            return None
        for key in ("id", "pod_id", "uuid"):
            value = data.get(key)
            if value:
                return str(value)
        pod = data.get("pod")
        if isinstance(pod, Mapping) and pod.get("id"):
            return str(pod["id"])
        return None

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key, "Accept": "application/json"}

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
            raise LiumError(f"Lium request {method} {path} failed") from exc

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
            raise _lium_http_error(method, path, response.status_code)
        return response


def _lium_http_error(method: str, path: str, status_code: int) -> LiumError:
    message = f"Lium {method} {path} returned {status_code}"
    if status_code == 401:
        return LiumAuthError(message, status_code=401)
    if status_code == 404:
        return LiumNotFoundError(message, status_code=404)
    if status_code == 429:
        return LiumRateLimitError(message, status_code=429)
    return LiumError(message, status_code=status_code)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_digest(value: Any) -> str | None:
    return _optional_str(value)


def _digests_allow_reuse(existing: str | None, requested: str | None) -> bool:
    """Reuse only when digests agree, or neither side pins a digest.

    A requested pin against a missing or different stored digest is a hard
    mismatch — never silent name-only reuse under a new pin.
    """
    if requested is None and existing is None:
        return True
    if requested is None:
        # Caller did not pin; reusing a digest-bearing template is still name-only
        # but does not introduce a false pin claim. Allowed for legacy callers.
        return True
    return existing is not None and existing == requested


def _as_list(data: Any, wrapper_key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, Mapping):
        inner = data.get(wrapper_key)
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []


def _parse_offer(item: Mapping[str, Any]) -> Offer | None:
    offer_id = item.get("id")
    price = _extract_price(item)
    if not offer_id or price is None:
        return None
    return Offer(
        id=str(offer_id),
        gpu_type=_extract_gpu_type(item),
        gpu_count=_extract_gpu_count(item),
        price_per_hour=price,
        provider="lium",
        raw=item,
    )


def _extract_price(item: Mapping[str, Any]) -> float | None:
    for key in ("price_per_hour", "price_per_gpu", "pending_price_per_hour"):
        value = item.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _extract_gpu_type(item: Mapping[str, Any]) -> str:
    name = item.get("machine_name") or item.get("gpu_type") or item.get("gpu_name")
    if not name:
        specs = item.get("specs")
        if isinstance(specs, Mapping):
            gpu = specs.get("gpu")
            if isinstance(gpu, Mapping):
                details = gpu.get("details")
                if isinstance(details, list) and details:
                    first = details[0]
                    if isinstance(first, Mapping):
                        name = first.get("name")
    return str(name) if name else ""


def _extract_gpu_count(item: Mapping[str, Any]) -> int:
    count = item.get("gpu_count")
    if count is None:
        specs = item.get("specs")
        if isinstance(specs, Mapping):
            gpu = specs.get("gpu")
            if isinstance(gpu, Mapping):
                count = gpu.get("count")
    try:
        return int(count) if count is not None else 0
    except (TypeError, ValueError):
        return 0


def _parse_instance(data: Any) -> Instance:
    if not isinstance(data, Mapping):
        raise LiumError("unexpected pod response shape")
    return Instance(
        id=str(data.get("id", "")),
        status=str(data.get("status", "")),
        provider="lium",
        raw=data,
    )
