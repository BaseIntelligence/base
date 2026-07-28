"""Small, secret-safe Phala Cloud REST adapter for self-deploy stages.

Auth matches the official Phala CLI (``phala`` node package):

* Header ``X-API-Key: <key>`` — **not** ``Authorization: Bearer``
  (Bearer returns HTTP 401 Invalid/expired token for Cloud API keys).
* Header ``X-Phala-Version: 2026-06-23`` — API version pin used by CLI.
* Header ``User-Agent: phala-cloud-cli/<version>`` — Cloudflare 1010 blocks bare
  Python-urllib agents; product sends the CLI-equivalent string.
* CVM listing uses ``GET /cvms/paginated`` (CLI authority). Unknown response
  shapes raise; they never degrade to an empty list / count 0.

Region selection must not hard-fail with ERR-02-002 (``No teepod found``)
on bare alias ``us-west`` when inventory capacity is only under ``us-west-1``.

Create-response CVM id helpers accept Phala's live schema (numeric ``id``,
alternate ``cvm_id`` / ``vm_uuid`` / ``instance_id``) and a safe list fallback
by ``app_id`` so product does not fail closed solely for
``product_create_response_missing_cvm_id_field`` when the CVM exists.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_challenge.selfdeploy.cvm_list import (
    CLI_PHALA_API_VERSION,
    CLI_PHALA_USER_AGENT,
    CvmListParseError,
    CvmListSnapshot,
    parse_cvms_list_response,
    resolve_cvm_id_from_snapshot,
)
from agent_challenge.selfdeploy.plan import PHALA_API_KEY_ENV, CredentialError

DEFAULT_PHALA_API = "https://cloud-api.phala.com/api/v1"

#: API version header accepted by cloud-api.phala.com (matches `phala` CLI `Lo`).
DEFAULT_PHALA_API_VERSION = CLI_PHALA_API_VERSION  # 2026-06-23

#: CLI-equivalent User-Agent (phala-cloud-cli; CF 1010 without it).
DEFAULT_PHALA_USER_AGENT = CLI_PHALA_USER_AGENT  # phala-cloud-cli/1.1.19

#: Preferred default region when caller omits one or alias maps to empty capacity.
#: Live inventory teepods (prod5/prod9) live under US-WEST-1; bare "us-west"
#: hits ERR-02-002. Empty string means "let the API/auto assign".
PREFERRED_PHALA_REGION = "us-west-1"

#: Region aliases that previously hard-failed against live capacity.
_US_WEST_ALIASES = frozenset({"us-west", "us_west", "uswest"})

#: Allowed GET paths for safe read helpers (list/details — never secrets).
_ALLOWED_GET_PATHS = frozenset({"/cvms", "/cvms/paginated"})

#: Allowed DELETE path shape: /cvms/{id} only (no nested paths).
_ALLOWED_DELETE_PATH_RE = re.compile(r"^/cvms/[A-Za-z0-9][A-Za-z0-9._-]*$")
_CVM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Create-response keys that may identify a CVM (ordered preference).
#: ``app_id`` is intentionally excluded: it names the app pin, not the CVM.
_CREATE_CVM_ID_FIELDS = ("id", "cvm_id", "vm_uuid", "instance_id", "uuid")


class PhalaApiError(RuntimeError):
    """A bounded Phala API failure without response-body disclosure."""


def _normalize_cvm_id_value(value: Any) -> str | None:
    """Coerce a create/list id field to a non-empty string identity.

    Live Phala create schema uses numeric ``id`` (CLI zod: ``id: number``).
    Product ack builders require string CVM ids; never invent identities.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        # Phala numbered CVM ids are positive; reject non-positive noise.
        if value <= 0:
            return None
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def extract_cvm_id_from_create_response(created: Mapping[str, Any]) -> str:
    """Extract a CVM id from a create/list item mapping.

    Accepts alternate field names used by Phala create (numeric ``id``,
    string ``cvm_id`` / ``vm_uuid`` / ``instance_id``). Never treats plain
    ``app_id`` as a CVM id (app pin is distinct). Raises ValueError with a
    secret-free message when no candidate field yields a usable identity.
    """

    if not isinstance(created, Mapping):
        raise ValueError("Phala create response does not identify the CVM")
    for name in _CREATE_CVM_ID_FIELDS:
        if name not in created:
            continue
        normalized = _normalize_cvm_id_value(created.get(name))
        if normalized is not None:
            return normalized
    raise ValueError("Phala create response does not identify the CVM")


def resolve_cvm_id_from_list(
    listing: Mapping[str, Any] | Sequence[Any],
    *,
    app_id: str,
    require_unique: bool = False,
) -> str | None:
    """Locate a CVM id in a GET /cvms listing by exact app_id match.

    Parses via :func:`parse_cvms_list_response` so unknown envelopes raise
    (:class:`CvmListParseError`) instead of silently matching nothing.
    Returns None when the listing is a known-empty/mismatched set.
    When ``require_unique`` is True (teardown), multiple matches raise.
    Secret bodies are never logged.
    """

    snapshot = parse_cvms_list_response(listing)
    try:
        return resolve_cvm_id_from_snapshot(
            snapshot, app_id=app_id, require_unique=require_unique
        )
    except CvmListParseError as exc:
        msg = str(exc)
        if "multiple CVMs match app_id" in msg:
            raise PhalaApiError(msg) from None
        raise


def normalize_phala_region(region: str | None) -> str:
    """Normalize a caller region to a capacity-safe Phala region string.

    * bare ``us-west`` (any case / underscore form) → ``us-west-1``
    * ``us-west-1`` / mixed case → lowercase ``us-west-1``
    * empty / None → empty (auto) so callers can omit the key
    * other regions → stripped lowercase
    """

    if region is None:
        return ""
    raw = str(region).strip()
    if not raw:
        return ""
    lowered = raw.lower().replace("_", "-")
    if lowered in _US_WEST_ALIASES:
        return PREFERRED_PHALA_REGION
    if lowered == "us-west-1":
        return PREFERRED_PHALA_REGION
    return lowered


def select_phala_region(
    preferred: str | None = None,
    *,
    available_regions: Sequence[str] | None = None,
) -> str:
    """Pick a capacity-aware region for provision requests.

    Rules (fail soft — never reintroduce bare ``us-west`` hard-fail):

    1. Normalize the preferred alias (``us-west`` → ``us-west-1``).
    2. If inventory is supplied and preferred is present (case-insensitive), use it.
    3. If preferred alias had no capacity / was empty, use the first available.
    4. If inventory is empty, use preferred if set else :data:`PREFERRED_PHALA_REGION`.
    """

    # Detect whether the raw preferred was only a bare us-west alias (capacity miss).
    raw = "" if preferred is None else str(preferred).strip()
    raw_alias = raw.lower().replace("_", "-") in _US_WEST_ALIASES if raw else False
    normalized = normalize_phala_region(preferred)
    inventory: list[str] = []
    if available_regions:
        for item in available_regions:
            n = normalize_phala_region(item)
            if n and n not in inventory:
                inventory.append(n)

    if inventory:
        if normalized and normalized in inventory:
            return normalized
        # Explicit non-alias preferred always wins (caller performant override).
        if normalized and not raw_alias and preferred is not None and str(preferred).strip():
            return normalized
        # Alias remapped but inventory only lists real capacity elsewhere: pick available.
        return inventory[0]

    if normalized:
        return normalized
    return PREFERRED_PHALA_REGION


class PhalaCloudClient:
    """HTTPS adapter for Phala provision/create and safe CVM list routes."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_PHALA_API,
        api_version: str = DEFAULT_PHALA_API_VERSION,
        user_agent: str = DEFAULT_PHALA_USER_AGENT,
        opener=urlopen,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = (
            api_key if api_key is not None else os.environ.get(PHALA_API_KEY_ENV, "")
        ).strip()
        if not self._api_key:
            raise CredentialError(
                f"{PHALA_API_KEY_ENV} is not set; set it before provisioning. "
                "The key value is never printed."
            )
        self._base_url = base_url.strip().rstrip("/")
        if not self._base_url.startswith("https://"):
            raise PhalaApiError("Phala API endpoint must use https://")
        self._api_version = (api_version or DEFAULT_PHALA_API_VERSION).strip()
        agent = (user_agent or DEFAULT_PHALA_USER_AGENT).strip()
        self._user_agent = agent or DEFAULT_PHALA_USER_AGENT
        self._opener = opener
        self._timeout = timeout

    def _base_headers(self, *, content_type: bool = False) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            # CLI-compatible identity + auth. Bare Python-urllib is CF-blocked.
            "User-Agent": self._user_agent,
            "X-API-Key": self._api_key,
            "X-Phala-Version": self._api_version,
        }
        if content_type:
            headers["Content-Type"] = "application/json"
        return headers

    def _decode_json_object(self, body: bytes) -> dict[str, Any]:
        decoded = self._decode_json_any(body)
        if not isinstance(decoded, dict):
            raise PhalaApiError("Phala provisioning returned a non-object response")
        return decoded

    def _decode_json_any(self, body: bytes) -> Any:
        if len(body) > 2 * 1024 * 1024:
            raise PhalaApiError("Phala provisioning response exceeded the bounded size")
        try:
            return json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PhalaApiError("Phala provisioning returned malformed JSON") from exc

    def _open(self, request: Request) -> dict[str, Any]:
        try:
            response = self._opener(request, timeout=self._timeout)
            body = response.read()
        except HTTPError as exc:
            raise PhalaApiError(f"Phala provisioning returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PhalaApiError("Phala provisioning endpoint is unreachable") from exc
        return self._decode_json_object(body)

    def _open_any(self, request: Request) -> Any:
        try:
            response = self._opener(request, timeout=self._timeout)
            body = response.read()
        except HTTPError as exc:
            raise PhalaApiError(f"Phala provisioning returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PhalaApiError("Phala provisioning endpoint is unreachable") from exc
        return self._decode_json_any(body)

    def get(self, path: str) -> dict[str, Any] | list[Any]:
        """GET an allowlisted read route (``/cvms`` or ``/cvms/paginated``)."""

        base_path = path.split("?", 1)[0]
        if base_path not in _ALLOWED_GET_PATHS:
            raise PhalaApiError("unsupported Phala read route")
        request = Request(
            f"{self._base_url}{path}",
            headers=self._base_headers(content_type=False),
            method="GET",
        )
        return self._open_any(request)

    def list_cvms(self, *, page_size: int = 50) -> CvmListSnapshot:
        """List account CVMs via CLI-authoritative ``GET /cvms/paginated``.

        Paginates until all pages are collected. Unknown response shapes raise
        :class:`CvmListParseError` (never under-report as total=0).
        """

        if page_size < 1 or page_size > 200:
            raise PhalaApiError("invalid CVM list page_size")
        page = 1
        all_items: list[Mapping[str, Any]] = []
        reported_total: int | None = None
        while True:
            path = f"/cvms/paginated?page={page}&page_size={page_size}"
            raw = self.get(path)
            snap = parse_cvms_list_response(raw)
            if reported_total is None:
                reported_total = snap.total
            elif snap.total != reported_total:
                raise CvmListParseError(
                    "unrecognized CVM list shape: total changed across pages "
                    f"({reported_total} -> {snap.total})"
                )
            all_items.extend(snap.items)
            if reported_total <= len(all_items):
                break
            if not snap.items:
                raise CvmListParseError(
                    "unrecognized CVM list shape: empty page before total reached "
                    f"(have={len(all_items)} total={reported_total})"
                )
            page += 1
            if page > 100:
                raise CvmListParseError(
                    "unrecognized CVM list shape: pagination exceeded page cap"
                )
        # Re-parse combined bare list so ids/total stay consistent.
        combined = parse_cvms_list_response(list(all_items))
        if reported_total is not None and combined.total != reported_total:
            # Prefer API total when we collected every page item count match.
            if len(all_items) != reported_total:
                raise CvmListParseError(
                    "unrecognized CVM list shape: collected items "
                    f"{len(all_items)} != total {reported_total}"
                )
            return CvmListSnapshot(
                items=combined.items,
                total=reported_total,
                ids=combined.ids,
                source_shape="paginated-merged",
            )
        return CvmListSnapshot(
            items=combined.items,
            total=reported_total if reported_total is not None else combined.total,
            ids=combined.ids,
            source_shape="paginated-merged",
        )

    def post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if path not in {"/cvms/provision", "/cvms"}:
            raise PhalaApiError("unsupported Phala mutation route")
        body_payload = dict(payload)
        # Capacity-safe region: remapped before send without logging secrets.
        if "region" in body_payload:
            region_value = body_payload.get("region")
            if isinstance(region_value, str) or region_value is None:
                region_arg: str | None = region_value
            else:
                region_arg = str(region_value)
            normalized = normalize_phala_region(region_arg)
            if normalized:
                body_payload["region"] = normalized
            else:
                # Empty → omit so API can auto-select from available teepods.
                body_payload.pop("region", None)
        raw = json.dumps(body_payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self._base_url}{path}",
            data=raw,
            headers=self._base_headers(content_type=True),
            method="POST",
        )
        return self._open(request)

    def delete_cvm(self, cvm_id: str) -> None:
        """DELETE ``/cvms/{id}``. 204 and 404 are success (idempotent teardown)."""

        cid = (cvm_id or "").strip()
        if not cid or not _CVM_ID_RE.fullmatch(cid):
            raise PhalaApiError("invalid CVM id for Phala delete")
        path = f"/cvms/{cid}"
        if not _ALLOWED_DELETE_PATH_RE.fullmatch(path):
            raise PhalaApiError("unsupported Phala mutation route")
        request = Request(
            f"{self._base_url}{path}",
            headers=self._base_headers(content_type=False),
            method="DELETE",
        )
        try:
            response = self._opener(request, timeout=self._timeout)
            # Drain body; 204 is empty. Never log response content.
            _ = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return
            raise PhalaApiError(f"Phala delete returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PhalaApiError("Phala delete endpoint is unreachable") from exc



__all__ = [
    "DEFAULT_PHALA_API",
    "DEFAULT_PHALA_API_VERSION",
    "DEFAULT_PHALA_USER_AGENT",
    "PREFERRED_PHALA_REGION",
    "CvmListParseError",
    "CvmListSnapshot",
    "PhalaApiError",
    "PhalaCloudClient",
    "extract_cvm_id_from_create_response",
    "normalize_phala_region",
    "parse_cvms_list_response",
    "resolve_cvm_id_from_list",
    "select_phala_region",
]
