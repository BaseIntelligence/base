"""Fail-loud Phala CVM list parsing (spend / teardown safety guard).

Legacy product code hit ``GET /cvms`` and treated any unrecognized envelope as
an empty list — under-reporting live CVMs as count 0. The Phala CLI (API
version ``2026-06-23``) lists via ``GET /cvms/paginated`` and returns
``{items, total, ...}``.

Unknown shapes raise; they never become total=0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: CLI-negotiated API version (phala@latest default).
CLI_PHALA_API_VERSION = "2026-06-23"

#: Cloudflare-safe CLI User-Agent (1010 without a phala-* agent string).
CLI_PHALA_USER_AGENT = "phala-cloud-cli/1.1.19"

_LIST_KEYS = ("items", "cvms", "data")
_CREATE_CVM_ID_FIELDS = ("id", "cvm_id", "vm_uuid", "instance_id", "uuid")


class CvmListParseError(ValueError):
    """GET /cvms payload shape is not one of the known envelopes."""


@dataclass(frozen=True, slots=True)
class CvmListSnapshot:
    """Parsed CVM listing. ``total`` is the authoritative account count."""

    items: tuple[Mapping[str, Any], ...]
    total: int
    ids: tuple[str, ...]
    source_shape: str


def _shape_hint(payload: Any) -> str:
    if payload is None:
        return "null"
    if isinstance(payload, list):
        return f"list(len={len(payload)})"
    if isinstance(payload, Mapping):
        keys = sorted(str(k) for k in payload.keys())
        return "object(keys=" + ",".join(keys[:12]) + ")"
    return type(payload).__name__


def _normalize_id(value: Any) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value <= 0:
            return None
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _item_id(item: Mapping[str, Any]) -> str | None:
    for key in _CREATE_CVM_ID_FIELDS:
        if key not in item:
            continue
        normalized = _normalize_id(item.get(key))
        if normalized is not None:
            return normalized
    return None


def _extract_cvm_id(item: Mapping[str, Any]) -> str:
    for name in _CREATE_CVM_ID_FIELDS:
        if name not in item:
            continue
        normalized = _normalize_id(item.get(name))
        if normalized is not None:
            return normalized
    raise ValueError("Phala create response does not identify the CVM")


def _as_item_dicts(raw_items: Sequence[Any], *, shape: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise CvmListParseError(
                f"unrecognized CVM list shape: {shape} item[{idx}] is not an object"
            )
        out.append(dict(item))
    return out


def _total_from_mapping(payload: Mapping[str, Any], item_count: int) -> int:
    """Prefer explicit total; validate against items on a single-page view."""

    raw_total = payload.get("total")
    has_page_meta = any(
        k in payload for k in ("page", "pages", "totalPages", "page_size", "pageSize")
    )
    if raw_total is None and not has_page_meta:
        return item_count
    if raw_total is None:
        raise CvmListParseError(
            "unrecognized CVM list shape: paginated object missing integer total "
            f"({_shape_hint(payload)})"
        )
    if isinstance(raw_total, bool) or not isinstance(raw_total, int):
        raise CvmListParseError(
            "unrecognized CVM list shape: total is not an int "
            f"({_shape_hint(payload)})"
        )
    if raw_total < 0:
        raise CvmListParseError(
            f"unrecognized CVM list shape: negative total={raw_total}"
        )

    page = payload.get("page")
    pages = payload.get("pages")
    if pages is None:
        pages = payload.get("totalPages")
    page_size = payload.get("page_size")
    if page_size is None:
        page_size = payload.get("pageSize")

    single_page = pages is None or pages in (0, 1)
    first_page = page is None or page == 1
    if first_page and single_page and raw_total == 0 and item_count > 0:
        raise CvmListParseError(
            "unrecognized CVM list shape: total=0 but items is non-empty "
            f"(items={item_count}; {_shape_hint(payload)})"
        )
    if (
        first_page
        and single_page
        and raw_total > 0
        and item_count == 0
        and (pages == 0 or (pages is None and page_size is None))
    ):
        raise CvmListParseError(
            "unrecognized CVM list shape: total>0 but items empty on single page "
            f"(total={raw_total}; {_shape_hint(payload)})"
        )
    return raw_total


def parse_cvms_list_response(payload: Any) -> CvmListSnapshot:
    """Parse a Phala CVM list body. Raises on any unrecognized shape.

    Known-good shapes:
    * bare ``list`` of CVM objects
    * ``{items|cvms|data: list, total?: int, ...}`` (API paginated + CLI wrap)
    * nested ``{data: {items, total}}``
    """

    if isinstance(payload, list):
        items = _as_item_dicts(payload, shape="bare-list")
        ids = tuple(i for i in (_item_id(x) for x in items) if i is not None)
        return CvmListSnapshot(
            items=tuple(items),
            total=len(items),
            ids=ids,
            source_shape="bare-list",
        )

    if not isinstance(payload, Mapping):
        raise CvmListParseError(
            f"unrecognized CVM list shape: {_shape_hint(payload)}"
        )

    # Nested CLI success wrapper: {success, data: {items, total}}
    if (
        "items" not in payload
        and "cvms" not in payload
        and "data" in payload
        and isinstance(payload.get("data"), Mapping)
    ):
        return parse_cvms_list_response(payload["data"])

    list_key: str | None = None
    raw_items: Any = None
    for key in _LIST_KEYS:
        if key not in payload:
            continue
        candidate = payload.get(key)
        if isinstance(candidate, list):
            list_key = key
            raw_items = candidate
            break
        raise CvmListParseError(
            f"unrecognized CVM list shape: {key!r} is not a list "
            f"({_shape_hint(payload)})"
        )

    if list_key is None:
        raise CvmListParseError(
            f"unrecognized CVM list shape: {_shape_hint(payload)}"
        )

    items = _as_item_dicts(raw_items, shape=f"object.{list_key}")
    total = _total_from_mapping(payload, len(items))
    ids = tuple(i for i in (_item_id(x) for x in items) if i is not None)
    return CvmListSnapshot(
        items=tuple(items),
        total=total,
        ids=ids,
        source_shape=f"object.{list_key}",
    )


def resolve_cvm_id_from_snapshot(
    snapshot: CvmListSnapshot,
    *,
    app_id: str,
    require_unique: bool = False,
) -> str | None:
    """Locate a CVM id in a parsed snapshot by exact app_id match."""

    if not isinstance(app_id, str) or not app_id.strip():
        return None
    target = app_id.strip()
    matches: list[str] = []
    for item in snapshot.items:
        item_app = item.get("app_id")
        if not isinstance(item_app, str) or item_app != target:
            continue
        try:
            matches.append(_extract_cvm_id(item))
        except ValueError:
            continue
    if not matches:
        return None
    if require_unique and len(matches) > 1:
        raise CvmListParseError(
            f"multiple CVMs match app_id ({len(matches)}); pass --cvm-id explicitly"
        )
    return matches[0]


__all__ = [
    "CLI_PHALA_API_VERSION",
    "CLI_PHALA_USER_AGENT",
    "CvmListParseError",
    "CvmListSnapshot",
    "parse_cvms_list_response",
    "resolve_cvm_id_from_snapshot",
]
