"""Strict canonical JSON v1 primitives shared by attested-review bindings."""

from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any


class CanonicalJsonError(ValueError):
    """The value cannot be represented by canonical_json_v1."""


def canonical_json_v1(value: Any) -> bytes:
    """Serialize a schema-closed JSON value with canonical Unicode and ordering.

    Schema validators own closed-object, integer and collection semantics. This
    helper makes the shared wire preimage deterministic and rejects values JSON
    would otherwise silently normalize, including floats and lone surrogates.
    """

    normalized = _normalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json_v1(value)).hexdigest()


def parse_json_object(raw: bytes | str) -> dict[str, Any]:
    """Parse one duplicate-key-free JSON object for schema validation."""

    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalJsonError("JSON must be UTF-8") from exc
    if not isinstance(raw, str):
        raise CanonicalJsonError("JSON input must be text or bytes")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise CanonicalJsonError(f"duplicate JSON key: {key!r}")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CanonicalJsonError(f"unsupported JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CanonicalJsonError("malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise CanonicalJsonError("JSON root must be an object")
    _normalize(parsed)
    return parsed


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError("JSON floats must be finite")
        raise CanonicalJsonError("canonical_json_v1 forbids floats")
    if isinstance(value, str):
        _reject_lone_surrogates(value)
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalJsonError("JSON object keys must be strings")
            normalized_key = _normalize(key)
            if normalized_key in normalized:
                raise CanonicalJsonError(
                    f"duplicate key after NFC normalization: {normalized_key!r}"
                )
            normalized[normalized_key] = _normalize(item)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_normalize(item) for item in value]
    raise CanonicalJsonError(f"unsupported canonical JSON type: {type(value).__name__}")


def _reject_lone_surrogates(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise CanonicalJsonError("strings must not contain surrogate code points")
