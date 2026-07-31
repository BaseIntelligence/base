#!/usr/bin/env python3
"""Canonical app-compose.json SHA-256 (dstack / base compose-hash rules).

Mirrors crates/compose-hash: strip object keys with null values, lexicographic
key order, compact separators, ensure_ascii=False, no trailing newline.
"""
from __future__ import annotations

import hashlib
import json
import sys
from typing import Any


def strip_null_object_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_null_object_keys(v)
            for k, v in value.items()
            if v is not None
        }
    if isinstance(value, list):
        return [strip_null_object_keys(v) for v in value]
    return value


def main() -> int:
    raw = sys.stdin.buffer.read()
    try:
        value = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        print(f"compose-hash: invalid JSON: {exc}", file=sys.stderr)
        return 2
    stripped = strip_null_object_keys(value)
    canonical = json.dumps(
        stripped,
        separators=(",", ":"),
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    sys.stdout.write(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
