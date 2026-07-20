#!/usr/bin/env python3
"""parity_diff.py — compare two harbor golden baselines (Task 4, harbor-independence-a3).

Compares per-task outcome records between two baselines (a single JSON file or a
directory of baseline files) and reports task-level deltas.

Comparison contract
-------------------
* EXACT (ε=0) on the canonical fields: ``status``, ``reason_code``, ``resolved``.
* ``reward`` uses the precision rule: ε=0 decimal-string compare (NaN == NaN).
  A non-zero ``--reward-eps`` widens the tolerance to ``abs(a-b) <= eps``
  (forward-compat for Task 9; default is exact).
* Any other field (provenance, blocker, observed_local, ...) is IGNORED.

Baseline shape
--------------
A baseline file is JSON with a top-level ``results`` mapping of
``task -> {reward, status, reason_code, resolved, ...}``. Files without a
``results`` key (e.g. ``dataset-digest.json``) are skipped. When a directory is
given, every ``*.json`` in it is loaded and records are keyed ``<file.name>::<task>``.

Exit code: 0 iff there are zero task deltas, else 1.
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Any

# Canonical fields compared EXACTLY (ε=0). Order is the report order.
CANONICAL_FIELDS: tuple[str, ...] = ("status", "reason_code", "resolved")
REWARD_FIELD = "reward"
COMPARED_FIELDS: tuple[str, ...] = (REWARD_FIELD,) + CANONICAL_FIELDS


def _to_decimal(value: Any) -> Decimal:
    """Decimal from the shortest round-trip string of a numeric value.

    Using ``str(float)`` gives Python's shortest repr (e.g. ``"0.3"``), so two
    values that print identically compare equal while genuine decimal
    differences (``0.30000000000000004`` vs ``0.3``) do not.
    """
    return Decimal(str(value))


def rewards_equal(a: Any, b: Any, reward_eps: float = 0.0) -> bool:
    """Compare two reward values under the precision rule.

    * NaN == NaN (both missing/undefined rewards are considered equal).
    * ``reward_eps == 0`` -> exact decimal-string comparison.
    * ``reward_eps > 0``  -> ``abs(a - b) <= reward_eps``.
    """
    a_nan = isinstance(a, float) and math.isnan(a)
    b_nan = isinstance(b, float) and math.isnan(b)
    if a_nan or b_nan:
        return a_nan and b_nan
    if reward_eps and reward_eps > 0:
        return abs(float(a) - float(b)) <= reward_eps
    return _to_decimal(a) == _to_decimal(b)


def _fields_equal(field: str, left: Any, right: Any, reward_eps: float) -> bool:
    if field == REWARD_FIELD:
        return rewards_equal(left, right, reward_eps=reward_eps)
    return left == right


def compare_records(
    left: dict[str, dict],
    right: dict[str, dict],
    reward_eps: float = 0.0,
) -> list[dict]:
    """Return a list of delta dicts between two ``task -> record`` maps.

    Each delta has keys: ``task``, ``kind``, and (for ``field_mismatch``)
    ``field``, ``left``, ``right``. ``kind`` is one of
    ``missing_in_right`` / ``missing_in_left`` / ``field_mismatch``.
    """
    deltas: list[dict] = []
    for task in sorted(set(left) | set(right)):
        if task not in right:
            deltas.append({"task": task, "kind": "missing_in_right"})
            continue
        if task not in left:
            deltas.append({"task": task, "kind": "missing_in_left"})
            continue
        lrec, rrec = left[task], right[task]
        for field in COMPARED_FIELDS:
            lval = lrec.get(field)
            rval = rrec.get(field)
            if not _fields_equal(field, lval, rval, reward_eps):
                deltas.append(
                    {
                        "task": task,
                        "kind": "field_mismatch",
                        "field": field,
                        "left": lval,
                        "right": rval,
                    }
                )
    return deltas


def _load_one_file(path: Path) -> dict[str, dict]:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    results = data.get("results")
    if not isinstance(results, dict):
        return {}
    return {f"{path.name}::{task}": rec for task, rec in results.items()}


def load_baseline(path: str | Path) -> dict[str, dict]:
    """Load a baseline from a file or a directory into a ``key -> record`` map.

    Keys are ``<file.name>::<task>``. Files without a ``results`` mapping are
    skipped (e.g. ``dataset-digest.json``).
    """
    path = Path(path)
    if path.is_dir():
        loaded: dict[str, dict] = {}
        for f in sorted(path.glob("*.json")):
            loaded.update(_load_one_file(f))
        return loaded
    return _load_one_file(path)


def _format_delta(d: dict) -> str:
    if d["kind"] == "missing_in_right":
        return f"  - {d['task']}: missing in RIGHT"
    if d["kind"] == "missing_in_left":
        return f"  + {d['task']}: missing in LEFT"
    return f"  ~ {d['task']}.{d['field']}: {d['left']!r} != {d['right']!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="baseline file or directory (reference)")
    parser.add_argument("right", help="baseline file or directory (candidate)")
    parser.add_argument(
        "--reward-eps",
        type=float,
        default=0.0,
        help="reward tolerance (default 0 = exact decimal compare)",
    )
    args = parser.parse_args(argv)

    left = load_baseline(args.left)
    right = load_baseline(args.right)
    deltas = compare_records(left, right, reward_eps=args.reward_eps)

    tasks_with_deltas = sorted({d["task"] for d in deltas})
    if not deltas:
        print(f"PARITY OK: {len(left)} record(s), 0 task delta(s) ({args.left} vs {args.right})")
        return 0

    print(f"PARITY FAIL: {len(tasks_with_deltas)} task(s) differ")
    for d in deltas:
        print(_format_delta(d))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
