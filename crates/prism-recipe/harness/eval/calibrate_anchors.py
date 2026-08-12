#!/usr/bin/env python3
"""Fill `anchors/v0.json` reference (and optional chance) from baseline METRICS.

After E6 GPU baseline runs (Transformer++ / hybrid delta-net), point this
script at one or more METRICS_JSON v2 blobs (or directories of them). It
reads `battery.metrics` for the scored `org.g1.bits_per_byte_*` keys and
rewrites the matching `reference` fields in the anchor set, clearing
per-metric `status: placeholder` when a finite value is present.

Does **not** flip the top-level `"status": "placeholder"` → `"active"`;
that is a governance step after hash-commit / pre-registration.

Usage:
  python3 calibrate_anchors.py \\
      --metrics /path/to/transformer_pp/METRICS.json \\
      --metrics /path/to/hybrid_delta/METRICS.json \\
      --anchors ../../anchors/v0.json \\
      --write

Without `--write`, prints the proposed patch as JSON on stdout.
Without GPU baselines, leave v0.json placeholders and run this later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Scored G1 keys (tokenizer-neutral). Efficiency / accuracy refs are left
# alone unless explicitly listed — chance floors for accuracy stay theory-real.
_G1_ORG = (
    "org.g1.bits_per_byte_code",
    "org.g1.bits_per_byte_prose",
    "org.g1.bits_per_byte_math",
    "org.g1.bits_per_byte_fresh_crawl",
    "org.g1.bits_per_byte_key_token",
)


def _metric_value(entry):
    if isinstance(entry, (int, float)) and math.isfinite(float(entry)):
        return float(entry)
    if isinstance(entry, dict):
        v = entry.get("value")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            return float(v)
    return None


def _load_metrics(path: Path) -> dict:
    blob = json.loads(path.read_text(encoding="utf-8"))
    battery = blob.get("battery") or {}
    metrics = battery.get("metrics") or blob.get("metrics") or {}
    if not isinstance(metrics, dict):
        raise SystemExit(f"{path}: no battery.metrics object")
    return metrics


def _collect(paths: list[Path]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in _G1_ORG}
    for p in paths:
        m = _load_metrics(p)
        for k in _G1_ORG:
            v = _metric_value(m.get(k))
            if v is not None:
                out[k].append(v)
    return out


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics",
        action="append",
        type=Path,
        required=True,
        help="METRICS_JSON v2 path (repeatable); means across files become references",
    )
    ap.add_argument(
        "--anchors",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "anchors" / "v0.json",
        help="Anchor set JSON to patch (default: crates/prism-recipe/anchors/v0.json)",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write the patched anchor file in place (default: print patch only)",
    )
    args = ap.parse_args(argv)

    collected = _collect(args.metrics)
    refs = {k: _mean(vs) for k, vs in collected.items()}
    missing = [k for k, v in refs.items() if v is None]
    patch = {
        "references": {k: v for k, v in refs.items() if v is not None},
        "missing": missing,
        "note": (
            "Apply after both E6 baselines finish private-tier eval. "
            "Top-level status stays placeholder until governance flip."
        ),
    }
    if not args.write:
        json.dump(patch, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    anchors = json.loads(args.anchors.read_text(encoding="utf-8"))
    g1 = anchors["groups"]["g1"]["metrics"]
    for key, value in refs.items():
        if value is None or key not in g1:
            continue
        g1[key]["reference"] = value
        g1[key]["status"] = "measured"
        note = g1[key].get("note") or ""
        if "PLACEHOLDER" in note:
            g1[key]["note"] = (
                f"Measured reference={value:.6f} bits/byte "
                f"(mean over {len(collected[key])} baseline METRICS); "
                "confirm chance floor before composite flip"
            )
    args.anchors.write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")
    json.dump(patch, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
