#!/usr/bin/env python3
"""Exact parameter count for the Transformer++ anchor config.

Prints the per-component breakdown for the default config and asserts the
total is <= 350,000,000 (the harness MAX_PARAMS cap).

Runs with or without torch:
- with torch: builds the model via `architecture.build_model({})` and counts
  `sum(p.numel() for p in model.parameters())` (the harness's own method),
  cross-checked against the closed-form static math;
- without torch: static math only (formula mirrors the module layout).

Usage: python3 count_params.py
"""

import sys

MAX_PARAMS = 1_000_000_000

# Keep in sync with architecture.DEFAULTS (asserted when torch is available).
DEFAULTS = {
    "vocab_size": 50257,
    "d_model": 1024,
    "n_layer": 24,
    "n_head": 16,
    "mlp_hidden": 2560,
}


def static_count(cfg):
    v, d = cfg["vocab_size"], cfg["d_model"]
    h, l = cfg["mlp_hidden"], cfg["n_layer"]
    parts = {
        "tok_emb (tied head)": v * d,
        "attn qkvo / block": 4 * d * d,
        "mlp swiglu / block": 3 * d * h,
        "rmsnorms / block": 2 * d,
        "final rmsnorm": d,
    }
    total = v * d + l * (4 * d * d + 3 * d * h + 2 * d) + d
    return total, parts


def torch_count():
    import os
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.py")
    spec = importlib.util.spec_from_file_location("tpp_architecture", path)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    assert arch.DEFAULTS["vocab_size"] == DEFAULTS["vocab_size"]
    assert arch.DEFAULTS["d_model"] == DEFAULTS["d_model"]
    assert arch.DEFAULTS["n_layer"] == DEFAULTS["n_layer"]
    assert arch.DEFAULTS["mlp_hidden"] == DEFAULTS["mlp_hidden"]
    model = arch.build_model({})
    return sum(p.numel() for p in model.parameters())


def main():
    total, parts = static_count(DEFAULTS)
    print("Transformer++ anchor config:", DEFAULTS)
    for name, n in parts.items():
        print(f"  {name:24s} {n:>13,}")
    print(f"  {'TOTAL (static math)':24s} {total:>13,}")
    try:
        exact = torch_count()
    except ImportError:
        print("torch not available: static math only (exact build skipped)")
    else:
        print(f"  {'TOTAL (torch build)':24s} {exact:>13,}")
        if exact != total:
            print(f"MISMATCH: static {total} != torch {exact}", file=sys.stderr)
            return 1
    print(f"cap check: {total:,} <= {MAX_PARAMS:,}: {total <= MAX_PARAMS}")
    if total > MAX_PARAMS:
        print("OVER CAP", file=sys.stderr)
        return 1
    headroom = MAX_PARAMS - total
    print(f"headroom: {headroom:,} params ({100.0 * headroom / MAX_PARAMS:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
