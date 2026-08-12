#!/usr/bin/env python3
"""Exact parameter count for the hybrid delta-net anchor config.

Prints the per-component breakdown for the default config and asserts the
total is <= 350,000,000 (the harness MAX_PARAMS cap).

Runs with or without torch (same dual mode as the Transformer++ counter):
with torch it builds via `architecture.build_model({})` and counts exactly,
cross-checked against the closed-form static math; without torch it prints
the static math only.

Usage: python3 count_params.py
"""

import sys

MAX_PARAMS = 350_000_000

# Keep in sync with architecture.DEFAULTS (asserted when torch is available).
DEFAULTS = {
    "vocab_size": 50257,
    "d_model": 1024,
    "n_layer": 24,
    "attn_heads": 16,
    "delta_heads": 8,
    "delta_key_dim": 128,
    "delta_value_dim": 128,
    "mlp_hidden": 2048,
    "conv_kernel": 4,
}


def static_count(cfg):
    v, d, l = cfg["vocab_size"], cfg["d_model"], cfg["n_layer"]
    h = cfg["mlp_hidden"]
    hd, dk, dv = cfg["delta_heads"], cfg["delta_key_dim"], cfg["delta_value_dim"]
    ck = cfg["conv_kernel"]
    n_attn = l // 4
    n_delta = l - n_attn
    mixer_proj = 3 * d * hd * dk + 3 * d * hd * dv  # wq/wk/wa + wv/wgate/wo
    delta_block = (
        mixer_proj
        + d * hd  # wbeta
        + hd * dk  # a_bias
        + ck * (2 * hd * dk + hd * dv)  # depthwise convs q/k/v
        + dv  # per-head output norm
        + 2 * d  # block rmsnorms
        + 3 * d * h  # swiglu mlp
    )
    attn_block = 4 * d * d + 3 * d * h + 2 * d
    parts = {
        "tok_emb (tied head)": v * d,
        "delta mixer proj / block": mixer_proj,
        "delta misc / block": delta_block - mixer_proj - 3 * d * h,
        "mlp swiglu / block": 3 * d * h,
        "attn block (qkvo+mlp+norms)": attn_block,
        "final rmsnorm": d,
        "blocks:": f"{n_delta} delta + {n_attn} attn",
    }
    total = v * d + n_delta * delta_block + n_attn * attn_block + d
    return total, parts, delta_block, attn_block


def torch_count():
    import os
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.py")
    spec = importlib.util.spec_from_file_location("hd_architecture", path)
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    for key in DEFAULTS:
        assert arch.DEFAULTS[key] == DEFAULTS[key], f"DEFAULTS drift on {key}"
    model = arch.build_model({})
    return sum(p.numel() for p in model.parameters())


def main():
    total, parts, delta_block, attn_block = static_count(DEFAULTS)
    print("Hybrid delta-net anchor config:", DEFAULTS)
    for name, n in parts.items():
        print(f"  {str(name):34s} {n if isinstance(n, str) else format(n, ','):>13}")
    print(f"  {'delta block total':34s} {delta_block:>13,}")
    print(f"  {'TOTAL (static math)':34s} {total:>13,}")
    try:
        exact = torch_count()
    except ImportError:
        print("torch not available: static math only (exact build skipped)")
    else:
        print(f"  {'TOTAL (torch build)':34s} {exact:>13,}")
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
