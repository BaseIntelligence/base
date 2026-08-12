"""G5 — long-context battery orchestration (pretrain-only, miner tokenizer).

Scored path (recipe ≥ 1.4.0): community protocols + natural documents —

| Adapter | Module | Grid (tokens of `ctx["tokenizer"]`) |
|---------|--------|--------------------------------------|
| RULER | `g5_ruler` | 4k/8k/16k/32k; **64k** on `niah_mk`+`vt` |
| BABILong | `g5_babilong` | 4k/8k/16k (QA1–QA5) |
| Natural | `natural_docs` | LongBench-v2 MCQ + HELMET RAG ≤16k |

All lengths are measured in tokens of the **submitted** tokenizer via
`eval.toklen` / `common.fit_to_tokens`. The old procedural pack in
`gen_longctx.py` (word-budget ≈ 0.75× tokens) stays in-tree as a
debug/fallback generator but **does not** drive ranked `org.g5.*`
metrics.

Fail-soft: each adapter is wrapped so one OOM/import/runtime failure
cannot kill the group; `g5.partial` is set when any adapter errors or
expires its share of the group budget.

`g5.lstar` / `org.g5.lstar` — length capability: the highest length L
(on the pooled RULER+BABILong per-length means) where
`acc(L) ≥ 0.9 × acc(L_min)` and `acc(L) ≥ 0.25`. When the short-length
base is below the floor, L* is 0.
"""

import time

from . import common, g5_babilong, g5_ruler, natural_docs

_FLOOR = 0.25
_REL = 0.9

# Internal shares of the G5 wall-clock budget (sum = 1).
_BUDGET_SHARE = {
    "ruler": 0.45,
    "babilong": 0.30,
    "natural": 0.25,
}


def _left(budget):
    """Seconds remaining on a shared group budget."""
    return max(1.0, budget.seconds - (time.time() - budget.t0))


def _soft(name, fn, out):
    """Run one adapter; record a fail-soft marker instead of raising."""
    try:
        part = fn()
    except Exception as exc:  # noqa: BLE001 — one adapter never kills G5
        out[f"g5.{name}.error"] = 1.0
        out["g5.partial"] = 1.0
        out.setdefault("_errors", []).append(f"{name}: {str(exc)[:160]}")
        return None
    if part:
        out.update(part)
    return part


def _per_length_means(out):
    """Pool RULER + BABILong `g5.*.L{N}.acc` into {L: mean_acc}."""
    buckets = {}
    for key, value in out.items():
        if not isinstance(value, (int, float)):
            continue
        # g5.ruler.L4096.acc / g5.babilong.L8192.acc (not probe-qualified).
        parts = key.split(".")
        if len(parts) != 4 or parts[0] != "g5" or parts[3] != "acc":
            continue
        if parts[1] not in ("ruler", "babilong"):
            continue
        tag = parts[2]
        if not tag.startswith("L") or not tag[1:].isdigit():
            continue
        buckets.setdefault(int(tag[1:]), []).append(float(value))
    return {L: sum(vs) / len(vs) for L, vs in buckets.items() if vs}


def compute_lstar(len_means):
    """Highest L with acc ≥ 0.9×acc(L_min) and ≥ absolute floor 0.25."""
    if not len_means:
        return None
    base_L = min(len_means)
    base = len_means[base_L]
    if base < _FLOOR:
        return 0.0
    lstar = float(base_L)
    for L in sorted(len_means):
        if len_means[L] >= _REL * base and len_means[L] >= _FLOOR:
            lstar = float(L)
    return lstar


def run(model, ctx):
    """Orchestrate RULER + BABILong + natural; emit flat `g5.*` metrics."""
    budget = common.Budget(common.group_budget_s("g5", 3600.0))
    out = {}
    tiny = common.tiny_caps()

    # --- RULER (partial 64k on niah_mk + vt under full caps) ---
    ruler_s = min(_left(budget), budget.seconds * _BUDGET_SHARE["ruler"])
    probe_grid = None if tiny else dict(g5_ruler.PROBE_GRID_64K)
    _soft(
        "ruler",
        lambda: g5_ruler.run(
            model, ctx, probe_grid=probe_grid, budget_s=ruler_s
        ),
        out,
    )

    # --- BABILong (plan grid caps at 16k) ---
    babi_s = min(_left(budget), budget.seconds * _BUDGET_SHARE["babilong"])
    babi_grid = g5_babilong.GRID_TINY if tiny else g5_babilong.GRID_PLAN
    _soft(
        "babilong",
        lambda: g5_babilong.run(
            model, ctx, grid=babi_grid, budget_s=babi_s
        ),
        out,
    )

    # --- Natural documents (LongBench-v2 MCQ + HELMET RAG) ---
    nat_s = min(_left(budget), budget.seconds * _BUDGET_SHARE["natural"])
    _soft(
        "natural",
        lambda: natural_docs.run(model, ctx, budget=common.Budget(nat_s)),
        out,
    )

    len_means = _per_length_means(out)
    for L, v in sorted(len_means.items()):
        common.emit(out, f"g5.mean.L{L}.acc", v)

    lstar = compute_lstar(len_means)
    common.emit(out, "g5.lstar", lstar)
    if len_means and lstar is not None:
        base_L = min(len_means)
        top_L = max(len_means)
        base = len_means[base_L]
        if top_L != base_L and base > 0:
            common.emit(out, "g5.retention", len_means[top_L] / base)

    # Drop the private error list before returning (not a float metric).
    errors = out.pop("_errors", None)
    if errors:
        out["g5.adapter_errors"] = float(len(errors))
    if not budget.ok():
        out["g5.partial"] = 1.0
    return out
