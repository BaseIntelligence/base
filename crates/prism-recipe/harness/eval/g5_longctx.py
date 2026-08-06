"""G5 — long-context capability (research/05 §3, §7).

Procedural pack on a nominal length grid — full: 1k–32k, tiny caps:
512–2k — covering NIAH multi-key, RULER variable tracking + frequent
words, BABILong qa1–qa5, GraphWalks, MRCR ordering, NoLiMa latent
needles. Aggregation is self-normalized: L* = max L where mean accuracy
≥ 0.9 × mean accuracy at the shortest grid point AND ≥ an absolute
floor (0.25) so uniformly-bad models do not get inflated L*. Needle
depths are randomized by the generators; per-item records carry the
(task, length) cluster for the clustered bootstrap.
"""

from . import common, gen_longctx as gl

_GRID_FULL = (1024, 2048, 4096, 8192, 16384, 32768)
_GRID_TINY = (512, 1024, 2048)
_FLOOR = 0.25

_TASKS = (
    ("niah", lambda seed, L: gl.niah_multikey(seed, L)),
    ("vt", lambda seed, L: gl.variable_tracking(seed, L)),
    ("freq", lambda seed, L: gl.freq_words(seed, L)),
    ("babi", lambda seed, L: [it for qa in (1, 2, 3, 4, 5) for it in gl.babilong(seed + qa, L, qa=qa)]),
    ("graph", lambda seed, L: gl.graphwalks(seed, L)),
    ("mrcr", lambda seed, L: gl.mrcr_order(seed, L)),
    ("nolima", lambda seed, L: gl.nolima(seed, L)),
)


def run(model, ctx):
    budget = common.Budget(common.group_budget_s("g5", 3600.0))
    out = {}
    secret = common.resolve_secret_seed(ctx)
    tiny = common.tiny_caps()
    grid = _GRID_TINY if tiny else _GRID_FULL
    n_items = 1 if tiny else 2
    per_len = {L: [] for L in grid}
    all_accs, nlls = [], []

    for task, gen in _TASKS:
        for L in grid:
            if not budget.ok():
                out["g5.partial"] = 1.0
                break
            accs = []
            for i in range(n_items):
                seed = common.task_seed(secret, f"g5/{task}/{L}/{i}")
                for it in gen(seed, L):
                    try:
                        acc, nll = common.score_choices(
                            model, ctx["tokenizer"], ctx["device"],
                            it["prompt"], it["choices"], it["gold"],
                        )
                    except Exception:  # noqa: BLE001 — e.g. forward OOM at L
                        continue
                    accs.append(acc)
                    nlls.append(nll)
                    common.record(ctx, "g5.item.acc", f"{task}@{L}", acc)
            v = common.mean(accs)
            common.emit(out, f"g5.{task}.L{L}.acc", v)
            if v is not None:
                per_len[L].append(v)
                all_accs.append(v)

    len_means = {}
    for L in grid:
        v = common.mean(per_len[L])
        if v is not None:
            len_means[L] = v
            common.emit(out, f"g5.mean.L{L}.acc", v)

    common.emit(out, "g5.longctx.mean_acc", common.mean(all_accs))
    common.emit(out, "g5.longctx.mean_nll", common.mean(nlls))

    if len_means:
        base_L = min(len_means)
        base = len_means[base_L]
        lstar = float(base_L) if base >= _FLOOR else 0.0
        for L in sorted(len_means):
            if len_means[L] >= 0.9 * base and len_means[L] >= _FLOOR:
                lstar = float(L)
        common.emit(out, "g5.lstar", lstar)
        top_L = max(len_means)
        if top_L != base_L and base > 0:
            common.emit(out, "g5.retention", len_means[top_L] / base)
    return out
