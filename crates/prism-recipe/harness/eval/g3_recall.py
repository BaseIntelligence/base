"""G3 — retrieval / associative recall (research/04 §6.3, research/05 §3).

MQAR, repeat-after-me copying with a gap sweep, induction-head probes,
passkey — the mechanistic probes that separate attention from
fixed-state architectures and that average PPL is blind to. Fully
procedural: fresh Cantor-lattice seeds per run ⇒ memorization-proof.
Every accuracy ships with an answer-NLL companion (Schaeffer rule).
"""

from . import common, generators


def _score_items(model, ctx, items, budget, out):
    accs, nlls = [], []
    for it in items:
        if not budget.ok():
            out["g3.partial"] = 1.0
            break
        try:
            acc, nll = common.score_and_record(
                ctx,
                "g3.item.acc",
                it["cluster"],
                model,
                ctx["tokenizer"],
                ctx["device"],
                it["prompt"],
                it["choices"],
                it["gold"],
                group="g3",
                task=it.get("task"),
            )
        except Exception:  # noqa: BLE001
            continue
        accs.append(acc)
        nlls.append(nll)
    return accs, nlls


def run(model, ctx):
    budget = common.Budget(common.group_budget_s("g3", 1800.0))
    out = {}
    secret = common.resolve_secret_seed(ctx)
    tiny = common.tiny_caps()
    n_items = common.eval_n_items(default_full=4, default_tiny=2)
    family_means, all_nll = [], []

    def go(gen, seed_name, variants):
        for tag, kwargs in variants:
            accs = []
            for i in range(n_items):
                seed = common.task_seed(secret, f"{seed_name}/{tag}/{i}")
                accs_i, nlls = _score_items(model, ctx, gen(seed, **kwargs), budget, out)
                accs.extend(accs_i)
                all_nll.extend(nlls)
            v = common.mean(accs)
            common.emit(out, f"g3.{seed_name}.{tag}.acc", v)
            if v is not None:
                variants_means.append(v)

    variants_means = []

    go(
        generators.mqar,
        "mqar",
        [(f"n{n}", {"n_pairs": n}) for n in ((4, 8) if tiny else (4, 8, 16))],
    )
    if variants_means:
        common.emit(out, "g3.mqar.acc", common.mean(variants_means))
        family_means.append(common.mean(variants_means))
    variants_means = []

    go(
        generators.copy_gap,
        "copy",
        [
            (f"g{g}", {"seq_len": 12 if tiny else 16, "gap": g})
            for g in ((0, 16) if tiny else (0, 16, 64))
        ],
    )
    if variants_means:
        common.emit(out, "g3.copy.acc", common.mean(variants_means))
        family_means.append(common.mean(variants_means))
    variants_means = []

    go(generators.induction, "induction", [("base", {"n_reps": 2, "gap": 8})])
    if variants_means:
        family_means.append(common.mean(variants_means))
    variants_means = []

    go(generators.passkey, "passkey", [("base", {"n_filler": 24 if tiny else 40})])
    if variants_means:
        family_means.append(common.mean(variants_means))

    common.emit(out, "g3.recall.mean_acc", common.mean(family_means))
    common.emit(out, "g3.recall.mean_nll", common.mean(all_nll))
    return out
