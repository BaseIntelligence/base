"""G4 — reasoning at small scale (research/10 §2, §4, §7).

S5 permutation composition (NC^1-complete no-CoT probe), templated
arithmetic (clause tiers, NoOp distractor, held-out operand range),
ProofWriter-style deductive closure depth 0–3, boolean expressions,
Dyck-k with length split, modular arithmetic ID/OOD, small-N Knights &
Knaves. All procedural with Cantor-lattice seeds; difficulty knobs place
the 100M–350M band in the discriminative 20–80% window. Every accuracy
has an answer-NLL companion.
"""

from . import common, gen_reasoning as gr


def _score(model, ctx, items, budget, out, nlls):
    accs = []
    for it in items:
        if not budget.ok():
            out["g4.partial"] = 1.0
            break
        try:
            acc, nll = common.score_choices(
                model, ctx["tokenizer"], ctx["device"], it["prompt"], it["choices"], it["gold"]
            )
        except Exception:  # noqa: BLE001
            continue
        accs.append(acc)
        nlls.append(nll)
        common.record(ctx, "g4.item.acc", it["cluster"], acc)
    return accs


def run(model, ctx):
    budget = common.Budget(common.group_budget_s("g4", 1800.0))
    out, nlls = {}, []
    secret = common.resolve_secret_seed(ctx)
    tiny = common.tiny_caps()
    n_items = common.eval_n_items(default_full=4, default_tiny=2)
    family_means = []

    def sweep(name, gen, variants):
        means = []
        for tag, kwargs in variants:
            accs = []
            for i in range(n_items):
                seed = common.task_seed(secret, f"{name}/{tag}/{i}")
                accs.extend(_score(model, ctx, gen(seed, **kwargs), budget, out, nlls))
            v = common.mean(accs)
            common.emit(out, f"g4.{name}.{tag}.acc", v)
            if v is not None:
                means.append(v)
        if means:
            common.emit(out, f"g4.{name}.acc", common.mean(means))
            family_means.append(common.mean(means))

    sweep("s5", gr.s5_compose, [(f"k{k}", {"k": k}) for k in ((3, 5) if tiny else (5, 10))])
    sweep(
        "arith",
        gr.arith,
        [("t1", {"tier": 1}), ("t2", {"tier": 2}), ("t3", {"tier": 3})]
        + ([] if tiny else [("noop", {"tier": 2, "noop": True}), ("extrap", {"tier": 2, "extrap": True})]),
    )
    sweep("proof", gr.proofwriter, [(f"d{d}", {"depth": d}) for d in [0, 1] + ([] if tiny else [2, 3])])
    sweep("bool", gr.boolean_expr, [("base", {"depth": 2})])
    sweep(
        "dyck",
        gr.dyck,
        [("k1", {"k": 1, "n_pairs": 10}), ("k2", {"k": 2, "n_pairs": 10})]
        + ([] if tiny else [("k2long", {"k": 2, "n_pairs": 28})]),
    )
    sweep("mod", gr.modular, [("id", {"ood": False}), ("ood", {"ood": True})])
    sweep("kk", gr.knights_knaves, [("n2", {"n": 2}), ("n3", {"n": 3})])

    id_v, ood_v = out.get("g4.mod.id.acc"), out.get("g4.mod.ood.acc")
    if id_v is not None and ood_v is not None:
        common.emit(out, "g4.mod.id_ood_gap", id_v - ood_v)
    common.emit(out, "g4.reasoning.mean_acc", common.mean(family_means))
    common.emit(out, "g4.reasoning.mean_nll", common.mean(nlls))
    return out
