"""G2 — commonsense/reading core (research/04 §2, §8: the
Pythia/Mamba six-plus, 0-shot, loglikelihood `acc_norm`, frozen prompts).

Self-contained evaluators — no lm-eval dependency. Prompts are frozen
inside the JSONL assets (fully-formed `prompt` + `choices` + `gold`),
so public anchors and private mirrors share one code path and one
extraction contract:

  $PRISM_EVAL_ASSETS_DIR/g2/<task>.jsonl   (private mirrors — scored)
  eval/public_dev/g2/<task>.jsonl          (public anchors — dev tier)

A task with no asset is skipped (`g2.assets.tasks_present` reports how
many ran); the group never errors on missing data.
"""

from . import common

TASKS = (
    "lambada",
    "hellaswag",
    "piqa",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "boolq",
    "openbookqa",
)


def run(model, ctx):
    tok = ctx["tokenizer"]
    device = ctx["device"]
    budget = common.Budget(common.group_budget_s("g2", 1800.0))
    out = {}
    per_task = {}
    nlls = []
    cap = 8 if common.tiny_caps() else 200
    for task in TASKS:
        path = common.assets_path(ctx, f"g2/{task}.jsonl")
        if path is None:
            continue
        rows = common.load_jsonl(path, cap=cap)
        accs = []
        for row in rows:
            if not budget.ok():
                out["g2.partial"] = 1.0
                break
            prompt, choices, gold = row.get("prompt"), row.get("choices"), row.get("gold")
            if not prompt or not choices or gold is None:
                continue
            try:
                acc, nll = common.score_choices(model, tok, device, prompt, choices, int(gold))
            except Exception:  # noqa: BLE001
                continue
            accs.append(acc)
            nlls.append(nll)
            common.record(ctx, f"g2.{task}.acc", f"g2/{task}", acc)
        v = common.mean(accs)
        common.emit(out, f"g2.{task}.acc_norm", v)
        if v is not None:
            per_task[task] = v
    out["g2.assets.tasks_present"] = float(len(per_task))
    if per_task:
        common.emit(out, "g2.core.mean_acc_norm", common.mean(per_task.values()))
    common.emit(out, "g2.core.mean_nll", common.mean(nlls))
    return out
