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

LAMBADA is additionally scored **strict** (`g2.lambada_strict.acc`):
unconstrained greedy last-word exact match — the canonical protocol.
The 4-way MC form (`g2.lambada.acc_norm`) saturates (~0.95 at 112M,
~0.985 for GPT-2 Large): the gold word is uniquely determined by the
long context, so random-word distractors measure nothing. The strict
variant keeps real headroom (GPT-2 Large ≈ 0.52-0.60) and feeds
anchors v2 (`org.g2.lambada_strict_acc`); the MC key keeps feeding
anchor sets v0/v1 unchanged. Same JSONL asset — the gold word is
recovered from `choices[gold]`, so no eval-pack rebuild is needed.
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

# Strip wrapping punctuation before exact-match compare (gold words from
# `text.rsplit(" ", 1)` may carry closing quotes; generations may add them).
_PUNCT = "\"'`“”‘’.,;:!?)(][}{<>«»—–-…"


def _norm_word(s):
    return (s or "").strip().strip(_PUNCT)


def _strict_lambada(ctx, model, tok, device, prompt, choices, gold):
    """Greedy last-word exact match on one row; returns acc01 or None."""
    gold_word = _norm_word(choices[int(gold)])
    if not gold_word:
        return None
    gen = common.greedy_word(model, tok, device, prompt)
    acc = 1.0 if _norm_word(gen) == gold_word else 0.0
    common.record(ctx, "g2.lambada_strict.acc", "g2/lambada_strict", acc)
    common.record_trace(
        ctx,
        {
            "kind": "gen",
            "group": "g2",
            "task": "lambada_strict",
            "metric": "g2.lambada_strict.acc",
            "prompt": prompt,
            "gen": gen,
            "gold_word": gold_word,
            "value": acc,
        },
    )
    return acc


def run(model, ctx):
    tok = ctx["tokenizer"]
    device = ctx["device"]
    budget = common.Budget(common.group_budget_s("g2", 1800.0))
    out = {}
    per_task = {}
    nlls = []
    cap = common.eval_asset_cap(200, 8, env_key="PRISM_EVAL_G2_CAP")
    for task in TASKS:
        path = common.assets_path(ctx, f"g2/{task}.jsonl")
        if path is None:
            continue
        rows = common.load_jsonl(path, cap=cap)
        accs = []
        strict_accs = []
        for row in rows:
            if not budget.ok():
                out["g2.partial"] = 1.0
                break
            prompt, choices, gold = row.get("prompt"), row.get("choices"), row.get("gold")
            if not prompt or not choices or gold is None:
                continue
            try:
                acc, nll = common.score_and_record(
                    ctx,
                    f"g2.{task}.acc",
                    f"g2/{task}",
                    model,
                    tok,
                    device,
                    prompt,
                    choices,
                    int(gold),
                    group="g2",
                    task=task,
                )
            except Exception:  # noqa: BLE001
                continue
            accs.append(acc)
            nlls.append(nll)
            if task == "lambada":
                try:
                    s = _strict_lambada(ctx, model, tok, device, prompt, choices, gold)
                except Exception:  # noqa: BLE001
                    s = None
                if s is not None:
                    strict_accs.append(s)
        v = common.mean(accs)
        common.emit(out, f"g2.{task}.acc_norm", v)
        if v is not None:
            per_task[task] = v
        if task == "lambada":
            common.emit(out, "g2.lambada_strict.acc", common.mean(strict_accs))
    out["g2.assets.tasks_present"] = float(len(per_task))
    if per_task:
        common.emit(out, "g2.core.mean_acc_norm", common.mean(per_task.values()))
    common.emit(out, "g2.core.mean_nll", common.mean(nlls))
    return out
