"""G5 — BABILong (2406.10149) qa1–qa5, base-LM scoring.

Adapter over `eval.vendor_babilong` (upstream noise injection + sampler +
bAbI v1.2 qa1–qa5 grammar, see `eval/VENDOR.md`). Each item hides a handful
of bAbI facts in filler prose drawn from the operator's **private pinned**
corpus (`vendor_babilong.FILLER_ASSET`), asks the upstream question and
scores the short answer by log-probability over the task's closed answer
vocabulary (locations, objects or actors) — upstream's exact-match target
without free generation, no instruction prompt and no judge.

Two no-noise shots of the same task open the prompt so a base model has the
answer format in context. Lengths are targets in tokens of the **submitted**
tokenizer: `eval.toklen` grows the filler until the whole prompt hits the
target and never truncates a fact. Instances regenerate per run from the
private secret seed.

Phase-C entrypoint: `run(model, ctx, grid=..., tasks=..., task_grid=...)` ->
flat `dict[str, float]`; nothing here touches the g5 orchestration, the
rollup or the composite.
"""

import random

from . import common, toklen, vendor_babilong as vb
from .g5_ruler import evidence_ok

GRID = (4096, 8192, 16384, 32768)
# The plan's protocol table caps BABILong at 16k; Phase C selects it by
# passing `grid=GRID_PLAN` — no code change.
GRID_PLAN = (4096, 8192, 16384)
# Tiny caps still have to clear facts + shots + template, which a byte-level
# test tokenizer inflates by ~6x versus a BPE vocabulary.
GRID_TINY = (2048, 4096)
TASKS = vb.TASK_IDS

_N_SHOTS = 2
_SHOT_SHARE = 0.25  # in-context examples never take more of the length target
_DOC_CAP = 512
_SEP = " "
# Background sentences per refill of the lazy filler stream.
_FILL_BATCH = 64


def _filler_docs(ctx, seed):
    """Private pinned filler pack, else the seeded synthetic pool."""
    path = common.assets_path(ctx, vb.FILLER_ASSET)
    if path is not None:
        docs = [
            str(r["text"])
            for r in common.load_jsonl(path, cap=_DOC_CAP)
            if r.get("text")
        ]
        if docs:
            return docs, True
    return vb.synthetic_filler(seed), False


def _split_template(question):
    """Upstream context framing, split around `{context}` for the fitter."""
    head, tail = vb.BASE_TEMPLATE.split("{context}")
    return head, tail.format(question=question)


def _filler_stream(seed, docs):
    """One background sentence per call, upstream `SentenceSampler` order."""
    buf, draw = [], [0]

    def _next():
        if not buf:
            draw[0] += 1
            buf.extend(
                vb.sample_sentences(common.cantor(seed, draw[0]), docs, _FILL_BATCH)
                or ["The record was never found again."]
            )
        return buf.pop(0)

    return _next


def _shots(tok, seed, qa, n_budget):
    """No-noise in-context examples of the same bAbI task.

    Shots are added while they fit `n_budget`, so a short length target keeps
    its context for the scored story instead of for the demonstrations.
    """
    blocks, used = [], 0
    for k in range(_N_SHOTS):
        st = vb.story(common.cantor(seed, 9001 + k), qa)
        if st is None:
            continue
        head, tail = _split_template(st["question"])
        block = f"{head}{' '.join(st['facts'])}{tail} {st['answer']}\n\n"
        cost = toklen.n_tokens(tok, block)
        if used + cost > n_budget:
            break
        blocks.append(block)
        used += cost
    return "".join(blocks)


def _item(tok, seed, qa, length, docs):
    """One scored BABILong item, or None when the story grammar came up dry.

    The facts are the fitter's load-bearing segments and background prose is
    the filler, spliced at random positions — upstream's rule that a fact may
    sit anywhere in the context, with the guarantee that none is ever cut.
    """
    st = vb.story(seed, qa)
    if st is None:
        return None
    head, tail = _split_template(st["question"])
    shots = _shots(tok, seed, qa, int(length * _SHOT_SHARE))
    prompt, n_prompt = toklen.fit_prompt(
        tok,
        shots + head,
        st["facts"],
        tail,
        length,
        _filler_stream(common.cantor(seed, 4243), docs),
        sep=_SEP,
        rng=random.Random(common.cantor(seed, 5077)),
    )
    choices = list(st["choices"])
    random.Random(common.cantor(seed, 6151)).shuffle(choices)
    return {
        "prompt": prompt,
        "choices": [" " + c for c in choices],
        "gold": choices.index(st["answer"]),
        "answer": st["answer"],
        "n_prompt": n_prompt,
        "intact": evidence_ok(prompt, st["facts"]),
        "n_facts": len(st["facts"]),
        "n_shots": shots.count("Answer:"),
        "depths": toklen.depths(prompt, st["facts"]),
    }


def run(
    model, ctx, grid=None, n_items=None, tasks=None, task_grid=None, budget_s=None
):
    """Score BABILong qa1–qa5 on the length grid. Returns flat float metrics."""
    tiny = common.tiny_caps()
    grid = tuple(grid or (GRID_TINY if tiny else GRID))
    tasks = tuple(tasks or TASKS)
    task_grid = dict(task_grid or {})
    n_items = int(n_items if n_items is not None else (1 if tiny else 2))
    budget = common.Budget(
        budget_s if budget_s is not None else common.group_budget_s("g5_babilong", 900.0)
    )
    probe_cap = common.float_env(
        "PRISM_EVAL_G5_BABILONG_PROBE_BUDGET_S", budget.seconds / max(1, len(tasks))
    )

    out = {}
    secret = common.resolve_secret_seed(ctx)
    tok = toklen.tokenizer_of(ctx)
    docs, from_pack = _filler_docs(ctx, common.task_seed(secret, "g5/babilong/filler"))
    common.emit(out, "g5.babilong.filler_pack", 1.0 if from_pack else 0.0)

    per_len, task_means, nlls, len_errs = {}, [], [], {}
    misses, depths, shots = [], [], []
    for qa in tasks:
        probe_budget = common.Budget(min(probe_cap, max(1.0, budget.seconds)))
        means = []
        for length in tuple(task_grid.get(qa, grid)):
            if not budget.ok() or not probe_budget.ok():
                out["g5.babilong.partial"] = 1.0
                break
            accs = []
            for i in range(n_items):
                seed = common.task_seed(secret, f"g5/babilong/qa{qa}/{length}/{i}")
                try:
                    item = _item(tok, seed, qa, length, docs)
                    if item is None:
                        continue
                    acc, nll = common.score_choices(
                        model,
                        tok,
                        ctx["device"],
                        item["prompt"],
                        item["choices"],
                        item["gold"],
                    )
                except Exception:  # noqa: BLE001 — e.g. forward OOM at this length
                    continue
                len_errs.setdefault(length, []).append(
                    abs(item["n_prompt"] - length) / length
                )
                misses.append(0.0 if item["intact"] else 1.0)
                depths.extend(item["depths"])
                shots.append(float(item["n_shots"]))
                accs.append(acc)
                nlls.append(nll)
                common.record(ctx, "g5.babilong.item.acc", f"qa{qa}@{length}", acc)
            value = common.mean(accs)
            common.emit(out, f"g5.babilong.qa{qa}.L{length}.acc", value)
            if value is not None:
                per_len.setdefault(length, []).append(value)
                means.append(value)
        value = common.mean(means)
        common.emit(out, f"g5.babilong.qa{qa}.acc", value)
        if value is not None:
            task_means.append(value)

    for length in sorted(per_len):
        common.emit(out, f"g5.babilong.L{length}.acc", common.mean(per_len[length]))
        common.emit(
            out, f"g5.babilong.L{length}.len_err", common.mean(len_errs.get(length))
        )
    common.emit(out, "g5.babilong.acc", common.mean(task_means))
    common.emit(out, "g5.babilong.mean_nll", common.mean(nlls))
    common.emit(out, "g5.babilong.evidence_miss", common.mean(misses))
    # Mean fact depth: ~0.5 means the facts really are scattered through the
    # filler instead of clustering at the head of the context.
    common.emit(out, "g5.babilong.mean_depth", common.mean(depths))
    common.emit(out, "g5.babilong.mean_shots", common.mean(shots))
    return out
