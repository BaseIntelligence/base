"""G5 — RULER (2404.06654) on the official generators, base-LM scoring.

Adapter over `eval.vendor_ruler` (upstream templates and generators, see
`eval/VENDOR.md`). Subsets: `niah_mk` / `niah_mq` / `niah_mv` (upstream
`niah_multikey`, `niah_multiquery`, `niah_multivalue` complexity knobs),
`vt` (variable tracking) and `qa` (documents + short answer).

Everything a *pretrained base* model needs and nothing it does not: the
prompt is the upstream template plus one in-distribution shot, ending on the
upstream answer prefix; the score is the log-probability argmax over
candidate short answers (`common.score_choices`, OLMES `acc_norm`), one
scored slot per upstream reference answer — the log-probability form of
upstream `string_match_all`. No chat template, no instruction tuning, no
generation, no judge.

Lengths are targets in tokens of the **submitted** tokenizer: `eval.toklen`
grows the upstream haystack with filler of the same grammar until the whole
prompt hits the target, never truncating a needle. Instances regenerate per
run from the private secret seed via the `common.task_seed` lattice.

Phase-C entrypoint: `run(model, ctx, grid=..., probes=..., probe_grid=...)`
-> flat `dict[str, float]`; nothing here touches the g5 orchestration, the
rollup or the composite.
"""

import random

from . import common, toklen, vendor_ruler as vr

GRID = (4096, 8192, 16384, 32768)
# Tiny caps still have to clear the task's structural minimum (needles,
# chains or documents plus the upstream template), which a byte-level test
# tokenizer inflates by ~6x versus a BPE vocabulary.
GRID_TINY = (2048, 4096)
PROBES = ("niah_mk", "niah_mq", "niah_mv", "vt", "qa")
# Partial upper tier (plan: 64k on niah_mk + vt only). Phase C passes this
# as `probe_grid` — no code change needed to light it up.
PROBE_GRID_64K = {"niah_mk": GRID + (65536,), "vt": GRID + (65536,)}

_SHOT_TOKENS = 512  # one in-distribution shot, short by design
_SHOT_SHARE = 0.25  # …and never more than this much of the item's budget
_SHOT_SLACK = 16  # tokens of tolerance before a shot is dropped
_QA_ROW_CAP = 256

# Upstream `synthetic.yaml` complexity knobs per subset.
_CFG = {
    "niah_mk": ("niah", {"num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1}),
    "niah_mq": ("niah", {"num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 4}),
    "niah_mv": ("niah", {"num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1}),
    "vt": ("variable_tracking", {"num_chains": 2, "num_hops": 4}),
    "qa": ("qa", {}),
}


def _maker(probe, seed, rows):
    gen, kwargs = _CFG[probe]
    fn = getattr(vr, gen)
    if probe == "qa":
        return lambda size: fn(seed, size, rows=rows)
    return lambda size: fn(seed, size, **kwargs)


def evidence_ok(prompt, evidence):
    """True when every answer-bearing span is present in the prompt.

    `eval.toklen` never truncates a load-bearing segment, so this is a guard
    rather than a filter; it is compared on whitespace-normalized text so a
    tokenizer whose decode normalizes spacing is not falsely rejected.
    """
    flat = " ".join(prompt.split())
    return all(" ".join(str(e).split()) in flat for e in evidence)


def _sized(tok, make, n_budget):
    """Upstream's haystack-size search, one measured step, biased to undershoot.

    Only needed by the probes whose sentences are numbered (`qa`): there the
    fitter tops up at the tail, so the generated documents must already come
    in under the budget. Probes that accept spliced filler start minimal.
    """
    inst = make(8)
    segments = inst["segments"]
    per = max(1.0, sum(toklen.n_tokens(tok, s) for s in segments) / len(segments))
    return make(max(1, int(0.9 * n_budget / per)))


def _build(tok, make, length, shot, seed=0):
    """Instance + prompt fitted to `length` tokens, evidence intact."""
    shot = shot or ""
    inst = make(4)
    if not inst["splice"]:
        overhead = toklen.n_tokens(tok, shot + inst["head"] + inst["tail"])
        inst = _sized(tok, make, max(8, int(length) - overhead))
    rng = random.Random(common.cantor(int(seed), 811)) if inst["splice"] else None
    prompt, n_prompt = toklen.fit_prompt(
        tok,
        shot + inst["head"],
        inst["segments"],
        inst["tail"],
        length,
        inst["filler"],
        sep=inst["sep"],
        rng=rng,
    )
    return inst, prompt, n_prompt, evidence_ok(prompt, inst["evidence"])


def _shot(tok, make, n_budget, seed=0):
    """One in-distribution completion example (upstream VT `add_fewshot`).

    Returns `""` when the smallest instance of this probe does not fit in
    `n_budget`: a shot is a courtesy to the base model, not a reason to
    overshoot the length the run is measuring. The slack absorbs the token or
    two a tokenizer can add when the shot is concatenated.
    """
    n_budget = int(n_budget)
    inst, prompt, n, _ok = _build(tok, make, n_budget, "", seed=seed)
    if n > n_budget + _SHOT_SLACK:
        return ""
    answer = ", ".join(slot["answer"] for slot in inst["slots"])
    return prompt.rstrip() + " " + answer + "\n\n"


def _shot_budget(length):
    """Tokens a shot may take: short, and never a large share of the item."""
    return min(_SHOT_TOKENS, int(int(length) * _SHOT_SHARE))


def _score_slots(model, ctx, prompt, slots):
    """Per-reference argmax accuracy, teacher-forced on the earlier golds.

    The log-probability analogue of upstream `string_match_all`: reference
    `j` is credited when the model prefers it over the in-distribution
    distractors at its position, given the true earlier references.
    """
    prefix = prompt.rstrip()
    accs, nlls = [], []
    for j, slot in enumerate(slots):
        sep = " " if j == 0 else ", "
        choices = [sep + c for c in slot["choices"]]
        detail = common.score_choices_detail(
            model, ctx["tokenizer"], ctx["device"], prefix, choices, slot["gold"]
        )
        acc, nll = detail["acc"], detail["gold_nll"]
        # Trace only here — bootstrap cluster is the item mean (`probe@L`).
        common.record_trace(
            ctx,
            {
                "kind": "mc",
                "group": "g5",
                "task": "ruler",
                "metric": "g5.ruler.item.acc",
                "cluster": f"slot{j}",
                "prompt": prefix,
                "choices": choices,
                "gold": int(slot["gold"]),
                "selected": detail["selected"],
                "value": acc,
                "gold_nll": nll,
                "choice_logprobs": detail["choice_logprobs"],
                "meta": {"slot": j},
            },
        )
        accs.append(acc)
        nlls.append(nll)
        prefix = prefix + sep + slot["answer"]
    return accs, nlls


def _qa_rows(ctx):
    path = common.assets_path(ctx, vr.QA_ASSET)
    if path is None:
        return None
    rows = [
        r
        for r in common.load_jsonl(path, cap=_QA_ROW_CAP)
        if r.get("question") and r.get("answers") and r.get("context")
    ]
    return rows or None


def run(model, ctx, grid=None, n_items=None, probes=None, probe_grid=None, budget_s=None):
    """Score the RULER subsets on the length grid. Returns flat float metrics."""
    tiny = common.tiny_caps()
    grid = tuple(grid or (GRID_TINY if tiny else GRID))
    probes = tuple(probes or PROBES)
    probe_grid = dict(probe_grid or {})
    n_items = int(
        n_items
        if n_items is not None
        else common.eval_g5_n_items(default_full=2, default_tiny=1)
    )
    budget = common.Budget(
        budget_s if budget_s is not None else common.group_budget_s("g5_ruler", 1200.0)
    )
    probe_cap = common.float_env(
        "PRISM_EVAL_G5_RULER_PROBE_BUDGET_S", budget.seconds / max(1, len(probes))
    )

    out = {}
    secret = common.resolve_secret_seed(ctx)
    tok = toklen.tokenizer_of(ctx)
    rows = _qa_rows(ctx)
    common.emit(out, "g5.ruler.qa.pack", 1.0 if rows else 0.0)

    per_len, probe_means, nlls, len_errs = {}, [], [], {}
    misses, depths, shots = [], [], []
    for probe in probes:
        probe_budget = common.Budget(min(probe_cap, max(1.0, budget.seconds)))
        lengths = tuple(probe_grid.get(probe, grid))
        means = []
        for length in lengths:
            if not budget.ok() or not probe_budget.ok():
                out["g5.ruler.partial"] = 1.0
                break
            accs = []
            for i in range(n_items):
                seed = common.task_seed(secret, f"g5/ruler/{probe}/{length}/{i}")
                make = _maker(probe, seed, rows)
                try:
                    shot = _shot(
                        tok, _maker(probe, seed + 1, rows), _shot_budget(length), seed + 1
                    )
                    inst, prompt, n_prompt, intact = _build(
                        tok, make, length, shot, seed
                    )
                    shots.append(1.0 if shot else 0.0)
                    item_accs, item_nlls = _score_slots(model, ctx, prompt, inst["slots"])
                except Exception:  # noqa: BLE001 — e.g. forward OOM at this length
                    continue
                if not item_accs:
                    continue
                misses.append(0.0 if intact else 1.0)
                depths.extend(toklen.depths(prompt, inst["evidence"]))
                len_errs.setdefault(length, []).append(abs(n_prompt - length) / length)
                accs.extend(item_accs)
                nlls.extend(item_nlls)
                value = sum(item_accs) / len(item_accs)
                # Cluster bootstrap mean (trace already recorded per-slot above).
                common.record(ctx, "g5.ruler.item.acc", f"{probe}@{length}", value)
            value = common.mean(accs)
            common.emit(out, f"g5.ruler.{probe}.L{length}.acc", value)
            if value is not None:
                per_len.setdefault(length, []).append(value)
                means.append(value)
        value = common.mean(means)
        common.emit(out, f"g5.ruler.{probe}.acc", value)
        if value is not None:
            probe_means.append(value)

    for length in sorted(per_len):
        common.emit(out, f"g5.ruler.L{length}.acc", common.mean(per_len[length]))
        common.emit(out, f"g5.ruler.L{length}.len_err", common.mean(len_errs.get(length)))
    common.emit(out, "g5.ruler.acc", common.mean(probe_means))
    common.emit(out, "g5.ruler.mean_nll", common.mean(nlls))
    common.emit(out, "g5.ruler.evidence_miss", common.mean(misses))
    # Mean evidence depth: ~0.5 means the answer spans really are spread
    # through the context instead of sitting near the prompt head.
    common.emit(out, "g5.ruler.mean_depth", common.mean(depths))
    common.emit(out, "g5.ruler.shot_rate", common.mean(shots))
    return out
