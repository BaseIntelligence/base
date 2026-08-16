"""G5 natural-document slices — LongBench-v2 MCQ + HELMET RAG (base LM only).

Library module, **not** a battery group module: `eval.discover()` keys on
the `gN` prefix and keeps the first module per group, so a second
`g5_*.py` file would be silently dropped (`g5_longctx` sorts first). The
name `natural_docs` therefore has no group prefix at all, and the G5
orchestrator wires it in explicitly:

    from . import natural_docs
    out.update(natural_docs.run(model, ctx, budget))   # in g5_longctx.run

    pairs += natural_docs.mirror_pairs(model, ctx)     # in rollup.build_mirrors

Per-slice accuracy is reported as `g5.<slice>.L<bucket>.acc`, the shape
`rollup.flatten_metrics` already folds into `org.g5.<slice>_acc` through
its `_G5_TASK_ORG` table, and per-item clusters go to `g5.item.acc` under
`<slice>@<cluster>` like the procedural G5 tasks — so integration is a
table entry, not new aggregation code.

Two slices, both scored the way a *pretrained* LM can be scored — no chat
template, no instruction turns, no judge, no long free-form generation:

| Slice | Items | Scoring |
|-------|-------|---------|
| `natural_mcq` | LongBench-v2 4-way MCQ over real documents | length-normalized logprob over the answer texts (`common.score_choices`, the OLMES `acc_norm` path G2 uses) |
| `helmet_rag`  | HELMET RAG (kilt nq/triviaqa/hotpotqa/popqa) | HELMET's own **non-chat** few-shot template + bounded greedy decode + substring EM, plus a closed-set logprob ranking companion |

Assets are raw text + choices + gold — **tokenizer-agnostic**. Every
token-exact operation runs on-pod through the tokenizer contract
(`common.tokenizer_of` / `encode` / `decode` / `token_len`), i.e. against
the tokenizer the miner submitted. The one piece of token math this module
owns is `_fit_middle`: LongBench's official over-length rule, keep the
first and last half of the budget and drop the middle. `common`'s own
`truncate_tokens` keeps a prefix instead, which would throw away the end of
the document, and `fit_to_tokens` grows a context rather than shrinking a
real one.

Which pool rows are scored, the MCQ choice order, the few-shot demos and
the ranking distractors are all redrawn from `PRISM_EVAL_SECRET_SEED` (via
`common.task_seed`) at eval time, so a leaked pack still does not reveal
which items or which gold indices a run will use.

Pack layout under the operator eval-assets dir (staged post-train, never
visible to the train phase; built by `cargo run -p xtask -- natural-pack`):

    g5/natural/natural_mcq.jsonl                 scored pool (private)
    g5/natural/helmet_rag.jsonl
    g5/natural/helmet_rag.demos.jsonl            few-shot demo pool
    g5/natural/public_dev/natural_mcq.jsonl      disjoint mirror pool
    g5/natural/public_dev/helmet_rag.jsonl
    g5/natural/public_dev/helmet_rag.demos.jsonl
    g5/natural/manifest.json                     provenance (not scored)

`eval/public_dev/g5/natural/` ships tiny synthetic fixtures in the same
format so the code path runs (and is smoke-testable) with no staged
assets; the real pools are never committed.
"""

import random
import re

from . import common

MCQ = "natural_mcq"
RAG = "helmet_rag"
SLICES = (MCQ, RAG)
PACK_DIR = "g5/natural"
MIRROR_DIR = "g5/natural/public_dev"

# Token budget per slice (miner-tokenizer tokens) and the reporting grid.
MAX_TOKENS = 16384
_BUCKETS = (4096, 8192, 16384)
# HELMET RAG answers are short entities; `stop_new_line` upstream.
_MAX_NEW_TOKENS = 8
_SHOTS = 2
_DISTRACTORS = 3

# HELMET's frozen non-chat RAG template (`configs/rag_short.yaml`,
# `use_chat_template: false`) — reproduced verbatim so the protocol is the
# upstream one and not a local paraphrase.
_RAG_INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the "
    "question. Write your answer in the following format:\nAnswer: [answer]\n\n"
)
_PASSAGE = "Document (Title: {title}): {text}"


# ------------------------------------------------------ token-exact cutting


def _fit_middle(tok, text, max_tokens):
    """(text, n_tokens) truncated to `max_tokens` by dropping the middle.

    LongBench's official over-length rule (`pred.py`): decode the first and
    last half of the budget and concatenate. `common.truncate_tokens` is not
    a substitute — it keeps a prefix, and the tail of a LongBench document
    is where a good part of the evidence lives.

    Re-encoding a decoded cut can come back a few tokens longer than the
    slice it came from (BPE merges across the seam), so the length is
    re-measured and the window shrunk until it holds.
    """
    ids = common.encode(tok, text)
    n = len(ids)
    if n <= max_tokens:
        return text, n
    budget, cut, got = max_tokens, text, n
    for _ in range(4):
        half = budget // 2
        cut = common.decode(tok, ids[:half]) + common.decode(tok, ids[n - (budget - half) :])
        got = common.token_len(tok, cut)
        if got <= max_tokens:
            break
        budget -= max(1, (got - max_tokens) * 2)
    return cut, got


# ------------------------------------------------------------- pack access


def _pool(ctx, rel):
    """(rows, staged) for one pack file.

    `staged` is True only when the file resolved inside the operator's
    post-train assets dir — the public_dev fixtures are a fallback, not a
    private pack, and the difference is reported.
    """
    path = common.assets_path(ctx, rel)
    if not path:
        return [], False
    base = ctx.get("eval_assets_dir")
    staged = bool(base) and str(path).startswith(str(base))
    return common.load_jsonl(path), staged


def _rng(secret, task_id):
    return random.Random(common.task_seed(secret, task_id))


def _sample(rows, n, secret, task_id):
    """Seeded held-out draw of `n` rows (the run's item set is secret)."""
    idx = list(range(len(rows)))
    _rng(secret, task_id).shuffle(idx)
    return [rows[i] for i in idx[:n]]


def _bucket(n_tokens):
    """Largest reporting grid point the item actually reached.

    Reported as `g5.<slice>.L<bucket>.acc`, the same shape `g5_longctx.py`
    uses for its procedural grid, so `rollup.flatten_metrics` folds these
    into `org.g5.<slice>_acc` through its existing G5 branch.
    """
    reached = [b for b in _BUCKETS if n_tokens >= b]
    return reached[-1] if reached else _BUCKETS[0]


# ---------------------------------------------------------- prompt building


def _mcq_prompt(tok, row, secret):
    """(prompt, choices, gold, n_tokens) for one MCQ row.

    Choice order is redrawn from the secret seed, so the gold index stored
    in the pack is never the index scored.
    """
    choices = [str(c) for c in row.get("choices") or []]
    gold = int(row.get("gold", -1))
    if len(choices) < 2 or not 0 <= gold < len(choices):
        return None
    order = list(range(len(choices)))
    _rng(secret, f"g5/natural/mcq/choices/{row.get('id')}").shuffle(order)
    scored = [" " + choices[i].strip() for i in order]
    tail = f"\n\nQuestion: {str(row.get('question', '')).strip()}\nAnswer:"
    # `score_choices` forwards prompt + one answer, so the reserve is the
    # question plus the *longest* answer — measured, not assumed, because
    # the miner's tokenizer decides what those cost.
    reserve = common.token_len(tok, tail) + max(common.token_len(tok, c) for c in scored)
    body, _ = _fit_middle(tok, str(row.get("context", "")), max(256, MAX_TOKENS - reserve))
    prompt = body + tail
    return prompt, scored, order.index(gold), common.token_len(tok, prompt)


def _documents(row):
    passages = row.get("passages") or []
    return "\n\n".join(
        _PASSAGE.format(title=str(p.get("title", "")), text=str(p.get("text", "")))
        for p in passages
    )


def _rag_prompt(tok, row, demos):
    """(prompt, n_tokens) — HELMET's non-chat few-shot RAG completion."""
    shots = "".join(
        f"{_documents(d)}\n\nQuestion: {str(d.get('question', '')).strip()}\n"
        f"Answer: {str(d.get('answer', '')).strip()}\n\n"
        for d in demos
    )
    head = _RAG_INSTRUCTION + shots
    tail = f"\n\nQuestion: {str(row.get('question', '')).strip()}\nAnswer:"
    room = MAX_TOKENS - common.token_len(tok, head + tail) - _MAX_NEW_TOKENS
    body, _ = _fit_middle(tok, _documents(row), max(256, room))
    prompt = head + body + tail
    return prompt, common.token_len(tok, prompt)


# ------------------------------------------------------------------ scoring


def _logits(out):
    if hasattr(out, "logits"):
        return out.logits
    return out[0] if isinstance(out, (tuple, list)) else out


def _greedy_line(model, tok, device, prompt):
    """Bounded greedy continuation: <= `_MAX_NEW_TOKENS`, first line only.

    No KV cache — the miner's model is an arbitrary `nn.Module` and the
    harness contract only promises `model(ids)`, so each step re-forwards,
    the same trade `common.continuation_logprob` already makes. That is why
    `_MAX_NEW_TOKENS` is small and a newline stops the loop early.
    """
    import torch

    ids = common.encode(tok, prompt)
    grown, text = [], ""
    for _ in range(_MAX_NEW_TOKENS):
        batch = torch.tensor([ids + grown], dtype=torch.long, device=device)
        with torch.no_grad():
            step = _logits(model(batch))
        grown.append(int(step[0, -1].float().argmax().item()))
        text = common.decode(tok, grown)
        if "\n" in text:
            break
    return text.split("\n")[0]


def _normalize(text):
    """SQuAD-style normalization for HELMET's substring exact match."""
    text = re.sub(r"\b(a|an|the)\b", " ", str(text).lower())
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def _substring_em(prediction, answers):
    pred = _normalize(prediction)
    if not pred:
        return 0.0
    return 1.0 if any(_normalize(a) and _normalize(a) in pred for a in answers) else 0.0


def _mcq_series(model, ctx, rows, secret, budget, out, prefix):
    """Mean acc + per-cluster values over MCQ rows; emits detail into `out`."""
    tok, device = common.tokenizer_of(ctx), ctx["device"]
    accs, nlls, clusters, per_bucket = [], [], {}, {}
    for row in rows:
        if not budget.ok():
            out["g5.natural.partial"] = 1.0
            break
        built = _mcq_prompt(tok, row, secret)
        if built is None:
            continue
        prompt, choices, gold, n_tokens = built
        cluster = str(row.get("cluster") or "unknown")
        try:
            if prefix:
                acc, nll = common.score_and_record(
                    ctx,
                    "g5.item.acc",
                    f"{MCQ}@{cluster}",
                    model,
                    tok,
                    device,
                    prompt,
                    choices,
                    gold,
                    group="g5",
                    task=MCQ,
                )
            else:
                acc, nll = common.score_choices(model, tok, device, prompt, choices, gold)
        except Exception:  # noqa: BLE001 — one OOM item never kills the slice
            continue
        accs.append(acc)
        nlls.append(nll)
        clusters.setdefault(cluster, []).append(acc)
        per_bucket.setdefault(_bucket(n_tokens), []).append(acc)
    if prefix:
        for b, vals in sorted(per_bucket.items()):
            common.emit(out, f"{prefix}.L{b}.acc", common.mean(vals))
        common.emit(out, f"{prefix}.mean_nll", common.mean(nlls))
        out[f"{prefix}.n"] = float(len(accs))
    if not accs:
        return None
    return {
        "value": sum(accs) / len(accs),
        "clusters": {c: sum(v) / len(v) for c, v in clusters.items()},
    }


def _rag_series(model, ctx, rows, demo_pool, secret, budget, out, prefix):
    """Mean substring-EM + per-cluster values over RAG rows.

    Each item is first scored by closed-set logprob ranking (cheap, smooth
    at 100M–1B) and then, while the budget allows, by bounded greedy
    decode + substring EM — the upstream HELMET RAG metric.
    """
    tok, device = common.tokenizer_of(ctx), ctx["device"]
    ems, ranks, nlls, clusters, per_bucket = [], [], [], {}, {}
    for row in rows:
        if not budget.ok():
            out["g5.natural.partial"] = 1.0
            break
        answers = [str(a) for a in row.get("answers") or []]
        if not answers:
            continue
        cluster = str(row.get("cluster") or "unknown")
        rid = row.get("id")
        demos = _sample(
            [
                d
                for d in demo_pool
                if str(d.get("cluster")) == cluster and d.get("question") != row.get("question")
            ],
            _SHOTS,
            secret,
            f"g5/natural/rag/demos/{rid}",
        )
        prompt, n_tokens = _rag_prompt(tok, row, demos)
        gold_forms = {_normalize(a) for a in answers}
        pool = dict.fromkeys(
            str((r.get("answers") or [""])[0])
            for r in rows
            if r is not row and str(r.get("cluster")) == cluster
        )
        pool = [a for a in pool if a and _normalize(a) not in gold_forms]
        choices = [answers[0]] + _sample(pool, _DISTRACTORS, secret, f"g5/natural/rag/distract/{rid}")
        order = list(range(len(choices)))
        _rng(secret, f"g5/natural/rag/order/{rid}").shuffle(order)
        scored_choices = [" " + choices[i] for i in order]
        gold_i = order.index(0)
        try:
            detail = common.score_choices_detail(
                model, tok, device, prompt, scored_choices, gold_i
            )
            rank, nll = detail["acc"], detail["gold_nll"]
        except Exception:  # noqa: BLE001
            continue
        # A cluster too small to supply distractors would score 1.0 for
        # free; the gold NLL is still meaningful, the ranking is not.
        if len(choices) > 1:
            ranks.append(rank)
        nlls.append(nll)
        line = None
        if budget.ok():
            try:
                line = _greedy_line(model, tok, device, prompt)
            except Exception:  # noqa: BLE001
                line = None
            if line is not None:
                em = _substring_em(line, answers)
                ems.append(em)
                clusters.setdefault(cluster, []).append(em)
                per_bucket.setdefault(_bucket(n_tokens), []).append(em)
                if prefix:
                    common.record(ctx, "g5.item.acc", f"{RAG}@{cluster}", em)
        if prefix:
            common.record_trace(
                ctx,
                {
                    "kind": "generative",
                    "group": "g5",
                    "task": RAG,
                    "metric": "g5.item.acc",
                    "cluster": f"{RAG}@{cluster}",
                    "prompt": prompt,
                    "choices": scored_choices,
                    "gold": gold_i,
                    "selected": detail["selected"],
                    "generated": line,
                    "gold_answers": answers,
                    "value": ems[-1] if line is not None and ems else rank,
                    "gold_nll": nll,
                    "choice_logprobs": detail["choice_logprobs"],
                    "meta": {"rank_acc": rank},
                },
            )
    if prefix:
        for b, vals in sorted(per_bucket.items()):
            common.emit(out, f"{prefix}.L{b}.acc", common.mean(vals))
        common.emit(out, f"{prefix}.rank_acc", common.mean(ranks))
        common.emit(out, f"{prefix}.gold_nll", common.mean(nlls))
        out[f"{prefix}.rank_n"] = float(len(ranks))
        out[f"{prefix}.n"] = float(len(ems))
    if not ems:
        return None
    return {
        "value": sum(ems) / len(ems),
        "clusters": {c: sum(v) / len(v) for c, v in clusters.items()},
    }


# -------------------------------------------------------------- entrypoints


def item_caps():
    """(mcq_items, rag_items) — tiny under test caps, env-overridable."""
    if common.tiny_caps():
        return 2, 2
    n = common.int_env("PRISM_EVAL_NATURAL_ITEMS", 40)
    return n, max(1, n // 2)


def _budget():
    # Direct-call fallback (g5_longctx passes its own share); derived from
    # the global battery budget so it cannot over-subscribe.
    return common.Budget(
        common.float_env(
            "PRISM_EVAL_G5_NATURAL_BUDGET_S",
            common.group_budget_s("g5") * common.G5_NATURAL_SHARE,
        )
    )


def run(model, ctx, budget=None):
    """Score both natural slices. Returns flat `g5.*` metrics (plain dict).

    `budget` lets the G5 orchestrator share one wall-clock budget; omitted,
    a private one is taken from `PRISM_EVAL_G5_NATURAL_BUDGET_S` (900 s).
    Missing packs are not an error — the slice is simply absent.
    """
    budget = budget or _budget()
    secret = common.resolve_secret_seed(ctx)
    n_mcq, n_rag = item_caps()
    out = {}
    staged = []

    mcq_pool, mcq_staged = _pool(ctx, f"{PACK_DIR}/{MCQ}.jsonl")
    if mcq_pool:
        staged.append(mcq_staged)
        out[f"g5.natural.pool_rows.{MCQ}"] = float(len(mcq_pool))
        rows = _sample(mcq_pool, n_mcq, secret, f"g5/natural/{MCQ}/items")
        series = _mcq_series(model, ctx, rows, secret, budget, out, f"g5.{MCQ}")
        if series:
            common.emit(out, f"g5.{MCQ}.acc", series["value"])

    rag_pool, rag_staged = _pool(ctx, f"{PACK_DIR}/{RAG}.jsonl")
    if rag_pool:
        staged.append(rag_staged)
        out[f"g5.natural.pool_rows.{RAG}"] = float(len(rag_pool))
        demos, _ = _pool(ctx, f"{PACK_DIR}/{RAG}.demos.jsonl")
        rows = _sample(rag_pool, n_rag, secret, f"g5/natural/{RAG}/items")
        series = _rag_series(model, ctx, rows, demos, secret, budget, out, f"g5.{RAG}")
        if series:
            common.emit(out, f"g5.{RAG}.acc", series["value"])

    # 1.0 only when every scored pool came from the operator's post-train
    # staging — a public_dev fixture fallback is reported honestly as 0.0.
    out["g5.natural.staged"] = 1.0 if staged and all(staged) else 0.0
    return out


def mirror_pairs(model, ctx, budget=None, cap=None):
    """Public-vs-private mirror pairs for the contamination gap (step 2.5).

    Same shape and orientation as `rollup.build_mirrors`, so its output can
    be appended to that list unchanged: mirror side = the scored private
    pool, public side = the disjoint `public_dev/` pool built from the same
    pinned revision with the same construction. With no staged mirror
    (public_dev tier) the run is its own mirror — gap 0 by construction and
    honestly labelled, exactly as `build_mirrors` does for G2. Budget and
    per-side cap default to that function's own knobs.
    """
    budget = budget or common.Budget(
        common.float_env("PRISM_EVAL_MIRROR_BUDGET_S", common.group_budget_s("mirror"))
    )
    secret = common.resolve_secret_seed(ctx)
    cap = cap or (2 if common.tiny_caps() else 4)
    sink = {}
    pairs = []
    for slice_name in SLICES:
        if not budget.ok():
            break
        private, _ = _pool(ctx, f"{PACK_DIR}/{slice_name}.jsonl")
        if not private:
            continue
        public, _ = _pool(ctx, f"{MIRROR_DIR}/{slice_name}.jsonl")
        sides = {}
        for side, rows in (("mirror", private), ("public", public or private)):
            drawn = _sample(rows, cap, secret, f"g5/natural/mirror/{slice_name}/{side}")
            if slice_name == MCQ:
                sides[side] = _mcq_series(model, ctx, drawn, secret, budget, sink, None)
            else:
                base = MIRROR_DIR if side == "public" and public else PACK_DIR
                demos, _ = _pool(ctx, f"{base}/{RAG}.demos.jsonl")
                sides[side] = _rag_series(
                    model, ctx, drawn, demos, secret, budget, sink, None
                )
        if sides.get("public") and sides.get("mirror"):
            pairs.append(
                {
                    "group": "g5",
                    "metric": f"org.g5.{slice_name}_acc",
                    "public": sides["public"],
                    "mirror": sides["mirror"],
                }
            )
    return pairs
