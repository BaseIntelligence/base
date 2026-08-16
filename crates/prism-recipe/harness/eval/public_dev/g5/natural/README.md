# G5 natural-document packs — format and fixtures

[`eval/natural_docs.py`](../../../natural_docs.py) scores two slices of
**real** long documents with base-LM metrics only: LongBench-v2 four-way
MCQ (length-normalized logprob over the answer texts) and HELMET RAG
(HELMET's own non-chat few-shot template, bounded greedy decode, substring
exact match). No chat template, no judge, no summarization.

The files in *this* directory are **tiny synthetic fixtures**, not dataset
content. They exist so the code path runs and is smoke-testable with no
staged assets, exactly as the hand-authored `g2/<task>.jsonl` anchors do.
Every row carries `meta.fixture: true`. The real pools are built by the
operator and are **never committed**.

## Row format (tokenizer-agnostic by construction)

Packs store raw text plus choices plus gold. All token math — length
measurement, over-length middle-truncation — happens on-pod against the
tokenizer the miner submitted (`ctx["tokenizer"]`), so one pack serves
every submission regardless of vocabulary.

`natural_mcq.jsonl`:

```json
{"id": "...", "slice": "natural_mcq", "cluster": "<bootstrap unit>",
 "question": "...", "choices": ["...", "...", "...", "..."], "gold": 0,
 "context": "<raw document text>",
 "meta": {"chars": 0, "source_chars": 0, "truncated": false}}
```

`helmet_rag.jsonl` and `helmet_rag.demos.jsonl`:

```json
{"id": "...", "slice": "helmet_rag", "cluster": "nq|triviaqa|hotpotqa|popqa",
 "question": "...", "answers": ["<alias>", "..."],
 "passages": [{"title": "...", "text": "..."}],
 "meta": {"dataset": "nq", "k": 3, "chars": 0}}
```

```json
{"id": "...", "cluster": "nq", "question": "...", "answer": "...",
 "passages": [{"title": "...", "text": "..."}]}
```

`cluster` is the unit of randomization for the clustered bootstrap in the
Rust composite: LongBench-v2 sub-domain for MCQ, source corpus for RAG.

## Operator pool layout (private, staged post-train)

Built by `cargo run -p xtask -- natural-pack --out "$PRISM_EVAL_ASSETS_DIR"`,
which writes into the operator's eval-assets dir so the existing
`prism-lium` post-train staging carries it to the pod unchanged:

```
$PRISM_EVAL_ASSETS_DIR/g5/natural/
  natural_mcq.jsonl              scored pool
  helmet_rag.jsonl
  helmet_rag.demos.jsonl
  public_dev/natural_mcq.jsonl   disjoint mirror pool (contamination gap)
  public_dev/helmet_rag.jsonl
  public_dev/helmet_rag.demos.jsonl
  manifest.json                  source URL + revision + SHA-256 + license
                                 + counts + length histogram + pack hash
```

The pool is a superset: which rows a run scores, the MCQ choice order, the
few-shot demos and the ranking distractors are all redrawn from
`PRISM_EVAL_SECRET_SEED` through `common.task_seed` at eval time. A leaked
pack therefore still does not reveal which items or which gold indices any
particular run used.

`public_dev/` is the mirror side of the contamination gap — same pinned
revision, same construction, disjoint rows. With no staged mirror the run
is its own mirror (gap 0, honestly labelled), the same degenerate case
`rollup.build_mirrors` already reports for G2.

## Emitted metrics

`natural_docs.run(model, ctx, budget=None)` returns a flat dict; the
`L<bucket>` keys follow `g5_longctx.py`'s grid convention so
`rollup.flatten_metrics` folds them into the canonical keys through its
existing G5 branch.

| Raw key | Canonical | Meaning |
|---------|-----------|---------|
| `g5.natural_mcq.L{4096,8192,16384}.acc` | `org.g5.natural_mcq_acc` | length-normalized logprob accuracy per token bucket |
| `g5.helmet_rag.L{...}.acc` | `org.g5.helmet_rag_acc` | substring exact match per token bucket |
| `g5.natural_mcq.acc`, `g5.helmet_rag.acc` | — | pooled means (debug) |
| `g5.helmet_rag.rank_acc` | — | closed-set logprob ranking companion; smoother than EM at ≤1B |
| `g5.natural_mcq.mean_nll`, `g5.helmet_rag.gold_nll` | — | mean per-token NLL of the gold text |
| `g5.natural_mcq.n`, `g5.helmet_rag.n`, `g5.helmet_rag.rank_n` | — | items actually scored |
| `g5.natural.pool_rows.<slice>` | — | rows found in the resolved pool |
| `g5.natural.staged` | — | 1.0 only when every scored pool came from the operator's staging |
| `g5.natural.partial` | — | present at 1.0 when the budget cut the slice short |

`natural_docs.mirror_pairs(model, ctx, budget=None, cap=None)` returns
`rollup.build_mirrors`-shaped pairs (`group`/`metric`/`public`/`mirror`)
for `org.g5.natural_mcq_acc` and `org.g5.helmet_rag_acc`.
