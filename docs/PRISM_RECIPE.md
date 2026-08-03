# PRISM recipe v1.0.1 — `prism-recipe-v1`

The official execution contract every miner submission is verified inside.
Miners ship **two scripts only** (`architecture.py` + `training.py`); the
harness and data pin are operator-owned. No third source file, no offline
weights, no network reach at pod runtime beyond the pinned dataset pull.

## Contract

```python
# architecture.py
def build_model(ctx):
    """Return a model given the recipe context (devices, dims, seed)."""

# training.py
def train(model, ctx):
    """Train the model; must respect ctx.budget():
    budget.max_steps <= 20000 and budget.max_seconds <= 21600 (6h train)."""
```

The pod runs [`prism_harness.py`](../crates/prism-recipe/harness/prism_harness.py),
which imports both scripts, downloads the **pinned** fineweb-edu shard,
verifies its SHA-256, times the run, and reports `METRICS_JSON` (bpb,
tokens_seen, steps, wall-clock seconds, gpu type) back to the master.

## Pinned dataset

| Field | Value |
|-------|-------|
| Ref | `HuggingFaceFW/fineweb-edu@sample/10BT` |
| URL | `…/resolve/main/sample/10BT/010_00000.parquet` |
| Bytes | 2 152 798 864 |
| SHA-256 | `e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6` |

The hash is a build-time pin in `prism-recipe` (env `PRISM_DATASET_SHA256`
may override in deployments). The pod harness re-verifies it on the file it
actually fetched; a mismatch ends the eval as `ChallengeInternal` — never a
score.

## Budget & caps

| Cap | Value |
|-----|-------|
| Train wall clock | 6.0 h per submission |
| Pod lifetime | 7.0 h (train + bootstrap margin) |
| Hard step cap | 20 000 (config may only lower) |
| Source size | 128 KiB per script |

## Recipe pin

`recipe_pin_hex()` = SHA-256 over the versioned descriptor (URL, dataset pin,
budget, caps, harness bytes, recipe version) — surfaced on `GET /v1/recipe`
and `GET /v1/status`. Any change to this file's parameters **must** bump
`prism-recipe`'s version string so old leaves stay unambiguous.

## Context-window rule (harness)

Architectures may self-truncate their context (the baseline applies
`block=512` internally at inference). At scoring time the harness aligns the
target window to the logits the model actually produced
(`tgt = ids[:, 1:][:, -logits.shape[1]:]`) so long validation texts never
fault against shorter model windows. Miners still train and score against
the same frozen texts; the rule only protects against an architecture's own
context clamp.

## Scoring v2

`final_score = 0.7·bpb_score + 0.3·llm_quality_score`
(`PRISM_LLM_WEIGHT` tunes the 0.3 at deploy). Anti-copy cups:

- similarity verdict `Copied` → hard **Score 0**
- similarity verdict `Suspicious` → hard **Score 0** until reviewed
- harness/antipattern failure → `ChallengeInternal` maps to `NoScore` reason

## Anti-copy review

Each submission faces an LLM review on the master (`OpenRouter` when the key
file `/run/base/openrouter/api_key` exists, else the deterministic
`SimReviewer`) over its source vs. the recipe **baseline plus every earlier
submission** (`prism_submission` history, capped at the 6 most recent
records). Verdicts: `Original` / `Suspicious` / `Copied`, with a similarity
score and evidence line — all stored append-only in `prism_stage_event`.
