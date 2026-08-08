# PRISM recipe v1.0.2 — `prism-recipe-v1`

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

## Miner telemetry hooks (required since recipe 1.1.0)

The harness registers a `prism_telemetry` module before miner code loads
(also at `ctx["telemetry"]`). `training.py` MUST:

```python
import prism_telemetry

prism_telemetry.report(loss=..., step=..., grad_norm=..., layer_stats=...)  # every N steps
prism_telemetry.finish_evaluation()  # optional early stop: score the model as-is
```

The harness captures the series into `METRICS_JSON.telemetry.loss_series`
(persisted master-side in `prism_telemetry` and surfaced on the site).
`finish_evaluation()` raises a `BaseException` through `train()`, so miner
`except Exception` blocks cannot swallow it; without it the eval ends when
`train()` returns or the wall-clock cap fires. **Missing hooks are a hard
contract violation**: review fails the submission
(`missing_telemetry_hooks` cheat code, zero score, terminal — no retry).

## Training-only submissions (recipe 1.2.0)

Instead of shipping both scripts, a miner may submit `training.py` +
`arch_id` referencing an already-**published** architecture (see
[`PRISM.md`](PRISM.md) § Architecture registry + competition). The master
pulls `architecture.py` from the registry; the same harness contract applies
unchanged. Published archs: `GET /v1/architectures`.

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
| Model parameters | ≤ **350 000 000** after `build_model` (`MAX_PARAMS`) |
| `train_rows` (descriptor) | **2048** — baseline / default cut advertised on `GET /v1/recipe` |
| `val_rows` | **256** — frozen val cut scored by the harness (not miner-chosen) |

### What `train_rows` means (and what it does not)

`train_rows: 2048` is the **baseline cut** and the value injected into
`ctx["train_rows"]`. The sealed baseline (`training.py`) reads that many texts
from the pinned parquet (~2M GPT-2 tokens for that slice — **not** billions).

Egalitarian constraints are the **pinned shard + seed + wall/step/param caps**.
The harness hands miners `ctx["dataset_path"]` to the **full** verified
parquet; competitive `training.py` may stream or multi-pass that shard until
the 6h / 20k-step guard fires. Token throughput therefore depends on the miner
loop and the rented GPU — a ~6h RTX 5090 run can report on the order of
**~2.6B** tokens in telemetry. That figure is **observed throughput**, not a
recipe-published “2.6B token window.”

Do not treat the marketing site’s loss-chart axis (or a leader’s telemetry
peak) as the recipe contract — always trust `GET /v1/recipe` + this doc.

**Harness note (follow-up, do not hot-fix mid-flight):** `METRICS_JSON.tokens_seen`
currently echoes `TRAIN_ROWS` (2048) even when telemetry `layer_stats.tokens`
shows billions. Changing that field would alter the recipe pin (harness bytes
are hashed) — coordinate a version bump if/when fixing it.

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

## Scoring v2 (bpb-only)

`final_score = score_from_bpb(measured_bpb)` — the integer lattice is
**pure bpb**. The LLM/coherence review is **not a grader**: it only verifies
that the miner is not cheating and that the submission is coherent. Its
verdict, quality notes and issues are kept as audit records
(`prism_stage_event`), never added nor subtracted from the score. The
review still gates eligibility:

- similarity verdict `Copied` → hard **Score 0**
- similarity verdict `Suspicious` → **Score 0** only when `score ≥ 0.9` and
  evidence is not generic-trope-only (else no wipe; agentic remains the
  structural judge)
- harness/antipattern failure → `ChallengeInternal` maps to `NoScore` reason

## Anti-copy review

A **pre-LLM copy gate** first compares the candidate `architecture.py`
against **champion** submissions (Score>0 current top + historical ex-tops)
from **other miners** (byte hash + AST fingerprints, `created_at` ordered;
same-`miner_hotkey` and same-`miner_coldkey` prior art excluded): a byte/AST
copy of a strictly-earlier champion architecture is terminal `rejected` with
zero score — no pod time, no LLM spend. The baseline is exempt (everyone may
start from it); created_at ties fall through to the LLM path below.

Each remaining submission then faces an LLM review on the master
(`OpenRouter` when the key file `/run/base/openrouter/api_key` exists, else
the deterministic `SimReviewer`) over its **architecture only** vs. the
recipe **baseline plus champions** (capped; same hotkey/coldkey exclusion).
Since similarity v2/v3, `training.py` is exempt from both candidate and
corpus: the same training script on two different architectures is
legitimate. Verdicts: `Original` / `Suspicious` / `Copied`, with a similarity
score and evidence line — all stored append-only in `prism_stage_event`.
Generic modern-LM components (RMSNorm, RoPE, SwiGLU, …) must not appear as
copy evidence; parsers coerce those false positives to `Original`.
