<!-- protocol_version: 1 -->

# Prism challenge — HTTP script submit

**challenge_id:** `prism`  
**scoring_version:** `2` live (bpb-only; LLM review is an anti-cheat gate, not a grader). **v3 (opt-in, shadow-by-default):** composite scoring runs alongside — your run is also measured on the G1–G8 battery; see *v3 scoring* below.  
**recipe_version:** `1.4.0` (miner-chosen tokenizer; G5 = RULER + BABILong + natural docs, **pretrain-only**)  
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../PRISM.md`](../PRISM.md), recipe [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md).

## What you submit

A **ZIP** (preferred) containing two Python scripts under the official recipe
contract, or JSON with the same fields / `zip_base64`:

- `architecture.py`
- `training.py`

Since recipe **1.3.0** you may instead submit a **source-tree ZIP**: the two
seam files plus optional extras — a `prism.toml` manifest (entry point),
`count_params.py`, a `kernels/` directory of custom ops implementing
`KERNEL_INTERFACE.md` (pure Python + torch; no prebuilt binaries, no
`ctypes`, no I/O or threads — intake scans for banned patterns and the
in-pod **cheatguard** AST audit re-checks them), and a `vendor.lock`. Trees
with `kernels/` are eligible for the 2×2 **attribution** decomposition
(`POST /v1/submissions/{id}/attribution`) with a hidden-shape correctness
gate on kernel-swapped cells.

Models must stay **≤ 350M parameters** after `build_model` — since 1.3.0 a
breach is a **terminal Score(0)** (`CAP_EXCEEDED`), not a retryable failure.

**The tokenizer is yours (no hardcoded GPT-2).** The harness resolves one
tokenizer per run and injects it as `ctx["tokenizer"]`, with its vocab at
`ctx["vocab_size"]` — size your embedding/head from that key. Declare yours by
exporting `build_tokenizer(ctx)` from `architecture.py`, beside `build_model`
(it must live there: the eval phase imports that module only, and a hook found
in `training.py` is rejected instead of silently falling back). Declare nothing
and you get the pinned `gpt2` fallback, exactly like earlier submissions.

```python
# architecture.py
def build_tokenizer(ctx):
    """Anything offline: train a BPE on ctx["dataset_path"], wrap a vendored
    implementation, or hand-roll a byte-level tokenizer. Must satisfy:

        tok(text, add_special_tokens=False)["input_ids"] -> list[int]
        tok.decode(ids) -> str            # roundtrips plain ASCII
        len(tok) or tok.vocab_size -> int # 256 .. 262144
        tok.eos_token_id -> int | None
    """
```

Your pod has **no network** (`unshare --net`), so `from_pretrained("<hub id>")`
inside your code fails closed — build the tokenizer from the pinned shard or
from files in your own submission. The harness validates it (vocab bounds,
probe ids in range, encode/decode roundtrip) and fingerprints it: the eval
phase re-resolves it and refuses to score a run whose tokenizer does not
reconstruct identically, so `build_tokenizer` must be deterministic. Shipping
raw `tokenizer/` files in a source-tree ZIP is supported: the whole validated
tree (kernels, helpers, `tokenizer/`) is staged under `submission/` on the
pod (≤ 12 tokenizer files, ≤ 8 MiB total — enough for a real HF
`tokenizer.json`). Fairness note:
different vocabs change tokenization, not the unit — `bits_per_byte` (bits over
UTF-8 bytes) is the tokenizer-neutral anchor, while the legacy `bpb` key is
bits per *token* and only comparable at equal tokenizers.

**Telemetry hooks (required, recipe ≥ 1.1.0).** Your `training.py` MUST import
the harness-provided `prism_telemetry` module and call
`prism_telemetry.report(loss=..., step=..., ...)` during training plus
`prism_telemetry.finish_evaluation()` to end the eval early. Missing hooks
are a hard contract violation — the review fails the submission
(`missing_telemetry_hooks`, zero score, terminal). See the baseline for the
exact pattern (it includes an offline fallback stub you can copy).

**Training-only entries (architecture competition, recipe ≥ 1.2.0).** To
compete on an already-published architecture, submit `training.py` +
`arch_id` (JSON field, or `X-Prism-Arch-Id` header with a `training.py`-only
ZIP). Do **not** include `architecture.py` — the source is pulled from the
registry. Published archs and their best bpb: `GET /v1/architectures`.

Evaluation runs on **miner-funded** Lium GPU pods (you pay the rent). Master
still operates the pod over SSH; you do **not** deploy a miner CVM. CI uses
`SimLiumBackend` and does not need a key.

## Pay for your own GPU (required on live)

Create a [Lium](https://lium.io) account, fund it, and pass your API key on
every live submit:

```http
X-Lium-Api-Key: <your Lium API key>
```

The key is held only in master memory for that submission (never stored in the
DB, never logged). Missing key on live → `400 missing_lium_api_key`. Cost
guardrails (`max_price_per_hour`, lifetime) still apply so a bad key cannot
rent unbounded SKUs through the orchestrator.

## Submit

```bash
# ZIP via gateway (preferred)
curl -sS -X POST "$BASE_GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: <64 lowercase hex>" \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  --data-binary @submission.zip

# JSON sources (local/CI convenience)
curl -sS -X POST "$BASE_GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  -d @submission.json

# Local / direct
curl -sS -X POST "http://127.0.0.1:28092/v1/submissions" \
  -H 'content-type: application/json' \
  -H "X-Lium-Api-Key: $LIUM_API_KEY" \
  -d @submission.json
```

Inspect recipe pins before coding:

```bash
curl -sS "$BASE_GATEWAY/challenge/prism/v1/recipe"
curl -sS "$BASE_GATEWAY/challenge/prism/v1/recipe/baseline"
```

Live production (recipe **1.2.0**) advertises `train_rows: 2048`,
`val_rows: 256`, `train_hours_cap: 6.0`, `max_train_steps: 20000`,
`max_params: 350000000`, and `pin_hex` (sha over version + caps + dataset +
harness). The sealed baseline only trains on the 2048-row cut (~2M GPT-2
tokens) and scores poorly by design; competitive entries may stream the full
pinned FineWeb-Edu shard for up to 6h. Site chart labels that show “~2.6B
tokens · single pass” were **observed leader telemetry**, not a fixed recipe
quota — trust `/v1/recipe`, not the chart meta line.

`POST /v1/submissions` is idempotent by `submission_id`.

## Submission gating (1-max)

- Your hotkey must be **registered on the subnet** (metagraph). Unknown hotkey
  → `403 hotkey_not_in_metagraph`; a fresh registration may lag the snapshot
  (`503 metagraph_unavailable` → retry shortly).
- **One accepted architecture submission per hotkey.** While yours is
  `registered` / `blocked` / `rejected`, a *different* architecture submission
  gets `409 submission_gated`. Re-POSTing the **identical** sources is always
  safe (idempotent `200 already-queued`).
- **Training-only entries are separate**: one accepted entry per
  `(hotkey, arch_id)`, same retry rules — you may train on many published
  archs, one script per arch.
- If your hotkey **leaves the metagraph**, the watcher reopens your slot(s)
  automatically — resubmit under your new uid.
- Infra failures (Lium pod, review/similarity/LLM infra) **auto-retry up to 3
  times**; cheat / rejected verdicts are terminal. Manual retry:
  `POST /v1/submissions/{id}/retry`.

## Anti-copy rule (architecture-only)

Copying another miner's `architecture.py` (byte-for-byte or renamed/shuffled)
from an **earlier** submission is terminal `rejected` with zero score — judged
automatically before any GPU time is spent, no appeal. Similarity is judged on
`architecture.py` **only**: reusing a known training loop on your own novel
architecture is fine, and training-only entries on a published arch are never
"copies" by construction. Starting from the published baseline is always
allowed.

## Causal LM contract (banned: non-causal label leak)

Prism scores **next-token** cross-entropy → BPB. Architectures must not let
position `t` read tokens `t+1…` (including the label). Dense sequence mixers —
MLP-Mixer-style `TokenMix` / `t_mix` / `nn.Linear` over the full time axis
after `transpose(1, 2)` — **without** a causal mask (`triu` / `tril` /
`is_causal` / attention mask) are a hard ban (`non_causal_label_leak`,
`Score(0)`, terminal, often caught **before** GPU rent). Channel mixing and
causal attention / causal conv are fine; bidirectional full-sequence mixes
used as a next-token LM are not.

### Precheck before you submit (recommended)

Dry-run the same pre-LLM copy gate **without** burning your 1-max slot or a
GPU eval:

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions/precheck" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

| Field | Meaning |
|-------|---------|
| `similar` | `true` → would hard-reject at intake copy gate |
| `verdict` | `clean` / `copied` / `skipped` (training-only) |
| `matched_against` | Corpus id only (never competitor source) |
| `score` | Similarity in `[0,1]` when compared |
| `quota` | `{ day, used, limit: 3, remaining, identity }` |

**Quota: 3 attempts per coldkey per UTC day** (falls back to hotkey when the
metagraph Owner coldkey is unknown). Rotating hotkeys under the same coldkey
does **not** reset the budget. A 4th call returns `429` /
`precheck_quota_exceeded` with `remaining=0`. Precheck never creates a scored
submission and never rents a Lium pod.

## Scoring (summary)

Final leaf score is pure bits-per-byte (bpb) on the lattice `[0, SCORE_MAX]`.
The shared **agentic** gate (AST + metrics/receipt) hard-zeros `cheat` /
`suspicious`. Cheap LLM similarity hard-zeros `Copied`, and `Suspicious` only
when confidence `≥ 0.9` with non-generic evidence (below that — e.g. 0.7 citing
RMSNorm/SwiGLU/LayerNorm — does **not** wipe your score). Copy/similarity
corpora are **champions only** (current top + historical Score>0 ex-tops) plus
baseline — not every past submission — and still exclude your own prior art
(same hotkey **or** same coldkey). Standard components (RMSNorm, RoPE, SwiGLU,
LayerNorm, gated/parallel residual, …) are **not** plagiarism signals. LLM
quality is coherence-only, not a grader.
Public gallery/leaderboard show champions only.
**Competition:** per epoch you are
credited the max of (a) your own best training result and (b) for each arch you
own, that arch's best result by *any* trainer — architecture owners are rewarded
for architectures people win with. Emission is **winner-take-all**: only the
single highest credit that epoch receives Prism's share (50% of the subnet);
ties break by lexicographically smallest hotkey. Scores first land in the leaf
set emitted at the first chain-epoch boundary **after** your run finalizes (a
long train that crosses epochs is normal — outbox assignment is exactly once).
Positive scores then keep participating in later epochs' competition sets until
a better valid score supersedes them (WTA still collapses to one leaf winner).
The global-best model (sources + `ARTIFACT.json` / checkpoint release) is
published to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism)
`top-model/`. See [`PRISM.md`](../PRISM.md).

## v3 scoring (shadow-by-default)

Recipe ≥ 1.3.0 harnesses run a **two-phase pod flow**: your code trains
(`phase=train`), checkpoints, and only then does the operator stage private
eval assets — the eval phase (`phase=eval`) is a fresh subprocess that runs
the frozen-val bpb plus the **G1–G8 battery**: intrinsic fit (G1),
commonsense/reading (G2), retrieval/recall (G3), reasoning (G4),
long-context (G5), sample efficiency from the train probe curve (G6),
inference efficiency (G7), and training stability/µP (G8). Everything the
battery reports is organizer-measured (**Zone A**, `org.*`) and is computed
inside the harness — your code never emits it.

**G5 is pretrain-only (recipe ≥ 1.4.0).** The long-context group scores a
**base LM**, not an instruction-tuned chat model: completion-style /
few-shot base prompts, short exact-match or multiple-choice logprob —
no chat templates, no free-form summarization, no LLM-as-judge on the
ranked path. Length targets are counted in tokens of **your** tokenizer
(`ctx["tokenizer"]`). Scored keys (group weight 0.15 total):
`org.g5.ruler_acc` (0.35), `org.g5.babilong_acc` (0.25),
`org.g5.natural_mcq_acc` (0.15), `org.g5.helmet_rag_acc` (0.15),
`org.g5.lstar` (0.10). L* is the highest length where pooled
RULER+BABILong accuracy stays ≥ 90% of the shortest-grid accuracy and
≥ 0.25 (else 0). Natural MCQ / HELMET RAG packs are mirrored like G2/G4.

Your `train()` return dict (`train_metrics` in METRICS_JSON v2) is
**Zone B**: participant-reported, displayed-but-labelled, validated at
ingest (scalars/series/histograms under `miner.<group>.<name>`, caps
64 scalars / 16 series / 10k points / 1 MB), and **never scored**. Do not
emit `org.*` keys — that quarantines the report as anti-cheat evidence.
You can also post additional self-reports out-of-band:
`POST /v1/submissions/{id}/zone-b` with a JSON envelope
`{"schema_version": "<recipe version>", "prev_hash": <previous report_hash,
optional>, "metrics": {"miner.<group>.<name>": {"kind": "scalar"|"series"|
"histogram", ...}}}`. Reports chain per submission (`prev_hash` → previous
`report_hash`; omit it for master-chained ingest), are validated against
organizer ground truth (token/step/wall-clock counters, MFU ceiling,
terminal-loss band) and the cross-miner cohort, and land a stored verdict
(`ok` / `flagged` / `quarantined`) — verdicts are evidence, never an
auto-zero. Malformed or over-cap envelopes reject `422` and store nothing.

While `PRISM_SCORING_MODE=shadow` (default) the leaf score stays pure bpb,
bit-identical to v2. After the reference baselines (**Transformer++** and
**hybrid delta** — published in-repo under `crates/prism-recipe/baselines/`)
are measured and the anchor set is pre-registered, governance may flip to
`composite`: group scores are anchor-normalized, gate-filtered
(`g3 ≥ 0.25`, `g8 ≥ 0.5`, budget + CI gates), combined as a weighted
geometric mean, and ranked by the bootstrap lower-confidence bound
(`lattice = round(SCORE_MAX × max(0, C − 1.645·SE))`). Inspect the anchor
registry and pre-registration commits at `GET /v1/anchors` and
`GET /v1/preregistration`; per-run Zone A / Zone B rows at
`GET /v1/submissions/{id}/metrics?zone=a|b`.

## Useful routes

| Route | Use |
|-------|-----|
| `POST /v1/submissions/precheck` | Advisory copy-gate (3/coldkey/UTC day); no submit |
| `GET /v1/status` | Backend mode, epoch, queue |
| `GET /v1/submissions/{id}` | Detail + receipt + scores + composite block (v3) |
| `GET /v1/submissions/{id}/events` | Stage timeline |
| `GET /v1/submissions/{id}/metrics?zone=a\|b` | Zone A battery rows / Zone B self-report chain (v3) |
| `POST /v1/submissions/{id}/zone-b` | Miner Zone B self-report intake: validated + chained + stored (v3) |
| `POST /v1/submissions/{id}/attribution` | 2×2 arch/kernel attribution run plans (v3) |
| `GET /v1/anchors` | v3 anchor-set registry + status |
| `GET /v1/preregistration` | v3 anchor pre-registration hash-commits |
| `GET /v1/architectures` | Published archs + per-arch best bpb |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Miner-reported loss curve / gradients / layer stats (from `prism_telemetry.report`) |
| `GET /v1/jobs` | Active/recent pods (ops) |
| `GET /health` | Liveness |

Emission share for prism is owner-controlled via the trust root. Current split is
`5000` bps prism / `5000` bps design (50/50) — see
[`../runbooks/prism-enable-lium-and-emission.md`](../runbooks/prism-enable-lium-and-emission.md)
and [`../runbooks/design-enable-and-emission.md`](../runbooks/design-enable-and-emission.md).
