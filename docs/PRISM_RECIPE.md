# PRISM recipe v1.0.2 — `prism-recipe-v1` (current harness `RECIPE_VERSION 1.4.0`)

The official execution contract every miner submission is verified inside.
Miners ship **two scripts only** (`architecture.py` + `training.py`) — or,
since recipe **1.3.0**, a **source-tree ZIP** (see *Source-tree
submissions* below); the harness and data pin are operator-owned. No
offline weights, no network reach at pod runtime beyond the pinned dataset
pull. Recipe **1.4.0** makes the tokenizer miner-chosen and replaces the
G5 long-context scored path with community protocols + natural documents
under a **pretrain-only** rule (base LM completion / few-shot; no IFT,
chat templates, or LLM judges on ranked metrics).

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

## Causal next-token contract

Val scoring is next-token CE → BPB on a frozen cut. Architectures that densify
mix across the **full** time axis (MLP-Mixer `TokenMix` / `t_mix` /
`nn.Linear(seq, …)` after `transpose(1, 2)`) without a causal mask let
position `t` read the label at `t+1` — that is a hard cheat
(`non_causal_label_leak`), caught by the pre-pod static screen before Lium
rent. Channel mixers and masked causal attention / causal conv remain allowed.

## Tokenizer (submitted, not imposed)

The tokenizer is **part of the submission**. The harness resolves it once per
phase and hands it to miner code as `ctx["tokenizer"]`, with its vocab at
`ctx["vocab_size"]` (size your embedding from that key, not from a constant).
Resolution order — first match wins, always offline:

| Order | Declaration | Notes |
|-------|-------------|-------|
| 1 | `tokenizer/` files in a source-tree ZIP | staged under `submission/tokenizer/` on the pod and loaded with `AutoTokenizer.from_pretrained(dir, local_files_only=True)`; ≤ 12 files, ≤ 8 MiB total (admits a real ~1.4 MiB HF `tokenizer.json`), extensions `json/txt/model/vocab/merges/bpe` |
| 2 | `def build_tokenizer(ctx)` in `architecture.py` | code hook path. Must sit beside `build_model`: the eval phase imports that module only, so a hook in `training.py` is a hard intake and in-pod error, never a silent fallback. Gets a ctx without harness internals; build/train/wrap whatever you like, offline |
| 3 | pinned fallback `gpt2` | what pre-1.4 submissions already got — a **default**, not a rule. Warmed into the pod HF cache by the parent before the netns child starts |

The miner subprocess runs under `unshare --net`: a tokenizer that would need a
download fails closed with a clear error instead of stalling inside
`transformers`. Never call `from_pretrained("<hub id>")` yourself.

Every resolved tokenizer is validated before your code sees it — callable,
`decode`, vocab in `[256, 262144]`, all probe ids inside that vocab, exact
encode/decode roundtrip on an ASCII probe — and fingerprinted (sha256 over
the ids of a fixed probe corpus). The train phase stores that fingerprint in
the checkpoint; the eval phase re-resolves the tokenizer and refuses to score
a run whose tokenizer does not reconstruct identically. `METRICS_JSON` carries
the resulting spec under `tokenizer`
(`{source, id, class, vocab_size, probe_tokens, fingerprint}`).

Minimal interface a tokenizer must satisfy (a byte-level tokenizer in ~30
lines qualifies):

```python
tok(text, add_special_tokens=False)["input_ids"] -> list[int]
tok.decode(ids) -> str                # roundtrips plain ASCII
len(tok) or tok.vocab_size -> int     # 256 .. 262144
tok.eos_token_id -> int | None        # document separator in the train stream
```

**Fairness.** Different vocabularies change how text is split, not the unit of
comparison: the tokenizer-neutral number is **bits per byte** — total bits
over the UTF-8 bytes of the scored region — reported as `bits_per_byte` in
`METRICS_JSON` and as `g1.bits_per_byte.*` beside every `g1.bpb.*` key in the
battery group view. The scored `bpb` (and the `org.g1.bpb_*` anchor keys)
stay bits *per token* (`CE / ln 2`) so earlier leaves and the anchor set keep
their meaning; they are only comparable across submissions that share a
tokenizer, and promoting the per-byte siblings into the anchor set is an
anchor-recalibration decision, not a harness one. Long-context length targets
are counted in tokens of the submitted tokenizer.

## Training-only submissions (recipe 1.2.0)

Instead of shipping both scripts, a miner may submit `training.py` +
`arch_id` referencing an already-**published** architecture (see
[`PRISM.md`](PRISM.md) § Architecture registry + competition). The master
pulls `architecture.py` from the registry; the same harness contract applies
unchanged. Published archs: `GET /v1/architectures`.

## Source-tree submissions (recipe 1.3.0 — v3)

A miner may submit the full program as a ZIP instead of two scripts:
`zip_base64` in the JSON intake body, or `application/zip` with the
`X-Miner-Hotkey` header (source-tree ZIPs are rejected on the raw-zip path
with a pointer to `zip_base64`, which validates and retains the full tree).

Layout:

```text
prism.toml            # optional manifest: entry = "train.py" (default entry;
                      #   `training.py` keeps the legacy two-script layout valid)
architecture.py       # seam: def build_model(ctx)
train.py              # seam: def train(model, ctx) (or training.py)
count_params.py       # optional static parameter-count check (prints one int)
kernels/<op>.py       # optional custom ops per KERNEL_INTERFACE.md
tokenizer/*.json      # tokenizer files (see Tokenizer above; staged on the pod)
vendor.lock           # optional vendored-dependency hash lock
```

The validated tree is persisted as a content-addressed USTAR blob and
staged onto the Lium pod under `submission/` (tar-over-SSH-stdin; never
base64-in-argv). Seam projections remain in the DB for copy-gate /
similarity; the harness loads the real tree so sibling imports
(`import kernels`) and `tokenizer/` resolve.

Validation at intake (`prism_recipe::zip_submit`): file count / per-file /
total-size budgets (128 files, 4 MiB/file, 16 MiB total), UTF-8 seam
projections (`architecture.py` must define
`build_model(`, the entry must define `train(`), tokenizer declaration rules
(§ *Tokenizer*), and a **banned-pattern scan** (prebuilt binaries, `ctypes`,
network/process/threads escapes, …) —
one shared list with the harness-side `prismlib/cheatguard.py` AST audit,
which re-screens the tree in-pod before train and again post-eval. The
canonical tree sha-256 is recorded; `kernels/` trees are attribution- and
hidden-shape-suite eligible.

## Two-phase pod flow + eval battery (recipe 1.3.0 — v3)

The multi-file harness (`main.py` entrypoint + `prismlib/` modules, miner
code inside an `unshare --net` subprocess) runs two fresh phases:

| Phase | Env | What happens |
|-------|-----|--------------|
| `train` | `PRISM_PHASE=train` | contract checks → `build_model` (**350M param cap**: breach → terminal `CAP_EXCEEDED` payload, `Score(0)`) → seeded train stream (authoritative token counter) → G6 probe curve → checkpoint |
| (gate) | — | parent prints `PHASE_TRAIN_DONE`, then holds on `$PRISM_EVAL_ASSETS_DIR/.ready`; the operator stages private eval assets + the secret seed **post-train only** (fail-closed: no `.ready` → error, never a silent public downgrade) |
| `eval` | `PRISM_PHASE=eval`, `PRISM_EVAL_ASSETS_DIR`, `PRISM_EVAL_SECRET_SEED` (env only, never on disk) | fresh subprocess → frozen-val bpb + the **G1–G8 battery** (`eval/` package: intrinsic, downstream, recall, reasoning, long-context, curve, inference, stability) → `METRICS_JSON` v2 |

**METRICS_JSON v2** (`metrics_version: 2`): every v1 key (`bpb`,
`tokens_seen`, `wall_clock_seconds`, `gpu_type`, `notes`, `val_rows`,
`n_params`, `recipe`, `telemetry`) plus `bits_per_byte` (tokenizer-neutral
frozen-val anchor), `tokenizer` (resolved spec, § *Tokenizer*),
`tokens_seen_source`
(`"train_stream"` | `"legacy"`), `probe_curve` (G6), `train_metrics`
(miner-returned flat scalar dict — the **Zone B** self-report source,
sanitized master-side, never scored), `pod_manifest` (nvidia-smi -q +
netns facts), `netns`, `harness_files_sha256`, and on v3 runs `flow`,
`eval_tier` (`"private"` | `"public_dev"`), `gate`, `battery`, `items`.
Cap breach: `cap_exceeded: true` + `n_params` with the `CAP_EXCEEDED`
terminal line instead of `EVAL_OK`.

**`battery` (v3 composite contract)**: an object with four members —
`groups` (nested per-group debug view `{status, module, metrics}` with
internal `gN.family.tag` keys), `metrics` (the **flat canonical map** the
composite ingests: `org.<group>.<name>` → bare float or
`{value, clusters}` where `clusters` are per-template means — the units of
randomization for the clustered bootstrap; a metric that was never
measured is **absent**, never fabricated), `mirrors` (contamination-gap
pairs `[{group, metric, public, mirror}]` for `g2`/`g4`: the same metric
scored on the public dev-seed/asset family vs the private mirror family;
in the `public_dev` tier no private assets exist so each pair is
degenerate — gap 0, honestly labelled), and `tier` (echoes `eval_tier`).
`eval/rollup.py` is the single reconciliation point between internal
metric names and the anchor set's `org.*` keys
(`crates/prism-recipe/anchors/v0.json`); ingestion
(`prism-eval-store/src/finalize.rs`) requires the flat map and skips the
composite when it is absent (fail-closed in composite mode).

**G5 long-context (recipe ≥ 1.4.0, pretrain-only).** Ranked metrics come
from RULER + BABILong + LongBench-v2 MCQ + HELMET RAG few-shot base —
short EM / choice logprob only. Length grids are in tokens of
`ctx["tokenizer"]` (4k–32k; 64k on RULER `niah_mk`+`vt`). Canonical keys:
`org.g5.ruler_acc` (0.35), `org.g5.babilong_acc` (0.25),
`org.g5.natural_mcq_acc` (0.15), `org.g5.helmet_rag_acc` (0.15),
`org.g5.lstar` (0.10). L* is the highest length L on pooled
RULER+BABILong `L{N}.acc` means with `acc(L) ≥ 0.9×acc(L_min)` and
`acc(L) ≥ 0.25` (else `0`), normalized as `efficiency_log_ratio` over
`[4096, 65536]`. Natural slices participate in the G5 mirror gap. Open-gen
sum/cite, chat, and judge protocols stay out of the ranked path.

## Reference baselines (recipe 1.3.0 — v3 anchors)

Two reference submissions ship in-repo (`crates/prism-recipe/baselines/`,
embedded as `prism_recipe::baselines`): **Transformer++**
(`transformer_pp`: modern GPT at the 350M cap) and **hybrid delta**
(`hybrid_delta`: 3:1 gated delta-net/attention hybrid). Each tree carries
`architecture.py` + `training.py` (contract-satisfying), `count_params.py`
(prints the static parameter count as a single integer), and `NOTES.md`.
They are the reference points the v3 anchor set (`anchors/v0.json`,
currently placeholder) is measured against before any
`PRISM_SCORING_MODE=composite` flip, and the attribution reference family.

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
| Source size | 128 KiB per script (two-script intake); tree budgets per `zip_submit` |
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

Caps are **unchanged** in v3 (350M params, 6h). The parameter-cap breach
semantics changed in 1.3.0: it is a terminal `Score(0)` (`CAP_EXCEEDED`),
not an infra retry.

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
