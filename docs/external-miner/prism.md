<!-- protocol_version: 1 -->

# Prism challenge — HTTP AutoModel patch submit

**challenge_id:** `prism`  
**competition_id:** `prism-v2.1` (`scoring_generation` `21`) — **new competition**. Old recipe `2.0.0` / 1.x scores are dead: they are not rescored and cannot win weights. Until the first eligible 2.1 submission terminates, subnet weights stay **burn** (uid 0 = 100%).  
**scoring_version:** `4` live (equal-weight G2 public-suite accuracies → lattice; LLM review is an anti-cheat gate, not a grader). **v3 harness (default):** every scored run executes the **G1–G8 battery**; the leaf uses G2 benches while `PRISM_SCORING_MODE=benchmarks` (default). Legacy `shadow` = bits/token bpb; `composite` = full G1–G8 lattice when anchors are ready.  
**recipe_version:** `2.1.0` (pinned [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) diff + 4-GPU CUDA 13/TE pod + attested dual cap; legacy 1.x layouts rejected)
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../PRISM.md`](../PRISM.md), recipe [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md).

## What you submit

A **ZIP** (preferred) — or JSON with the same members / `zip_base64` — that is
**not** a free-form `architecture.py` / `training.py` project. Recipe **2.1.0**
accepts only an AutoModel pin id plus your git diff against that pin:

```text
automodel.base          # required — pin id from GET /v1/recipe (live: automodel@v0.5.0)
automodel.patch         # required — unified diff vs that pin (git diff pin...HEAD)
prism.toml              # optional — entry / model-config knobs
```

**Workflow: fork pin → edit → `git diff` → submit**

1. Read the live pin from `GET /v1/recipe` (`automodel_pin_id`,
   `automodel_repo_url`, `automodel_git_commit`, `automodel_content_sha256`).
2. Check out that exact AutoModel commit (or extract the staged archive and
   verify `automodel_content_sha256` matches `/v1/recipe`).
3. Edit under the AutoModel layout — new model modules / configs are allowed;
   trainer / data-path edits get high scrutiny.
4. Produce a unified diff against the pin commit, e.g.
   `git diff <automodel_git_commit> > automodel.patch`.
5. Write `automodel.base` as a single line equal to `automodel_pin_id`, pack
   the ZIP, and `POST /v1/submissions` with your hotkey + **`X-Lium-Api-Key`**.

Models must stay in **850M–1B parameters** (total unique; tied embeddings
once). A 215M pack is **invalid** for recipe 2.1. Miner **model code** (build/train/
eval) runs with **no network** (`unshare --net`) beyond the operator-owned
dataset pull — do not call Hub downloads from `build_model` / `train`.

**Bring your own dependencies (recipe-v10).** The pod image is a complete
CUDA 13 base with PyTorch, a build toolchain (`nvcc`, `ninja`), Transformer
Engine (NVFP4 training), and common accelerators. You may additionally ship
**one** of:

- `requirements.txt` — installed with `pip install -r requirements.txt`
- `pyproject.toml` — installed with `pip install .`

by **adding the file at the repo root in your `automodel.patch`** (or at
the ZIP root on the legacy two-script path). It is installed in a
**network-on install phase before** your model code is sandboxed — so
`flash-attn`, `mamba-ssm`, custom Triton/CUDA kernels, etc. compile and
install, then train/eval run offline. `requirements.txt` wins if you ship
both. Check `GET /v1/recipe` for `pod_image_ref`, `miner_install_supported`,
`miner_deps_members`, and `install_timeout_secs`.

**GPU train contract.** Recipe-v10 1B pods default to **one NVIDIA B200**
(`ctx["gpu_count"]==1`, ~180–192 GB, TE on, mb≥8, world=1). Set
`PRISM_POD_GPU_COUNT=4` for the **four RTX 5090** fallback, or `2`/`8` for
**RTX PRO 6000**. The organizer eval stays on GPU 0.
Training must consume global batches from the harness-owned
`ctx["train_stream"]`, because that stream enforces step/wall/FLOPs caps and
owns token/byte accounting for G6. For DDP/ZeRO/FSDP, keep rank 0 as the
stream owner and scatter/shard each accounted global batch to workers over
the local process group. At 850M–1B, default the example pack to ZeRO-1. Do not let each worker create an independent dataset stream:
v3 rejects a trainer that returns with zero harness-accounted tokens. The
isolated network namespace brings `127.0.0.1` loopback up for rendezvous but
has no external route. Rank 0 should write `dense_1b_ddp/telemetry.json`
(loss series + probe curve): spawned workers do not share the parent
`prism_telemetry` hook, and without that sidecar G6/G8 used to omit keys.

**Resubmit at will on your own failures.** If your dependency install fails
(`install_deps`) or your `training.py` crashes at build/train time
(`train_script`), the run fails **without burning your one-submission
slot** — fix the manifest or the script and resubmit immediately, no time
window. (Operator-side infra hiccups keep the existing 30-minute resubmit
window.)

**Bring your own tokenizer — verified.** The tokenizer is part of your
submission: ship `tokenizer/` files in your tree (≤ 12 files, ≤ 8 MiB,
loaded offline with `AutoTokenizer.from_pretrained(dir,
local_files_only=True)`) or export `def build_tokenizer(ctx)` next to
`build_model` in `architecture.py` (train/wrap anything, offline). The
harness hands it back as `ctx["tokenizer"]` / `ctx["vocab_size"]`; the
`gpt2` pin is only the default when you ship nothing. Two things keep this
fair: (1) G1 scores **bits/byte**, tokenizer-neutral — exotic vocabularies
buy you nothing on the headline metric; (2) every run emits an objective
**tokenizer card** (compression on a fixed probe, roundtrip fidelity,
vocab-shape scan) that the LLM anti-cheat review reads. A tokenizer
engineered to game metrics — multi-word answer phrases as single tokens,
vocab stuffed with eval-looking strings, a `decode()` that rewrites output,
memorizing compression — is a **cheat** (`tokenizer_gaming`, Score 0). A
merely weak tokenizer is not a cheat; it just hurts your own score.

**Legacy recipe 1.x rejected on live.** Two-script ZIPs
(`architecture.py` + `training.py`), 1.3 source-tree ZIPs, and training-only
`arch_id` submissions return `400 unsupported_layout` or `400 recipe_version`
once 2.0 is advertised. Do not ship Megatron-Bridge or other non-AutoModel
frameworks.

**Telemetry.** The harness wrap still requires `prism_telemetry` reporting /
`finish_evaluation` under the AutoModel train entry. Patches that remove or
bypass those hooks fail review (`missing_telemetry_hooks`, zero score,
terminal).

**Example: dense 1B (1× B200, NVFP4).** A reference AutoModel patch
that honors `ctx["train_stream"]`, rank-0 stream ownership, and NVFP4 TE
on 180 GB-class (mb≥8, ckpt off; 4×5090 stays BF16 + activation checkpoint)
lives at
[`examples/dense-1b/`](examples/dense-1b/). It is a **dense** ~975M
transformer (GQA + SwiGLU). The harness uses analytic 6N for the parent
FLOPs cap so GPU0 stays free for spawn. Fine-grained MoE at 1B is a miner
experiment, not the reference. Pack `automodel.base` + `automodel.patch`
(+ optional `prism.toml` / `requirements.txt`) as in this document. It is
an example, not a scored baseline.

**Diff visibility.** After intake, inspect your applied delta at
`GET /v1/submissions/{id}/diff` (full unified diff + diffstat / classification).

Evaluation runs on **miner-funded** Lium GPU pods (you pay the rent). Master
still operates the pod over SSH; you do **not** deploy a miner CVM. CI uses
`SimLiumBackend` and does not need a key.

## Pay for your own GPU (required on live)

Create a [Lium](https://lium.io) account, fund it, and pass your API key on
every live submit:

```http
X-Lium-Api-Key: <your Lium API key>
```

The key is held in master memory for that submission and may also land in a
**TTL-bounded encrypted seal file** on the master host (default ≥36h; never in
Postgres, never logged). Master **re-seals** on measure start and heartbeats
so a full 4h train wall cannot outlive the seal across a control-plane
restart. Missing key on live → `400 missing_lium_api_key`. Cost guardrails
(`max_price_per_hour`, lifetime) still apply so a bad key cannot rent
unbounded SKUs through the orchestrator.

**Pod lifetime ceiling is 7.0h.** You are billed for time actually used, not
the ceiling. It must contain build (≤15m) + the **4.0 h / 240 min** train
wall + checkpoint (≤30m) + eval (≤1.5h) ≈ 6.28 h. Isolated proofs set
`PRISM_TEST_TRAIN_MINUTES=60`; unset or `240` is the operator default (same
as the recipe constant). The eval battery itself runs under one global 1h
budget with per-group shares, and if a group hits its ceiling the run
reports it (`budget.truncated` / `budget.partial_groups` in the battery
blob) rather than silently scoring fewer items.

If the challenge process restarts mid-run while your Lium pod is still
training/evaling, master **reattaches** quietly (same submission id; pod is
not killed). You only see `control_plane_restart` / `harness_detached` when
the pod is already dead or the sealed key cannot be restored and master
cannot talk to Lium — then stop the pod yourself and resubmit with
`X-Lium-Api-Key`. Poll `GET /v1/submissions/{id}/events` and
`GET /v1/submissions/{id}/logs?since=` for live stage heartbeats and harness
tails while the run is healthy.

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

Inspect recipe + AutoModel pin before coding:

```bash
curl -sS "$BASE_GATEWAY/challenge/prism/v1/recipe"
```

Live recipe **2.1.0** advertises `version: "2.1.0"` and AutoModel pin fields
(`automodel_pin_id` = `automodel@v0.5.0`, `automodel_repo_url`,
`automodel_git_ref`, `automodel_git_commit`, `automodel_content_sha256`),
plus caps such as `train_flops_cap: 3.0e18` (the budget currency),
`train_hours_cap: 4.0` (240 min anti-DoS wall; operator default
`PRISM_TEST_TRAIN_MINUTES=240` is the same as unset), `min_spend_fraction: 0.5`
(voluntary-stop floor; a step/wall/FLOPs-bound run stays eligible),
`max_train_steps: 20000`, `min_params: 850000000`, `max_params: 1000000000`, FineWeb dataset pin,
and `pin_hex` (sha over the versioned descriptor). Trust `/v1/recipe`,
not marketing chart labels.

The FLOPs probe retries OOM at progressively smaller row counts down to one
sequence and reports whether it reduced the batch; a deliberately
memory-heavy model cannot turn probe OOM into an unmetered train.

`POST /v1/submissions` is idempotent by `submission_id` (hash of **pin id ‖
`0x00` ‖ patch bytes**).

## Submission gating (1-max)

- Your hotkey must be **registered on the subnet** (metagraph). Unknown hotkey
  → `403 hotkey_not_in_metagraph`; a fresh registration may lag the snapshot
  (`503 metagraph_unavailable` → retry shortly).
- **One accepted patch submission per hotkey.** While yours is `registered` /
  `rejected`, or `blocked` **outside** the infra recovery window, a *different*
  patch submission gets
  `409 submission_gated`. Re-POSTing the **identical** pin+patch is always
  safe (idempotent `200 already-queued`).
- If your hotkey **leaves the metagraph**, the watcher reopens your slot(s)
  automatically — resubmit under your new uid.
- Infra failures (Lium pod, review/similarity/LLM infra) **auto-retry up to 3
  times**; harness `EVAL_FAIL` (miner/model code) is terminal for that attempt
  and is **not** auto-retried. Cheat / rejected verdicts are terminal. After an
  infra failure (`ChallengeInternal`), you may **recover within 30 minutes**
  via `POST /v1/submissions/{id}/retry` with **`X-Lium-Api-Key`** (required on
  live when another GPU rent is needed). After 30 minutes the slot stays
  blocked until your hotkey leaves the metagraph.

### Retry vs re-POST

| Action | When | Headers |
|--------|------|---------|
| Re-POST the **same** ZIP | Always safe | Same as submit | Returns `200 already-queued` — **no new GPU run**; does not recover a failed row |
| `POST /v1/submissions/{id}/retry` | Row status is **`failed`** only | **`X-Lium-Api-Key`** on live (infra recovery); admin Bearer for operator non-infra retries | Requeues measure; wrong/missing Lium key → `400 missing_lium_api_key` |
| `/retry` on non-failed | — | — | `409 not_failed` — hotkey or Bearer alone does not change that |

Do **not** expect `X-Miner-Hotkey` or admin Bearer alone to fund a new Lium
pod. Seal TTL is ≥36h and master re-seals on measure + heartbeats; the key is
kept across measure Err so auto-/miner-retry can re-rent without a new submit.

## Anti-copy rule (patch / delta)

Copying another miner's **patch** (or an equivalent touched-file rewrite of
an earlier champion delta) is terminal `rejected` with zero score — judged
before or without burning GPU when the gate can decide from the diff alone.
Review focuses on your unified diff and touched files (`arch` / `trainer` /
`data` / `other`), not the whole AutoModel tree. Starting from the operator
pin and submitting only your delta is the intended path.

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

Dry-run the copy / layout gate **without** burning your 1-max slot or a
GPU eval (send the same AutoModel ZIP you would submit):

```bash
curl -sS -X POST "$GATEWAY/challenge/prism/v1/submissions/precheck" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: $HOTKEY" \
  --data-binary @submission.zip
```

| Field | Meaning |
|-------|---------|
| `similar` | `true` → would hard-reject at intake copy gate |
| `verdict` | `clean` / `copied` / `skipped` |
| `matched_against` | Corpus id only (never competitor source) |
| `score` | Similarity in `[0,1]` when compared |
| `quota` | `{ day, used, limit: 3, remaining, identity }` |

**Quota: 3 attempts per coldkey per UTC day** (falls back to hotkey when the
metagraph Owner coldkey is unknown). Rotating hotkeys under the same coldkey
does **not** reset the budget. A 4th call returns `429` /
`precheck_quota_exceeded` with `remaining=0`. Precheck never creates a scored
submission and never rents a Lium pod.

## Scoring (summary)

Final leaf score (live `scoring_version` **4**) is the **equal-weight mean of
available G2 public accuracies** mapped to `round(SCORE_MAX × mean)` — not
bits/token bpb. Tokenizer length cannot farm the rank. Bits/token bpb and
tokenizer-neutral `bits_per_byte` remain recorded for display / G1. The shared
**agentic** gate (AST + metrics/receipt) hard-zeros `cheat` /
`suspicious`. Cheap LLM similarity hard-zeros `Copied`, and `Suspicious` only
when confidence `≥ 0.9` with non-generic evidence (below that — e.g. 0.7 citing
RMSNorm/SwiGLU/LayerNorm — does **not** wipe your score). Copy/similarity
corpora are **champions only** (current top + historical Score>0 ex-tops) plus
baseline — not every past submission — and still exclude your own prior art
(same hotkey **or** same coldkey). Standard components (RMSNorm, RoPE, SwiGLU,
LayerNorm, gated/parallel residual, …) are **not** plagiarism signals. LLM
quality is coherence-only, not a grader.
Public gallery/leaderboard show champions only.
**New competition (`prism-v2.1`).** Only harvests finalized under recipe
**2.1.0** / `scoring_generation` **21** can receive leaves. A prior 2.0
AutoModel run — even a high lattice score — is a different contest and
does not carry, win WTA, or get paid. Re-submit under 2.1 if you want to
compete. Until someone finishes an eligible 2.1 run, weights burn.

**Competition (temporary):** emission uses **your own best training score
only** — architecture-owner credit (rewarding arch owners when others train
well on their code) is **disabled** for now so the best-scoring trainer keeps
Prism's weights. Emission remains **winner-take-all**: only the single highest
own score that epoch receives Prism's share (50% of the subnet); ties break by
lexicographically smallest hotkey. Two **v2.1 opt-in** emission knobs exist
but are **off by default** (operators announce any flip): `top3` mode pays
the top three positive scores at 100 % / 50 % / 25 % of their own lattice
score instead of winner-take-all, and an architecture-owner split can carve
up to 50 % of the winner's leaf to the **registry owner** of the winning
architecture — publishing a strong architecture that someone else trains to
the top then earns you a share. A third mode, `sig` (**significance-gated**,
also off by default), is described in
[what actually earns emission](#what-actually-earns-emission-significance-gating)
below. Scores first land in the leaf
set emitted at the first chain-epoch boundary **after** your run finalizes (a
long train that crosses epochs is normal — outbox assignment is exactly once).
Positive scores then keep participating in later epochs' competition sets until
a better valid score supersedes them (WTA still collapses to one leaf winner).
The global-best model by **G2 lattice score** (sources + `ARTIFACT.json` /
checkpoint release) is published to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism)
`top-model/` and (when configured) a HuggingFace model repo
`BaseIntelligence/top-prism-architecture` (custom-arch / AutoModel novelty +
weights, `trust_remote_code`). See [`PRISM.md`](../PRISM.md).

## What Prism does and does not claim about your architecture

Read this before optimizing anything, because it tells you what the ranking
means.

**Prism ranks architectures at a pinned, small budget.** Every submission trains
under the same fixed recipe, the same fixed data, the same wall-clock cap, on the
same GPU class. That is a genuine, well-controlled comparison — the pinned-recipe
discipline is what makes it meaningful at all — and it is the thing the leaderboard
measures.

**Prism does not claim to select architectures that will scale.** This is not
modesty; it is a measured result we would rather publish than hide. Tay et al.,
*Scaling Laws vs Model Architectures* (EMNLP Findings 2023,
[arXiv 2207.10551](https://arxiv.org/abs/2207.10551)) pretrained **>100 models
across 10 architectures from 15 M to 40 B parameters** and found:

- "**The best performing model can fluctuate at different scales.**"
- The **vanilla Transformer has the best scaling exponent** while *not* being the
  best at every individual compute point — the winner at one budget is not
  necessarily the best scaler.
- **Concrete rank flips:** Evolved Transformer beats vanilla at small scale on
  downstream tasks and falls behind when scaled up; **ALBERT scales negatively
  downstream** (α = −0.12), and ALBERT's mechanism is cross-layer weight sharing,
  the same family as looped / recurrent-depth designs.

So a win here is evidence that your architecture is better **at this budget**, not
a prediction about 70 B. The authors also state the converse, which is Prism's
honest positive claim: not every practitioner needs models that scale to billions,
and inductive biases tailored to small or low-compute regimes are valuable in
their own right. That is the regime Prism measures.

Two practical consequences for you:

- Improvements that only appear at larger scale will not be visible here, and
  that is a limitation of the instrument, not a judgement on your idea.
- Tuning tricks that exploit the pinned budget specifically may win here without
  transferring. We would rather you know that than discover it later.

## What actually earns emission (significance gating)

Live emission today is **winner-take-all**: the single highest own score takes
Prism's share. The `sig` mode described here is **implemented but off by
default** — operators announce any flip, and it cannot be enabled until the
run-to-run (seed) noise floor has been measured and published. It is documented
now so you can see where the incentives are going.

Under `sig`, **beating the champion requires clearing a measurement-uncertainty
margin, not just posting a better point estimate.**

**Why.** Two runs of the *same* architecture do not produce the same number.
Eval sampling and training-seed noise both move the score. So a challenger that
scores 0.1 % better has not shown it is better — it may simply have drawn a
luckier run. Under the old rule that coin flip won the entire share, which is
also exactly what makes copying the champion profitable: a copy has the *same*
true quality, so it wins the flip about half the time. Requiring real evidence of
improvement removes that.

**How the comparison works, in plain terms.**

1. You and the champion are scored on the **same** private eval slice.
2. Your scores are compared **example by example**, not as two totals. Hard
   examples are hard for both models, and pairing cancels that out.
3. Differences smaller than a fixed **dead zone** (0.01 in absolute metric units
   — bits/byte, or accuracy) are treated as *undecided*. Hairline differences do
   not vote.
4. You must win **≥ 55 % of the decided examples** — and not on the point
   estimate: on a **99 % lower confidence bound** from a 10 000-resample
   bootstrap with a fixed, published seed. Anyone can recompute the verdict.
5. Your **average margin** on decided examples must also clear the dead zone, so
   you cannot win a majority of near-ties while being much worse where you lose.
6. There must be at least **100 decided examples**. Below that the win rate
   cannot be estimated closely enough to mean anything, and the champion holds.
   On a task where everyone scores nearly the same, that is the normal outcome:
   the comparison refuses rather than crowning a coin flip.

One consequence worth knowing: you and the champion must have been measured on
the **same** eval slice for any of this to run. If the slice rotated between the
champion's run and yours, there is no valid comparison and the champion holds
until it is re-measured on your slice. That re-measurement is the operator's job,
not yours.

The bar is 55 % rather than something higher on purpose: a genuinely better
architecture with a wide per-example spread sits near 55 %, so demanding much
more would select for **low-variance submissions instead of good ones**.

**What you get paid.** Prism's share splits: **60 %** champion (dropping toward
a 50 % floor if the win is real but marginal, with the difference **burned**),
**15 / 10 / 5 %** to ranks 2–4, and **10 %** split across up to five entries that
pass every gate and **hold the best measured value on any single axis** `g1..g8`.

That last pool is the one worth understanding. You do **not** have to win overall
to earn from it. If you are third on the composite but **first on G3**
(associative recall) or **first on G7** (inference cost), you produced real
information and you are paid for it. Note what this is not: there is **no reward
for being different**. Renaming variables, reordering statements, or otherwise
looking novel earns nothing — the axes are real measurements, and you have to
actually be best on one. Anything unallocated **burns** rather than being
redistributed.

Also under `sig`: the champion is **re-measured** on fresh private slices at
unannounced times (operator-funded, eval-only). If a champion's score was propped
up by fitting the public anchors, that shows up and costs it the title.

## v3 scoring (battery always; leaf mode via env)

Recipe ≥ 1.3.0 harnesses run a **two-phase pod flow**: your code trains
(`phase=train`), checkpoints, and only then does the operator stage private
eval assets — the eval phase (`phase=eval`) is a fresh subprocess that runs
the frozen-val bpb plus the **G1–G8 battery**: intrinsic fit (G1),
commonsense/reading (G2), retrieval/recall (G3), reasoning (G4),
long-context (G5), sample efficiency from the train probe curve (G6),
inference efficiency (G7), and training stability/µP (G8). Everything the
battery reports is organizer-measured (**Zone A**, `org.*`) and is computed
inside the harness — your code never emits it.

The v3 metric surface is structurally complete: G1 includes code, prose,
math, fresh crawl, and key-token bits/byte (news/crawl alias into prose/fresh
when a pack omitted those files; otherwise a chance-floor bits/byte so the
org keys are never silently dropped); G6 includes byte-denominated AUC,
bytes-to-threshold, and bpb at half of the organizer FLOPs cap, plus
fail-closed `org.g6.auc_log_tokens` / `org.g6.tokens_to_threshold` when the
probe curve is empty; G7 includes measured-or-censored 32k TTFT/TPOT/state,
board energy, throughput, and reasoning throughput (32k is skipped without
allocating when a shorter length OOM'd); G8 emits `org.g8.loss_spike_score`
and µP LR-transfer. Unsupported 32k/OOM/power cases receive explicit
worst-case censored values rather than silently disappearing. Live emission nevertheless remains G2 benchmarks +
WTA until operators announce calibrated v3 anchors and a separate governance
flip.

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

While `PRISM_SCORING_MODE=benchmarks` (default) the leaf score is the
**equal-weight mean of available G2 public accuracies** (HellaSwag, ARC-E/C,
PIQA, WinoGrande, BoolQ, LAMBADA strict when present, OpenBookQA), mapped to
`round(SCORE_MAX × mean)`. Missing every listed bench → `0` (fail-closed).
**Tokenizer length no longer farms the rank** — bits/token bpb is still
recorded (and tokenizer-neutral `bits_per_byte` feeds G1) but does **not**
drive emission. `PRISM_SCORING_MODE=shadow` restores legacy pure bits/token
bpb (v2). After the reference baselines (**Transformer++** and
**hybrid delta** — published in-repo under `crates/prism-recipe/baselines/`)
are measured and the anchor set is pre-registered, governance may flip to
`composite`: group scores are anchor-normalized (**arithmetic** mean within
each group; a single zero sub-metric does not zero the group), gate-filtered
(`g3 ≥ 0.25` currently **disarmed** until G3 item counts stabilize;
`g8 ≥ 0.5`, budget + CI gates), combined as a weighted
**geometric** mean across groups (`C = ∏ g_k^{w_k}` — a full group score of 0
collapses C), and ranked by the bootstrap lower-confidence bound
(`lattice = round(SCORE_MAX × max(0, C − 1.645·SE))`). Inspect the anchor
registry and pre-registration commits at `GET /v1/anchors` and
`GET /v1/preregistration`; per-run Zone A / Zone B rows at
`GET /v1/submissions/{id}/metrics?zone=a|b`.

**G8 µP probe.** The stability sweep builds 1× and 4× width from a **fixed
small** width/depth base (not your full 850M–1B scored model), then scales with
`ctx["prism_width_multiplier"]`. Honor top-level / `arch` geometry overrides
and that multiplier in `build_model` (reference baselines do) or the sweep
fail-closes `org.g8.mup_lr_stability = 0.0`.

### v2.1 battery additions (anchor set v1, opt-in)

Two extra organizer-measured keys ship with the v2.1 harness on every real
run (inert until operators select anchor set v1; you will see them in
`GET /v1/submissions/{id}/metrics?zone=a`):

- `org.g7.reasoning_throughput` — mean G4 accuracy × decode toks/s.
  Compute-normalized reasoning: architectures that spend extra inference
  compute to reason (loops, adaptive depth, recursion) are credited for
  the accuracy they buy in the same key that charges its cost — raw
  throughput alone no longer structurally penalizes them.
- `org.g8.mup_scaling_slope` — a local scaling-exponent probe measured on
  the existing µP 1×/4× width sweep (how fast your architecture improves
  with scale). Support the `prism_width_multiplier` build knob (already
  required for G8) and this costs you nothing extra; a failed sweep
  fail-closes the key to 0.0.

Practical consequence for architecture design: under anchor set v1,
"thinks more when it's hard" designs and "scales steeper" designs earn
score on dedicated axes instead of only paying G7/G6 penalties.

### v2.2: LAMBADA scored strict (anchor set v2, opt-in)

The G2 LAMBADA item used to be a 4-way multiple choice against random
distractor words — nearly free points (0.95+ for everyone, 0.985 for the
GPT-2 Large reference), because LAMBADA's gold word is uniquely determined
by its long context. The harness now **also** emits
`org.g2.lambada_strict_acc`: unconstrained **greedy last-word exact match**
(the canonical protocol — GPT-2 Large lands around 0.52–0.60, small 1-hour
models around 0.10–0.30). Under anchor set v2 the strict key replaces the
saturated MC key in the composite; v0/v1 scoring is unchanged. For your
model this means last-word prediction quality is measured for real: test
locally by greedy-decoding the final word of LAMBADA passages, not by
ranking four candidate words.

### v2.2: G6 sample-efficiency scoring corrected (anchor set v2, opt-in)

Two G6 defects are fixed. Both only affect anchor set **v2**; v0 and v1 are
pre-registered and byte-frozen, so their scoring is unchanged.

- **Never reaching the CE threshold no longer scores well.**
  `org.g6.tokens_to_threshold` is lower-better, and a curve that never
  reaches CE 4.0 used to report the small token count it stopped at — so
  training *less* scored **better**. A right-censored curve now scores the
  **0.0 floor**. There is no longer any advantage in stopping early; get the
  probe loss down and actually cross the threshold. The raw endpoint is
  still reported for you as `g6.tokens_to_ce4.0.observed`, and
  `g6.tokens_to_ce4.0.censored` tells you it happened.
- **`org.g6.auc_log_tokens` now discriminates.** It is the mean probe
  cross-entropy per decade of tokens — **lower is better**. The v0/v1 anchor
  treated it as higher-better over `[0.5, 0.95]`, so every plausible run
  clipped to a perfect 1.0 and the metric measured nothing. Under v2 the
  **shape** of your learning curve is scored: reaching a low loss early, and
  staying low, beats a late crossover with the same final loss.

### G2 item counts raised on the tasks that discriminate

LAMBADA, HellaSwag, PIQA and ARC-easy are now scored over **~1000 items**
each instead of 200. At 850M–1B params / 6h, Winogrande and OpenBookQA sit at
chance and ARC-challenge / BoolQ at or below their floors, so those keep 200
items — more items there would not separate two submissions. **No group or
task weights changed.** Practical consequence: a 2–3 point difference on a
G2 task was inside the noise floor at 200 items; on the four raised tasks
the floor is roughly 3× tighter, so real gains there now show up in your
score instead of being washed out.

## Useful routes

| Route | Use |
|-------|-----|
| `POST /v1/submissions/precheck` | Advisory copy/layout gate (3/coldkey/UTC day); no submit |
| `GET /v1/status` | Backend mode, epoch, queue |
| `GET /v1/recipe` | Caps + AutoModel pin (`automodel_pin_id`, commit, content sha) |
| `GET /v1/submissions/{id}` | Detail + receipt + scores + composite block (v3) |
| `GET /v1/submissions/{id}/diff` | Unified diff + diffstat / classification (recipe ≥ 2.0) |
| `GET /v1/submissions/{id}/events` | Stage timeline |
| `GET /v1/submissions/{id}/metrics?zone=a\|b` | Zone A battery rows / Zone B self-report chain (v3) |
| `POST /v1/submissions/{id}/zone-b` | Miner Zone B self-report intake: validated + chained + stored (v3) |
| `GET /v1/anchors` | v3 anchor-set registry + status |
| `GET /v1/preregistration` | v3 anchor pre-registration hash-commits |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Miner-reported loss curve / gradients / layer stats (from `prism_telemetry.report`) |
| `GET /v1/jobs` | Active/recent pods (ops) |
| `GET /health` | Liveness |

Emission share for prism is owner-controlled via the trust root. Current split is
`5000` bps prism / `5000` bps design (50/50) — see
[`../runbooks/prism-enable-lium-and-emission.md`](../runbooks/prism-enable-lium-and-emission.md)
and [`../runbooks/design-enable-and-emission.md`](../runbooks/design-enable-and-emission.md).
