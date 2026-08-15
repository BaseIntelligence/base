<!-- protocol_version: 1 -->

# Prism challenge — HTTP AutoModel patch submit

**challenge_id:** `prism`  
**scoring_version:** `4` live (equal-weight G2 public-suite accuracies → lattice; LLM review is an anti-cheat gate, not a grader). **v3 harness (default):** every scored run executes the **G1–G8 battery**; the leaf uses G2 benches while `PRISM_SCORING_MODE=benchmarks` (default). Legacy `shadow` = bits/token bpb; `composite` = full G1–G8 lattice when anchors are ready.  
**recipe_version:** `2.0.0` (pinned [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) base + miner unified diff; legacy 1.x layouts rejected on live)  
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../PRISM.md`](../PRISM.md), recipe [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md).

## What you submit

A **ZIP** (preferred) — or JSON with the same members / `zip_base64` — that is
**not** a free-form `architecture.py` / `training.py` project. Recipe **2.0.0**
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

Models must stay **≤ 350M parameters**. The pod has **no network**
(`unshare --net`) beyond the operator-owned dataset pull — do not call Hub
downloads from miner code.

**Legacy recipe 1.x rejected on live.** Two-script ZIPs
(`architecture.py` + `training.py`), 1.3 source-tree ZIPs, and training-only
`arch_id` submissions return `400 unsupported_layout` or `400 recipe_version`
once 2.0 is advertised. Do not ship Megatron-Bridge or other non-AutoModel
frameworks.

**Telemetry.** The harness wrap still requires `prism_telemetry` reporting /
`finish_evaluation` under the AutoModel train entry. Patches that remove or
bypass those hooks fail review (`missing_telemetry_hooks`, zero score,
terminal).

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
so a full 6h train wall cannot outlive the seal across a control-plane
restart. Missing key on live → `400 missing_lium_api_key`. Cost guardrails
(`max_price_per_hour`, lifetime) still apply so a bad key cannot rent
unbounded SKUs through the orchestrator.

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

Live recipe **2.0.0** advertises `version: "2.0.0"` and AutoModel pin fields
(`automodel_pin_id` = `automodel@v0.5.0`, `automodel_repo_url`,
`automodel_git_ref`, `automodel_git_commit`, `automodel_content_sha256`),
plus caps such as `train_hours_cap: 6.0`, `max_train_steps: 20000`,
`max_params: 350000000`, FineWeb dataset pin, and `pin_hex` (sha over the
versioned descriptor). Trust `/v1/recipe`, not marketing chart labels.

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
**Competition (temporary):** emission uses **your own best training score
only** — architecture-owner credit (rewarding arch owners when others train
well on their code) is **disabled** for now so the best-scoring trainer keeps
Prism's weights. Emission remains **winner-take-all**: only the single highest
own score that epoch receives Prism's share (50% of the subnet); ties break by
lexicographically smallest hotkey. Scores first land in the leaf
set emitted at the first chain-epoch boundary **after** your run finalizes (a
long train that crosses epochs is normal — outbox assignment is exactly once).
Positive scores then keep participating in later epochs' competition sets until
a better valid score supersedes them (WTA still collapses to one leaf winner).
The global-best model (sources + `ARTIFACT.json` / checkpoint release) is
published to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism)
`top-model/` and (when configured) a HuggingFace model repo
`BaseIntelligence/top-prism-architecture` (custom-arch / AutoModel novelty +
weights, `trust_remote_code`). See [`PRISM.md`](../PRISM.md).

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
(`g3 ≥ 0.25`, `g8 ≥ 0.5`, budget + CI gates), combined as a weighted
**geometric** mean across groups (`C = ∏ g_k^{w_k}` — a full group score of 0
collapses C), and ranked by the bootstrap lower-confidence bound
(`lattice = round(SCORE_MAX × max(0, C − 1.645·SE))`). Inspect the anchor
registry and pre-registration commits at `GET /v1/anchors` and
`GET /v1/preregistration`; per-run Zone A / Zone B rows at
`GET /v1/submissions/{id}/metrics?zone=a|b`.

**G8 µP probe.** The stability sweep builds 1× and 4× width from a **fixed
small** width/depth base (not your full ≤350M scored model), then scales with
`ctx["prism_width_multiplier"]`. Honor top-level / `arch` geometry overrides
and that multiplier in `build_model` (reference baselines do) or the sweep
fail-closes `org.g8.mup_lr_stability = 0.0`.

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
