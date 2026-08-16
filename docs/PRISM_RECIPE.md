# PRISM recipe v2.0.0 — AutoModel base + miner git diffs

**Live contract:** recipe **`2.0.0`**. Miners submit a **unified diff against a
pinned [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) checkout** —
not a free-form `architecture.py` / `training.py` project. Megatron-Bridge is
**out of scope**. Legacy recipe **1.x** two-script / source-tree / training-only
layouts are **rejected on live** once 2.0 is enabled (`400 recipe_version` /
`unsupported_layout`). Local Sim may keep a tiny fixture patch for CI.

Caps, FineWeb dataset pin, telemetry, two-phase train/eval, and the G1–G8
battery from 1.4 remain operator-owned unless a later bump says otherwise.

**Harness staging (pod).** Operator harness (`prismlib/automodel.py`) materialises
the pin + applied tree under `$PRISM_WORKDIR/automodel/` and invokes the train
entry (default `nemo_automodel/recipes/llm/train_ft.py`, overridable via
`prism.toml`). Env contract shared with crate `prism-automodel`:
`PRISM_AUTOMODEL_APPLIED_DIR` (preferred), or `PRISM_AUTOMODEL_PIN_DIR` +
`PRISM_AUTOMODEL_PATCH_PATH`, plus `PRISM_AUTOMODEL_BASE` / optional
`PRISM_AUTOMODEL_PRISM_TOML`. CI/Sim uses `PRISM_AUTOMODEL_FIXTURE=1` with the
tiny `FIXTURE_PIN` + `happy.patch` — fixture markers are **not** proof of a
real NVIDIA AutoModel train.

Historical 1.x contract text is preserved below under
[*Legacy recipe 1.x*](#legacy-recipe-1x-historical--rejected-on-live-under-20)
for leaf/audit continuity.

---

## Recipe 2.0.0 — product contract

```text
pinned AutoModel commit ──┐
                          ├─► git apply (fail-closed) ─► materialized tree
miner unified diff ───────┘         │
                                    ├─► diff view + agentic (delta-focused)
                                    └─► Lium pod train / eval (harness wrap)
```

1. **Operator pin** — recipe freezes AutoModel at a tagged git commit plus a
   content-addressed archive hash (tarball staged like today’s FineWeb pin).
2. **Miner submit** — ZIP (or JSON equivalent) with `automodel.base` +
   `automodel.patch` (+ optional `prism.toml`). Recipe-v10: the patch may
   **add `requirements.txt` or `pyproject.toml` at the repo root** for
   custom deps (`requirements.txt` wins if both). See **Modular pod image +
   miner dependencies** below.
3. **Apply fail-closed** — master applies the patch onto a clean pin checkout;
   reject on conflict / path escape / binary blobs / oversized diff.
4. **Visibility** — persist full unified diff, `diffstat`, and file
   classification (`arch` / `trainer` / `data` / `other`); expose
   `GET /v1/submissions/{id}/diff` (site complete-view panel).
5. **Allowed novelty** — new model modules / configs under the AutoModel
   layout are OK. Trainer edits OK but **high scrutiny**.
6. **Cheat focus on the delta** — agentic + AST primarily on diff hunks and
   touched files (not the whole AutoModel tree). Harness gates remain:
   netns, telemetry hooks, budget, causal screen, no eval-asset reads in train.
7. **Harness wrap** — pod still owns dataset pin, `prism_telemetry`,
   wall-clock/step caps, eval battery. Miner code must call into AutoModel’s
   train entry under those constraints (thin operator adapter — not miner-owned).

### Modular pod image + miner dependencies (recipe-v10)

The pod image (`/v1/recipe` `pod_image_ref` =
`ghcr.io/baseintelligence/prism-pod:v10-cuda13-te`, built from
[`../deploy/prism-pod/Dockerfile`](../deploy/prism-pod/Dockerfile)) is a
complete CUDA 13 base: PyTorch, `nvcc`/`ninja`/`build-essential`,
Transformer Engine (NVFP4 training), and common accelerators. A submission
may ship `requirements.txt` (`pip install -r`) or `pyproject.toml`
(`pip install .`) — patch-added at the AutoModel repo root (slim delivery
always keeps both names; the harness searches the workdir root and
`submission/`). The harness installs it in a **network-on install phase in
the parent, before** the `unshare --net` train/eval children
([`prismlib/deps.py`](../crates/prism-recipe/harness/prismlib/deps.py)). So
dependency installs (FlashAttention, `mamba-ssm`, custom kernels) have
network; model code that later sees private eval assets does not.

Descriptor keys: `pod_image_ref`, `miner_install_supported` (bool),
`miner_deps_members` (`["requirements.txt","pyproject.toml"]`),
`install_timeout_secs` (1800). Image is env-overridable for staged rollout:
`PRISM_POD_IMAGE` + `PRISM_POD_IMAGE_TAG` (the Lium template name flips to
`prism-recipe-v10` automatically — template identity is name-based, so a new
image must ship under a new name). Ops: build + push the image and validate
`transformer_engine` import on a GPU node **before** repinning live.

**Miner-fixable retry classes.** A failed custom-deps install (`install_deps`)
or a `training.py` build/train crash (`train_script`) fails **without
burning the 1-max slot** and is resubmittable at will (unbounded — no time
window), distinct from operator infra classes
(`install`/`ast_infra`/`llm_infra`, windowed 30 min). Classification:
harness `EVAL_FAIL` + `{"stage": …}` → `orchestrator::classify_eval_fail` →
`submission_gating::{is_miner_fixable_class, resubmit_allowed}`.

### AutoModel pin metadata (apply-lib fields)

Operator freeze writes these fields into the recipe descriptor (surfaced on
`GET /v1/recipe` as `automodel_*` keys). Live pin identity is frozen in
`crates/prism-automodel` (`AUTOMODEL_PIN`) including the content SHA of the
staged checkout.

| Field (JSON on `/v1/recipe`) | Type | Live / notes |
|------------------------------|------|----------------|
| `automodel_pin_id` | string | `automodel@v0.5.0` — tag-shaped id miners must echo in `automodel.base` |
| `automodel_repo_url` | string | `https://github.com/NVIDIA-NeMo/Automodel` |
| `automodel_git_ref` | string | `v0.5.0` (informational tag label) |
| `automodel_git_commit` | 40-hex | `d02f49cb314554715aabb97e8dba6599c9f6e9e0` — exact commit the pin checkout must match |
| `automodel_content_sha256` | 64-hex | SHA-256 of the staged pin tree (`tree_content_sha256`; `.git`/dotfiles skipped) |
| archive name (ops only) | string | suggested `automodel@v0.5.0.tar.zst` (or `.tar.gz`) under the operator pin store |

**Operator staging (required on live):** set `PRISM_AUTOMODEL_PIN_DIR` to a
clean checkout of `automodel_git_commit`. Use
`deploy/scripts/stage-automodel-pin.sh` (clone + hash verify). Intake
fail-closes when the dir is missing or the content SHA mismatches.
Prod/staging Compose (`deploy/compose/env-prod.yml` / `env-staging.yml`)
mount `/var/lib/prism/automodel-pin` read-only into `prism-challenge` and set
the env var; stage the tree **before** `remote-deploy` recreates the service:

```bash
sudo ./deploy/scripts/stage-automodel-pin.sh --dir /var/lib/prism/automodel-pin
# then recreate prism-challenge (or full remote-deploy)
```

**CI / Sim fixture (not live):** pin id `automodel@fixture-v1` + vendored tree
`crates/prism-automodel/fixtures/automodel-pin/` + `fixtures/patches/happy.patch`.
Accepted **only** when `PRISM_AUTOMODEL_FIXTURE=1`. Fixture markers are not
proof of a real NVIDIA AutoModel train.

**Example shape (live constants):**

```text
automodel_pin_id         = automodel@v0.5.0
automodel_repo_url       = https://github.com/NVIDIA-NeMo/Automodel
automodel_git_ref        = v0.5.0
automodel_git_commit     = d02f49cb314554715aabb97e8dba6599c9f6e9e0
automodel_content_sha256 = f8af64ef572e2e3634dcbae7b351fdcd3c8d458caf2fe974aff26d301a11d838
```

Rules:

- Submit `automodel.base` **must equal** the recipe’s current
  `automodel_pin_id` (byte-identical ASCII, no whitespace). Mismatch →
  intake reject (`pin`).
- `submission_id` hashes **`pin_id` + `0x00` + patch bytes** (not ephemeral
  keys).
- Apply-lib resolves the live pin on production; the CI fixture pin requires
  `PRISM_AUTOMODEL_FIXTURE=1`. Live always verifies `content_sha256` against
  `PRISM_AUTOMODEL_PIN_DIR` before `git apply`.
- Local/CI fixture env must not be set on live hosts.

### Submission ZIP layout (recipe ≥ 2.0.0)

Preferred: `application/zip` (or JSON with the same members / `zip_base64`).

```text
automodel.base          # required — single-line pin id (ASCII), must match recipe pin_id
automodel.patch         # required — unified diff vs that pin (git diff / format-patch style)
prism.toml              # optional — entry / recipe knobs (train script path, model config)
```

| Member | Required | Notes |
|--------|----------|-------|
| `automodel.base` | yes | Exact recipe `automodel_pin_id` (live: `automodel@v0.5.0`; CI fixture: `automodel@fixture-v1`). Trailing newline OK; no other files aliased to this name. |
| `automodel.patch` | yes | Text unified diff against the pin tree. No binary blobs; size budgets enforced in intake (fail-closed). |
| `prism.toml` | no | Optional knobs: `entry` / train script path under the applied tree, model config path. Unknown keys ignored or rejected per intake strictness. |

**Not accepted on live (recipe ≥ 2.0.0):**

- Two-script ZIPs (`architecture.py` + `training.py`)
- Recipe 1.3+ source-tree ZIPs (`train.py` / `kernels/` / … without AutoModel members)
- Training-only `training.py` + `arch_id` / `X-Prism-Arch-Id`
- Megatron-Bridge or any non-AutoModel framework tree

Those layouts return **`400 unsupported_layout`** (or **`400 recipe_version`**
when the advertised recipe is ≥ 2.0.0 and the payload declares/implies 1.x).

### Miner workflow (normative)

1. Clone the pin: checkout `automodel_git_commit` from `automodel_repo_url`
   (or extract the staged archive and verify `automodel_content_sha256` once
   set).
2. Edit under the AutoModel layout (new modules / configs OK).
3. Produce a unified diff: `git diff <automodel_git_commit>` (or equivalent
   `git format-patch` series folded into one `automodel.patch`).
4. Pack ZIP with `automodel.base` = recipe `automodel_pin_id`,
   `automodel.patch`, optional `prism.toml`.
5. `POST /v1/submissions` with miner hotkey + **`X-Lium-Api-Key`** (BYOK;
   unchanged from 1.x live).

### Apply / reject (fail-closed)

Master applies `automodel.patch` onto a clean pin checkout. Reject when:

| Condition | Typical code / note |
|-----------|---------------------|
| Patch does not apply cleanly | patch apply failure (conflict / missing context) |
| Path escape / touches outside allowlisted roots | hard reject |
| Binary blobs in patch | hard reject |
| Oversized diff / too many files | hard reject (budgets in intake) |
| Wrong / unknown `automodel.base` | pin mismatch |
| Legacy 1.x layout | `unsupported_layout` / `recipe_version` |
| Removes or disables telemetry / introduces network exfil / eval-set leakage | anti-cheat hard reject |

### Caps that carry forward (unchanged unless bumped)

| Cap | Value |
|-----|-------|
| **Train budget (currency)** | **`3.0e18` attested FLOPs** (`TRAIN_FLOPS_CAP`) |
| Train wall clock | **5.0 h** per submission — **safety bound, not the currency** |
| Underspend floor | **0.5 ×** the FLOPs cap (`MIN_SPEND_FRACTION`); below it the run is ineligible |
| Pod lifetime | **7.5 h** (derived; see *Budget currency* below) |
| Eval battery | **3600 s** global, per-group ceilings are fractional shares |
| Hard step cap | 20 000 (config may only lower) |
| Model parameters | ≤ **1 000 000 000** (`MAX_PARAMS`) |
| Dataset pin | FineWeb-Edu shard below (*Pinned dataset*) |
| GPU funding | Miner `X-Lium-Api-Key` on live |

### Budget currency: attested FLOPs, dual-capped

The budget is **attested FLOPs**, with wall-clock demoted to an anti-DoS
bound. **Whichever cap binds first stops the run**, and the metrics record
which one did (`org.diag.binding_cap` ∈ `flops｜wall｜steps`). The miner picks
`N` (params) and `D` (tokens) freely underneath both.

**Why not wall-clock.** A fixed wall makes MFU a *scored* quantity: two
identical architectures differ in score by kernel maturity (no `sm_120`
FlashAttention cubins, Triton version rent), and a looped model at `r=4` pays
~3.3× FLOPs/token so it sees ~3.3× fewer tokens — charged to the architecture
as if it were a defect. Measured FLOPs price looping, MoE sparsity and
vocabulary size **automatically**, so the budget adapts to the architecture
class with no tier to declare and none to shop for.

**How FLOPs are attested — the miner never reports a number.**

```
f_tok      = median over 8 harness-driven fwd+bwd passes under
             torch.utils.flop_counter.FlopCounterMode, on batches drawn from
             the real train stream at SECRET indices        (prismlib/flops.py)
C_attested = f_tok × stream.tokens_seen                     (both harness-owned)
```

Enforcement is inside `SeededTrainStream.next_batch`, which **refuses to yield
more tokens** once a cap is reached — a hard stop, not the cooperative
`ctx["guard"]` closure it replaces. Reaching your budget raises
`BudgetExhausted`, which routes to the same graceful checkpoint-then-eval path
as `finish_evaluation()`: spending the full budget is the *expected* outcome,
not a way to score zero.

If the probe cannot run at all, the FLOPs cap is left **disarmed** (the wall
bound still contains the run) and `org.diag.flops_probe_error` is emitted —
a failed measurement is never treated as a zero-cost model.

**Cheat surface, and what is still open.**

| Attack | Hardening |
|---|---|
| Under-report FLOPs | Structurally impossible: the miner never reports them |
| Input-dependent cost (MoE routing cheaply on probe-shaped inputs, early exit) | Probes are real training batches at secret indices; `org.diag.flops_probe_cv` is published, and above `FLOPS_PROBE_CV_MAX = 0.15` the estimator switches from the **median to the max** — the expensive branch is charged |
| Bypass the harness stream | `tokens_seen_source != "train_stream"` ⇒ FLOPs are **not** attested (the product inherits the token count's trust) |
| Physically impossible claim | `flops_attested ≤ peak × n_gpu × wall × 1.05` asserted; `n_gpu` is attested, not declared |
| **Opaque fused kernel** (the real hole) | `FlopCounterMode` only sees what the PyTorch dispatcher sees, and recipe-v10 lets miners install their own dependencies — a fused Triton/CUDA op registered as one opaque dispatch is **invisible**. Cross-checked against an analytic model (below) with the gap published as `org.diag.flops_analytic_ratio` / `_gap`. **Evidence for review, never a silent pass.** |

**The analytic cross-check.** `C = 6ND` is wrong at this scale:

```
F_tok = 6·N_body·r_eff·active + 6·d·V + 12·L·d·S
        (body matmuls)          (lm_head)  (attention quadratic)
```

At `d=512, V=32768` the `lm_head` alone is ~36 % of FLOPs/token, so `6·N_body`
captures only ~55 % of the true cost — `6ND` overstates the affordable token
count by **1.3–1.8×** here. Body params **exclude** embeddings and the head
(the head is charged once, by `6·d·V`); MoE counts **active** experts only;
only the body loops, which is why `r=4` costs ~3.3× and not 4×. The quadratic
attention term is charged **only when attention is detected**, so a
delta-net/SSM is not billed for a phantom cost. A gap above
`FLOPS_ANALYTIC_GAP_MAX = 0.25` sets `flops_analytic_mismatch`.

**Pod lifetime is derived, not guessed.** The pod must strictly contain both
children, and the payer's model must reconstruct it exactly:

```text
train child : build 900 + train 18000 (5.0 h) + grace 120 + checkpoint 1800 = 20820 s
eval  child : PRISM_EVAL_TIMEOUT_S 5400  (battery 3600 + load/rollup/score reserve 1800)
worst case  : 26220 s = 7.28 h      ⇒ POD_LIFETIME_HOURS_CAP = 7.5 h (780 s margin)
payer       : TRAIN_WALL_SECS + EVAL_BUDGET_SECS == 7.5 h exactly (derived from these constants)
```

`prism_lium_payer::sealed` derives its TTL from these same constants rather
than duplicating them, which is how the old 6 h/2 h payer model came to
disagree with a 7.0 h pod cap.

**Calibration status.** `TRAIN_FLOPS_CAP = 3.0e18` is sized so that any
implementation at **≥ 20 % MFU is FLOPs-bound** inside the 5.0 h wall
(4.97 h at 20 %, 3.98 h at 25 %). Real MFU is **measured, not assumed**:
Phase 0 measured **40.2 %** on 1×RTX 5090 with the reference Transformer++
(`d=1024, L=24`), so the full budget needs ≈ 2.5 h on 4 GPUs — comfortably
FLOPs-bound. See `deploy/scripts/prism-phase0-seed-variance.sh` and the
evidence under [`docs/evidence/prism-v3-phase0/`](evidence/prism-v3-phase0/).

> ### ⚠ Batch size is now load-bearing (measured, not theoretical)
>
> The **step cap and the FLOPs cap are only mutually reachable at a large
> batch.** At the reference `batch 8 × seq 512 = 4096` tokens/step, the
> `MAX_TRAIN_STEPS = 20 000` cap buys `8.2e7` tokens — and at the measured
> `F_tok = 2.22e9` for `d=1024, L=24` that is **`1.8e17` FLOPs, only ~6 % of
> `TRAIN_FLOPS_CAP`**. Phase 0 observed exactly this: the reference baseline
> stopped at **20 006 steps** with `binding_cap = none` (its own step budget),
> not at either cap.
>
> Reaching the cap inside 20 000 steps needs **batch ≈ 132 at seq 512**
> (~68k tokens/step, 16× the reference). So:
>
> - `MIN_SPEND_FRACTION = 0.5` **must not be enforced** until miners are told
>   this, or it marks the *reference baseline itself* `Ineligible` — a
>   quality-neutral failure for a batch-size reason.
> - The step cap is **cooperative** (`ctx["max_train_steps"]`; nothing
>   harness-side enforces it), so it is a *convention* the reference honors,
>   while the FLOPs cap is a hard stop inside the stream. Under the dual cap
>   the step cap is the redundant one.
>
> Asserted in `prism_recipe::tests::step_cap_and_flops_cap_are_only_mutually_reachable_at_large_batch`,
> so the interaction fails a test rather than being rediscovered on a pod.

### Recipe pin hex

`recipe_pin_hex()` remains SHA-256 over the versioned descriptor (URL, dataset
pin, AutoModel pin fields, budget, caps, harness bytes, recipe version) —
surfaced on `GET /v1/recipe` and `GET /v1/status`. Any change to these
parameters **must** bump the recipe version string so old leaves stay
unambiguous.

---

## Legacy recipe 1.x (historical — rejected on live under 2.0)

> The sections below document harness **`RECIPE_VERSION 1.4.0`** and earlier
> miner layouts (`architecture.py` + `training.py`, source-tree ZIP,
> training-only + `arch_id`). They remain for audit / leaf interpretation.
> **Do not submit 1.x layouts to live once recipe 2.0.0 is advertised.**

# PRISM recipe v1.0.2 — `prism-recipe-v1` (harness `RECIPE_VERSION 1.4.0`)

The official execution contract every miner submission was verified inside
under 1.x. Miners shipped **two scripts only** (`architecture.py` +
`training.py`) — or, since recipe **1.3.0**, a **source-tree ZIP** (see
*Source-tree submissions* below); the harness and data pin are
operator-owned. No offline weights, no network reach at pod runtime beyond
the pinned dataset pull. Recipe **1.4.0** made the tokenizer miner-chosen
and replaced the G5 long-context scored path with community protocols +
natural documents under a **pretrain-only** rule (base LM completion /
few-shot; no IFT, chat templates, or LLM judges on ranked metrics).

## Contract (1.x)

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

Under 2.0 the same telemetry contract is enforced via the harness wrap around
the AutoModel entry (miner patches must not remove or disable hooks).

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

**Anti-cheat verification (v2.2).** Tokenizer freedom is not a cheat
surface: `validate()` also computes an objective **tokenizer card**
(`METRICS_JSON["tokenizer"]["card"]`): `probe_tokens_per_byte` on a fixed
paragraph, `probe_roundtrip_ok`, a sampled vocab-shape scan
(`vocab_multiword_frac`, `vocab_max_token_bytes`) and soft `flags`
(`extreme_compression` < 0.08 tokens/byte, `multiword_tokens` — BPE/SP
pre-tokenization never merges across spaces, so alpha-space-alpha tokens
are engineered — and `lossy_roundtrip`). Flags never fail the pod run; the
card is **evidence** for the metrics-aware agentic pass, whose domain rules
(`agentic_v5`) judge **intent to game** (`tokenizer_gaming`: answer-phrase
single tokens, vocab stuffing, decode-side output rewriting, memorizing
compression) as `cheat`, while an honestly weak tokenizer (byte-level,
small vocab) is explicitly NOT a cheat. G1 already scores tokenizer-neutral
bits/byte, so a weak tokenizer only hurts its owner.

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
battery group view. Scored G1 composite anchors are `org.g1.bits_per_byte_*`
(roll-up maps those internals). Historical per-token `g1.bpb.*` stays in the
group view for debugging; leaf v2 `bpb` is still bits/token. Fill measured
`reference` values with `harness/eval/calibrate_anchors.py` after E6 baseline
GPU runs. Long-context length targets are counted in tokens of the submitted
tokenizer.

> **2.0 note:** tokenizer resolution under AutoModel patches is harness-defined
> (offline HF / AutoModel paths inside the applied tree). Do not assume the
> 1.x `build_tokenizer` / `tokenizer/` ZIP rules still apply as intake layout.

## Training-only submissions (recipe 1.2.0)

Instead of shipping both scripts, a miner may submit `training.py` +
`arch_id` referencing an already-**published** architecture (see
[`PRISM.md`](PRISM.md) § Architecture registry + competition). The master
pulls `architecture.py` from the registry; the same harness contract applies
unchanged. Published archs: `GET /v1/architectures`.

> **2.0 conflict:** live 2.0 rejects this layout (`unsupported_layout`). A
> 2.0-native architecture-competition model (if any) is **not** specified in
> this freeze — see open conflicts in the rollout plan.

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
| `train` | `PRISM_PHASE=train` | contract checks → `build_model` (**1B param cap**: breach → terminal `CAP_EXCEEDED` payload, `Score(0)`) → seeded train stream (authoritative token counter) → G6 probe curve → checkpoint |
| (gate) | — | parent prints `PHASE_TRAIN_DONE`, then holds on `$PRISM_EVAL_ASSETS_DIR/.ready`; the operator stages the public HF held-out pack (default `eval_tier=public`) + generator seed **post-train only** (fail-closed: no `.ready` → error, never a silent downgrade to embedded `public_dev`) |
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
`eval_tier` (`"public"` | `"private"` | `"public_dev"`), `gate`, `battery`, `items`.
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
(`transformer_pp`: modern GPT, ~341M params) and **hybrid delta**
(`hybrid_delta`: 3:1 gated delta-net/attention hybrid). Each tree carries
`architecture.py` + `training.py` (contract-satisfying), `count_params.py`
(prints the static parameter count as a single integer), and `NOTES.md`.
They are the reference points the v3 anchor set (`anchors/v0.json`,
currently placeholder) is measured against before any
`PRISM_SCORING_MODE=composite` flip, and the attribution reference family.

> **2.0 note:** 1.x baselines are not AutoModel patch fixtures. CI will ship a
> separate fixture patch that applies cleanly to the 2.0 pin (apply-lib /
> ci-docs todos).

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
| Train budget (currency) | **`3.0e18` attested FLOPs** — see *Budget currency* above |
| Train wall clock | **5.0 h** per submission (safety bound, not the currency) |
| Pod lifetime | **7.5 h** (derived from the phase ceilings) |
| Hard step cap | 20 000 (config may only lower) |
| Source size | 128 KiB per script (two-script intake); tree budgets per `zip_submit` |
| Model parameters | ≤ **1 000 000 000** after `build_model` (`MAX_PARAMS`) |
| `train_rows` (descriptor) | **2048** — baseline / default cut advertised on `GET /v1/recipe` |
| `val_rows` | **256** — frozen val cut scored by the harness (not miner-chosen) |

### What `train_rows` means (and what it does not)

`train_rows: 2048` is the **baseline cut** and the value injected into
`ctx["train_rows"]`. The sealed baseline (`training.py`) reads that many texts
from the pinned parquet (~2M GPT-2 tokens for that slice — **not** billions).

Egalitarian constraints are the **pinned shard + seed + FLOPs/wall/step/param
caps**. The harness hands miners `ctx["dataset_path"]` to the **full** verified
parquet; competitive `training.py` may stream or multi-pass that shard until a
cap binds — under the dual cap that is normally the **FLOPs** cap, and the
stream stops yielding batches at that point rather than relying on a guard the
miner must call. Token throughput therefore depends on the miner loop and the
rented GPU, and the affordable token count is now **explicit**: `D =
TRAIN_FLOPS_CAP / F_tok`, so ~2.5 B tokens at `d=1024, L=12` and ~10.8 B at
`d=512, L=8`. Those are **budget arithmetic**, not a recipe-published token
window.

Do not treat the marketing site’s loss-chart axis (or a leader’s telemetry
peak) as the recipe contract — always trust `GET /v1/recipe` + this doc.

**Harness note (follow-up, do not hot-fix mid-flight):** `METRICS_JSON.tokens_seen`
currently echoes `TRAIN_ROWS` (2048) even when telemetry `layer_stats.tokens`
shows billions. Changing that field would alter the recipe pin (harness bytes
are hashed) — coordinate a version bump if/when fixing it.

The **parameter cap is 1B** (raised from 350M alongside the 4×RTX 5090
recipe-v10 pod). Under the iso-FLOPs currency that cap is **non-binding**: at
`C_MAX` the compute optimum is around `N_body ≈ 143 M`, and the 0.02-nat
plateau spans ~88–236 M body params (a 2.7× range), so *compute* binds and the
cap is a VRAM/checkpoint parameter rather than a scientific one. Placeholder
anchors and the public GPT-2 Large reference row MUST be re-measured under the
dual cap before any `PRISM_ANCHOR_VERSION` / composite governance flip
(v0/v1/v2 anchors stay byte-frozen with their own pre-registration hashes, so
neither the raise nor the currency change silently invalidates them).
The parameter-cap breach semantics changed in 1.3.0: it is a terminal
`Score(0)` (`CAP_EXCEEDED`), not an infra retry.

## Recipe pin (1.x descriptor)

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

> **2.0:** copy / similarity use a **patch fingerprint** (`patch_sha256` in the
> store architecture surface) plus **touched-file AST** (concatenated post-apply
> `.py` bodies), not the whole AutoModel tree. Agentic primaries are
> `.prism/automodel.patch`, `.prism/diffstat.json`, `.prism/review_brief.md`,
> and touched paths only. Trainer/data-class edits get higher scrutiny in the
> brief + domain rules; static screens hard-reject telemetry disable, network
> exfil, and eval-set leakage patterns in added hunk lines. Harness gates
> (netns, telemetry wrap, budget, causal screen) stay unchanged.
