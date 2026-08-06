# Appendix 01 — Current State: Prism Implementation (Technical Map)
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 by codebase exploration. Non-normative spike document.

# Prism challenge — technical map

Prism is **not** “miners upload trained weights.” Miners submit **Python recipe code** (`architecture.py` + `training.py`, or training-only + published `arch_id`). The operator harness trains and scores on a **public** fineweb-edu parquet pin; the leaf score is **pure bits-per-byte (bpb)** on a fixed val cut, mapped into `[0, SCORE_MAX]`. Evaluation and leaf emission run **master-only** on Lium GPUs (Sim in CI). Validators only fetch sealed weights.

Normative docs: `/root/gbase/docs/PRISM.md`, `/root/gbase/docs/PRISM_RECIPE.md`, miner mirror `/root/gbase/docs/external-miner/prism.md`.

---

## 1. Where the service lives and crate graph

### Binary
| Path | Role |
|------|------|
| `/root/gbase/bins/prism-challenge/` | Operator binary `prism-challenge` on **:8092** |
| `/root/gbase/bins/prism-challenge/src/main.rs` | Backend selection, workers, emitter, gating |

### Library crates (`crates/`)

| Crate | Role |
|-------|------|
| `prism-challenge` | HTTP API, orchestrator, `combine_final`, leaf helper |
| `prism-challenge-task` | Identity: `challenge_id=prism`, `SCORING_VERSION=2`, `SCORE_MAX` |
| `prism-recipe` | Contract, dataset pin, caps, baseline, embedded harness |
| `prism-pipeline` | Intake validation + eval pipeline + `score_from_bpb` |
| `prism-lium` | Lium client / SSH exec / Sim backend / receipts |
| `prism-review` | OpenRouter quality + arch-only similarity (or Sim) |
| `prism-store` | Postgres/memory submissions, arch registry, emit outbox |
| `prism-registry` | Competition emission math + top-model GitHub publish |
| `prism-emit` | Epoch-close D24 leaf emission → gateway |
| Shared | `challenge-agentic`, `challenge-ast`, `challenge-common`, `submission-gating`, `bundle`, `db` |

`prism-challenge` deps (from `/root/gbase/crates/prism-challenge/Cargo.toml` lines 11–40): `prism-challenge-task`, `prism-emit`, `prism-lium`, `prism-pipeline`, `prism-recipe`, `prism-registry`, `prism-review`, `prism-store`, `challenge-agentic`, `challenge-common`, `submission-gating`, `bundle`, `db`, …

Identity constants:

```19:47:/root/gbase/crates/prism-challenge-task/src/lib.rs
pub const CHALLENGE_ID: &str = "prism";
// ...
pub const SCORING_VERSION: u16 = 2;
// ...
pub const SCORE_MAX: u64 = 1_000_000;
```

---

## 2. Miner submissions — what and how

### What miners submit
**Not** model weight files. Two kinds:

1. **Architecture submission**: `architecture.py` (`build_model(ctx)`) + `training.py` (`train(model, ctx)`).
2. **Training-only** (recipe ≥1.2.0): `training.py` + `arch_id` (`arch_<16 hex>`); architecture pulled from registry.

ZIP preferred; JSON sources / `zip_base64` for CI. Optional header `X-Prism-Arch-Id` for training-only ZIP.

Contract validation (`/root/gbase/crates/prism-pipeline/src/submission.rs` 160–203; recipe `check_contract` at `/root/gbase/crates/prism-recipe/src/lib.rs` 149–163): size ≤128 KiB/script, required entrypoints, hotkey = 64 hex chars.

Idempotency: `submission_id = sha256(hotkey ‖ architecture_py ‖ training_py)` (lines 241–249 of `submission.rs`).

### HTTP routes (challenge service, not gateway weights)

From `/root/gbase/crates/prism-challenge/src/api.rs` 56–69:

| Route | Purpose |
|-------|---------|
| `POST /v1/submissions` | Accept submission |
| `GET /v1/submissions` | List |
| `GET /v1/submissions/{id}` | Detail + scores/receipt |
| `GET /v1/submissions/{id}/events` | Stage timeline |
| `POST /v1/submissions/{id}/retry` | Manual retry |
| `GET /v1/architectures` | Published arch registry |
| `GET /v1/status`, `/v1/jobs`, `/v1/recipe`, `/v1/recipe/baseline` | Ops / recipe pin |
| `GET /health` | Liveness |

Public path via gateway proxy: `$BASE_GATEWAY/challenge/prism/v1/submissions` (`docs/external-miner/prism.md` 40–55).

### Relation to `POST /v1/weights/raw`
Miners **do not** POST weights/raw. The **challenge service** (epoch emitter) signs D24 leaves and posts them to the **gateway**:

```126:126:/root/gbase/crates/challenge-common/src/submit.rs
        let url = format!("{}/v1/weights/raw", self.cfg.base_url.trim_end_matches('/'));
```

Gateway mounts that route in `/root/gbase/crates/gateway/src/weights.rs` 26–29.

---

## 3. Scoring / evaluation (detail)

### Pipeline (measure)
Orchestrator phases (`orchestrator.rs` 327–398):

0. Pre-LLM **copy gate** (arch only; skip if training-only)  
1. **Measure**: Lium provision → SSH upload harness + sources → run harness → parse `METRICS_JSON` → terminate  
2. LLM **coherence review** (audit; not a grader)  
3. Cheap **similarity** (arch-only; training-only forced `Original`)  
4. **Agentic** anti-cheat  
5. `combine_final` → store score → registry/top-model hooks  
6. Leaf emission later via `prism-emit` outbox (not inline)

### Metric today: **bits-per-byte (bpb)** from val CE

Harness (`/root/gbase/crates/prism-recipe/harness/prism_harness.py`):

```289:318:/root/gbase/crates/prism-recipe/harness/prism_harness.py
    val_texts = texts[TRAIN_ROWS : TRAIN_ROWS + VAL_ROWS]
    # Score: mean cross-entropy over frozen tokens -> bpb.
    model.eval()
    losses = []
    # ...
            l = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="mean")
            losses.append(l.item())
    ce = sum(losses) / len(losses)
    bpb = ce / 0.6931471805599453  # ln 2
```

- Tokenizer for scoring: **GPT-2** (`TOKENIZER = "gpt2"`, line 49).  
- Val: **256** texts at indices `[2048, 2304)` of filtered shard texts.  
- Mean CE across val texts → bpb = CE / ln(2).  
- Miner cannot own the val loop; harness scores after `train()`.

Receipt parsing rejects non-finite / non-positive bpb (`prism-lium` `client.rs` 478–483).

### Lattice map: `score_from_bpb`

```21:38:/root/gbase/crates/prism-pipeline/src/score.rs
/// Invert BPB into lattice score: lower bpb → higher score.
/// Uses a soft map: `score = SCORE_MAX * (1 / (1 + bpb))` clamped.
pub fn score_from_bpb(bpb: f64) -> u64 {
    if !bpb.is_finite() || bpb < 0.0 {
        return 0;
    }
    let quality = 1.0 / (1.0 + bpb);
    let v = (quality * (SCORE_MAX as f64)).round();
    // ...
}
```

### Final gates: `combine_final` (scoring v2)

```40:64:/root/gbase/crates/prism-challenge/src/score.rs
pub fn combine_final(outcome: &FinalOutcome) -> prism_store::FinalScore {
    match outcome {
        FinalOutcome::ChallengeInternal => { /* NoScore ChallengeInternal */ }
        FinalOutcome::Measured { bpb, quality: _, similarity, agentic } => {
            if matches!(agentic, VerdictKind::Cheat | VerdictKind::Suspicious) {
                return FinalScore::Score(0);
            }
            if matches!(similarity, Copied | Suspicious) {
                return FinalScore::Score(0);
            }
            FinalScore::Score(score_from_bpb(*bpb))
        }
    }
}
```

**LLM `quality_score` is ignored for the number** (explicitly `_`). Docs/tests assert quality never moves the score (`score.rs` 97–112).

### Competition aggregation → leaves

Per epoch batch, `/root/gbase/crates/prism-registry/src/competition.rs` 27–79:

- Challenger credit = max own `Score` across batch rows  
- Owner credit = max score on each owned arch (any trainer) → credited to owner  
- Emission = `max(own, owner)` (never sum)  
- Then `prism-emit` fills D24 set with `NoScore(NotAttempted)` for other metagraph hotkeys and submits via `POST /v1/weights/raw`

```181:211:/root/gbase/crates/prism-emit/src/lib.rs
pub fn build_epoch_leaves(...) -> Result<...> {
    let by_miner = prism_registry::competition_scores(batch, arch_owners);
    // ... Score / NoScore / NotAttempted per expected hotkey
    challenge_common::emit_signed_leaf_set(secret, CHALLENGE_ID_BYTES, epoch, ...)
}
```

---

## 4. Datasets / data pipeline

| Item | Value | Source |
|------|-------|--------|
| Dataset | `HuggingFaceFW/fineweb-edu@sample/10BT` | `prism-recipe` `DATASET_REF` L58 |
| Shard URL | `…/sample/10BT/010_00000.parquet` | `DATASET_URL` L54–55 |
| Size | 2 152 798 864 bytes | `dataset_len_bytes()` L63–65 |
| SHA-256 | `e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6` | `DATASET_SHA256` L71; overridable via `PRISM_DATASET_SHA256` L75–80 |
| Filter | strings with `len(t) >= 100` | harness `load_texts` L179–187 |
| Train rows (ctx hint) | **2048** | `TRAIN_ROWS` L121–122; harness L48 |
| Val rows | **256** = `texts[2048:2304]` | harness L291 |
| Seed | `0x00505249534D` | `RECIPE_SEED` L116 |

**Critical redesign facts (from recipe crate docs L14–18):**
- **No held-out private val.** Val cut is published as part of the pin.  
- Miners get `dataset_path` to the full verified parquet → can read **beyond** the first 2048 texts (including the “val” slice) if they choose. Anti-overfit relies on anti-cheat + seed lattice, **not** data secrecy.  
- Dedup: only length filter; no cross-doc dedup in harness.  
- Pod network: dataset URL + GPT-2 tokenizer download only (harness docstring L7–9).

---

## 5. Architecture / compute constraints

| Constraint | Value | Where |
|------------|-------|--------|
| Model API | Any `torch.nn.Module` from `build_model` | harness L258–259 |
| Transformer required? | **No** — baseline is TinyGPT (~12M) but contract is Module + logits | baseline `architecture.py` L11–46 |
| Vocab / tokenizer assumption | Scoring uses **GPT-2** (50257); model must emit logits over that vocab | harness L49, L212–316 |
| Max params | **350 000 000** after `build_model` | `MAX_PARAMS` L47–48; harness L263–266 |
| Source size | 128 KiB per script | `MAX_SOURCE_BYTES` L143 |
| Train wall clock | **6 h** | `TRAIN_HOURS_CAP` L84; harness env |
| Pod lifetime | **7 h** (orchestrator uses this) | `POD_LIFETIME_HOURS_CAP` L87; `OrchestratorConfig` default in `orchestrator.rs` L72 / `main.rs` L424 |
| Step cap | **20 000** | `MAX_TRAIN_STEPS` L113 |
| Offline weights | Forbidden by recipe (“no offline weights”) | `PRISM_RECIPE.md` L4–6 |
| Sandbox | Miner code runs **inside operator harness on rented Lium GPU pod** (SSH upload); not Phala/TDX; CUDA required | harness L207; `PRISM.md` L6 |
| Concurrent evals | Default **1** | `PRISM_MAX_CONCURRENT_EVALS` |
| Price guard | ~$2.5/h (orchestrator/main) | `main.rs` L423 |

Context-window rule: harness aligns targets to model logits length so self-truncating archs (baseline `block=512`) still score (`PRISM_RECIPE.md` L85–93; harness L305–311).

---

## 6. Anti-cheat / overfitting / plagiarism / contamination

| Layer | Mechanism | Effect |
|-------|-----------|--------|
| Pre-LLM copy gate | Byte hash + AST vs earlier submissions; ≥ **9500 bps** AST = cheat | Terminal `Rejected` + `Score(0)`; no pod/LLM (`orchestrator.rs` 332–337, 441–495; `challenge-ast/src/gate.rs` 10–11, 56–85) |
| Baseline exempt | Corpus ids `baseline*` skipped | `BASELINE_CORPUS_PREFIX` |
| Training-only skip | Copy gate + similarity skipped | `copy_gate_step` L450–452; `similarity_step` L644–653 |
| LLM similarity v2 | Arch-only vs baseline + recent (cap 6) | `Copied`/`Suspicious` → Score 0 |
| LLM quality review | Coherence / telemetry checklist | Audit only; low quality does **not** zero via `combine_final` — agentic is the hard enforcer for telemetry |
| Agentic (`PRISM_DOMAIN_RULES`) | Tools: list/read/AST/metrics; codes: `missing_telemetry_hooks`, `inconsistent_metrics`, `eval_short_circuit`, `ast_architecture_copy`, … | Cheat/Suspicious → Score 0; missing verdict → `ChallengeInternal` |
| Telemetry contract | Must call `prism_telemetry.report` + `finish_evaluation` | SimAgent string check (`sim.rs` 182–204); live LLM via domain rules |
| Metrics integrity | Operator-collected `METRICS_JSON`; miner-printed metrics = short-circuit | Domain rules in `challenge-agentic/src/prompts.rs` L8–14 |
| Receipt / terminate | Pod terminate verified; image digest optional | pipeline / measure path |
| Dataset pin | SHA mismatch → fail, never score | harness L173–174 |
| Contamination | **Weak by design**: public val; full shard on disk | Recipe docs L14–18 |

Cheat taxonomy also in `docs/PRISM.md` L184–192.

---

## 7. Round / epoch / quota structure

There is **no Design-style admin round**. Structure:

**Intake gating (`submission_gating`):**
- Hotkey must be in metagraph (`403` / `503`).  
- **One** accepted architecture submission per hotkey under key `"prism"`.  
- Training-only: one per `(hotkey, arch_id)` under `"prism:train:<arch_id>"` (`submission.rs` 228–238).  
- Infra failures (`install`, `ast_infra`, `llm_infra`): auto-retry up to `PRISM_AUTO_RETRY_MAX` (default 3).  
- Cheat/copy: terminal `rejected`.  
- Watcher reopens when hotkey leaves metagraph (`BASE_GATING_WATCH_SECS`, default 120).

**Eval cadence:** On submit → queue → worker claims when concurrency free. Prod train up to 6h; not epoch-aligned.

**Emission cadence:** One D24 leaf set **per chain epoch** at first emitter tick observing epoch `E` (`prism-emit` module docs L1–24; poll `emit_poll` 15s in `main.rs` L428). Acceptance epoch on the row is metadata only; score lands in the **next** epoch-close batch after finalize.

Stuck rows: sweeper after **7h** grace (`stuck_grace_secs: 7 * 3600`, `main.rs` L431).

---

## 8. Config knobs (env)

| Env | Effect |
|-----|--------|
| `BASE_CHALLENGE_BIND` | Bind (default `0.0.0.0:8092`) |
| `BASE_CHALLENGE_SK_FILE` / `PRISM_CHALLENGE_SK_FILE` | Leaf signing secret |
| `BASE_DATABASE_URL` | Postgres vs memory |
| `BASE_NETUID` | Netuid / emit cursor |
| `BASE_CHAIN_ENDPOINT` / `BASE_CHAIN_ENDPOINTS` | Epoch resolution |
| `BASE_CHALLENGE_GATEWAY_ENDPOINT` | Gateway for `/v1/weights/raw` |
| `BASE_SUBMISSION_GATING=0` | Disable metagraph/1-max (dev) |
| `BASE_GATING_WATCH_SECS` | Watcher cadence |
| `PRISM_FORCE_SIM` | Force Sim Lium |
| `PRISM_MAX_CONCURRENT_EVALS` | Worker concurrency (default 1) |
| `PRISM_AUTO_RETRY_MAX` | Infra auto-retries (default 3) |
| `PRISM_SIM_STAGE_DELAY_MS` | E2E mid-flight delay (never staging/prod) |
| `PRISM_DATASET_SHA256` | Override dataset pin |
| `PRISM_TEST_TRAIN_MINUTES` | Shrink train wall (staging/e2e) |
| `PRISM_TEST_MAX_PARAMS` | Tiny param cap (staging/e2e) |
| `LIUM_API_KEY` / `_FILE` / credentials | Real Lium |
| `LIUM_SSH_PUBLIC_KEY_FILE` / `LIUM_SSH_PRIVATE_KEY` | Pod SSH |
| `PRISM_SSH_ATTEMPTS` / `PRISM_SSH_RETRY_SECS` / `PRISM_SSH_RUNNING_TIMEOUT_SECS` | SSH settle |
| `OPENROUTER_API_KEY_FILE` | Real LLM review + agentic; else Sim |
| `PRISM_TOPMODEL_GITHUB_TOKEN_FILE` | Publish global-best to public repo |

Harness-side (set by Lium client): `PRISM_DATASET_URL`, `PRISM_DATASET_SHA256`, `PRISM_MAX_TRAIN_STEPS`, `PRISM_TRAIN_HOURS_CAP`, `PRISM_GPU_TYPE`, optional test forwards (`client.rs` 436–451, 515–529).

Example env file: `/root/gbase/deploy/env/prism-challenge.env.example`.

---

## Essential snippets (quick reference)

**Submission shape** (`submission.rs` 26–46):

```rust
pub struct SubmissionRequest {
    pub miner_hotkey: String,
    pub architecture_py: String,  // empty if arch_id set
    pub training_py: String,
    pub zip_base64: Option<String>,
    pub arch_id: Option<String>,
    pub label: Option<String>,
}
```

**Val scoring** — see harness block quoted in §3 (lines 289–318).

**Score map** — `score_from_bpb` in `prism-pipeline/src/score.rs` 25–38; gates in `combine_final` `prism-challenge/src/score.rs` 40–64.

**Data loading**:

```179:187:/root/gbase/crates/prism-recipe/harness/prism_harness.py
def load_texts(parquet_path):
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    texts = [t for t in texts if isinstance(t, str) and len(t) >= 100]
```

---

## Redesign-relevant gaps (facts only)

1. **Public val + full parquet path** → classic contamination/overfit surface.  
2. **Scoring assumes GPT-2 vocab/logits**; non-Transformer OK if logits shape matches.  
3. **`quality_score` does not gate** numerically; missing telemetry must be caught by agentic (Sim is string-based; live LLM can miss aliases).  
4. **`tokens_seen` in METRICS_JSON is hardcoded to `TRAIN_ROWS` (2048)**, not real tokens (`harness.py` L321) — weakens metrics-consistency checks.  
5. Emission is **epoch-batch / competition max**, not per-submission immediate leaf.  
6. Eval concurrency and Lium cost caps dominate throughput, not chain epoch length (~72 min vs ≤6h train).

For operator ceremony / Lium enablement: `/root/gbase/docs/runbooks/prism-enable-lium-and-emission.md`.
