# base Hypertraining Challenge Specification

**Status:** DRAFT (task 3 freeze skeleton: structure locked; implementation fills remaining pins)  
**Normative design source:** [`/root/challenge-training-fork.md`](/root/challenge-training-fork.md) (v1.0)  
**Normative for (when FROZEN):** `hypertraining-challenge` service, miner submit API, SimBackend tournament path, leaf emission under `challenge_id = hypertraining`  
**challenge_id:** `hypertraining`  
**challenge_scoring_version:** `1`  
**Bundle leaf protocol:** [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) **`protocol_version = 1`** (unchanged)

This file is the freeze surface for the base **hypertraining** challenge: miner-owned training-code forks, validator-owned measurement, three independent guards, integer-only leaf scores, and a software-first path that runs today on **SimBackend** without live B300 hardware.

Where this document and any other source disagree on **hypertraining-challenge behaviour**, **this document wins** once status is FROZEN.  
Where this document and [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) disagree on **leaf bytes, signatures, aggregation, or bundle verify**, **BUNDLE_SPEC wins**.  
Where this document and the design brief disagree on **tournament economics, sealed surface, or statistical gates**, the brief is the design source until this freeze is completed (task 17).

Checklist map: [`HYPERTRAINING_CHECKLIST.md`](./HYPERTRAINING_CHECKLIST.md).  
CI gate (planned, task 17): `cargo run -p xtask -- hypertraining-check`.

### Draft gate note (task 3)

This skeleton freezes **identifiers, topology split, emission posture, sealed-surface summary, three guards, integer scoring path, D24 coverage, and must-not claims**. Implementation crates and Sim E2E land in later plan todos. Do **not** treat Real B300, live MFU measurement, or non-zero emission as current state.

---

## 0. Document conventions

| Notation | Meaning |
|----------|---------|
| `u8`/`u16`/`u32`/`u64` | SCALE fixed-width little-endian unsigned integers |
| `[u8; N]` | Fixed-length byte array |
| `Bytes` / `Vec<T>` | SCALE compact length + payload |
| `scale(T)` | Canonical SCALE encoding |
| HTTP JSON | Allowed on miner submit and challenge admin hops. Forbidden in bundle leaves (BUNDLE_SPEC §1) |

Hotkeys inside SCALE structures are `[u8; 32]` raw public keys, not SS58 strings.

Cross-reference: leaf `ScoreOrAbsence`, `NoScoreReasonCode`, and `base-rawweight-v1` signing are defined in BUNDLE_SPEC §3. This document defines **when** each reason is chosen and **how** `Score { value: u64 }` is computed under `challenge_scoring_version = 1` for hypertraining.

### 0.1 Invariants (H1-H8)

| id | invariant |
|----|-----------|
| H1 | Miner owns the training fork; validator owns measurement. Miner-reported metrics are never scoring inputs |
| H2 | Exactly one leaf per expected `(challenge_id=hypertraining, hotkey)` per epoch; silence is a bug (D24) |
| H3 | No floating point in the final public score path (integer / fixed-point only → `u64`) |
| H4 | Challenge signing key never inside untrusted miner train jobs |
| H5 | Validators verify and recompute bundle crypto; they do not re-run the tournament scorer (D19 posture) |
| H6 | Bundle `protocol_version` stays **1** |
| H7 | Trust-root `emission_share_bps` for hypertraining is **0** until a separate owner ceremony |
| H8 | Live path is **SimBackend**; Real B300 is deferred (stub / runbook only) |

---

## 1. What runs where (topology)

Three trust domains. Do not collapse them.

```text
Operator / master host (compose profile master)
  postgres · gateway · validator · updater
  + hypertraining-challenge service (registered challenge backend)
           | HTTPS (gateway TLS terminates)
           | proxy path /challenge/hypertraining/*
           v
Challenge service process (orchestrator)
  - challenge signing secret (mounted file only)
  - expected set from local trust root + metagraph (D24)
  - sealed-surface admission, hermetic build, kernel gate
  - ClusterBackend: SimBackend (now) | RealBackend stub (later)
  - eval guards 2-3, promotion machine, anti-noise, pay map
  - score, sign leaves, POST /v1/weights/raw
           | miner submit HTTPS (repo/commit/tree + topology + precision_attestation)
           v
Miner (training-code owner)
  - submits full training fork (allowlisted paths only)
  - does NOT own eval, dataset order, token accounting, or leaf signing
  - does NOT get credit for self-reported wallclock or loss
           |
           v
Validator
  - attestation policy (when required by profile)
  - bundle verify per BUNDLE_SPEC; does NOT re-score tournament internals (D19)
```

| Component | Role | Holds challenge sk? | Signs leaves? |
|-----------|------|---------------------|---------------|
| Miner fork / train job | Untrusted training code under sealed surface | No | No |
| hypertraining-challenge | Admit, build, measure, score, emit leaves | **Yes** (file) | **Yes** |
| ClusterBackend::SimBackend | Deterministic / controllable wallclock + fake checkpoints | No | No |
| ClusterBackend::RealBackend | Deferred B300 path; returns NotConfigured until enabled | No | No |
| Gateway | Master routing only | No | Bundle only |
| Validator | Attestation + bundle crypto | No | Dissent / peer-root |

**Normative split (director principle):**

1. **Miner** owns the training implementation (kernels, parallelism strategy, communication overlap within allowlist).  
2. **Challenge / validator measurement path** owns data slice, eval image, wallclock, quality guards, and payment.  
3. **Only artifact that crosses the trust boundary for quality** is a **checkpoint** (hash-verified), reloaded and evaluated in a validator-controlled image. Miner metrics are ignored by construction.

**Hardware posture (current):**

| Mode | Status |
|------|--------|
| SimBackend | **Current** implementation target; CI and E2E use sim only |
| Real B300 dual-cluster A/B | **Not live**; operator runbook later (plan todo 18) |
| Live MFU §10.4 measurement | **Not claimed**; first real-cluster day-one op when HW arrives |

---

## 2. Identifiers and versions

| Field | Value |
|-------|-------|
| `challenge_id` | UTF-8 `hypertraining` (SCALE `Bytes`) |
| `challenge_scoring_version` | `u16 = 1` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| Trust-root row | Owner-signed `config/challenges.toml`: id `hypertraining`, public key, **`emission_share_bps = 0`**, `ParticipantPolicy` |
| agent-v1 emission | Remains **`10000` bps** (sole non-zero share until ceremony) |
| TE pin | `2.18.0+e7c550c5` |
| mlm_commit | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` |

Bump `challenge_scoring_version` when sealed surface, guard math, score map, submit schema, or compose service/port contract changes in a score-affecting way.  
Leaf SCALE layout changes go through BUNDLE_SPEC `protocol_version` (still **1** for this freeze).

### 2.1 Emission posture (normative)

| Challenge | `emission_share_bps` | Notes |
|-----------|----------------------|--------|
| `agent-v1` | `10000` | Unchanged by hypertraining work |
| `hypertraining` | `0` | Registered; aggregate may skip zero-share; **no pay from chain emission until owner ceremony** |

Non-zero hypertraining emission requires a **separate** owner ceremony (resign `challenges.toml`, restart services). This document MUST NOT describe non-zero emission as current state.

### 2.2 Branding

User-facing product name is **base**. Do not use the monorepo checkout directory name as a product or challenge brand in strings, docs prose, or miner-facing errors. Paths on disk may still name the checkout folder; that is not product branding.

---

## 3. Attestation precondition (profile-aware)

### 3.1 Production posture

When `require_attestation = true` (production default intent):

**A same-epoch `Verified` attestation is a precondition for emitting `Score { value }` for that miner**, matching agent-v1 I1 semantics under D10 binding.

| Attestation outcome this epoch | Leaf for an expected miner |
|--------------------------------|----------------------------|
| `Verified` | May be `Score` or `NoScore` (other reasons) |
| `Rejected` / `Parked` / Missing | MUST `NoScore { reason: AttestationNotVerified }` (code **3**) |

### 3.2 Sim / offline posture

Sim E2E and unit tests may set `require_attestation = false` so the tournament pipeline can be proven without a live attest channel. That flag is **test/sim configuration**, not a claim that production skips attestation.

### 3.3 What attestation does not prove

Attestation (when used) proves measured identity binding for the epoch. It does **not** prove:

- honesty of operator measurement (D19);
- that a fork is free of subtle numeric bias (guards 1-3 do that work);
- that CUDA sandboxing confined the miner (it does not; see §11).

---

## 4. Miner submit protocol (challenge ↔ miner)

Transport: HTTPS. `Content-Type: application/json; charset=utf-8`.

### 4.1 Challenge service endpoints (planned)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness |
| `POST` | `/v1/submissions` | Accept fork submission (brief §7) |
| Admin / internal | TBD in implementation | enqueue, status |

Default compose port for hypertraining-challenge: **8091** (agent-challenge remains **8090**).

Gateway registration (master-only admin API, D3/D18):

```http
POST /v1/admin/backends
Content-Type: application/json

{
  "challenge_id": "hypertraining",
  "base_url": "http://hypertraining-challenge:8091"
}
```

| Field | Value |
|-------|-------|
| `challenge_id` | UTF-8 `hypertraining` (must match trust-root row in `config/challenges.toml`) |
| `base_url` | Compose service URL `http://hypertraining-challenge:8091` (no trailing slash; `http://` or `https://` required) |
| `weight` | Optional; default `1` |

Response: `201 Created` with backend view (`id`, `challenge_id`, `base_url`, `weight`, `healthy`, `fail_count`, `ejected`). **No signing keys** in request or response (D18 — keys stay in owner-signed trust root only).

Proxy path after registration: `/challenge/hypertraining/*` → round-robin among healthy backends for that `challenge_id`. List filter: `GET /v1/admin/backends?challenge_id=hypertraining`.

`agent-v1` remains registered separately (`http://agent-challenge:8090` / port **8090**). Multi-challenge registries are first-class; do not hardcode a single challenge id in gateway routing.

### 4.2 Submission body (brief §7)

```json
{
  "repo_url":   "https://...",
  "commit_sha": "...",
  "tree_sha":   "...",
  "topology":   { "tp": 4, "pp": 2, "ep": 8, "cp": 1 },
  "precision_attestation": {
    "format":              "fp8_e4m3 | bf16 | mixed",
    "accumulate_dtype":    "fp32",
    "accumulate_interval": 128,
    "scaling_recipe":      "delayed | current | block",
    "allow_tf32":          false
  }
}
```

`precision_attestation` is **mandatory and binding**. `allow_tf32 = false` is forced in the harness when required by policy.

### 4.3 Pipeline order (cheap filters first)

```text
1. ADMISSION     sealed surface: allowlist + denylist hashes + sealed symbol AST
2. BUILD         offline, validator lock/wheelhouse, immutable image digest
3. GATE KERNEL   guard 1 (κ=2), single-device / CPU fixture path OK in sim
4. TRAIN         exclusive slot via ClusterBackend (Sim now)
5. EXTRACT       checkpoint + integrity; validator re-hash
6. SCORE         eval in validator image only
7. GATES 2 & 3   quality non-inferiority + physical plausibility
8. PROMOTION     state machine §9
9. PAY MAP       marginal Δ → integer Score
10. LEAVES       D24 full cover → gateway
```

---

## 5. Sealed surface summary

Full tables live in the design brief §6. Freeze summary:

### 5.1 Allowlist (miner may change)

```text
megatron/core/fusions/**
megatron/core/extensions/**
megatron/core/transformer/**            except moe_logging.py
megatron/core/tensor_parallel/**
megatron/core/pipeline_parallel/**
megatron/core/distributed/**
megatron/core/optimizer/**
megatron/core/parallel_state.py
megatron/core/model_parallel_config.py
miner_ext/**                            dedicated custom extensions package
```

### 5.2 Denylist (any touch → admit reject)

```text
megatron/core/datasets/**
megatron/core/dist_checkpointing/**
megatron/core/num_microbatches_calculator.py
megatron/training/checkpointing.py
megatron/bridge/data/**
megatron/bridge/training/eval.py
pyproject.toml, uv.lock, 3rdparty/
```

### 5.3 Sealed symbols (AST fingerprint)

Frozen symbols include (brief §6.4):

| Symbol role | Intent |
|-------------|--------|
| `consumed_train_samples` increment | token accounting |
| `while iteration < args.train_iters` | stop condition |
| `num_floating_point_operations()` | FLOP accounting |
| `update_num_microbatches(...)` | GBS / MBS coherence |

### 5.4 Manifest pin fields

```yaml
sealed_surface.v1:
  base_commit:     <sha Megatron-Bridge>
  mlm_commit:      cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54
  te_version:      2.18.0+e7c550c5
  denylist_hashes: { <path>: <sha256> }
  sealed_symbols:  { <path>:<symbol>: <ast_hash> }
  dataset_pin:     { corpus: fineweb-edu, revision: <sha>, order_seed: <fixed> }
  segment:         { tokens: T_seg, gbs: <fixed>, seq_len: <fixed> }
```

---

## 6. Three guards summary

Guards are **independent**. All required for promotion pay.

### 6.1 Guard 1: Kernel numeric correctness (κ = 2)

```text
max|candidate − ref_fp32|  ≤  κ · max|baseline_same_dtype − ref_fp32|
κ = 2
```

Forward and backward: outputs, input grads, parameter grads. Runs before exclusive train slot allocation. Precision attestation checked mechanically (`allow_tf32=false` when forced).

### 6.2 Guard 2: Quality non-inferiority

```text
d_i = L_champion^(i) − L_candidate^(i)     paired seeds i = 1..K

H0 : E[d] ≤ −ε
Promotion allowed only if H0 is rejected (one-sided paired test, α = 0.05)
ε = min(0.25% · L , 0.5 · σ̂_d)
```

Primary metric: **continuous validation loss** (never discrete bench as primary).  
Until real calibration, implementations MAY use `MUST_CALIBRATE` placeholders for σ̂_d with documented defaults.

### 6.3 Guard 3: Physical plausibility

**FLOP count is not used** as a cheat detector (FlashAttention counterexample). Signals:

| Counter | Detects |
|---------|---------|
| DRAM bytes vs analytic model | skipped work |
| Tensor-core ops vs expected Θ | skipped tiles |
| MMA instruction family | silent precision downgrade |
| Roofline: speedup vs peak bandwidth | physically impossible speedup |

In **SimBackend**, telemetry is fixture/sim counters compared to analytic thresholds (not live Nsight).

---

## 7. Score meaning and scoring rule (`challenge_scoring_version = 1`)

### 7.1 What the score means

Hypertraining pays **marginal wallclock gain** over the current champion, not absolute performance and not validation-loss improvement.

```text
Δ(candidate) = T_champion − T_candidate     (saved compute time)
pay ∝ max(Δ, 0)  if and only if guards 1-3 passed and promotion rules allow
```

Resubmitting the champion unchanged → Δ = 0 → score 0.

### 7.2 Integer map

| Field | Value |
|-------|-------|
| Leaf score type | `Score { value: u64 }` or `NoScore { reason }` |
| Range | `0 ..= SCORE_MAX` |
| `SCORE_MAX` | `1_000_000` |
| Domain tag for leaf sig | `base-rawweight-v1` |
| Floating point in final score API | **Forbidden** (fixed-point internal math OK if public API is integer-only) |

Mapping from vested marginal reward to `u64` is implementation-defined under scoring_version 1 but MUST be monotone in vested Δ, MUST clamp to `[0, SCORE_MAX]`, and MUST yield `0` when Δ ≤ 0 or guards fail.

### 7.3 Vesting and clawback (summary)

```text
release = 1/V per subsequent segment over V segments
clawback = unpaid remainder suspended on detected regression
```

### 7.4 Anti-noise (summary, brief §12)

| Binary similarity to champion | Promotion K |
|-------------------------------|-------------|
| < 0.30 | 5 (base) |
| 0.30 - 0.60 | 7 |
| 0.60 - 0.85 | 11 |
| > 0.85 | **automatic reject, no measure** |

Fingerprint dedupe: same normalized binary fingerprint from the same miner rejected for N segments.  
**LLM is never a gate:** no admit, no reject, no pay decision. Advisory only (default Noop).

### 7.5 Promotion state machine (summary)

```text
ADMITTED → SCREENED (K=3) → DUELLED (K=5) → CONFIRMED (holdout) → CHAMPION
                │ fail              │ fail           │ disagree
                └──────────────────┴────────────────┴──► REJECTED
CHAMPION ── later regression ──► ROLLBACK to prior hashed checkpoint
```

Screen K = 3, promotion K = 5, calibration K = 10 (config; real HW when available). α = 0.05 with Benjamini-Hochberg across challengers.

---

## 8. Key custody (challenge signing key)

| Rule | Requirement |
|------|-------------|
| Algorithm | sr25519 (same as BUNDLE_SPEC leaf sigs) |
| Public key | Committed in owner-signed `config/challenges.toml` for `hypertraining` |
| Secret key | **Never** in git. Project convention: `~/.base-secrets/challenge-hypertraining.age` (or equivalent); runtime file mode 0600 |
| Runtime load | `BASE_CHALLENGE_SK_FILE` |
| Signing domain | `base-rawweight-v1` over `RawWeightBodyV1` |
| Gateway DB | Must not store challenge secrets (D18) |

Validators verify leaf signatures only against their **local** trust-root copy.

---

## 9. Declared participant set and `NoScore` reasons (D24)

### 9.1 Who defines the set

1. Read `ParticipantPolicy` for `hypertraining` from local owner-signed `config/challenges.toml`.  
2. Read `metagraph_at(block_hash)` for the epoch seal block.  
3. Run BUNDLE_SPEC `expected_participants(hypertraining, policy, block_hash, chain)`.  
4. That set `E` is the coverage obligation.

### 9.2 Coverage obligation

For every `h ∈ E`, the challenge MUST produce exactly one signed leaf `(challenge_id=hypertraining, miner_hotkey=h, epoch)` with either `Score` or `NoScore`.  
**Silence is a bug.** Missing leaves fail gateway seal and validator verify.

### 9.3 Reason selection (normative priority, draft)

Evaluate in order; first match wins:

| Priority | Condition | Reason |
|----------|-----------|--------|
| 1 | Not in `E` | Do not emit a leaf |
| 2 | Attestation not `Verified` when required | `AttestationNotVerified` (3) |
| 3 | Deadline / timeout / transport | `Timeout` (1) |
| 4 | Rate limited | `RateLimited` (5) |
| 5 | Schema / bad submit / admit reject | `InvalidResponse` (2) |
| 6 | Miner / job fault | `MinerError` (4) |
| 7 | Challenge / operator fault | `ChallengeInternal` (6) |
| 8 | Else scored | `Score(value)` per §7 |

---

## 10. Leaf emission and gateway POST

Leaves use BUNDLE_SPEC `LeafV1` under domain `base-rawweight-v1` with:

- `challenge_id` = `hypertraining`
- `scoring_version` = `1`
- integer `Score` or `NoScore` only

Challenge service POSTs raw weights to gateway per existing base patterns (same as agent-challenge). Bundle `protocol_version` remains **1**.

---

## 11. Compose services, ports, image contract (draft)

| Service | Port | Notes |
|---------|------|-------|
| `agent-challenge` | 8090 | Unchanged |
| `hypertraining-challenge` | **8091** | New; must not remove agent-challenge |

Image pins and Dockerfile stages follow repo patterns when the binary lands (plan todo 13). No `:latest` floating tags in production compose.

**Out of scope for this challenge:**

- Harbor CLI / Harbor packs / HarborVerifier (agent-v1 only)
- AWS infrastructure as a required dependency
- Claiming live dual physical clusters A/B

---

## 12. Security claim boundary

| Claim | Status |
|-------|--------|
| Artifact boundary (checkpoint hash + validator eval image) | **In scope** |
| Sealed surface admission before resource spend | **In scope** |
| Three statistical / numeric guards | **In scope** |
| LLM as security or pay gate | **Forbidden** |
| CUDA / container sandbox as security boundary | **Not claimed** (artifact boundary only; brief §2.4 / §3) |
| Confidential Computing for multi-node train | **Not a fallback** (GPUDirect RDMA disabled in CC) |
| Real B300 isolation (PKey partitions, exclusive slots) | **Design target**; sim approximates measurement, does not claim fabric security |

---

## 13. ClusterBackend contract (draft)

```text
trait ClusterBackend:
  exclusive_slot(...)
  topology_mirror_check(tp, pp, ep, cp)
  run_segment(seeds, budget_tokens) -> wallclock + checkpoint_handle + telemetry
```

| Backend | Behavior now |
|---------|----------------|
| `SimBackend` | Deterministic or seeded wallclock from code fingerprint + noise param; fake checkpoint hash; API surface for partition ids without real IB |
| `RealBackend` | Stub: `Err(NotConfigured)` with message that owner B300 enablement is deferred |

Must NOT pretend RealBackend produces GPU timing until the enablement runbook is executed.

---

## 14. Verification checklist (implementers)

See [`HYPERTRAINING_CHECKLIST.md`](./HYPERTRAINING_CHECKLIST.md).  
Planned: `cargo run -p xtask -- hypertraining-check` (task 17).  
agent-v1 gate remains: `cargo run -p xtask -- agent-challenge-check` (must stay green).

---

## Appendix A. Cross-links

| Doc | Role |
|-----|------|
| [`/root/challenge-training-fork.md`](/root/challenge-training-fork.md) | Design source (v1.0) |
| [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) | Leaf / bundle crypto |
| [`AGENT_CHALLENGE.md`](./AGENT_CHALLENGE.md) | Sibling challenge (agent-v1); do not merge domains |
| [`HYPERTRAINING_CHECKLIST.md`](./HYPERTRAINING_CHECKLIST.md) | Pin table for xtask |
| `docs/runbooks/hypertraining-enable-real-and-emission.md` | Planned (todo 18): Real B300 + emission unlock |

---

## Appendix B. Normative pins (config defaults)

| Param | Value |
|-------|--------|
| challenge_id | `hypertraining` |
| challenge_scoring_version | `1` |
| emission_share_bps | `0` (agent-v1: `10000`) |
| bundle protocol_version | `1` |
| TE pin | `2.18.0+e7c550c5` |
| mlm_commit | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` |
| kernel κ | `2` |
| screen K | `3` |
| promotion K | `5` |
| calibration K | `10` |
| α | `0.05` + Benjamini-Hochberg |
| ε | `min(0.25%·L, 0.5·σ̂_d)` with MUST_CALIBRATE placeholders |
| binary sim reject | `> 0.85` without measure |
| SCORE_MAX | `1_000_000` |
| compose port | `8091` |

---

## Appendix C. Domain tags (hypertraining, draft)

Distinct from `base-agent-*` tags. Exact strings land with `hypertraining-challenge-task` crate:

| Purpose | Tag family (intent) |
|---------|---------------------|
| Task / blob / receipt domains | `base-hypertraining-*` (crate pins) |
| Leaf signatures | `base-rawweight-v1` (unchanged) |

---

## Appendix D. Must-not (guardrails)

1. Do not touch agent-v1 scoring, Harbor packs, or AGENT_CHALLENGE freeze pins.  
2. Do not claim live B300 / MFU runs.  
3. Do not set hypertraining emission non-zero without owner ceremony.  
4. Do not use Harbor or AWS as required path.  
5. Do not brand the product with the monorepo checkout directory name.  
6. Do not use LLM as admit/reject/pay gate.  
7. Do not claim CUDA sandbox-as-security.  
8. Do not bump bundle `protocol_version` for this challenge.  
9. Do not drop any of guards 1-3, promotion machine, or anti-noise core in an "MVP".  
10. Do not commit as non-echobt identity.
