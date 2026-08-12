# Appendix 02 — Current State: Prism Documentation, Contracts and CI Gates
> Research appendix for the Prism v3 evaluation proposal (`docs/spikes/prism-v3/`). Produced 2026-08-06 by codebase exploration. Non-normative spike document.

# Prism documentation picture

## 1. Frozen / normative contract — `/root/gbase/docs/PRISM.md`

**Status:** Architecture lists it as **live** (not FROZEN like Design/BUNDLE). Still treated as normative in `docs/AGENTS.md`.

| Field | Value |
|-------|--------|
| `challenge_id` | `prism` |
| `scoring_version` | **2** (bpb-only; v1 had 0.3 LLM quality blend) |
| `recipe_version` | **1.2.0** (telemetry 1.1.0 + arch registry / training-only) |
| Port | `:8092` |
| Emission (doc) | `10000` bps sole until design ceremony |
| GPU | Master-centralized **Lium** (no Phala CVM); Sim in CI only |

### What miners do
- Submit **`architecture.py` + `training.py`** (recipe contract), or **training-only**: `training.py` + published `arch_id` (`architecture_py` empty / ZIP + `X-Prism-Arch-Id`).
- Eval on operator-rented Lium GPU; harness produces `METRICS_JSON` (bpb, tokens, steps, wall-clock, GPU).
- `training.py` **must** call `prism_telemetry.report(...)` + `prism_telemetry.finish_evaluation()` or → `missing_telemetry_hooks` → `Score(0)`.

### What is scored
- **Pure bpb** → lattice `[0, SCORE_MAX]` via `score_from_bpb`. LLM quality is **coherence gate / audit only**, never a grader.
- Hard-zero: agentic `cheat`/`suspicious`, cheap `Copied`/`Suspicious`, copy-gate rejects.
- Fail-closed: missing agentic verdict → `NoScore(ChallengeInternal)`.

### Competition (recipe ≥ 1.2.0) — epoch-local, SCORE_MAX preserved
- **Challenger credit:** hotkey’s best lattice score in the epoch batch.
- **Architecture-owner credit:** each arch’s best batch `Score` credited to the **owner**.
- **Emission:** `max(own, owner)` — never summed.
- `Score(0)` never sets arch best; all-`NoScore` keeps absence.

### Round / epoch structure
- Not Design-style timed rounds. **Chain-epoch emission** via `prism-emit`:
  - Intake `epoch` is metadata only.
  - One D24-complete leaf set per chain epoch at close; batch = finalized-since-last-emit.
  - Exactly-once cursor (`prism_emit_cursor`); crash replays identical set.
  - Long trains (≤6h ≫ ~72m epochs) score once at the first epoch boundary after finalize.
- State machine: `Queued → Provisioning → Running → Reviewing → AgenticReview → Scoring → Terminated` (or Failed/Rejected). Events in `prism_stage_event`.

### Data sources / recipe pin
Delegated to `/root/gbase/docs/PRISM_RECIPE.md` (`prism-recipe-v1`, header says **v1.0.2**):

| Cap / pin | Value |
|-----------|--------|
| Dataset | `HuggingFaceFW/fineweb-edu@sample/10BT` parquet `010_00000.parquet` |
| SHA-256 | `e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6` |
| Train wall clock | 6h |
| Pod lifetime | 7h |
| Max steps | 20 000 |
| Source size | 128 KiB / script |
| Max params | ≤ **350M** after `build_model` |

### Constraints / must-not
- No Phala/TDX for Prism GPUs.
- No emission moves without owner trust-root ceremony.
- Don’t commit Lium/OpenRouter/challenge secrets.
- Master-only eval; validators fetch sealed weights only.
- Gating: metagraph hotkey; 1 accepted arch submit per `(prism, hotkey)`; training-only under `prism:train:<arch_id>`; infra auto-retry ≤3; cheat terminal.

### Roadmap / TODOs in PRISM.md
No explicit TODO list. Implied forward work:
- Emission rebalance with Design after ceremony.
- Top-model publish to public `BaseIntelligence/prism` (graceful no-op without GitHub token).
- Lium ops notes (image matrix, template v9, cost baseline) as operational truth, not open product TODOs.

---

## 2. Miner-facing mirror — `/root/gbase/docs/external-miner/`

| File | Role |
|------|------|
| `/root/gbase/docs/external-miner/prism.md` | Primary Prism miner guide |
| `/root/gbase/docs/external-miner/README.md` | Index; scoring axis table; gateway paths |
| `/root/gbase/docs/external-miner/troubleshoot.md` | Prism reject / Score 0 / Provisioning / idempotency |
| **No** `examples/prism-*` | Only Design baseline under `examples/` |

**Today’s miner instructions (`prism.md`):**
- ZIP (preferred) or JSON: `architecture.py` + `training.py`; ≤350M params.
- Telemetry hooks required (≥1.1.0); training-only + `arch_id` (≥1.2.0); `GET /v1/architectures`.
- Submit via gateway `/challenge/prism/v1/submissions` with `X-Miner-Hotkey`; inspect `/v1/recipe` + `/baseline`.
- 1-max gating, arch-only anti-copy, pure-bpb + competition credits, scores at next epoch boundary after finalize.
- Public repo: [`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism) — README notes “miner docs forthcoming”; top-model publishes under `top-model/`.

**Companion recipe for miners:** `/root/gbase/docs/PRISM_RECIPE.md` (contract, telemetry, dataset pin, budgets, scoring v2, anti-copy).

---

## 3. Completeness — `/root/gbase/docs/COMPLETENESS.md`

**§ prism-challenge:** crates/binary/compose/Dockerfile/GHCR **done**; emission doc’d as **10000 bps** sole share.

**§ Challenge backends (all done):** Lium backend, orchestration + epoch-close emit, recipe v1, LLM review, full API.

**Known gaps:** Design emission ceremony / mainnet owner — not Prism-product gaps. Phala miner path **removed**.

---

## 4. Architecture fit — `/root/gbase/docs/ARCHITECTURE.md`

- Spec table: `PRISM.md` = live Lium GPU HTTP challenge.
- Topology: `prism-challenge` on **master** compose; miners HTTP-submit scripts; other validators have **no** challenge exec.
- Epoch data flow: challenges sign `Score`/`NoScore` leaves (D24) → gateway seal → validators verify with local `challenges.toml`.
- Role: “Lium (or sim) recipe eval, review gate, **sign leaves**”.
- Emission posture in ARCHITECTURE: `prism = 10000`, `design = 0` (see drift note below).

---

## 5. Design patterns Prism already shares / may borrow

From `/root/gbase/docs/DESIGN_CHALLENGE.md` §8 (and shared crates):

| Pattern | Design | Prism |
|---------|--------|-------|
| Shared `challenge-agentic` | Yes (harness/pages) | Yes (arch + metrics/receipt) |
| Pre-LLM copy gate (`created_at`, baseline exempt) | Harness corpus | **Architecture-only** corpus |
| `cheat`/`suspicious` → `Score(0)` | Explicit “same policy as Prism” | Same |
| Fail-closed missing verdict → `ChallengeInternal` | Yes | Yes |
| Infra auto-retry ≤3; cheat terminal | Yes | Yes |
| Admin winners / Elo / elimination / rounds | Design-only | **N/A** — bpb + competition credits |
| Containerized review image | `design-review` | In-process / OpenRouter on master |

---

## 6. Other docs mentioning Prism (high-signal)

| Path | Why it matters |
|------|----------------|
| `/root/gbase/docs/PRISM_RECIPE.md` | Execution contract / dataset / caps |
| `/root/gbase/docs/runbooks/prism-enable-lium-and-emission.md` | Lium + emission ceremony |
| `/root/gbase/docs/runbooks/design-enable-and-emission.md` | Prism sole-share / rebalance |
| `/root/gbase/docs/runbooks/local-testnet-e2e.md` | `prism_sk`, `:28092`, weights smoke |
| `/root/gbase/docs/runbooks/staging-testnet-e2e.md` | Register + health + identity |
| `/root/gbase/docs/SITE_API.md` | `/v1/site/arenas/prism/*` proxy |
| `/root/gbase/docs/AGENTS.md` | Normative list; public repo; verification invariants |
| `/root/gbase/config/challenges.toml` | **Live:** design `2000`, prism `8000` (comment: activated 2026-08-06) |
| `/root/gbase/config/CEREMONY.md` | Still describes prism@10000 / design@0 |
| `/root/gbase/deploy/AGENTS.md`, compose, secrets, `prod-burn-seal.sh` | Ops: Postgres `prism_*`, secrets, D24 seal coordination |

**Doc drift:** Frozen/ops docs still say prism **10000** sole share; committed trust root is already **8000/2000**. Updating Prism emission text may conflict with `design-check`’s `prism_bps_sole` pin (below).

---

## Documentation constraints (any Prism spec change must respect)

### Frozen / normative sections to keep coherent
1. **`docs/PRISM.md`** — identity (`challenge_id`, scoring_version 2), orchestration, gating keys, arch registry + competition math (`max` not sum), epoch-close emit, agentic taxonomy, API table, Must not.
2. **`docs/PRISM_RECIPE.md`** — script contract, telemetry, dataset SHA, budgets/caps; **any pin change must bump `prism-recipe` version** so old leaves stay unambiguous.
3. **`docs/external-miner/prism.md` (+ README + troubleshoot)** — miner-facing mirror; public repo `BaseIntelligence/prism` when product/API changes (root `AGENTS.md`).
4. **`docs/ARCHITECTURE.md` / `COMPLETENESS.md` / runbooks** — topology, emission, enablement.
5. **Leaf/bundle axis** — scoring_version ≠ bundle `protocol_version` (stay 1); D24 complete sets; gateway first-write-wins per `(challenge, epoch, hotkey)`.

### CI doc gates (xtask)

| Gate | Enforces for Prism? |
|------|---------------------|
| `spec-check` | **No** — only `BUNDLE_SPEC.md` letter/content pins |
| `design-check` | **Indirect:** `DESIGN_CHALLENGE.md` must contain substring **`10000`** (`prism_bps_sole`) and design `emission_share_bps = 0`. Changing Design’s emission narrative without updating `xtask/src/design_check.rs` + checklist breaks CI |
| `external-docs-check` | **Yes:** `docs/external-miner/{README,design,prism,troubleshoot}.md` required; every `*.md` needs `<!-- protocol_version: N -->` matching `bundle::PROTOCOL_VERSION`; README must contain pins `prism`, `PRISM.md`, `HTTP`, `no Phala/CVM`, `DESIGN_CHALLENGE.md`, `BUNDLE_SPEC.md`; banned strings (`phala deploy`, `install.sh`, …) |
| **No `prism-check`** | Unlike Design, **PRISM.md has no section/content pin gate** — structure is not CI-enforced beyond external-miner presence |

### Product invariants called out in AGENTS / docs
- Master-only challenge exec; validators only seal path consumers.
- Digest-only images; secrets via age/files.
- Don’t enable evil-gateway in prod.
- After API/quota/scoring changes: update **both** public Prism repo and `docs/external-miner/`.
