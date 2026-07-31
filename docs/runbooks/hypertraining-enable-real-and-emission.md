# Runbook: enable Real B300 cluster + hypertraining emission unlock

**Status:** operator procedure for a **future** day. This runbook does **not** claim that B300 hardware is live, that MFU has been measured, or that hypertraining emission is non-zero today.

**Current production posture (do not rewrite without ceremony):**

| Item | Today |
|------|--------|
| `ClusterBackend` | **SimBackend** (CI / E2E). `RealBackend` returns `NotConfigured` |
| Live B300 dual-cluster A/B | **Not live** |
| Live MFU §10.4 | **Not claimed** |
| `config/challenges.toml` `hypertraining.emission_share_bps` | **`0`** |
| `agent-v1.emission_share_bps` | **`10000`** (sole non-zero share) |

Normative freeze: [`../HYPERTRAINING.md`](../HYPERTRAINING.md) (H7, H8, §2.1, §13, Appendix D).  
Design source: [`/root/challenge-training-fork.md`](/root/challenge-training-fork.md) §10.4, §14, §15.  
Trust-root ceremony: [`../../config/CEREMONY.md`](../../config/CEREMONY.md).  
Rotation mechanics: [`trust-root-rotation.md`](./trust-root-rotation.md).  
Promote / restart: [`promote-rollback-restore.md`](./promote-rollback-restore.md).

**Must not (this runbook and any PR that only documents it):**

1. Commit a non-zero `hypertraining` `emission_share_bps` without a completed owner ceremony and dual-accept window.
2. Claim Real B300, live MFU, or calibrated σ̂ as current state in freeze docs or release notes.
3. Fabricate GPU wallclock from `RealBackend` before hardware enablement (crate still stubs `NotConfigured` until code lands).
4. Drop `agent-v1` or leave emission bps sum ≠ **10000**.
5. Hot-drop an unsigned `challenges.toml` on a live host.

---

## Preconditions (before step 1)

- [ ] Physical dual-cluster A/B (B300) is racked, networked, and operator-accessible. Isolation design target: PKey partitions + exclusive slots ([`HYPERTRAINING.md`](../HYPERTRAINING.md) §12).
- [ ] Megatron-Bridge reference recipe for MoE 200B is available on the cluster (design brief §10.4).
- [ ] TE pin and `mlm_commit` match freeze pins in [`HYPERTRAINING.md`](../HYPERTRAINING.md) Appendix B unless a scored bump of `challenge_scoring_version` is planned.
- [ ] Owner signing material offline only (`~/.base-secrets/`, mode `0700`). Production owner key is **not** the throwaway CI key unless this is a non-prod lab.
- [ ] Staging stack can load dual trust-root versions ([`trust-root-rotation.md`](./trust-root-rotation.md)).
- [ ] Change ticket records: target hypertraining bps, reduced agent-v1 bps, MFU result, calibration artifact paths.

---

## Checklist (execute in order)

### 1. MFU measure on B300 (design brief §10.4)

**Goal:** Decide whether the tournament is worth opening. First op on cluster arrival, before controller cutover.

1. On cluster A (or the designated measure host), run the **Megatron-Bridge reference recipe** for MoE 200B on B300.
2. Record **MFU** (model FLOPs utilization) for the reference créneau. Do not use miner self-reports.
3. Apply the brief verdict table:

| MFU observed | Verdict |
|--------------|---------|
| 20–25% | Tournament rich (~2× headroom). Proceed. |
| 30–35% | Acceptable (~1.4×). Prefer short seasons. |
| ≥ 40% | Thin margin (~1.2×). Re-open algorithmic track question before emission unlock. |

4. Attach raw logs + one-line summary to the change ticket.  
5. **Stop gate:** if MFU is unusable or hardware is not ready, **do not** continue to RealBackend cutover or emission ceremony. Stay on SimBackend and `emission_share_bps = 0`.

**Done when:** ticket has dated MFU number, recipe commit/image pins, and a go / no-go decision. Still **no** claim that production scoring uses RealBackend.

---

### 2. K=10 calibration for σ̂ (sigma)

**Goal:** Replace `MUST_CALIBRATE` placeholders (`hypertraining-eval` `EpsilonParams::must_calibrate_defaults`, freeze Appendix B) with published σ̂ on target hardware.

Calibrate **K = 10** independent paired runs (brief §14 / §9.1) and estimate at least:

| Symbol | Role |
|--------|------|
| `σ̂_loss` | Loss noise under sealed data order |
| `σ̂_d` | Paired loss-diff noise (feeds `ε = min(0.25%·L, 0.5·σ̂_d)`) |
| `σ̂_wallclock` | Wallclock noise (feeds effective K / anti-noise) |

Operator actions:

1. Run K=10 calibration segments on the **same** topology mirror (TP/PP/EP/CP) intended for production slots.
2. Publish calibration artifact (path + hash) outside git secrets; record micro-loss units consistent with `hypertraining-eval`.
3. Wire calibrated `σ̂_d` via `EpsilonParams::with_calibrated_sigma` (or successor config surface) in a **code/config PR**, not by editing freeze docs to pretend calibration already happened.
4. Re-derive `ε`, effective K, and `T_seg` (brief §15 steps 2–3). Bump `challenge_scoring_version` if score-affecting defaults change.

**Done when:** calibration artifact is reviewed, σ̂ values are pinned in the release that enables RealBackend scoring, and sim CI still passes with fixtures.

---

### 3. Configure RealBackend

**Goal:** Switch the challenge orchestrator from SimBackend-only measurement to the real cluster path **after** steps 1–2.

Today `crates/hypertraining-cluster` `RealBackend` is a **stub**: every method returns `ClusterError::NotConfigured` ("B300 path is deferred"). Enabling hardware means **shipping and deploying** a real implementation plus operator config, not flipping a doc bit.

1. Land / promote the RealBackend implementation that:
   - Enforces topology mirror (`check_topology_mirror`)
   - Allocates exclusive slots bound to PKey ids
   - Returns real wallclock + checkpoint handles from `run_segment` (never fabricated numbers)
2. Configure dual-cluster endpoints, credentials, and PKey partitions via the deploy env / secret layout used by `hypertraining-challenge` (see `deploy/secrets/`, compose service on **8091**). Prefer digest-pinned images ([`promote-rollback-restore.md`](./promote-rollback-restore.md)).
3. Staging smoke:
   - `curl -fsS http://127.0.0.1:8091/health` (or `/healthz` if that is the live path)
   - One sealed segment on RealBackend completes with checkpoint hash + wallclock
   - Topology mismatch is rejected
   - SimBackend remains available for CI (do not delete sim path)
4. Only then point production orchestrator config at RealBackend.

**Done when:** staging shows RealBackend `Ok` segment results under operator control, and freeze docs still say production cutover is **operator-gated** until this checklist is complete for that environment.

**Must not:** claim "B300 live" in `HYPERTRAINING.md` status tables until this step is finished **for that environment** and recorded in the ticket.

---

### 4. Ceremony: raise hypertraining bps (reduce agent-v1 so sum = 10000)

**Goal:** Owner economic decision. Emission is **not** unlocked by code alone.

1. Choose non-negative integers:

   ```text
   hypertraining.emission_share_bps = H    # H > 0 only after owner approval
   agent-v1.emission_share_bps      = 10000 - H
   sum over all challenges          = 10000
   ```

2. Edit a **next** body (do not destroy `v(n)` early):

   ```bash
   cp config/challenges.toml config/challenges.vNEXT.toml
   # set hypertraining emission_share_bps = H
   # set agent-v1 emission_share_bps = 10000 - H
   # keep public keys and policies correct
   ```

3. Confirm no other challenge rows break the sum. Zero-share challenges may remain registered; aggregate may skip zero-share until H > 0.

**Done when:** `vNEXT` body is reviewed offline and sum is exactly 10000. **Do not merge** until step 5 signs it.

**Must not:** set non-zero hypertraining bps in the live committed `config/challenges.toml` from a docs-only or software-only PR. This step is owner ceremony content.

---

### 5. Resign `challenges.toml`

**Goal:** Owner-signed trust root only ([`CEREMONY.md`](../../config/CEREMONY.md), D18/D21).

```bash
# Offline. Secrets never in git.
cargo run -q -p trustroot-bin -- sign \
  --key ~/.base-secrets/owner-throwaway.age \
  --age-identity ~/.base-secrets/age-identity.txt \
  --input config/challenges.vNEXT.toml \
  --kind challenges \
  --out config/challenges.vNEXT.toml.sig

cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/challenges.vNEXT.toml \
  --kind challenges
```

Production: use the real owner secret path from the ceremony, not the CI throwaway, unless the environment is explicitly non-prod.

Rollout:

1. PR adds `v(n+1)` beside `v(n)` (dual-accept for `rotation_epochs`, default 3).
2. CI green; merge; promote staging then prod ([`trust-root-rotation.md`](./trust-root-rotation.md)).
3. After the window, PR drops `v(n)`.

**Abort:** verify fails → do not promote. Bad root on staging → roll back release digest and restore previous signed pair.

**Done when:** verify exits 0 on the new pair and validators accept the new root in the dual window.

---

### 6. Restart hypertraining-challenge + gateway backends

**Goal:** Processes reload trust root and backend registration after config/image promote.

On the master host (paths assume `/opt/base`; adjust):

```bash
set -euo pipefail
cd /opt/base

# After materialize-env / updater promote as needed:
docker compose --profile master up -d hypertraining-challenge gateway

# Health
curl -fsS "http://127.0.0.1:8091/health" || curl -fsS "http://127.0.0.1:8091/healthz"

# Re-register challenge backend if your deploy does not auto-register (D3/D18).
# No signing keys in this request — public trust root only.
curl -fsS -X POST "http://127.0.0.1:${GATEWAY_ADMIN_PORT:-8080}/v1/admin/backends" \
  -H 'Content-Type: application/json' \
  -d '{
    "challenge_id": "hypertraining",
    "base_url": "http://hypertraining-challenge:8091"
  }'
```

Verify:

- [ ] `hypertraining-challenge` healthy on **8091** (agent-challenge remains **8090**).
- [ ] Gateway lists backend `challenge_id=hypertraining`, not ejected.
- [ ] Trust root load shows hypertraining `emission_share_bps = H` and agent-v1 `10000 - H`.
- [ ] One epoch path: expected set D24 leaves still emit (Score or NoScore); silence remains a bug.
- [ ] Rollback plan: previous compose digest + previous signed challenges pair ready ([`promote-rollback-restore.md`](./promote-rollback-restore.md)).

**Done when:** health + backend registration + one clean epoch observation are attached to the ticket.

---

## Post-enable verification (environment-specific)

Only after steps 1–6 succeed for **that** environment may operators update environment run notes to say RealBackend is enabled and emission is non-zero **there**. The monorepo freeze doc (`HYPERTRAINING.md`) stays honest about default git posture until the owner deliberately refreshes status after production cutover.

Suggested ticket evidence pack:

1. MFU log path + verdict  
2. Calibration artifact hash (K=10)  
3. RealBackend staging segment receipt  
4. Signed `challenges` verify transcript  
5. Compose restart + gateway backend list  
6. First epoch leaf sample (redacted)

---

## Rollback (short)

| Failure | Action |
|---------|--------|
| MFU / calibration bad | Stay SimBackend; keep emission 0; no ceremony |
| RealBackend unstable | Point orchestrator back to SimBackend; do not unlock emission |
| Bad trust root | Restore previous signed `challenges` pair; dual-accept or full rollback |
| Gateway / challenge unhealthy | Previous image digest via updater; re-register backend |

---

## Related code (orientation only)

| Path | Role |
|------|------|
| `crates/hypertraining-cluster` | `SimBackend` / `RealBackend` / `ClusterBackend` |
| `crates/hypertraining-eval` | `ε`, `MUST_CALIBRATE`, guards |
| `bins/hypertraining-challenge` | Service on 8091 |
| `config/challenges.toml` | Owner-signed emission shares (sum 10000) |
| `docker-compose.yml` | `hypertraining-challenge` service |

---

## Document history

| Date | Note |
|------|------|
| 2026-07-31 | Task 18: initial operator runbook. Does not enable HW or emission in-repo. |
