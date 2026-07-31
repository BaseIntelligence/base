# Task 47 close-out pointer (todo 35 + live-complete residual)

Date: 2026-07-31T00:12:00Z  
Netuid: 541 (Bittensor test)  
Branch: `reborn`  
Base SHA at live-complete: `6c1dd9b` (prior close-out base `5acb665…`)

## Purpose

Tracked in-repo pointer for parent-plan task 47 testnet E2E close-out and the
**2026-07-31 live-complete residual** pass. Full transcripts live under
`.omo/evidence/` (operator host).

## Evidence paths (operator host)

| file | role |
|------|------|
| `/root/.omo/evidence/base-agent-challenge-deepagent/live-complete-residual-matrix.txt` | **Master** residual matrix A–H (this session) |
| `/root/.omo/evidence/base-agent-challenge-deepagent/task-35-testnet-e2e-close.txt` | Close-out + dated live-complete refresh |
| `/root/.omo/evidence/base-agent-challenge-deepagent/task-35-unmet-criteria.txt` | Honest PARTIAL/UNMET (refreshed; (a) no longer unmet) |
| `/root/.omo/evidence/base-rust-subnet/task-47-e2e.txt` | Parent evidence + Live-complete refresh section |
| `/root/.omo/plans/base-rust-subnet.md` task 47 | Parent plan status remains `[~]` PARTIAL |

## Criteria snapshot (2026-07-31 live-complete)

| id | STATUS | summary |
|----|--------|---------|
| (a) Match | **PASS** | continuous staging Match; fresh epoch 102 seal→Match |
| (b) TimelockedWeightsCommitted | PASS | extrinsic `0xf9b806d4a89823a6a0abca63555c10d3cbbccf87748cd3917100e99b716cccc2` block 7671064 |
| (c) weights after reveal | PASS | uid=2 row `[(0, 3641), (1, 65535), (2, 3641)]` |
| (d) revealed == local | PASS | u16 quantization max abs diff < 5e-6 |
| (e) attest Verified | PASS | staging fixture certify `outcome=verified` (not live Phala CVM) |
| (f) pack scores | PARTIAL | scoring_version=2 FixtureVerifier Score{1000000}; Harbor docker blocked; on-chain CRV4 still toy lattice |

**Emission is explicitly NOT a pass criterion.** Not claimed.

## TAO

Free balances ~0.028 / 0.334 / 0.346 τ (validator/miner/owner). **No new CRV4 submit.**
Proven path reused only.

## Staging

Host `68.183.23.51` `/opt/base`: validator, gateway, agent-challenge (scoring_version=2),
postgres, socket-proxy healthy. Latest sealed epoch=102. Continuous Match in validator logs.
Updater left stopped (local-only pull fails).

## What remains for full `[x]` on task 47

1. Pack-derived weight vector (Harbor or multi-pack fixture path) sealed on staging with multi-uid honesty.
2. CRV4 commit+reveal of that pack-derived vector (fee TAO + schedule).
3. Optional for full HOW text: live Phala CVM miner quote (not fixture).
