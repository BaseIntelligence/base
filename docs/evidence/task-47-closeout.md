# Task 47 close-out pointer (todo 35)

Date: 2026-07-30T23:12:20Z  
Netuid: 541 (Bittensor test)  
Branch: `reborn`  
Base SHA at evidence capture: `5acb66550375ff093f27a200c653f7d3a78e100a`

## Purpose

Tracked in-repo pointer for the parent-plan task 47 testnet E2E close-out performed under
`gbase-agent-challenge-deepagent` todo 35. Full transcripts live outside the git tree under
`.omo/evidence/` (operator host).

## Evidence paths (operator host)

| file | role |
|------|------|
| `/root/.omo/evidence/gbase-agent-challenge-deepagent/task-35-testnet-e2e-close.txt` | Primary close-out transcript; criteria (a)–(f) table |
| `/root/.omo/evidence/gbase-agent-challenge-deepagent/task-35-unmet-criteria.txt` | Honest PARTIAL/UNMET list |
| `/root/.omo/evidence/gbase-rust-subnet/task-47-e2e.txt` | Parent evidence with dated close-out section |
| `/root/.omo/plans/gbase-rust-subnet.md` task 47 | Parent plan status remains `[~]` PARTIAL |

## Criteria snapshot

| id | STATUS | summary |
|----|--------|---------|
| (a) Match | PARTIAL | full_local_e2e + lib Match; staging validator-bin no continuous Match log |
| (b) TimelockedWeightsCommitted | PASS | extrinsic `0xf9b806d4a89823a6a0abca63555c10d3cbbccf87748cd3917100e99b716cccc2` block 7671064 |
| (c) weights after reveal | PASS | uid=2 row `[(0, 3641), (1, 65535), (2, 3641)]` |
| (d) revealed == local | PASS | u16 quantization max abs diff < 5e-6 |
| (e) attest Verified | PASS | staging fixture certify `outcome=verified` (not live Phala CVM) |
| (f) pack scores | PARTIAL | scoring_version=2 Score{1000000} local e2e; on-chain CRV4 still toy lattice |

**Emission is explicitly NOT a pass criterion.** Not claimed.

## TAO

Free balances ~0 τ at close-out. **No new CRV4 submit.** Proven path reused only.

## Staging

Host `68.183.23.51` `/opt/gbase`: validator, gateway, agent-challenge (scoring_version=2),
postgres, socket-proxy healthy after QA. Updater left stopped (local-only pull fails).

## What remains for full `[x]` on task 47

1. Continuous validator-bin epoch loop logging `Match` against a sealed staging gateway bundle.
2. Pack-derived weight vector committed via CRV4 (requires fee TAO + seal path).
