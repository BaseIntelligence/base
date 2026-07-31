# base miner docs (external-facing)

<!-- protocol_version: 1 -->
**Bundle `protocol_version`:** `1`  
**Challenge `scoring_version`:** `2` (Harbor SWE packs — **not** the same axis as bundle protocol)

This badge must match `bundle::PROTOCOL_VERSION` in crate `bundle`.  
CI gate: `cargo run -p xtask -- external-docs-check`.

Do **not** conflate the two version numbers (Metis S4):

| Axis | Value | Meaning |
|------|-------|---------|
| Bundle `protocol_version` | **1** | Leaf / merkle / weight bytes ([`BUNDLE_SPEC.md`](../BUNDLE_SPEC.md)) |
| `challenge_scoring_version` | **2** | Pack dispatch, `model.patch`, pure correctness ([`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md)) |

Scoring_version **1** (SHA-256 echo answer + latency decay) is **retired**. Live scoring is pack-based only.

These pages are the miner-facing guide (funding Phala, deploy, certify, troubleshoot).  
Normative challenge contract: [`../AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) (**FROZEN**, scoring_version 2).  
Normative bundle bytes: [`../BUNDLE_SPEC.md`](../BUNDLE_SPEC.md) (**FROZEN**, protocol_version 1).  
Security claim (what validators prove, and what they do not): [`../THREAT_MODEL.md`](../THREAT_MODEL.md) §1 (D19).

| Page | Topic |
|------|--------|
| [funding-phala.md](./funding-phala.md) | Fund your own Phala account + miner-funded model key |
| [deploy.md](./deploy.md) | `install.sh`, compose-hash, CVM deploy, concurrency |
| [certify.md](./certify.md) | Bind quote and submit to validator |
| [troubleshoot.md](./troubleshoot.md) | Common failures |

---

## Quick path (recommended): one-command self-deploy

From a clean box with Docker Compose:

```bash
# Public hotkey only (64 lowercase hex). Never paste mnemonics.
export BASE_MINER_HOTKEY_HEX='<64 hex>'

# Miner-funded inference key (Q3=A) — file path only; never put key bytes in env.
umask 077
printf '%s' "$YOUR_PROVIDER_API_KEY" > /secure/model_key
chmod 0600 /secure/model_key
export BASE_MODEL_KEY_FILE=/secure/model_key

# Concurrency knob (clamped 1..=5 on the runner)
export BASE_MAX_CONCURRENCY=2

# Digest-pinned agent image (operators publish pins; local dev may use preloaded images)
# export BASE_AGENT_IMAGE='ghcr.io/baseintelligence/base/base-agent@sha256:<64 hex>'

./install.sh
# → prints compose-hash=…, starts local agent-runner
curl -sS http://127.0.0.1:8080/v1/capacity
# → {"max_concurrency":2,"current_load":0}
```

`install.sh` is **idempotent**, **fail-closed** on missing Docker / unreadable model-key / bad hotkey, and **never echoes secrets**.

---

## Pack flow (scoring_version 2)

1. **Deploy** measured CVM (`agent` + measured `socket-proxy` + `attest-helper`) — see [deploy.md](./deploy.md).  
2. **Certify** each epoch — [certify.md](./certify.md).  
3. Orchestrator **dispatches** a stripped Harbor pack (`base-agent-dispatch-v1`, `scoring_version: 2`).  
4. Your runner pulls the digest-pinned environment image **through the measured socket-proxy**, runs the agent with **OPEN egress** (default), and returns `model.patch` + a signed work receipt.  
5. Operator grades offline with held-out tests; pure correctness → leaf `Score` / `NoScore`. Bundle leaves stay on **protocol_version 1**.

Miner inference is **miner-funded** (your model key file). The subnet owner does not pay your LLM bill or your Phala CVM bill.

### Egress posture (todo 21 — LOCKED default)

**OPEN** by default: the pack-environment container may use the network. Honest claim: stripping protects **grading-channel integrity** (held-out `solution/` / `tests/` / `grader.py` never reach the miner), **not** miner honesty. D19 already disclaims score honesty. Optional allowlisted proxy is off unless you set it explicitly.

---

## CLI path (compose-hash / Phala)

```bash
# From base repo root (Rust 1.96 toolchain) — offline compose-hash
cargo build -q -p miner-bin
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1

# After funding Phala and installing `phala` CLI — real deploy
# cargo run -q -p miner-bin -- deploy --deploy --netuid 1

# Certify (fixture mode for offline smoke; live needs agent URL + validator)
# cargo run -q -p miner-bin -- certify \
#   --fixture-mode \
#   --validator-url http://127.0.0.1:8081 \
#   --epoch 0 \
#   --miner-hotkey-hex <64 hex>
```

---

## Version pin

When `protocol_version` bumps in `bundle`, update:

1. The HTML comment and bold badge at the top of **this file**.  
2. Any copy in sibling pages that states the bundle protocol version.  
3. Re-run `cargo run -p xtask -- external-docs-check`.

When only challenge scoring changes, bump `challenge_scoring_version` in [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) — **leave** bundle `protocol_version` at **1** unless leaf bytes change.
