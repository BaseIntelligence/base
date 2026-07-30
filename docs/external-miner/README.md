# gbase miner docs (external-facing)

<!-- protocol_version: 1 -->
**Bundle `protocol_version`:** `1`

This badge must match `bundle::PROTOCOL_VERSION` in crate `bundle`.  
CI gate: `cargo run -p xtask -- external-docs-check`.

These pages are the miner-facing guide (funding Phala, deploy, certify, troubleshoot).  
Normative challenge contract: [`../AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) (**FROZEN**).  
Normative bundle bytes: [`../BUNDLE_SPEC.md`](../BUNDLE_SPEC.md) (**FROZEN**).  
Security claim (what validators prove, and what they do not): [`../THREAT_MODEL.md`](../THREAT_MODEL.md) §1 (D19).

| Page | Topic |
|------|--------|
| [funding-phala.md](./funding-phala.md) | Fund your own Phala account |
| [deploy.md](./deploy.md) | Render compose-hash and deploy CVM |
| [certify.md](./certify.md) | Bind quote and submit to validator |
| [troubleshoot.md](./troubleshoot.md) | Common failures |

---

## Quick path

```bash
# From gbase repo root (Rust 1.96 toolchain)
cargo build -q -p miner-bin

# 1) Offline compose-hash (no Phala call)
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1

# 2) After funding Phala and installing `phala` CLI — real deploy
# cargo run -q -p miner-bin -- deploy --deploy --netuid 1

# 3) Certify (fixture mode for offline smoke; live needs agent URL + validator)
# cargo run -q -p miner-bin -- certify \
#   --fixture-mode \
#   --validator-url http://127.0.0.1:8081 \
#   --epoch 0 \
#   --miner-hotkey-hex <64 hex>
```

Miners **fund their own** Phala account. The subnet owner does not pay your CVM bill.

---

## Version pin

When `protocol_version` bumps in `bundle`, update:

1. The HTML comment and bold badge at the top of **this file**.
2. Any copy in sibling pages that states the bundle protocol version.
3. Re-run `cargo run -p xtask -- external-docs-check`.
