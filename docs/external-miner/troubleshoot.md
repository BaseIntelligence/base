# Miner troubleshooting

<!-- protocol_version: 1 -->

---

## Deploy

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `deploy --no-deploy` non-zero | Bad image ref / hash format | Use `repo@sha256:` + 64 hex; check `--launch-token-hash` is 64 hex |
| `phala` not found with `--deploy` | CLI missing | Install Phala CLI; or pass `--phala-bin` |
| Deploy fails: insufficient balance | Unfunded account | [funding-phala.md](./funding-phala.md) |
| Compose-hash differs from operator expect | Image/tag drift or template skew | Rebuild from same gbase commit; compare AGENT_CHALLENGE image pins |

```bash
cargo run -q -p gbase-miner-bin -- deploy --no-deploy --netuid 1
echo exit=$?
```

---

## Certify

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| HTTP errors to validator | Wrong URL / TLS / firewall | Confirm operator URL; curl `/healthz` if exposed |
| `Rejected` | Quote/event-log/policy | Check measurements allowlist rotation; redeploy measured image |
| `Parked` | Collateral/TCB outage | Wait; do not expect prior Verified to carry forward |
| Missing `--agent-url` | Live mode without URL | Pass `--agent-url` or use `--fixture-mode` only for smoke |
| Hotkey parse error | Not 64 hex | Export raw 32-byte public key as lowercase hex |

---

## Scoring / weights (miner view)

| Symptom | Notes |
|---------|--------|
| You are up but weight is zero | Challenge may emit `NoScore`; attestation Parked; or you are outside expected set (stake/policy) |
| Gateway "looks wrong" | Validators recompute from local trust roots; a deviant gateway is a validator-side detection problem (D19), not something miners fix by trusting gateway HTML |

Miners do not re-run aggregation. Bundle math is validator-side per BUNDLE_SPEC.

---

## Protocol version mismatch

If operators announce a new bundle `protocol_version` and your docs/binary lag:

1. Upgrade gbase to the release validators run.
2. Confirm badge in [`README.md`](./README.md) matches.
3. Redeploy CVM if the challenge compose contract changed (`challenge_scoring_version` bump).

```bash
# From repo: must exit 0
cargo run -q -p xtask -- external-docs-check
```

---

## Getting help

Provide: gbase git SHA, `compose-hash=`, certify `outcome=`, netuid, epoch, **public** hotkey hex only.  
Never send mnemonics, age identities, Phala API keys, or coldkey files.
