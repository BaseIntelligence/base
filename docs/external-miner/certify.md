# Certify miner attestation

<!-- protocol_version: 1 -->

Binds a TDX quote to this epoch via D10 `report_data` and submits it to a validator.  
Normative details: [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md).  
Policy outcomes: Verified / Rejected / Parked (park does **not** carry prior Verified forward).

---

## 1. Inputs you need

| Input | Source |
|-------|--------|
| `--validator-url` | Operator-published validator base URL |
| `--netuid` | Subnet netuid |
| `--epoch` | Current epoch (operator / chain) |
| `--miner-hotkey-hex` | Your 32-byte hotkey as 64 lowercase hex chars |
| Live: `--agent-url` | Public base URL of your CVM agent |
| Offline smoke: `--fixture-mode` | Uses embedded/fixture quote material |

---

## 2. Offline / fixture smoke

Useful when no CVM is up. Still exercises CLI wiring:

```bash
# Requires a reachable validator attest endpoint OR will fail on HTTP —
# for pure CLI parse checks, prefer unit tests:
cargo test -q -p miner

# Deploy dry-run remains the zero-dependency smoke:
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1
```

When a local validator is running with fixture-friendly config:

```bash
cargo run -q -p miner-bin -- certify \
  --fixture-mode \
  --validator-url "http://127.0.0.1:8081" \
  --netuid 1 \
  --epoch 0 \
  --miner-hotkey-hex "11$(printf '22%.0s' {1..31})"
```

(Replace hotkey with a real 64-hex key in real use. The example above is a structural placeholder pattern only if you generate a valid 64-hex string yourself.)

Safer explicit hotkey example (64 hex zeros is valid length for dry wiring tests only):

```bash
HK=$(printf 'ab%.0s' {1..32})   # 64 hex chars
echo "hotkey_len=${#HK}"
```

---

## 3. Live certify

```bash
export GBASE_VALIDATOR_URL='https://validator.example'  # operator URL
export GBASE_NETUID=1
export GBASE_EPOCH=123
export GBASE_MINER_HOTKEY_HEX='<64 hex>'
export GBASE_AGENT_URL='https://<your-cvm-host>'

cargo run -q -p miner-bin -- certify \
  --validator-url "$GBASE_VALIDATOR_URL" \
  --netuid "$GBASE_NETUID" \
  --epoch "$GBASE_EPOCH" \
  --miner-hotkey-hex "$GBASE_MINER_HOTKEY_HEX" \
  --agent-url "$GBASE_AGENT_URL"
```

Interpret stdout:

| Field | Meaning |
|-------|---------|
| `outcome=Verified` | Attestation credit this epoch |
| `outcome=Rejected` | Cryptographic / policy failure |
| `outcome=Parked` | Collateral/TCB outage path — **no** credit; not last-known-good |
| `grants_credit=` | Whether this epoch grants credit |
| `carries_prior_verified=` | Must be false on Parked (D13) |

---

## 4. What certify does **not** mean

- It does not prove your agent scores are "good" (D19(i)).
- It does not prove env secret **values** (D11).
- It does not put anything into the on-chain weight merkle field (there is none; D5).

---

## Next

[troubleshoot.md](./troubleshoot.md)
