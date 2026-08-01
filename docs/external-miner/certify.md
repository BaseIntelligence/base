# Certify miner attestation

<!-- protocol_version: 1 -->

**Bundle `protocol_version`:** `1` · **Challenge `scoring_version`:** `2`

Binds a TDX quote to this epoch via D10 `report_data` and submits it to a validator.  
Normative details: [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md).  
Policy outcomes: Verified / Rejected / Parked (park does **not** carry prior Verified forward).

Certify is the **attestation** path. It is independent of pack scoring: a Verified quote does not mean your `model.patch` scores are “good” (D19). Live scoring uses Harbor packs + operator grade under `challenge_scoring_version = 2`, not the retired echo-answer path.

---

## 1. Inputs you need

| Input | Source |
|-------|--------|
| `--validator-url` | Operator-published validator base URL (`BASE_VALIDATOR_URL`) |
| `--netuid` | Subnet netuid |
| `--epoch` | Current epoch (operator / chain) |
| `--miner-hotkey-hex` | Your 32-byte hotkey as 64 lowercase hex chars (`BASE_MINER_HOTKEY_HEX`) |
| Live: `--agent-url` | Public base URL of your CVM agent |
| Offline smoke: `--fixture-mode` | Uses embedded/fixture quote material |

After `./install.sh`, your local runner should already answer capacity (not a substitute for live TDX certify):

```bash
curl -sS http://127.0.0.1:8080/v1/capacity
curl -sS http://127.0.0.1:8080/healthz
```

---

## 2. Offline / fixture smoke

Useful when no CVM is up. Still exercises CLI wiring:

```bash
# Requires a reachable validator attest endpoint OR will fail on HTTP —
# for pure CLI parse checks, prefer unit tests:
cargo test -q -p miner

# Deploy dry-run remains the zero-dependency smoke:
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1
# or: ./install.sh  (needs Docker + model-key file)
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
export BASE_VALIDATOR_URL='https://validator.example'  # operator URL
export BASE_NETUID=1
export BASE_EPOCH=123
export BASE_MINER_HOTKEY_HEX='<64 hex>'
# Phala publishes each container port on its own hostname, so 8081 is a
# subdomain, not a `:8081` suffix:
export BASE_AGENT_URL='https://<app-id>-8081.<node>.phala.network'
export BASE_LAUNCH_TOKEN_FILE='./launch_token'   # the file `deploy` provisioned

cargo run -q -p miner-bin -- certify \
  --validator-url "$BASE_VALIDATOR_URL" \
  --netuid "$BASE_NETUID" \
  --epoch "$BASE_EPOCH" \
  --miner-hotkey-hex "$BASE_MINER_HOTKEY_HEX" \
  --agent-url "$BASE_AGENT_URL" \
  --launch-token-file "$BASE_LAUNCH_TOKEN_FILE"
```

The attest-helper serves `/v1/quote` only to a caller presenting the launch
token as `Authorization: Bearer <token>`; without it the CVM answers **401**,
and a CVM deployed with no launch token answers **503**. Anyone able to obtain
an unauthenticated quote could bind it to *their* hotkey, so keep the token
file secret and re-deploy if it leaks.

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
- It does not re-enable scoring_version 1 echo answers — packs + `model.patch` only.

---

## Next

[troubleshoot.md](./troubleshoot.md)
