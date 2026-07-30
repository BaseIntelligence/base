# Deploy the miner CVM

<!-- protocol_version: 1 -->

Normative compose service/port/image rules: [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) § measured app-compose.  
CLI: `gbase-miner deploy`.

---

## 1. Offline compose-hash (always do this first)

From the gbase repo root:

```bash
cargo run -q -p gbase-miner-bin -- deploy --no-deploy --netuid 1
```

Expected stdout includes:

```text
compose-hash=<64 hex>
phala_invoked=false
```

Exit code must be **0**.

Optional: write the rendered compose JSON:

```bash
cargo run -q -p gbase-miner-bin -- deploy --no-deploy --netuid 1 \
  --out /tmp/gbase-miner-app-compose.json
test -f /tmp/gbase-miner-app-compose.json
```

Flags of interest:

| Flag | Meaning |
|------|---------|
| `--agent-image` | Digest-pinned agent image (`repo@sha256:…`) |
| `--attest-helper-image` | Digest-pinned helper image |
| `--launch-token-hash` | Lowercase hex SHA-256 of the launch token (measured) |
| `--netuid` | Subnet netuid embedded as non-secret env |
| `--no-deploy` | Default path: hash only |
| `--deploy` | Invoke `phala deploy` after hashing |
| `--phala-bin` | Path to `phala` (default `phala`) |

---

## 2. Live deploy

Preconditions: [funding-phala.md](./funding-phala.md), `phala` on PATH, images reachable.

```bash
export GBASE_AGENT_IMAGE='<repo>@sha256:<64 hex>'
export GBASE_ATTEST_HELPER_IMAGE='<repo>@sha256:<64 hex>'
export GBASE_LAUNCH_TOKEN_HASH='<64 hex of token>'
export GBASE_NETUID=1   # publish real netuid when live

cargo run -q -p gbase-miner-bin -- deploy --deploy --netuid "$GBASE_NETUID"
```

Notes printed by the CLI:

- You fund your own Phala account.
- Secrets are file mounts under `/run/gbase` (not env values).

Record the public agent URL Phala assigns. You need it for certify.

---

## 3. After deploy

1. Confirm the CVM is healthy and the agent HTTP port from AGENT_CHALLENGE is reachable from the challenge service path (via gateway when in production topology).
2. Proceed to [certify.md](./certify.md) each epoch (or per operator schedule).
3. When measurements or images rotate, redeploy and re-hash; validators fail closed on unknown measurements.

---

## 4. Spot-check (no Phala account required)

```bash
cargo run -q -p gbase-miner-bin -- deploy --no-deploy --netuid 1
```

Must exit 0.
