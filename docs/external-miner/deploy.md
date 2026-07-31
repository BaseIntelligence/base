# Deploy the miner CVM

<!-- protocol_version: 1 -->

**Bundle `protocol_version`:** `1` · **Challenge `scoring_version`:** `2`

Normative compose service/port/image rules: [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) § measured app-compose.  
CLI: `miner deploy`. One-command box setup: repo-root [`install.sh`](../../install.sh).

---

## 0. One-command self-deploy (`install.sh`)

On a clean miner host (Docker + Compose required):

```bash
export BASE_MINER_HOTKEY_HEX='<64 lowercase hex public hotkey>'
export BASE_MODEL_KEY_FILE=/path/to/model_key   # mode 0600; miner-funded (Q3=A)
export BASE_MAX_CONCURRENCY=2                   # 1..=5
# export BASE_AGENT_IMAGE='repo@sha256:<64 hex>'
# export BASE_ATTEST_HELPER_IMAGE='repo@sha256:<64 hex>'
# export BASE_LAUNCH_TOKEN_HASH='<64 hex>'
# export BASE_VALIDATOR_URL='https://validator.example'

./install.sh
curl -sS http://127.0.0.1:8080/v1/capacity
```

What it does:

1. **Prereq checks** — Docker daemon, Compose, `python3`, readable model-key file, valid hotkey hex, concurrency in `1..5`. Failures are distinct, non-zero, and leave nothing half-installed.  
2. **Pull** digest-pinned images (no `:latest`).  
3. **Materialize secret files** under the install dir (`miner_hotkey`, `model_key`, `receipt_sk`) — **never** prints secret bytes.  
4. **Render** measured CVM `app-compose.json` (services `socket-proxy`, `agent`, `attest-helper`) and print `compose-hash=<64 hex>`.  
5. **Start** a local `agent-runner` answering `GET /v1/capacity` with your chosen concurrency (stub pack backend until the full CVM/pack path is live).

Re-running `./install.sh` with the same inputs is **idempotent**.

For Phala production, take the rendered `app-compose.json` / hash from the install dir (or `miner deploy`) and deploy with the Phala CLI after [funding-phala.md](./funding-phala.md).

---

## 1. Offline compose-hash (always do this first)

From the base repo root:

```bash
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1
```

Expected stdout includes:

```text
compose-hash=<64 hex>
phala_invoked=false
```

Exit code must be **0**.

Optional: write the rendered compose JSON:

```bash
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1 \
  --out /tmp/miner-app-compose.json
test -f /tmp/miner-app-compose.json
```

Flags of interest:

| Flag | Meaning |
|------|---------|
| `--agent-image` | Digest-pinned agent image (`repo@sha256:…`) |
| `--attest-helper-image` | Digest-pinned helper image |
| `--socket-proxy-image` | Digest-pinned measured socket-proxy |
| `--launch-token-hash` | Lowercase hex SHA-256 of the launch token (measured) |
| `--netuid` | Subnet netuid embedded as non-secret env |
| `--no-deploy` | Default path: hash only |
| `--deploy` | Invoke `phala deploy` after hashing |
| `--phala-bin` | Path to `phala` (default `phala`) |

Env surface (same names used by `install.sh`):

| Env | Role |
|-----|------|
| `BASE_AGENT_IMAGE` | Digest-pinned runner image |
| `BASE_ATTEST_HELPER_IMAGE` | Digest-pinned attest helper |
| `BASE_LAUNCH_TOKEN_HASH` | Measured launch-token hash |
| `BASE_VALIDATOR_URL` | For certify (not required to start runner) |
| `BASE_MINER_HOTKEY_HEX` | Public hotkey hex |
| `BASE_MODEL_KEY_FILE` | Path to miner-funded model key file |
| `BASE_MAX_CONCURRENCY` | Runner concurrency `1..=5` |

---

## 2. Measured CVM services (scoring_version 2)

| Service | Role | Port |
|---------|------|------|
| `agent` | HTTP runner (`/healthz`, `/v1/capacity`, `/v1/task`, …) | `8080` public |
| `socket-proxy` | Allowlisted Docker Engine API for pack env containers | internal only |
| `attest-helper` | Quote + event log for certify | `127.0.0.1:8081` |

**Raw `/var/run/docker.sock` must not be mounted into `agent`.** Only `socket-proxy` mounts the host socket (read-only), and that proxy is part of the measured compose-hash (RTMR3).

### Concurrency

`BASE_MAX_CONCURRENCY` (or `install.sh --max-concurrency`) is clamped to **1..=5**. `GET /v1/capacity` reports the effective max and current load. Over-capacity task accepts return HTTP **503** `capacity_exhausted`.

### Miner-funded inference (Q3=A)

Mount the provider API key as a **file** (e.g. `/run/base/model_key`). Env carries the **path** only. Never put key bytes in compose `environment:` values, tickets, or logs.

### Egress (OPEN default)

Default agent egress is **OPEN** ([`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) §4.2.1). Stripping protects grading-channel integrity, not miner honesty (D19).

---

## 3. Live Phala deploy

Preconditions: [funding-phala.md](./funding-phala.md), `phala` on PATH, images reachable.

```bash
export BASE_AGENT_IMAGE='<repo>@sha256:<64 hex>'
export BASE_ATTEST_HELPER_IMAGE='<repo>@sha256:<64 hex>'
export BASE_LAUNCH_TOKEN_HASH='<64 hex of token>'
export BASE_NETUID=1   # publish real netuid when live

cargo run -q -p miner-bin -- deploy --deploy --netuid "$BASE_NETUID"
```

Notes printed by the CLI:

- You fund your own Phala account.  
- Secrets are file mounts under `/run/base` (not env values).

Record the public agent URL Phala assigns. You need it for certify.

---

## 4. After deploy

1. Confirm the CVM is healthy: `curl -sS "$BASE_AGENT_URL/v1/capacity"`.  
2. Proceed to [certify.md](./certify.md) each epoch (or per operator schedule).  
3. When measurements or images rotate, redeploy and re-hash; validators fail closed on unknown measurements.

---

## 5. Spot-check (no Phala account required)

```bash
./install.sh --skip-pull   # if images already local
# or:
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1
```

Must exit 0 and print `compose-hash=`.
