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
| `--launch-token-file` | Host path of the **raw** launch token (mode 0600, generated if missing) |
| `--receipt-sk-host-path` | Host path of the work-receipt mini-secret (mode 0600, generated if missing) |
| `--miner-hotkey-hex` | Your **public** hotkey hex; required with `--deploy` |
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
| `attest-helper` | Quote + event log for certify | `8081` public, launch-token authenticated |

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
export BASE_MINER_HOTKEY_HEX='<64 lowercase hex public hotkey>'
export BASE_NETUID=1   # publish real netuid when live

cargo run -q -p miner-bin -- deploy --deploy --netuid "$BASE_NETUID" \
  --launch-token-file "$HOME/.base/launch_token" \
  --receipt-sk-host-path "$HOME/.base/receipt_sk"
```

`--deploy` requires all three of `--launch-token-file`, `--receipt-sk-host-path` and
`--miner-hotkey-hex`. The first two files are generated (mode 0600) on first use and
reused afterwards; keep them — the launch token is the bearer credential you present to
your own `attest-helper`, and the receipt key is what signs your work receipts.

### How the secrets reach the CVM

The measured `app-compose.json` is **public**: it is submitted to validators, hash-checked
against RTMR3, and served by Phala. It therefore carries only the secret variable *names*.
The values are sent as **Phala encrypted secrets** (X25519 + AES-256-GCM, decrypted only
inside the TEE), and the measured `pre_launch_script` writes them to the bind sources the
compose mounts under `/run/base`:

| Encrypted secret | Becomes |
|------------------|---------|
| `BASE_RECEIPT_SK_HEX` | `/run/base/receipt_sk` (mode 0400) |
| `BASE_LAUNCH_TOKEN` | `/run/base/launch_token` |
| `BASE_MINER_HOTKEY_HEX` | `/run/base/miner_hotkey` |

If a value is missing the CVM aborts at boot instead of starting with an empty file.
**Never** paste a key into the compose, a pre-launch script literal, a ticket, or a log:
anything in the compose is published with the measurement.

Because only the names are measured, the compose-hash from step 1 is unchanged by your
secret values.

Notes printed by the CLI:

- You fund your own Phala account.  
- Secret values travel as encrypted env and land as file mounts under `/run/base`; they are never measured.

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
