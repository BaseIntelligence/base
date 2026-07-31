# Fund your Phala account

Miners pay for their own Phala TDX CVM **and** their own model inference (Q3=A). The base owner does not sponsor miner compute or LLM tokens.

<!-- protocol_version: 1 -->

**Bundle `protocol_version`:** `1` · **Challenge `scoring_version`:** `2`

---

## 1. Create a Phala account

1. Sign up at the Phala Cloud console used by your region (operator will publish the exact portal URL for the subnet epoch).  
2. Create an API key / CLI login for the `phala` binary if required by current Phala tooling.  
3. Confirm the account can deploy **TDX** CVMs (not only non-TEE instances).

---

## 2. Add credits

1. Top up the account with enough balance for:
   - one CVM at the size required by [`AGENT_CHALLENGE.md`](../AGENT_CHALLENGE.md) compose contract,
   - continuous uptime across epochs you intend to mine,
   - headroom for redeploys after measurement or image rotations.
2. Re-check balance before every `--deploy` invocation.

There is no base CLI that spends owner funds on your behalf.

---

## 3. Miner-funded model key (Q3=A)

Scoring_version **2** runs a reference agent against Harbor pack instructions. The agent needs a provider API key **you** pay for:

```bash
umask 077
printf '%s' "$YOUR_PROVIDER_API_KEY" > /secure/model_key
chmod 0600 /secure/model_key
export BASE_MODEL_KEY_FILE=/secure/model_key
```

- Pass the **path** into `./install.sh` / CVM mounts (`BASE_MODEL_KEY_FILE`).  
- Never put key bytes in compose `environment:` values, git, chat, or tickets.  
- `install.sh` refuses to start if the file is missing, empty, or unreadable (fail-closed).

---

## 4. Install tools on your machine

```bash
# Docker + Compose (required for ./install.sh)
docker info
docker compose version

# Optional: Rust pin matches the repo (for miner-bin CLI)
rustc --version   # expect 1.96.x via rust-toolchain.toml when inside the repo

# Phala CLI must be on PATH when you pass --deploy
command -v phala
phala --help >/dev/null
```

Clone or pull `base` at the release your validators run (same bundle `protocol_version` **1**; challenge scoring is **2**).

Quick start after funding:

```bash
export BASE_MINER_HOTKEY_HEX='<64 hex>'
export BASE_MODEL_KEY_FILE=/secure/model_key
export BASE_MAX_CONCURRENCY=1
./install.sh
```

---

## 5. Wallet (Bittensor)

You need a miner hotkey registered on the base netuid (when the subnet is live). Hotkey **public** bytes (32-byte hex) are what `miner certify` and `install.sh` embed. Never paste mnemonics into tickets, git, or chat.

```bash
# Example overview (network/netuid as published by operators)
btcli wallet overview --wallet-name <your-wallet> --network <network>
```

---

## 6. Secrets on the CVM

Per D11 / AGENT_CHALLENGE:

- Secrets are **file mounts** under the measured layout (for example `/run/base/...`).  
- Env **values** are not attested. Only allowed env **names** and the launch-token **hash** are in the measured compose.  
- Do not put coldkey material inside the CVM unless a future challenge doc explicitly requires a file mount pattern.  
- Model key and receipt mini-secret follow the same file-mount rule.

---

## Next

[deploy.md](./deploy.md)
