# Fund your Phala account

Miners pay for their own Phala TDX CVM. The gbase owner does not sponsor miner compute.

<!-- protocol_version: 1 -->

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

There is no gbase CLI that spends owner funds on your behalf.

---

## 3. Install tools on your machine

```bash
# Rust pin matches the repo
rustc --version   # expect 1.96.x via rust-toolchain.toml when inside the repo

# Phala CLI must be on PATH when you pass --deploy
command -v phala
phala --help >/dev/null
```

Clone or pull `gbase` at the release your validators run (same `protocol_version`).

---

## 4. Wallet (Bittensor)

You need a miner hotkey registered on the gbase netuid (when the subnet is live). Hotkey **public** bytes (32-byte hex) are what `gbase-miner certify` embeds. Never paste mnemonics into tickets, git, or chat.

```bash
# Example overview (network/netuid as published by operators)
btcli wallet overview --wallet-name <your-wallet> --network <network>
```

---

## 5. Secrets on the CVM

Per D11 / AGENT_CHALLENGE:

- Secrets are **file mounts** under the measured layout (for example `/run/gbase/...`).
- Env **values** are not attested. Only allowed env **names** and the launch-token **hash** are in the measured compose.
- Do not put coldkey material inside the CVM unless a future challenge doc explicitly requires a file mount pattern.

---

## Next

[deploy.md](./deploy.md)
