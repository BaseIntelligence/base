# Miner hub

Mine BASE in minutes. BASE is the multi-challenge coordination layer behind
**[joinbase.ai](https://joinbase.ai)**. You pick a challenge, submit work signed with your
Bittensor hotkey, and earn when that challenge exports a raw weight for your hotkey.

**Day-1 target:** wallet ready → challenge chosen → first submission path understood in under
15 minutes. Deep science, TEE self-deploy, and GPU worker ops stay in Concepts / How-to, not
on the first page.

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | joinbase + chain URLs, wallet, pick a challenge, first submit |
| [Concepts](concepts.md) | Multi-challenge BASE, absolute **50/50** emission (Prism + Agent Challenge) |
| [How-to](how-to.md) | Links into Prism and Agent Challenge miner guides |
| [Reference](reference.md) | Public routes, headers, discovery |
| [Troubleshooting](troubleshooting.md) | 401 / 429 / 502 and common auth failures |
| [Worker plane (Prism GPU)](worker-plane.md) | Optional miner-funded GPU workers (advanced) |

## Canonical public URLs

| Surface | URL |
|---------|-----|
| Product / dashboard | https://joinbase.ai |
| Base master API (proxy, registry, weights) | https://chain.joinbase.ai |
| Master health | `GET https://chain.joinbase.ai/health` → `role=master`, `ready=true` |
| Active challenges | `GET https://chain.joinbase.ai/v1/registry` |

Do **not** use historical hostnames (for example `chain.platform.network`) as the shipping
master URL.

## Active emission split (network default)

| Challenge slug | Emission share | What you build |
|----------------|---------------:|----------------|
| `prism` | **50%** (absolute) | Neural architecture + training loop packages |
| `agent-challenge` | **50%** (absolute) | Software-engineering agents (Terminal-Bench) |

Shares are **absolute** registry percents. Missing or unscored side burns that share (uid0
policy on seal). BASE never calls `set_weights`; independent validators fetch
`GET /v1/weights/latest` and submit on-chain under their own wallets.

## Quick path

1. Open [joinbase.ai](https://joinbase.ai) and link / register your miner hotkey wallet.
2. Confirm the network is live: `curl -fsS https://chain.joinbase.ai/health`.
3. List challenges: `curl -fsS https://chain.joinbase.ai/v1/registry`.
4. Pick **Prism** or **Agent Challenge** and follow that challenge’s Getting started
   ([How-to](how-to.md)).
5. Submit through the signed public path; watch the challenge leaderboard.

Full walkthrough: [Getting started](getting-started.md).
