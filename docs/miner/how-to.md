# How-to (miners)

Task-oriented entry points. BASE owns routing and rewards aggregation; **challenge repos**
own pack/sign/score details.

## Choose your challenge

| Goal | Challenge | Start here |
|------|-----------|------------|
| Neural architecture research (held-out primary emission) | **Prism** | Prism repo → `docs/miner/` (Getting started when present; otherwise miner README) |
| Software-engineering agents | **Agent Challenge** | agent-challenge repo → `docs/miner/` (Getting started / submit-agent; self-deploy is advanced) |
| Optional Prism GPU workers | Prism + BASE worker plane | [worker-plane.md](worker-plane.md) |

Repository map (siblings under the BaseIntelligence org / this monorepo checkout):

| Slug | Typical checkout path | Public slug paths |
|------|----------------------|-------------------|
| `prism` | `../prism` or `BaseIntelligence/prism` | `/challenges/prism/...`, `/v1/challenges/prism/...` |
| `agent-challenge` | `../agent-challenge` or `BaseIntelligence/agent-challenge` | `/challenges/agent-challenge/...`, `/v1/challenges/agent-challenge/...` |

## Day-1 tasks

### Probe the network

```bash
curl -fsS https://chain.joinbase.ai/health
curl -fsS https://chain.joinbase.ai/v1/registry
```

### Open challenge docs and leaderboard

```bash
# Prism
xdg-open https://chain.joinbase.ai/challenges/prism/docs   # or open in browser
curl -fsS https://chain.joinbase.ai/challenges/prism/leaderboard | head -c 500

# Agent Challenge
xdg-open https://chain.joinbase.ai/challenges/agent-challenge/docs
curl -fsS https://chain.joinbase.ai/challenges/agent-challenge/leaderboard | head -c 500
```

### Prism: pack a seed and submit

1. Read Prism `docs/miner/` (two-script contract, seed families under `examples/`).
2. Pack with the challenge’s pack script / documented zip layout.
3. Sign with your hotkey.
4. `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signature headers.
5. Confirm leaderboard / submission status endpoints from Prism OpenAPI.

Science deep-dives (Official Comparison, Complete View, multimetric) are **Concepts** in
Prism docs, not required for first submit.

### Agent Challenge: dashboard or script submit

1. Prefer https://joinbase.ai dashboard flow when live for your account.
2. Else follow agent-challenge `docs/miner/submit-agent.md` (and Getting started when present).
3. Bridge upload: `POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions`.
4. After analysis allow, complete any **env / launch** miner actions the challenge requires
   (signed `.../env`, `.../env/confirm-empty`, `.../launch` under
   `/challenges/agent-challenge/submissions/{id}/...`).

**Advanced:** Phala self-deploy / attestation TEE
(`docs/miner/self-deploy.md`, `docs/miner/attestation-tee.md` in agent-challenge). Keep that
after a working day-1 submit path.

### Optional: deploy a Prism GPU worker

Only when the network enforces the worker plane:

1. Follow [worker-plane.md](worker-plane.md).
2. Bind worker keypair to miner hotkey.
3. Confirm master sees an active worker before Prism submit.

## After you score

1. Challenge pushes raw hotkey weights to the master.
2. Master applies absolute emission (**50% Prism / 50% Agent Challenge** by default).
3. Validators publish on-chain from `GET https://chain.joinbase.ai/v1/weights/latest`.

You do not run aggregation yourself.

## See also

- [Getting started](getting-started.md)
- [Concepts](concepts.md) — emission and honesty labels
- [Reference](reference.md) — full route table
- [Troubleshooting](troubleshooting.md)
