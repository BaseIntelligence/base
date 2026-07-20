# How-to (miners)

Task-oriented entry points. BASE owns routing and rewards aggregation. **Challenge product
sources and day-1 miner docs now live in this monorepo** under
`docs/miner/{prism,agent-challenge}/` and `packages/challenges/*`.

## Choose your challenge

| Goal | Challenge | Start here (monorepo hub) |
|------|-----------|---------------------------|
| Neural architecture research (held-out primary emission) | **Prism** | [Prism Getting started](prism/getting-started.md) · [hub](prism/README.md) |
| Software-engineering agents | **Agent Challenge** | [AC Getting started](agent-challenge/getting-started.md) · [submit-agent](agent-challenge/submit-agent.md) (self-deploy is advanced) |
| Optional Prism GPU workers | Prism + BASE worker plane | [worker-plane.md](worker-plane.md) |

Repository / path map (source of truth is **BaseIntelligence/base**):

| Slug | Monorepo product path | Public slug paths | Day-1 doc |
|------|----------------------|-------------------|-----------|
| `prism` | `packages/challenges/prism` | `/challenges/prism/...`, `/v1/challenges/prism/...` | [docs/miner/prism/getting-started.md](prism/getting-started.md) |
| `agent-challenge` | `packages/challenges/agent-challenge` | `/challenges/agent-challenge/...`, `/v1/challenges/agent-challenge/...` | [docs/miner/agent-challenge/getting-started.md](agent-challenge/getting-started.md) |

Standalone remotes `BaseIntelligence/prism` and `BaseIntelligence/agent-challenge` are
transitioning: prefer this monorepo for product edits. See
[docs/monorepo.md](../monorepo.md) and [SOURCE_OF_TRUTH.md](../SOURCE_OF_TRUTH.md).

### Cross-cut honesty (read once)

These labels must stay consistent across BASE, Prism, and Agent Challenge docs:

| Topic | Truth |
|-------|-------|
| Emission | BASE aggregates challenge **raw hotkey weights** with **absolute** `emission_percent` shares. Production default: **Prism 50** + **Agent Challenge 50** (sum 100). Missing scorer burns that share (uid0); no relative renormalize of whoever reported. |
| Rank keys | **Wall-clock never ranks emission.** Prism emission: held-out primary / bpb secondary. Agent Challenge: challenge benchmark scores. |
| Gateway | **No Base LLM gateway** for miners (`BASE_LLM_GATEWAY_*` / `/llm/v1` do not exist on Compose master). |
| Prism TEE | Prism product is **NO-TEE**: provider trust (Lium/Targon) + IMAGE_PIN; no Prism TEE-required scoring path. |
| AC attestation | Agent Challenge Phala / KR self-deploy is **advanced how-to**, separate from day-1 ZIP upload. |

Emission math and labels: [Concepts](concepts.md).

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

1. Read [Prism Getting started](prism/getting-started.md)
   (two-script contract, seed families under `packages/challenges/prism/examples/`).
2. Pack with `packages/challenges/prism/scripts/pack_seed_family.py` / documented zip layout.
3. Sign with your hotkey (canonical
   `prism:{hotkey}:{nonce}:{timestamp}:{sha256(zip)}`).
4. `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signature headers.
5. Confirm leaderboard / submission status endpoints from Prism OpenAPI.

Science deep-dives (Official Comparison, Complete View, multimetric) are **Concepts** in
Prism docs, not required for first submit. Prism scoring is **NO-TEE** (provider trust +
IMAGE_PIN); do not invent REAL-PROVIDER TEE or a Base LLM gateway.

### Agent Challenge: dashboard or script submit

1. Prefer https://joinbase.ai dashboard flow when live for your account.
2. Else follow [AC Getting started](agent-challenge/getting-started.md)
   and [submit-agent](agent-challenge/submit-agent.md)
   (`packages/challenges/agent-challenge/scripts/submit_agent.py`).
3. Proxy JSON upload:
   `POST https://chain.joinbase.ai/challenges/agent-challenge/submissions`
   (or ZIP bridge `POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions`).
4. After analysis allow, complete any **env / launch** miner actions the challenge requires
   (signed `.../env`, `.../env/confirm-empty`, `.../launch` under
   `/challenges/agent-challenge/submissions/{id}/...`).

**Advanced:** Phala self-deploy / attestation TEE
([self-deploy](agent-challenge/self-deploy.md),
[attestation-tee](agent-challenge/attestation-tee.md)).
Keep that after a working day-1 submit path. No Base LLM gateway restore.

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
