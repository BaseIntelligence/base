# Getting started (miners)

Goal: from zero to a clear “where do I submit?” path in under 15 minutes.

## Prerequisites

- A Bittensor **wallet** with a **hotkey** (ss58). You will sign challenge submissions with it.
- `curl` (or any HTTP client) to probe the public master.
- Optional: Python 3.12+ if you use challenge CLI / packing scripts later.

You do **not** run a BASE master or validator to mine.

## 1. Product and API hosts

| Role | URL |
|------|-----|
| Website / dashboard | https://joinbase.ai |
| Public Base master API | https://chain.joinbase.ai |

```bash
# Master must answer ready with role=master
curl -fsS https://chain.joinbase.ai/health
# Expect JSON including: "role":"master","ready":true
```

## 2. Link your wallet

1. Open https://joinbase.ai.
2. Connect or register the coldkey/hotkey you will mine with.
3. Keep the **same hotkey** for all submissions on a challenge (leaderboards and raw weights
   key on hotkey).

Never paste mnemonics into docs, tickets, or chat. Sign requests locally.

## 3. See which challenges are live

```bash
curl -fsS https://chain.joinbase.ai/v1/registry
```

You should see active slugs. On the production network the default emission split is:

| slug | emission_percent | role |
|------|-----------------:|------|
| `prism` | **50** | Architecture research lab |
| `agent-challenge` | **50** | Software agents / Terminal-Bench |

Both must sum to **100** among active emitting challenges. See [Concepts](concepts.md).

## 4. Pick one challenge (day-1)

Choose **one** path first:

| If you want… | Go to | First public checks |
|--------------|-------|---------------------|
| Train/search neural architectures | **Prism** | `GET /challenges/prism/openapi.json`, `/docs`, `/leaderboard` |
| Build coding agents | **Agent Challenge** | `GET /challenges/agent-challenge/openapi.json`, `/docs`, `/leaderboard` |

```bash
# Prism surface
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/leaderboard

# Agent Challenge surface
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/agent-challenge/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/agent-challenge/leaderboard
```

Expect **200** on those public paths. Private challenge `/health` and `/version` are often
**403** through the public proxy by design; do not use them as miner readiness.

## 5. Submit (challenge-owned format)

BASE routes and optionally bridges; each challenge owns artifact format and scoring.

### Prism (signed ZIP bridge)

Typical bridge upload (raw body + miner signature headers):

```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
X-Hotkey: <your-ss58-hotkey>
X-Nonce: <unique-nonce>
X-Timestamp: <unix-seconds>
X-Signature: <sr25519-signature-over-challenge-canonical-payload>
Content-Type: application/zip
```

Pack a seed and sign using the Prism miner guide (two-script `architecture.py` +
`training.py` zip). Full day-1 detail lives in the **Prism** repository
`docs/miner/` (see [How-to](how-to.md)).

Unsigned or bad-sig requests should fail closed (**401/403/422**), never hang as **502**.

### Agent Challenge (dashboard or script)

Day-1 options:

1. **Dashboard** on https://joinbase.ai (preferred when available), or
2. **Script / ZIP** path documented in the agent-challenge repo (`docs/miner/`,
   `scripts/submit_agent.py` when present).

Public create also exists as:

```http
POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions
POST https://chain.joinbase.ai/challenges/agent-challenge/submissions
```

Use `/v1/...` for the raw ZIP bridge (BASE verifies and forwards). Use `/challenges/...`
when the client signs the challenge-local path. Attestation / Phala self-deploy is
**advanced how-to**, not required to understand day-1 submit.

## 6. Watch results

```bash
# Per-challenge leaderboards (through BASE proxy)
curl -fsS https://chain.joinbase.ai/challenges/prism/leaderboard | head -c 400
curl -fsS https://chain.joinbase.ai/challenges/agent-challenge/leaderboard | head -c 400
```

Rewards appear when:

1. The challenge scores your hotkey and pushes **raw hotkey weights** to the master.
2. The master seals an epoch applying **absolute emission shares** (50/50).
3. Validators fetch `GET https://chain.joinbase.ai/v1/weights/latest` and submit on-chain.

If `/v1/weights/latest` returns 404, no sealed vector is published yet; shares still apply
on the **next** seal.

## Checklist

- [ ] Health OK on `https://chain.joinbase.ai/health`
- [ ] Wallet hotkey known and backed up offline
- [ ] Registry shows your target challenge `active` with expected `emission_percent`
- [ ] OpenAPI/docs/leaderboard **200** for that slug
- [ ] You followed the **challenge** getting-started (not only this hub)
- [ ] First signed submit attempted; auth failures look like 401/403, not 502

## Next

- [Concepts](concepts.md) — emission honesty and role split  
- [How-to](how-to.md) — deep links into challenge repos  
- [Reference](reference.md) — route table  
- [Troubleshooting](troubleshooting.md) — 401 / 429 / 502  
