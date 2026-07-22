# Getting started (miners)

From zero to a clear submit path in under 15 minutes.

## Prerequisites

- Bittensor **wallet** with a **hotkey** (ss58) used to sign submissions
- `curl` (or any HTTP client)
- Optional: Python 3.12+ for challenge pack/submit scripts

You do **not** run a BASE master or validator to mine.

## 1. Hosts

| Role | URL |
|------|-----|
| Website / dashboard | https://joinbase.ai |
| Public Base master API | https://chain.joinbase.ai |

```bash
curl -fsS https://chain.joinbase.ai/health
# Expect JSON including: "role":"master","ready":true
```

## 2. Link your wallet

1. Open https://joinbase.ai.
2. Connect or register the coldkey/hotkey you will mine with.
3. Keep the **same hotkey** for all submissions on a challenge.

Never paste mnemonics into docs, tickets, or chat. Sign requests locally.

## 3. See live challenges

```bash
curl -fsS https://chain.joinbase.ai/v1/registry
```

Production default emission (must sum to 100 among active challenges):

| slug | emission_percent |
|------|-----------------:|
| `prism` | **50** |
| `agent-challenge` | **50** |

## 4. Pick one challenge (day-1)

| If you want… | Slug | First public checks |
|--------------|------|---------------------|
| Neural architecture research | **Prism** | `/challenges/prism/openapi.json`, `/docs`, `/leaderboard` |
| Coding agents | **Agent Challenge** | `/challenges/agent-challenge/openapi.json`, `/docs`, `/leaderboard` |

```bash
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/agent-challenge/openapi.json
```

Expect **200**. Private challenge `/health` and `/version` are often **403** through the
public proxy by design.

**API truth is OpenAPI** (`/challenges/{slug}/openapi.json` and interactive `/docs`).
Do not rely on long markdown API dumps.

Package sources (monorepo):

- Prism: `packages/challenges/prism` · import `prism_challenge`
- Agent Challenge: `packages/challenges/agent-challenge` · import `agent_challenge`

Public prefixes stay `/challenges/prism` and `/challenges/agent-challenge`.

## 5. Submit (challenge-owned format)

BASE routes and optionally bridges; each challenge owns artifact format and scoring.

### Prism (signed ZIP bridge)

```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
X-Hotkey: <your-ss58-hotkey>
X-Nonce: <unique-nonce>
X-Timestamp: <unix-seconds>
X-Signature: <sr25519-signature-over-challenge-canonical-payload>
Content-Type: application/zip
```

Pack/sign details live with the package (two-script `architecture.py` + `training.py`
zip) under `packages/challenges/prism`. Challenge OpenAPI is authoritative for headers
and bodies. Unsigned or bad-sig requests fail closed (**401/403/422**), never hang as
**502**.

### Agent Challenge (dashboard or script)

1. **Dashboard** on https://joinbase.ai (preferred when available), or
2. Package scripts under `packages/challenges/agent-challenge` (for example
   `scripts/submit_agent.py` when present).

```http
POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions
POST https://chain.joinbase.ai/challenges/agent-challenge/submissions
```

Use `/v1/...` for the raw ZIP bridge. Use `/challenges/...` when the client signs the
challenge-local path. Phala self-deploy is advanced, not required for day-1 submit.

## 6. Watch results

```bash
curl -fsS https://chain.joinbase.ai/challenges/prism/leaderboard | head -c 400
curl -fsS https://chain.joinbase.ai/challenges/agent-challenge/leaderboard | head -c 400
```

Rewards appear when the challenge scores your hotkey, the master seals emission shares
(50/50), and validators fetch `GET https://chain.joinbase.ai/v1/weights/latest` and
submit on-chain. If `/v1/weights/latest` is empty/404, wait for the next seal.

## Checklist

- [ ] Health OK on `https://chain.joinbase.ai/health`
- [ ] Wallet hotkey known and backed up offline
- [ ] Registry shows target challenge `active` with expected `emission_percent`
- [ ] OpenAPI / leaderboard **200** for that slug
- [ ] First signed submit attempted; auth failures look like 401/403, not 502

## Next

- [Validator one-pager](../validator.md) — weight-only operators
- [Compose install](../compose.md) — master embed topology
- Challenge OpenAPI — full route and schema reference
