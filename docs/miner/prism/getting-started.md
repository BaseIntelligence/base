# Getting started (Prism miners)

Goal: wallet ready → pack a seed zip → signed submit on joinbase → see the leaderboard,
in under 15 minutes. Deep scoring science stays in [Concepts](concepts.md) and the linked
docs, not on this page.

## Prerequisites

- A Bittensor **wallet** with a miner **hotkey** (ss58). You sign every submission with it.
- Python 3.12+ and [`uv`](https://github.com/astral-sh/uv) to pack seeds from this repo.
- `curl` (or any HTTP client).

You do **not** run a BASE master, a Prism validator, or a GPU node for day-1 submit.
Eval is challenge-owned after your zip is accepted.

## Canonical public URLs

| Surface | URL |
|---------|-----|
| Product / dashboard | https://joinbase.ai |
| Base master API | https://chain.joinbase.ai |
| Prism OpenAPI (via proxy) | https://chain.joinbase.ai/challenges/prism/openapi.json |
| Prism docs UI | https://chain.joinbase.ai/challenges/prism/docs |
| Prism leaderboard | https://chain.joinbase.ai/challenges/prism/leaderboard |
| **Submit bridge (day-1)** | `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` |

Do **not** use historical hostnames (for example `chain.platform.network`) as the shipping
master URL.

Prism currently receives **50%** absolute emission share on the BASE network (paired with
Agent Challenge at 50%). See the BASE miner [Concepts](https://github.com/BaseIntelligence/base/blob/main/docs/miner/concepts.md)
hub for emission honesty.

## 1. Confirm the network and Prism surface

```bash
# Master must answer ready with role=master
curl -fsS https://chain.joinbase.ai/health

# Prism public readiness (prefer these over /health)
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/leaderboard
```

Expect **200** on OpenAPI and leaderboard. Public challenge `/health` and `/version` often
return **403** through the proxy by design; that is not a miner bug.

Optional registry check (emission share):

```bash
curl -fsS https://chain.joinbase.ai/v1/registry
# Look for slug "prism" with status active and emission_percent 50
```

## 2. Link / prepare your hotkey

1. Open https://joinbase.ai and connect or register the wallet you will mine with.
2. Note the **hotkey** ss58. Leaderboard rows and raw weights key on this hotkey.
3. Keep the coldkey offline. Never paste mnemonics into tickets, chat, or git.

All signature headers below use this same hotkey.

## 3. Pack an example seed (two-script zip)

Clone this repo and pack a lab seed. The default Transformer explore shape is
`examples/tiny-1m` (family id `transformer-tiny-1m`). Mamba pure-torch is
`examples/mamba-tiny` (`mamba-tiny-1m`).

```bash
git clone https://github.com/BaseIntelligence/prism.git
cd prism
uv sync --extra dev   # once per machine

# Pack the tiny transformer seed into a submit-shaped zip
uv run python packages/challenges/prism/scripts/pack_seed_family.py \
  --family transformer-tiny-1m \
  --output-dir dist/seed-packages
```

You should get a zip under `dist/seed-packages/` (name includes the family id). The zip is
the **two-script** contract:

```text
architecture.py   # build_model(ctx) — model only
training.py       # train(ctx) — miner-owned loop
prism.yaml        # optional entrypoints / tokenizer
```

That is enough for day-1. Novel architectures under the AST sandbox and the dual param
ladder (124M explore / 350M promote) are expected later; start from a seed.

Details: [Two-script contract](README.md#the-two-script-contract) and
[Submission format](../../../packages/challenges/prism/docs/submissions.md).

## 4. Sign headers and submit (joinbase bridge)

Production day-1 path is the **BASE verified bridge** (raw ZIP body + miner signature
headers). BASE checks the signature and forwards to Prism.

### Required headers

| Header | Meaning |
|--------|---------|
| `X-Hotkey` | Your miner hotkey ss58 |
| `X-Nonce` | Unique string per attempt (replay protection) |
| `X-Timestamp` | Unix seconds (string); must be fresh within signature TTL |
| `X-Signature` | sr25519 signature over the **canonical Prism payload** (hex; optional `0x` prefix) |
| `Content-Type` | `application/zip` |

### Canonical payload (what you sign)

Prism defines the bytes you sign as:

```text
prism:{hotkey}:{nonce}:{timestamp}:{sha256_hex(body)}
```

where `body` is the **raw zip bytes** of the request, and `timestamp` / `nonce` are the
exact header string values. Sign those UTF-8 bytes with your hotkey keypair
(`keypair.sign(message)` → hex).

Dev-only networks may accept an HMAC dev signature when insecure signatures are enabled;
mainnet joinbase expects a real hotkey signature.

### HTTP shape

```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
Content-Type: application/zip
X-Hotkey: <your-ss58-hotkey>
X-Nonce: <unique-nonce>
X-Timestamp: <unix-seconds>
X-Signature: <hex-signature>
```

```bash
# Sketch only — plug in a real signed headers helper for your wallet stack
ZIP=dist/seed-packages/transformer-tiny-1m.zip   # adjust to pack output name
# HOTKEY / NONCE / TS / SIG must match the canonical payload above
curl -sS -X POST "https://chain.joinbase.ai/v1/challenges/prism/submissions" \
  -H "Content-Type: application/zip" \
  -H "X-Hotkey: ${HOTKEY}" \
  -H "X-Nonce: ${NONCE}" \
  -H "X-Timestamp: ${TS}" \
  -H "X-Signature: ${SIG}" \
  --data-binary @"${ZIP}"
```

Unsigned or bad-sig requests should fail closed with **401** / **403** / **422**, never hang
as **502**. If you get 502, see [Troubleshooting](troubleshooting.md).

Local direct Prism (`POST /v1/submissions` with JSON base64) is only for operators with a
local challenge; miners on the network use the joinbase bridge above.

## 5. Watch the leaderboard

```bash
curl -fsS https://chain.joinbase.ai/challenges/prism/leaderboard | head -c 800
```

Also useful after admit (paths via proxy → Prism OpenAPI for exact schemas):

```bash
# OpenAPI surface
curl -fsS https://chain.joinbase.ai/challenges/prism/openapi.json | head -c 200
```

Rewards path after score:

1. Prism ranks your hotkey (held-out primary emission; see Concepts).
2. Prism pushes **raw hotkey weights** to the BASE master.
3. Master seals with absolute emission shares (**Prism 50%** + Agent Challenge 50%).
4. Validators fetch `GET https://chain.joinbase.ai/v1/weights/latest` and submit on-chain.

`/v1/weights/latest` returning **404** means no sealed vector yet; shares still apply on the
next seal.

## Checklist

- [ ] `https://chain.joinbase.ai/health` → `role=master`, `ready=true`
- [ ] Prism OpenAPI + leaderboard return **200**
- [ ] Hotkey known, backed up offline, linked on https://joinbase.ai
- [ ] Seed packed via `packages/challenges/prism/scripts/pack_seed_family.py` (two-script zip)
- [ ] Canonical message is `prism:{hotkey}:{nonce}:{timestamp}:{sha256(body)}`
- [ ] `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signature headers
- [ ] Auth failures look like **401/403/422**, not **502**
- [ ] Your hotkey appears (or updates) on the Prism leaderboard after eval

## What not to do on day-1

- Do not chase Official Comparison / multimetric / Complete View first — those are
  **scientific grade** surfaces ([Concepts](concepts.md)).
- Do not enable or expect a **TEE verifier** on Prism product scoring (NO-TEE; provider
  trust + IMAGE_PIN). REAL-PROVIDER TEE is retired for Prism.
- Do not invent a Base LLM gateway.
- Do not run Swarm or call `set_weights` yourself.

## Cross-cut honesty

| Topic | Truth |
|-------|-------|
| Emission | BASE absolute shares: **Prism 50%** + **Agent Challenge 50%**. Master aggregates raw weights; validators `set_weights`. |
| Rank | Held-out primary / bpb secondary inside Prism. **Wall-clock never ranks.** |
| Gateway | **No Base LLM gateway** on joinbase. |
| TEE | Prism product is **NO-TEE** (provider trust + IMAGE_PIN). AC Phala/KR is a different challenge. |

## Next

- [Miner hub](README.md) — nav + two-script summary  
- [Concepts](concepts.md) — emission vs science, honesty labels  
- [Troubleshooting](troubleshooting.md) — 401 / 409 / 429 / 502  
- [Submission format](../../../packages/challenges/prism/docs/submissions.md) — full contract  
- [Scoring](../../../packages/challenges/prism/docs/scoring.md) — held-out primary, bpb secondary  
- BASE miner hub: https://github.com/BaseIntelligence/base/tree/main/docs/miner  
- Agent Challenge miner hub (sibling): https://github.com/BaseIntelligence/agent-challenge/tree/main/docs/miner  

