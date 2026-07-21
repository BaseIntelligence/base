# Challenges

![BASE Banner](../assets/banner.jpg)

Simple miner + operator entry for shipping BASE.

## What ships today

| Challenge | Slug | Emission (default) | What miners build |
|-----------|------|-------------------:|-------------------|
| **Prism** | `prism` | **50%** absolute | Neural architecture + training packages |
| **Agent Challenge** | `agent-challenge` | **50%** absolute | Software-engineering agents (Terminal-Bench) |

Product sources live in this monorepo:

| Package | Path |
|---------|------|
| Prism | `packages/challenges/prism` |
| Agent Challenge | `packages/challenges/agent-challenge` |

Public path prefixes are **unchanged**:

- `/challenges/prism/...`
- `/challenges/agent-challenge/...`
- Bridges: `/v1/challenges/{slug}/...` where documented

GHCR image **names** for emergency dual-run / historical pins are also unchanged
(`ghcr.io/baseintelligence/prism`, `ghcr.io/baseintelligence/agent-challenge`, …).
Standalone challenge remotes are transition-only; see
[SOURCE_OF_TRUTH.md](SOURCE_OF_TRUTH.md) and [monorepo.md](monorepo.md).

## Public network (miners)

| Surface | URL |
|---------|-----|
| Website / dashboard | https://joinbase.ai |
| Master API | **https://chain.joinbase.ai** |
| Health | `GET https://chain.joinbase.ai/health` → `role=master`, `ready=true` |
| Registry | `GET https://chain.joinbase.ai/v1/registry` |
| Prism OpenAPI / docs / leaderboard | `/challenges/prism/openapi.json`, `/docs`, `/leaderboard` |
| AC OpenAPI / docs / leaderboard | `/challenges/agent-challenge/openapi.json`, `/docs`, `/leaderboard` |
| Sealed weights (validators) | `GET https://chain.joinbase.ai/v1/weights/latest` |

Private challenge `/health` and `/version` often return **403** through the public
proxy by design. Prefer openapi / docs / leaderboard for miner readiness.

Day-1 miner path (under 15 minutes): [miner hub](miner/README.md) →
[getting started](miner/getting-started.md) → pick
[Prism](miner/prism/getting-started.md) or
[Agent Challenge](miner/agent-challenge/getting-started.md).

## Master-embed topology (ops default)

Shipping Compose is **master + PostgreSQL only**. Prism and Agent Challenge run as
**localhost uvicorn** processes inside the `base-master` container (supervisor
entrypoint), not as separate required `challenge-*` Compose services.

| Process (inside master) | Bind |
|-------------------------|------|
| `base master proxy` | `0.0.0.0:8081` (host-published) |
| Prism ASGI | `127.0.0.1:18080` |
| Agent-challenge ASGI | `127.0.0.1:18081` |

Data under the master volume (sole writer for challenge SQLite):

- `/var/lib/base/challenges/prism`
- `/var/lib/base/challenges/agent-challenge`

Registry `internal_base_url` is loopback only. The public proxy still rewrites and
forwards `/challenges/{slug}/...` via httpx — public prefixes stay the same.

There is **no LLM gateway** on the Compose master. Scoring and admission are
challenge-owned. Base does not launch short-lived evaluator containers for these
challenges as part of the shipping master lifecycle.

Emergency dual-run (proxy-only master + external `challenge-*` container) is
**operator-only** (`BASE_MASTER_EMBED_CHALLENGES=0` + registry URL override). It is
**not** the required production path.

## Validators are weight-only

Independent validator Compose projects:

1. Point at **`https://chain.joinbase.ai`** (`--master-url`).
2. Fetch **`GET /v1/weights/latest`**.
3. Call **`set_weights`** with their own wallet when gated on.

They do **not** host master, control-plane Postgres, or challenge writers. Master
**never** constructs or invokes `set_weights`. Details:
[validator guide](validator/README.md), [compose.md](compose.md).

## Miner submit surfaces (chain.joinbase.ai)

### Prism

```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
X-Hotkey: <ss58>
X-Nonce: <unique>
X-Timestamp: <unix-seconds>
X-Signature: <sr25519 over challenge-canonical payload>
Content-Type: application/zip
```

Pack/sign details: [Prism miner hub](miner/prism/README.md).

### Agent Challenge

Day-1 options: https://joinbase.ai dashboard, or signed upload:

```http
POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions
POST https://chain.joinbase.ai/challenges/agent-challenge/submissions
```

Guide: [Agent Challenge miner hub](miner/agent-challenge/README.md).
Phala self-deploy / attestation is **advanced**, not day-1.

## Required challenge API surface

```text
GET /health
GET /version
```

Raw weight publication is an authenticated **push** from the challenge to the
master. Challenges also expose public routes the proxy rewrites under
`/challenges/{slug}/...`. Integration contract:
[challenge-integration.md](challenge-integration.md).

## Weights

Challenges export raw **hotkey** weights. Master aggregates, applies absolute
emission shares, and serves the final vector. Validators fetch and call
`set_weights`. Challenges never submit final UID vectors and never receive master
database credentials.

## Create a new challenge package (developers)

```bash
uv run base challenge create code-arena --out packages/challenges/code-arena
cd packages/challenges/code-arena
uv run --extra dev pytest
```

New packages still own scoring and state; shipping multi-challenge topology stays
master-embed unless an operator deliberately enables dual-run.

## Proxy notes (ops)

The BASE proxy preserves challenge-origin non-2xx when the challenge answered
safely. Transport failures become safe 502 responses.

Checklist for challenge 502s:

1. Confirm ingress routes `/challenges` to the BASE master proxy.
2. Confirm the slug is ACTIVE and loopback ASGI is up inside master (`18080` /
   `18081`) — or, for emergency dual-run only, that an external challenge service
   matches `internal_base_url`.
3. Prefer public openapi/docs/leaderboard over private `/health` through the edge.
4. Distinguish proxy transport 502 from origin error bodies.

Agent Challenge private control-plane work (work units, fold, weights, internal
launch) stays challenge-direct or master-internal, never on the public edge.
Attestation path notes:
[Architecture: Agent Challenge Phala path](architecture.md#agent-challenge-phala-intel-tdx-path).

## See also

| Audience | Doc |
|----------|-----|
| Miners | [miner/README.md](miner/README.md) |
| Validators | [validator/README.md](validator/README.md) |
| Operators | [compose.md](compose.md), [deploy.md](deploy.md) |
| Layout | [monorepo.md](monorepo.md), [architecture.md](architecture.md) |
