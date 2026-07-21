# Challenges

![BASE Banner](../assets/banner.jpg)

## Model

A challenge is a package (first-party: monorepo `packages/challenges/*`) and
historically a standalone Docker image for emergency dual-run. It owns its logic,
public routes, submissions, scoring data, database schema, and challenge-local
files.

**Shipping master-embed topology:** Prism and agent-challenge run as **localhost
uvicorn** processes inside the `base-master` container (supervisor entrypoint),
not as separate Compose `challenge-*` services. Data lives under the master
volume:

- `/var/lib/base/challenges/prism`
- `/var/lib/base/challenges/agent-challenge`

Registry `internal_base_url` is loopback (`http://127.0.0.1:18080` /
`http://127.0.0.1:18081`). The public proxy still rewrites and forwards
`/challenges/{slug}/...` via httpx — **public path prefixes are unchanged**.

Emergency dual-run may still use a dedicated challenge container with a named
`/data` volume and
`CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3`; that path
is operator-only, not the default install. Challenges never receive control-plane
Postgres credentials. Master is the sole control-plane writer; challenge SQLite
is not multi-writer across containers.

There is **no LLM gateway**. Scoring and admission are challenge-owned. Base and
Prism do not create short-lived evaluator containers for challenge evaluation;
external TEE (agent-challenge Phala when enabled) or miner-funded workers are
outside the master Compose service lifecycle.

## Required API surface

```text
GET /health
GET /version
```

Raw weight publication in the target path is an authenticated **push** from the
challenge to the master (not a master-polled scrap of standalone weights alone).
Challenges also expose challenge-local public routes the proxy rewrites under
`/challenges/{slug}/...`.

## Create a challenge

```bash
uv run base challenge create code-arena --out ../code-arena
cd ../code-arena
uv run --extra dev pytest
```

## Public routes

Public routes are exposed through `/challenges/{slug}/...`. The master blocks
`/internal/*`, `/health`, `/version`, Agent Challenge internal launch paths such
as `POST /internal/v1/submissions/{submission_id}/launch`, and generic benchmark
execution-shaped routes such as `/benchmark-executions` from the public proxy.

## Lifecycle on Compose

1. Register the challenge image and metadata with the master registry (immutable
   digest pin for production).
2. Activate the challenge. The master reconcile / adoption path installs a
   long-lived Compose service for ACTIVE challenges only.
3. The master-resident **digest-aware watcher** keeps that service aligned with
   the approved pin (controlled pull, targeted recreate, health/version verify,
   rollback and bounded backoff on failure). See [compose.md](compose.md).
4. Deactivate stops and removes the managed container while keeping the named
   volume for reactivation.
5. Inactive, draft, and disabled challenges never start.

## Proxy failure behavior

The BASE proxy preserves challenge-origin non-2xx responses when the challenge
answered safely. Transport failures, unreachable services, Compose service DNS
failures on the private network, and connection timeouts become safe 502
responses. Frontends should render unavailable copy and retry with backoff
instead of showing raw text such as `BASE request failed with status 502`.

Operator checklist for challenge 502s:

1. Confirm ingress includes `/challenges` and routes it to the BASE master proxy.
2. Confirm the slug maps to a running long-lived challenge service in the master
   Compose project.
3. Confirm challenge health (`/health`), Compose service name, connectivity on
   the private `app` network, and the challenge listen port.
4. Determine whether the response came from proxy transport handling or the
   challenge origin. Only transport failures should be rewritten to 502.

Agent Challenge env and public launch routes are public proxy routes, but BASE
stores neither their request bodies nor per-submission env values. Allowed BASE
paths are `GET/PUT /challenges/agent-challenge/submissions/{id}/env`,
`POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, and
`POST /challenges/agent-challenge/submissions/{id}/launch`; the challenge-local
paths are `GET/PUT /submissions/{id}/env`,
`POST /submissions/{id}/env/confirm-empty`, and `POST /submissions/{id}/launch`.
Only the signed miner headers `X-Hotkey`, `X-Signature`, `X-Nonce`, and
`X-Timestamp` are preserved for those routes.
`POST /internal/v1/submissions/{submission_id}/launch` is a bridge/internal API
only, not a public miner API, and the proxy must not expose generic benchmark
execution routes.

## Weights

Challenges export raw **hotkey** weights. The master aggregates and serves the
final vector; validators fetch and call `set_weights`. Challenges never submit
final UID vectors and never receive master database credentials.

## Agent Challenge Phala attestation notes

Private control-plane work (work units, fold, weights, bridge launch) stays on challenge-direct or master-internal channels, never on the public edge. Full attested mode evaluation is one miner-funded external eval (R=1) with **no** BASE validator multi-replica re-exec assignment for those units; cross-repo review→eval behavior is available after PR merge. See [Architecture: Agent Challenge Phala path](architecture.md#agent-challenge-phala-intel-tdx-path).
5. If attested routes are enabled, confirm the client hit an allowlisted review/eval/status path (not a private result or capability route).
