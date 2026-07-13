# Miner Guide

## Purpose

BASE is the routing and coordination layer for multiple challenge subnets. As a miner you do not
build against BASE-specific scoring logic: you choose a challenge, follow that challenge's
submission contract, and use BASE to reach its public surface.

## Miner Flow

1. Choose a challenge and read its repository and miner guide.
2. Build the required submission artifact and submit through the challenge's public BASE route.
3. Track challenge-specific status, reports, and leaderboards, and improve on feedback.
4. Earn rewards when the challenge exports a raw weight for your hotkey and BASE normalizes it
   into final subnet weights.

## How BASE Routes Miner Traffic

Each challenge has a slug (such as `agent-challenge`, `data-fabrication`, `bounty-challenge`, or
`prism`); BASE proxies public requests by slug to the correct isolated challenge service:

```http
POST /challenges/{challenge_slug}/...
GET /challenges/{challenge_slug}/...
```

The exact path after the slug belongs to the challenge repository. BASE does not define the
artifact format, task rules, scoring rubric, or leaderboard fields.


## Agent Challenge Frontend API

Frontend reads for Agent Challenge should use the BASE master/proxy base:

```http
GET /v1/registry
GET /challenges/agent-challenge/benchmarks
GET /challenges/agent-challenge/submissions/{id}/status
GET /challenges/agent-challenge/submissions/{id}/events
GET /challenges/agent-challenge/submissions/{id}/env
PUT /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
GET /challenges/agent-challenge/leaderboard
```

Uploads have two public paths:

```http
POST /v1/challenges/agent-challenge/submissions
POST /challenges/agent-challenge/submissions
```

Use the `/v1/...` path for raw ZIP bridge uploads (BASE verifies and forwards them to Agent
Challenge); use the `/challenges/...` path for the JSON base64 generic proxy when the client signs
the challenge-local `/submissions` request.

For v1 lists, `/challenges/agent-challenge/submissions` returns the latest 100 submissions
newest-first and `/challenges/agent-challenge/leaderboard` returns one best-scoring row per hotkey.
Pagination, filtering, and client-selected sorting are deferred to v2.

The public proxy blocks `/internal/*`, `/health`, and `/version`.

## Agent Challenge Miner Env Actions

After Agent Challenge analysis allows an artifact, a master validator pauses it at public state `Waiting for miner action`. The exact challenge lifecycle is `analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`. The miner must either save env vars or confirm that no env vars are needed before launch.

BASE public paths, including the exact shorthand `GET/PUT /challenges/agent-challenge/submissions/{id}/env`:

```http
GET /challenges/agent-challenge/submissions/{id}/env
PUT /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
```

Agent Challenge local paths behind the proxy, including the exact shorthand `GET/PUT /submissions/{id}/env`:

```http
GET /submissions/{id}/env
PUT /submissions/{id}/env
POST /submissions/{id}/env/confirm-empty
POST /submissions/{id}/launch
```

These requests are signed by the miner. Docs and examples must use fake placeholders only:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <nonce>
X-Timestamp: <timestamp>
```

Env keys must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`. Each request can contain at most 64 keys, each value is limited to 16 KiB, and the total payload is limited to 128 KiB. `PUT /env` replaces the full env set. `POST /env/confirm-empty` is required when the agent needs zero env vars, so it does not stay stuck waiting for miner action. `POST /launch` locks env metadata and starts Terminal-Bench queueing.

Env values are write-only. Responses expose metadata only: keys, count, empty confirmation, lock state, and timestamps. Values are scoped to the master validator, encrypted at rest in Agent Challenge storage, injected into the Harbor/Terminal-Bench runtime, and cannot be retrieved after submission. BASE registry and BASE proxy do not store per-submission env values. BASE only forwards the allowed signed miner headers for these env and launch routes and keeps other sensitive caller headers stripped.

## Agent Challenge 502 Troubleshooting

A 502 under `/challenges/agent-challenge/...` is a safe unavailable state from BASE proxy transport handling. Frontends should render safe unavailable copy, not raw text such as `BASE request failed with status 502`.

Checklist:

1. Confirm ingress routes `/challenges` to the BASE proxy (a `/v1/challenges` route alone is not enough), and that slug routing points at the Agent Challenge service while still blocking `/internal/*`, `/health`, and `/version`.
2. Confirm Agent Challenge health: a running long-lived challenge container in the master Compose project (not stuck restarting), service reachability on the private app network, and the challenge listen port.
3. Separate proxy transport failures (rewritten to safe 502s) from challenge-origin non-2xx responses (validation, auth, replay, rate-limit, or challenge errors), which pass through.
4. For an env action, confirm it uses one of the `.../env`, `.../env/confirm-empty`, or `.../launch` paths above and includes only the signed miner header names.

## Division Of Responsibility

**BASE provides:** one public entry point for multiple challenges, routing by slug, central
challenge discovery, final normalization across challenge emissions, Bittensor hotkey-to-UID
mapping, and final on-chain weight submission.

**Each challenge defines:** accepted submission format, authentication and signature rules, task or
project requirements, scoring algorithm, evaluation limits, leaderboard output, and public
status/result endpoints.

## Rewards

Challenge scores are not submitted directly to Bittensor. The challenge evaluates miner work and
exports raw hotkey weights; BASE applies the challenge emission share, normalizes across active
challenge outputs, maps hotkeys to Bittensor UIDs, and validators submit final weights on-chain.
So a strong score in one challenge contributes according to that challenge's configured emission
share.

## Miner Checklist

Before submitting: confirm the challenge slug and repository, read its miner guide, use the
required artifact format, sign requests when the challenge requires hotkey signatures, monitor the
challenge leaderboard (not only the BASE layer), keep your hotkey consistent, and never assume two
challenges share scoring rules. For detailed rules, use the challenge repository directly:

- **Agent Challenge**: software engineering agents and benchmark tasks.
- **Data Fabrication**: agentic coding conversation dataset generation.
- **Bounty Challenge**: owner-created project bounties.
- **PRISM**: neural architecture search and training variants.
