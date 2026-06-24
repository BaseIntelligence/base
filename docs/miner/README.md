# Miner Guide

## Purpose

BASE is the routing and coordination layer for multiple challenge subnets. As a miner, you do
not build against BASE-specific scoring logic. You choose a challenge, follow that challenge's
submission contract, and use BASE to reach the challenge's public surface.

## Miner Flow

1. Choose the challenge you want to compete in.
2. Read the challenge repository and miner guide.
3. Build the required submission artifact for that challenge.
4. Submit through the challenge's public route as exposed by BASE.
5. Track challenge-specific status, reports, and leaderboards.
6. Improve your submission based on challenge feedback.
7. Earn rewards when the challenge exports a raw weight for your hotkey and BASE normalizes it
   into final subnet weights.

## How BASE Routes Miner Traffic

Each challenge has a slug, such as `agent-challenge`, `data-fabrication`, `bounty-challenge`, or
`prism`. BASE uses that slug to proxy public challenge requests to the correct isolated
challenge service.

Challenge-specific examples:

```http
POST /challenges/{challenge_slug}/...
GET /challenges/{challenge_slug}/...
```

The exact path after the challenge slug belongs to the challenge repository. BASE does not define
the artifact format, task rules, scoring rubric, or leaderboard fields for each challenge.


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

Use `POST /v1/challenges/agent-challenge/submissions` for raw ZIP bridge uploads. BASE verifies the miner upload and forwards it to Agent Challenge. Use `POST /challenges/agent-challenge/submissions` for the JSON base64 generic proxy path when the client signs the challenge-local `/submissions` request.

For v1 lists, `/challenges/agent-challenge/submissions` returns the latest 100 submissions newest-first. `/challenges/agent-challenge/leaderboard` returns one best scoring row per hotkey. Pagination, filtering, and client-selected sorting are deferred to future v2.

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

1. Confirm ingress routes `/challenges` to BASE proxy. A route for `/v1/challenges` alone is not enough.
2. Confirm BASE proxy slug routing points at the Agent Challenge service and still blocks `/internal/*`, `/health`, and `/version`.
3. Confirm Agent Challenge health, that the Swarm service has a running task, service DNS on the overlay network, and the service port.
4. Confirm the challenge service is running on the manager node and is not stuck pending placement.
5. Separate proxy transport failures from challenge-origin non-2xx responses. Transport failures become safe 502 responses. Challenge-origin non-2xx responses should pass through as safe validation, auth, replay, rate-limit, or challenge error responses.
6. If the failing request is an env action, confirm the request uses `GET/PUT /challenges/agent-challenge/submissions/{id}/env`, `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, or `POST /challenges/agent-challenge/submissions/{id}/launch`, and includes only the signed miner header names above.

## What BASE Does For Miners

BASE provides:

- one public entry point for multiple challenges;
- challenge routing by slug;
- central challenge discovery;
- final normalization across challenge emissions;
- Bittensor hotkey-to-UID mapping;
- final on-chain weight submission.

## What Challenge Repositories Define

Each challenge defines:

- accepted submission format;
- authentication and signature rules;
- task or project requirements;
- scoring algorithm;
- evaluation limits;
- leaderboard output;
- public status and result endpoints.

## Rewards

Challenge scores are not submitted directly to Bittensor. The flow is:

1. The challenge evaluates miner work.
2. The challenge exports raw hotkey weights.
3. BASE applies the challenge emission share.
4. BASE normalizes across active challenge outputs.
5. BASE maps hotkeys to Bittensor UIDs.
6. Validators submit final weights on-chain.

This means a strong score in one challenge contributes according to that challenge's configured
emission share.

## Miner Checklist

Before submitting:

- Confirm the challenge slug and repository.
- Read the challenge miner guide.
- Use the challenge's required artifact format.
- Sign requests if the challenge requires hotkey signatures.
- Monitor the challenge leaderboard, not only the BASE layer.
- Keep your hotkey consistent across submissions.
- Do not assume two challenges share the same scoring rules.

## Where To Find Challenge Rules

Use the specific challenge repository for detailed mining instructions:

- Agent Challenge: software engineering agents and benchmark tasks.
- Data Fabrication: agentic coding conversation dataset generation.
- Bounty Challenge: owner-created project bounties.
- PRISM: neural architecture search and training variants.
