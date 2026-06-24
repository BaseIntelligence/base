# Challenges

![BASE Banner](../assets/banner.jpg)

## Model

A challenge is an independent repository and Docker image. It owns its logic, public routes, submissions, scoring data, database schema, and challenge-local files.

Challenge state is SQLite on the challenge `/data` Swarm volume. BASE injects `CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3` and mounts `/data` for the SQLite file. There is no Postgres server per challenge; the `/data` volume is the single home for the database, artifacts, analyzer output, uploaded files, and local files.

The challenge service runs on the manager node as a Swarm replicated service (placement `node.role==manager`) on an encrypted overlay network. The `/data` Swarm volume is retained by default when the service is removed.

## Required API

```text
GET /health
GET /version
GET /internal/v1/get_weights
```

The internal endpoint is authenticated with a per-challenge shared token mounted by the master.

## Create a challenge

```bash
uv run base challenge create code-arena --out ../code-arena
cd ../code-arena
uv run --extra dev pytest
```

## Public routes

Public routes are exposed through:

```text
/challenges/{slug}/...
```

The master blocks `/internal/*`, `/health`, `/version`, Agent Challenge internal launch paths such as `POST /internal/v1/submissions/{submission_id}/launch`, and generic benchmark execution-shaped routes such as `/benchmark-executions` from the public proxy.

## Proxy failure behavior

BASE proxy should preserve challenge-origin non-2xx responses when the challenge answered with a safe response. Transport failures, unreachable services, Swarm service DNS failures, and connection timeouts become safe 502 responses at BASE. Frontends should render unavailable copy and retry with backoff instead of showing raw text such as `BASE request failed with status 502`.

Operator checklist for challenge 502s:

1. Confirm ingress includes `/challenges` and routes it to BASE proxy.
2. Confirm the slug maps to a running challenge service.
3. Confirm challenge service health, the Swarm service name, service DNS on the overlay network, the service port, and that the service has at least one running task.
4. Check whether the response came from proxy transport handling or from the challenge origin. Only transport failures should be rewritten to 502.

Agent Challenge env and public launch routes are public proxy routes, but BASE does not store their request bodies or per-submission env values. The allowed BASE paths are `GET/PUT /challenges/agent-challenge/submissions/{id}/env`, `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, and `POST /challenges/agent-challenge/submissions/{id}/launch`. The challenge-local paths are `GET/PUT /submissions/{id}/env`, `POST /submissions/{id}/env/confirm-empty`, and `POST /submissions/{id}/launch`. Only signed miner headers `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp` are preserved for those routes. `POST /internal/v1/submissions/{submission_id}/launch` is a bridge/internal API only, not a public miner API, and the BASE proxy must not expose generic benchmark execution routes. The generic BASE broker remains the execution substrate for controlled BASE SDK jobs behind the challenge boundary.
