# AGENTS.md — Base monorepo developer guide

Product repo: `/projects/platform-network/platform` (BaseIntelligence/base).

This file is the **product** developer guide. Keep it short. Do not turn it into a
mission log, cutover diary, or Swarm novel.

## What you are building

BASE is a Bittensor subnet control plane:

- **Master** aggregates challenge weights, serves registry + sealed vectors, and
  **never** calls on-chain `set_weights`.
- **Prism** and **Agent Challenge** run **embedded** in the master container
  (ASGI on localhost via supervisor + reverse proxy). Separate challenge Compose
  services are not required.
- **Validators** are weight-only clients of `https://chain.joinbase.ai`: they
  `GET /v1/weights/latest` and submit that vector with their own wallet.

Public surfaces:

| Surface | URL |
|---------|-----|
| API host | `https://chain.joinbase.ai` |
| UI | `https://joinbase.ai` |
| API shapes | OpenAPI in code (`/openapi.json`, `/challenges/{slug}/openapi.json`) |

**OpenAPI is API truth.** Do not invent long markdown API dumps.

## Layout

```text
src/base/                         # master, proxy, validator client, CLI
packages/challenges/
  prism/                          # import: prism_challenge
  agent-challenge/                # import: agent_challenge
deploy/compose/                   # supported install (master embed)
tests/                            # unit + integration
docs/                             # keep minimal (see Docs policy)
```

Invariants that do not rename:

- GHCR image **names**
- Public paths `/challenges/prism` and `/challenges/agent-challenge`
- Python packages `prism_challenge` and `agent_challenge`

This is a **uv workspace**. Root `pyproject.toml` / `uv.lock` own the workspace;
challenge packages are members under `packages/challenges/*`.

## Day-1 commands

```bash
cd /projects/platform-network/platform
uv sync

# unit smoke (prefer scoped first)
UV_CACHE_DIR=/var/tmp/uv-cache uv run pytest -q --maxfail=5

# lint
uv run ruff check src packages/challenges
```

Challenge package import smoke (root `uv run` alone may not put challenge packages
on `PYTHONPATH`):

```bash
uv run --package prism-challenge python -c "import prism_challenge"
uv run --package agent-challenge python -c "import agent_challenge"
```

Scoped AC / sealer regression:

```bash
UV_CACHE_DIR=/var/tmp/uv-cache uv run pytest \
  packages/challenges/agent-challenge/tests \
  -k "miner_env or residual or tree_sha or review_api or key_release" -q

UV_CACHE_DIR=/var/tmp/uv-cache uv run pytest tests/unit \
  -k "sealer or aggregation or weights" -q
```

## Runtime topology (production)

Supported install is **Docker Compose master + PostgreSQL only**:

```text
Master container
  ├─ master proxy / API
  ├─ continuous weights sealer (in-process)
  ├─ Prism ASGI      :18080 (localhost)
  └─ Agent Challenge :18081 (localhost)
PostgreSQL (control-plane durability)
```

- Master embeds challenges; no separate required `challenge-*` Compose app
  containers.
- Continuous **in-process weight sealer** keeps `GET /v1/weights/latest` at 200.
  CLI weights paths are emergency/debug only.
- Validators: independent Compose projects → `chain.joinbase.ai`. They do not host
  master Postgres or challenge writer DBs.
- **Swarm is not** the supported shipping path. Compose embed is.

## Hard invariants (never violate)

1. **No** master `set_weights`. **No** product path through `burn_weights_24h.py`
   (leave that script untracked if present).
2. **No** LLM Base gateway for Agent Challenge scoring.
3. **Secrets hygiene:** names / digests / shas only in logs, docs, and evidence.
   Never paste private keys, wallet mnemonics, API tokens, or full secret values.
4. **AC miner env:** API keys / tokens only. Reject URL, proxy, and host-shaped
   env keys.
5. **AC review callback** hard-pinned to
   `https://chain.joinbase.ai/challenges/agent-challenge`.
6. **AGATE eval gate:** package LLM rules **residual** + `package_tree_sha` proof
   **before** TEE auth; otherwise no eval / attestation.
7. **Agent models:** no closed model catalog; **ban personal finetunes**.
8. **Tbench 2.1 tasks:** baked image content + digest; no miner-supplied task URL.
9. Do not wipe production Postgres / `KEY_FILE` / wallets unless a feature
   explicitly requires it.
10. Do not force-live Swarm mutate. Prefer Compose master-embed.
11. GHCR names and public challenge slugs stay stable (mineable digests matter).

## Network surfaces (must stay green)

| Path | Role |
|------|------|
| `GET /health` | master ready |
| `GET /v1/registry` | challenges + emission |
| `GET /v1/weights/latest` | sealed weight vector |
| `GET /v1/validators/public` | validator directory |
| `GET /challenges/{slug}/openapi.json` | challenge API schema |
| `GET /challenges/{slug}/leaderboard` | challenge stats |
| `POST /v1/challenges/{slug}/submissions` | signed submit bridge |

## Docs policy

Keep shipping docs **minimal**:

- short root `README.md`
- miner getting-started
- short validator / compose note
- OpenAPI + code for everything else

Do not reintroduce monorepo essays, SDK wheel SHA tables, or giant doc indexes in
README. Mission diaries and validation logs belong in local untracked evidence,
not in product docs.

## Commits and hygiene

- Author/committer identity: **echobt**
  (`154886644+echobt@users.noreply.github.com`)
- No force-push to protected tips
- Leave `scripts/burn_weights_24h.py` untracked if present
- Prefer targeted pytest + ruff on touched paths before commit
- Frontend is out of this monorepo

## When unsure

1. Code + OpenAPI beat markdown essays.
2. Compose master-embed + weight-only validators is the product topology.
3. Continuous sealer owns healthy `/v1/weights/latest`; do not add master
   on-chain weight submission.
4. AC anti-cheat locks (env keys-only, review URL pin, residual + tree_sha) are
   intentional fail-closed gates — do not "helpfully" loosen them.
