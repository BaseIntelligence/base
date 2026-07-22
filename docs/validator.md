# Validators (weight-only)

BASE validators are **weight-only** clients of the public master. They do **not** host
master, control-plane Postgres, or challenge writers. They **never write** challenge
`submissions` or `leaderboard` state; those stay sole-writer on the master-embedded
challenge ASGI processes (`/challenges/prism`, `/challenges/agent-challenge`).

## Public master

| Item | Value |
|------|-------|
| Master API | `https://chain.joinbase.ai` |
| Weights | `GET /v1/weights/latest` |
| Registry | `GET /v1/registry` |
| Health | `GET /health` → `role=master` |
| Validators directory | `GET /v1/validators/public` |
| Challenge board SVG | `GET /v1/challenges/dashboard.svg` |
| Challenge leaderboards | `GET /challenges/{slug}/leaderboard` |

`master_url` is the validator **coordination** root (Base master). Public
`registry_url` / weights default to the same joinbase host unless overridden.

```bash
curl -fsS https://chain.joinbase.ai/health
curl -fsS https://chain.joinbase.ai/v1/weights/latest
curl -fsS https://chain.joinbase.ai/v1/registry
curl -fsS https://chain.joinbase.ai/v1/validators/public
```

Weights URL example: `https://chain.joinbase.ai/v1/weights/latest`.

### Network stats (composed, not a single `/v1/stats`)

There is **no** dedicated `/v1/stats` or `/v1/network` aggregate route. Production
network visibility is **composed** from the live surfaces above plus per-challenge
OpenAPI/leaderboard:

| Need | Compose from |
|------|----------------|
| Master readiness | `GET /health` |
| Active challenges + emission shares | `GET /v1/registry` |
| Sealed UID vector | `GET /v1/weights/latest` |
| Validator directory | `GET /v1/validators/public` |
| Challenge ranks / scores | `GET /challenges/{slug}/leaderboard` |
| Challenge schemas | `GET /challenges/{slug}/openapi.json` |
| Compact board graphic | `GET /v1/challenges/dashboard.svg` |

`/v1/challenges/dashboard.svg` is the only dedicated network board endpoint; it
summarizes registry challenges (status + emission). Everything else is the
canonical JSON APIs (OpenAPI truth). Silent 404 “stats” paths are intentionally
absent, not advertised.

## Install (Compose)

Independent validator Compose project with your own wallet:

```bash
./deploy/compose/install-validator.sh \
  --project-name base-validator \
  --master-url https://chain.joinbase.ai
```

Artifact: `deploy/compose/docker-compose.validator.yml`.

Defaults (see `config/validator.example.yaml` and the installer):

- `challenge_execution_enabled: false` (weight-only; adapters off)
- `registry_url` / weights root: `https://chain.joinbase.ai`
- validator calls **`set_weights`** under **its own** hotkey
- master **never** `set_weights`

Optional **audit** re-exec (when explicitly enabled) is **non-write**: it must not
become a second challenge writer for submissions/leaderboard.

Local disposable master smoke only (secondary):

```bash
./deploy/compose/install-validator.sh \
  --project-name base-validator-local \
  --master-url http://127.0.0.1:3180
```

## Runtime notes

- Mount host `docker.sock` as the installer does; image runs as uid **1000**
- Writable state under `/var/lib/base/state` (and `.bittensor` home) even when the rootfs
  is `read_only`
- Docker Compose only (no Kubernetes path)

## Secrets

Use file-backed secrets (`*_FILE`, mode `0600`). Never paste private keys, mnemonics, or
tokens into docs, tickets, or chat.

## API truth

Route shapes come from live OpenAPI, not markdown dumps:

- `https://chain.joinbase.ai/openapi.json`
- `https://chain.joinbase.ai/challenges/{slug}/openapi.json`

See also: [compose.md](compose.md), [miner getting started](miner/getting-started.md).
