# Challenges (miner + ops index)

Shipping entry for BASE challenges. Full write-up: [../challenges.md](../challenges.md).

## Network

- Website: https://joinbase.ai  
- Master API: **https://chain.joinbase.ai**  
- Validators: weight-only → `GET /v1/weights/latest` → own-wallet `set_weights`

## Active challenges

| Slug | Emission | Monorepo package | Miner day-1 |
|------|---------:|------------------|-------------|
| `prism` | 50% | `packages/challenges/prism` | [../miner/prism/getting-started.md](../miner/prism/getting-started.md) |
| `agent-challenge` | 50% | `packages/challenges/agent-challenge` | [../miner/agent-challenge/getting-started.md](../miner/agent-challenge/getting-started.md) |

Public prefixes: `/challenges/prism`, `/challenges/agent-challenge` (unchanged).

## Topology (do not over-build)

- **Required:** master-embed Compose (`base-master` + `master-postgres`). Challenges are localhost ASGI inside master.
- **Not required:** separate multi-repo clones, separate `challenge-*` Compose app containers, Swarm challenge services.
- Emergency dual-run is operator-only, not miner day-1.

## Miner quick checks

```bash
curl -fsS https://chain.joinbase.ai/health
curl -fsS https://chain.joinbase.ai/v1/registry
curl -fsS -o /dev/null -w '%{http_code}\n' https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' https://chain.joinbase.ai/challenges/agent-challenge/openapi.json
```

Start mining from the [miner hub](../miner/README.md).
