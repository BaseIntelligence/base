<div align="center">

# BASE

**Multi-challenge Bittensor subnet control plane.**

[![CI](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml/badge.svg)](https://github.com/BaseIntelligence/base/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/BaseIntelligence/base)](https://github.com/BaseIntelligence/base/blob/main/LICENSE)
[![Bittensor](https://img.shields.io/badge/Bittensor-subnet-black.svg)](https://bittensor.com/)

![BASE Banner](assets/banner.jpg)

</div>

## What it is

BASE coordinates independent challenges (**Prism**, **Agent Challenge**), aggregates raw
hotkey weights, seals a final vector, and serves it to validators. The master **never**
calls on-chain `set_weights`. Challenges run **embedded** in the master container
(localhost ASGI). Public API: **https://chain.joinbase.ai** · UI: **https://joinbase.ai**

## Miners

Day-1: [docs/miner/getting-started.md](docs/miner/getting-started.md)

Probe live surfaces and follow challenge OpenAPI for submit shapes:

```bash
curl -fsS https://chain.joinbase.ai/health
curl -fsS https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS https://chain.joinbase.ai/challenges/agent-challenge/openapi.json
```

## Validators

Weight-only: `GET https://chain.joinbase.ai/v1/weights/latest`, then `set_weights` with
your own wallet. Install: [docs/validator.md](docs/validator.md).

```bash
./deploy/compose/install-validator.sh \
  --project-name base-validator \
  --master-url https://chain.joinbase.ai
```

## Deploy

Docker Compose is the only supported shipping operator path (`deploy/compose/`).

```bash
./deploy/compose/install-master.sh --project-name base-master --port 8081
```

Details: [docs/compose.md](docs/compose.md). Swarm is not a supported install path.

## API truth

OpenAPI in code, not markdown dumps:

- Master: `https://chain.joinbase.ai/openapi.json`
- Prism: `https://chain.joinbase.ai/challenges/prism/openapi.json`
- Agent Challenge: `https://chain.joinbase.ai/challenges/agent-challenge/openapi.json`

## License

Apache-2.0
