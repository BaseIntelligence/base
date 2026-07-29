# Agent Challenge package docs

Shipping docs for this challenge stay **minimal**. Day-1 miners should not start
from Phala self-deploy essays.

| Need | Where |
|------|--------|
| **Day-1 + current prod scoring** | Host-trust unattested after joinbase signed ZIP. Canonical pin: [`host-trust.md`](host-trust.md). Operator depth: [`no-phala-mode.md`](no-phala-mode.md). |
| Miner walkthrough (primary) | Agent-challenge repo [`docs/miner/getting-started.md`](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/getting-started.md) and [`docs/miner/submit-agent.md`](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/submit-agent.md) |
| Miner hub (reference) | [`docs/miner/README.md`](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/README.md) |
| API shape | OpenAPI: `https://chain.joinbase.ai/challenges/agent-challenge/openapi.json` |
| Interactive API | `https://chain.joinbase.ai/challenges/agent-challenge/docs` |
| Product UI | https://joinbase.ai |
| Package product pin | [`../README.md`](../README.md) |
| Self-deploy CLI accuracy fixtures (legacy / not current prod scoring) | [`miner/self-deploy.md`](miner/self-deploy.md), [`validator/self-deploy.md`](validator/self-deploy.md) |

**API truth is OpenAPI** (and the in-process challenge app `/openapi.json`).

**Honesty:** production scores are host-trust only (`attested: false`). Never claim
TEE-grade verification for the current path. UI STATUS is lifecycle; honesty may
show **Unattested · Host trust**.
