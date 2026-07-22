# Agent Challenge package docs

Shipping docs for this challenge are **not** maintained as a large essay tree
inside the package.

| Need | Where |
|------|--------|
| Day-1 miner path | Repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md) |
| API shape | OpenAPI: `https://chain.joinbase.ai/challenges/agent-challenge/openapi.json` |
| Interactive API | `https://chain.joinbase.ai/challenges/agent-challenge/docs` |
| Package product pin | [`../README.md`](../README.md) |
| Self-deploy CLI accuracy fixtures | [`miner/self-deploy.md`](miner/self-deploy.md), [`validator/self-deploy.md`](validator/self-deploy.md) |

**API truth is OpenAPI** (and the in-process challenge app `/openapi.json`).
Audience essays (lifecycle dumps, route catalogs, architecture novels) were
collapsed; keep only the short pointers above plus CLI accuracy fixtures required
by tests.
