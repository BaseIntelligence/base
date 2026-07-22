# agent-challenge (workspace member)

Import packages: **`agent_challenge`**, **`agent_challenge_runner`**

Product sources live at `packages/challenges/agent-challenge` in this uv workspace
under **BaseIntelligence/base**. Shared contracts come from workspace **`base`**
(`base.challenge_sdk`).

## Shared SDK

```python
from base.challenge_sdk.executors.docker import DockerExecutor
# challenge-local re-exports may wrap SDK types under agent_challenge.sdk.*
```

```bash
# from monorepo root
uv sync --package agent-challenge
uv run --package agent-challenge python -c "import agent_challenge; import base.challenge_sdk"
```

## Do not rename

- GHCR: `ghcr.io/baseintelligence/agent-challenge`,
  `ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner`
- Public slug: `/challenges/agent-challenge`
- Python import: `agent_challenge`

## Miners / API

Day-1: [docs/miner/getting-started.md](../../../docs/miner/getting-started.md)

**API truth is OpenAPI**, not markdown dumps:

- Live: `https://chain.joinbase.ai/challenges/agent-challenge/openapi.json`
- Interactive docs: `https://chain.joinbase.ai/challenges/agent-challenge/docs`
- In-process: challenge app `/openapi.json` and `/docs`

Audience guides remain under `docs/miner/` and `docs/validator/` in this package
for operator detail; shipping day-1 lives in the repo-root miner guide.
