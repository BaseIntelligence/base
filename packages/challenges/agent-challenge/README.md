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

Package docs tree: [docs/README.md](docs/README.md) (short pointer only).
Self-deploy CLI accuracy fixtures remain under `docs/miner/self-deploy.md` and
`docs/validator/self-deploy.md` for command/route pin tests.

## Agent-driven eval gate (product pin)

Scored path order: package + measured LLM rules residual → `package_tree_sha`
proof → fresh TEE authorization → **ONLY THEN** eval prepare / KR / score.
Host-static analyzer alone is not enough (**no eval prepare** without residual +
tree SHA). **no closed catalog** of models; **personal finetunes** banned.
