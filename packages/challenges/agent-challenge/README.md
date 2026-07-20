# agent-challenge (workspace member)

Import packages: **`agent_challenge`**, **`agent_challenge_runner`**

> **Source of truth:** this tree inside **BaseIntelligence/base**. The historical
> standalone remote `BaseIntelligence/agent-challenge` is a transition dual-source
> / archive surface — prefer monorepo paths for product edits, images, and miner
> docs ([docs/SOURCE_OF_TRUTH.md](../../../docs/SOURCE_OF_TRUTH.md),
> [docs/miner/agent-challenge/](../../../docs/miner/agent-challenge/README.md)).

Product sources live in this uv workspace member under the Base monorepo
(`BaseIntelligence/base`). Shared contracts come from workspace **`base`**
(`base.challenge_sdk`), not floating `git+base` HEAD.

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

See [`docs/monorepo.md`](../../../docs/monorepo.md).
