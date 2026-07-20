# agent-challenge (workspace member)

Import packages: **`agent_challenge`**, **`agent_challenge_runner`**

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
