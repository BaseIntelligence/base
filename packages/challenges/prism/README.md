# prism-challenge (workspace member)

Import package: **`prism_challenge`**

Product sources live at `packages/challenges/prism` in this uv workspace under
**BaseIntelligence/base**. Shared contracts come from workspace **`base`**
(`base.challenge_sdk`).

## Shared SDK

```python
from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.roles import Capability, Role, role_contract
```

```bash
# from monorepo root
uv sync --package prism-challenge
uv run --package prism-challenge python -c "import prism_challenge; import base.challenge_sdk"
```

## Do not rename

- GHCR: `ghcr.io/baseintelligence/prism`, `ghcr.io/baseintelligence/prism-evaluator`
- Public slug: `/challenges/prism`
- Python import: `prism_challenge`

## Miners / API

Day-1: [docs/miner/getting-started.md](../../../docs/miner/getting-started.md)

**API truth is OpenAPI**, not markdown dumps:

- Live: `https://chain.joinbase.ai/challenges/prism/openapi.json`
- Interactive docs: `https://chain.joinbase.ai/challenges/prism/docs`
- In-process: challenge app `/openapi.json` and `/docs`
