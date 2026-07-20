# prism-challenge (workspace member)

Import package: **`prism_challenge`**

> **Source of truth:** this tree inside **BaseIntelligence/base**. The historical
> standalone remote `BaseIntelligence/prism` is a transition dual-source / archive
> surface — prefer monorepo paths for product edits, images, and miner docs
> ([docs/SOURCE_OF_TRUTH.md](../../../docs/SOURCE_OF_TRUTH.md),
> [docs/miner/prism/](../../../docs/miner/prism/README.md)).

Product sources live in this uv workspace member under the Base monorepo
(`BaseIntelligence/base`). Shared contracts come from workspace **`base`**
(`base.challenge_sdk`), not the standalone release wheel.

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

See [`docs/monorepo.md`](../../../docs/monorepo.md).
