# ADR: Base monorepo layout (Prism + agent-challenge)

**Status:** Accepted (import in progress)  
**Date:** 2026-07-20  
**Decision owners:** Base monorepo residual (`mono-skeleton` → `mono-import-challenges` → later milestones)

## Context

Prism and agent-challenge have been developed as separate git remotes that pin
Base either as a release wheel (Prism) or floating git HEAD (agent-challenge).
Validator-runtime images git-clone challenge SHAs at build time. That multi-repo
setup blocks shared `challenge_sdk` evolution, dual CI matrices, and consistent
miner docs.

The residual goal is one source of truth: **`BaseIntelligence/base`**.

## Decision

### 1. Keep the Base package at repo root (`src/base`)

**Chosen:** leave the installable `base` distribution at the monorepo root with
package sources under `src/base/` (current layout).

**Rejected for now:** moving Base into `packages/base/`.

| Option | Pros | Cons |
|--------|------|------|
| **Root `src/base` (chosen)** | Zero path churn for import `base.*`, Docker COPY paths, hatch wheel config, Alembic, CLI entrypoints, release wheel identity `base-3.1.2` | Root `pyproject.toml` is both workspace root and package |
| `packages/base` | Symmetric with challenges | Touches every Docker/CI/docs path; risks release-artifact identity confusion; high churn for no runtime win |

Shared components stay where they already live: `base.challenge_sdk` under
`src/base/challenge_sdk/`. Challenges will path-depend on the workspace `base`
member after import (not a separate `challenge_sdk` distribution in M1).

### 2. Challenge members under `packages/challenges/*`

```text
platform/                                 # BaseIntelligence/base
├── pyproject.toml                        # uv workspace root + base package
├── src/base/                             # base + challenge_sdk (unchanged path)
├── packages/
│   └── challenges/
│       ├── prism/                        # dist: prism-challenge → import prism_challenge
│       └── agent-challenge/              # dist: agent-challenge → import agent_challenge
├── deploy/
├── docker/
└── docs/
    └── monorepo.md                       # this ADR
```

### 3. uv workspace membership

Root `pyproject.toml` declares:

```toml
[tool.uv.workspace]
members = [
  "packages/challenges/prism",
  "packages/challenges/agent-challenge",
]

[tool.uv.sources]
base = { workspace = true }
```

M1 shipped stubs; **M2 (`mono-import-challenges`) imported product sources** for
both challenges under `packages/challenges/*` (history-preserving import from the
standalone remotes / local checkouts). No git submodules.

Challenge `pyproject.toml` files depend on the plain requirement `"base"`, which
uv resolves to the workspace root package via `[tool.uv.sources]`. The previous
Prism release-wheel pin and agent-challenge floating `git+base` HEAD pin are
**removed** so shared `base.challenge_sdk` always comes from monorepo source.

Default `uv sync --extra dev --extra master` continues to install **Base** for
local/CI gates. Installing a challenge member also pulls workspace `base`:

```bash
uv sync --package prism-challenge
uv sync --package agent-challenge
```

### 3b. Shared `base.challenge_sdk` usage

Challenges import shared contracts from the workspace Base package, for example:

```python
from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.executor import DockerExecutor, DockerRunSpec
from base.challenge_sdk.proof import ExecutionProof  # prism
from base.challenge_sdk.roles import Capability, Role, role_contract
from base.challenge_sdk.schemas import ExternalResultEnvelope
```

There is no separate `challenge_sdk` distribution. Evolving shared schemas,
auth, or app factory happens once under `src/base/challenge_sdk/` and is visible
to both challenges on the next workspace lock/sync.

Import package names stay stable:

| Distribution | Import package(s) |
|--------------|-------------------|
| `prism-challenge` | `prism_challenge` |
| `agent-challenge` | `agent_challenge`, `agent_challenge_runner` |

### 4. Invariants (never break production)

- **GHCR image names** stay
  `ghcr.io/baseintelligence/{prism,prism-evaluator,agent-challenge,agent-challenge-terminal-bench-runner,base,base-master,base-validator-runtime}`.
- **Public slugs** stay `/challenges/prism` and `/challenges/agent-challenge`.
- **Python imports** stay `prism_challenge` and `agent_challenge`.
- **No frontend monorepo** in this residual (joinbase Next stays separate).
- **No `set_weights`** from master; secrets names-only in docs/evidence.
- **No submodules**; prefer `git subtree` for history-preserving import.

## Consequences

### Positive

- Single lockfile root and one place for shared `challenge_sdk` changes.
- Later milestones can COPY `packages/challenges/*` into validator-runtime
  (drop `AGENT_CHALLENGE_REF` / `PRISM_REF` clones).
- Miner and deploy docs can converge under Base without renaming public URLs.

### Trade-offs / follow-ups

| Milestone | Work |
|-----------|------|
| `mono-import-challenges` | **Done:** import prism + agent-challenge; workspace `base` path dep; smoke imports |
| `mono-ci-images` | Challenge Docker + CI publish from monorepo contexts; **same GHCR names** |
| `mono-validator-runtime` | Runtime image installs in-tree packages; import smoke for validator_dispatch |
| `mono-deploy-docs-archive` | Deploy/miner docs + standalone-repo SoT notes |

### CI path matrix (placeholder)

M1 adds a cheap workspace-path check job so the member directories and this ADR
cannot silently disappear. Full challenge image matrix path-filters land in
`mono-ci-images`; Base ruff/mypy/pytest jobs remain root-scoped.

## Alternatives considered

1. **Polyrepo forever** — rejected; blocks shared SDK and forces clone pins.
2. **Submodules** — rejected by residual hard rule (prefer subtree).
3. **Move Base under `packages/`** — deferred; churn outweighs symmetry at M1.

## References

- Mission residual layout: monorepo Prism+AC into Base
- Validation: `VAL-MONO-001` (workspace root), `VAL-MONO-002` (Base still alone),
  `VAL-MONO-003`..`006` (import + workspace base + challenge_sdk sharing)
