# First-party challenges (workspace members)

In-tree challenge packages imported into the Base monorepo:

| Path | Distribution name | Import package | Status |
|------|-------------------|----------------|--------|
| `prism/` | `prism-challenge` | `prism_challenge` | Imported (product sources) |
| `agent-challenge/` | `agent-challenge` | `agent_challenge` (+ `agent_challenge_runner`) | Imported (product sources) |

These directories are **uv workspace members** from the repo-root `pyproject.toml`.
Both depend on workspace `base` (shared `base.challenge_sdk`) instead of the
legacy release wheel (Prism) or floating `git+base` (agent-challenge).

Public contracts that must not change:

- GHCR image names (`ghcr.io/baseintelligence/prism`, `…/agent-challenge`, …)
- Public master paths `/challenges/prism` and `/challenges/agent-challenge`
- Python import names `prism_challenge` and `agent_challenge`

See [`docs/monorepo.md`](../../docs/monorepo.md).
