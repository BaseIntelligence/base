# First-party challenges (workspace members)

Future home of in-tree challenge packages after monorepo import:

| Path | Distribution name | Import package | Status |
|------|-------------------|----------------|--------|
| `prism/` | `prism-challenge` | `prism_challenge` | Stub (subtree import pending) |
| `agent-challenge/` | `agent-challenge` | `agent_challenge` | Stub (subtree import pending) |

These directories are declared as **uv workspace members** from the repo-root
`pyproject.toml`. They are intentionally empty product shells until the
`mono-import-challenges` milestone runs `git subtree add` from the standalone
remotes.

Public contracts that must not change when real code lands:

- GHCR image names (`ghcr.io/baseintelligence/prism`, `…/agent-challenge`, …)
- Public master paths `/challenges/prism` and `/challenges/agent-challenge`
- Python import names `prism_challenge` and `agent_challenge`

See [`docs/monorepo.md`](../../docs/monorepo.md).
