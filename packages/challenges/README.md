# First-party challenges (workspace members)

In-tree challenge packages in the Base monorepo:

| Path | Distribution name | Import package |
|------|-------------------|----------------|
| `prism/` | `prism-challenge` | `prism_challenge` |
| `agent-challenge/` | `agent-challenge` | `agent_challenge` (+ `agent_challenge_runner`) |

uv workspace members from the repo-root `pyproject.toml`. Both depend on workspace
`base` (`base.challenge_sdk`).

Do not rename:

- GHCR image names (`ghcr.io/baseintelligence/prism`, `…/prism-evaluator`,
  `…/agent-challenge`, `…/agent-challenge-terminal-bench-runner`)
- Public master paths `/challenges/prism` and `/challenges/agent-challenge`
- Python import names `prism_challenge` and `agent_challenge`

Images: [`.github/workflows/challenge-images.yml`](../../.github/workflows/challenge-images.yml)
(BuildKit `monorepo=.`). Shipping day-1: [docs/miner/getting-started.md](../../docs/miner/getting-started.md).
API truth: challenge `/openapi.json` (not markdown dumps).
