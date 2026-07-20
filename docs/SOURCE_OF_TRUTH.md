# Source of truth — monorepo transition

**Authoritative first-party product tree for Base + Prism + Agent Challenge is
[`BaseIntelligence/base`](https://github.com/BaseIntelligence/base)** (this repository).

| Concern | Location in monorepo |
|---------|----------------------|
| Base package + `challenge_sdk` | `src/base/` |
| Prism product | `packages/challenges/prism/` (`import prism_challenge`) |
| Agent Challenge product | `packages/challenges/agent-challenge/` (`import agent_challenge`) |
| Unified miner docs | `docs/miner/` · `docs/miner/prism/` · `docs/miner/agent-challenge/` |
| Compose / deploy | `deploy/compose/` (supported); `deploy/swarm/` historical only |
| Challenge images CI | `.github/workflows/challenge-images.yml` |
| Layout ADR | [monorepo.md](monorepo.md) |

## Public contracts (unchanged)

These **must not** change when working in the monorepo:

- GHCR image names:
  `ghcr.io/baseintelligence/{prism,prism-evaluator,agent-challenge,agent-challenge-terminal-bench-runner,base,base-master,base-validator-runtime}`
- Public master / proxy slugs: `/challenges/prism`, `/challenges/agent-challenge`
- Python imports: `prism_challenge`, `agent_challenge` (+ `agent_challenge_runner`)
- Frontend (joinbase Next) stays a **separate** repo — not part of this monorepo residual

## Standalone remotes (transition)

Historically, Prism and Agent Challenge lived as separate remotes:

- `BaseIntelligence/prism`
- `BaseIntelligence/agent-challenge`

Those remotes may remain for dual-source / archive / redirect during cutover. **Prefer this
monorepo** for product edits, image builds, validator-runtime installs, and miner docs.

If you still clone a standalone remote:

1. Treat its README banner as advisory: source of truth is **base**.
2. For new image builds, use monorepo Dockerfiles under `packages/challenges/*` with
   BuildKit `--build-context monorepo=.` (see [deploy.md](deploy.md#monorepo-local-image-builds)).
3. Do not reintroduce release-wheel / floating `git+base` pins for workspace challenges.

## Out of scope for monorepo residual

- joinbase frontend (Vercel / Next)
- Live Swarm mutation, `set_weights` from mission workers
- Renaming GHCR images or public API slugs
