# Versioning

BASE follows **Semantic Versioning**. Source of the package version is root
`pyproject.toml` (kept in lockstep with `uv.lock`).

## Git and GitHub Release

- Tags: `vMAJOR.MINOR.PATCH` on `main`
- GitHub Release notes generated from the tag
- Release body points operators at this policy

## GHCR images

Published images (names never rename):

- `ghcr.io/baseintelligence/base-master`
- `ghcr.io/baseintelligence/base-validator-runtime`
- `ghcr.io/baseintelligence/prism` / `prism-evaluator`
- `ghcr.io/baseintelligence/agent-challenge` / `agent-challenge-terminal-bench-runner`

CI tag strategy (see `.github/workflows/ci.yml`):

- `type=semver,pattern={{version}}`
- `type=semver,pattern={{raw}}`
- digest pin via `sha256` for production
- movable `latest` is non-production convenience only

## Production

Production pulls digest-pinned images (`@sha256:…`). Do not treat floating `latest` as
a production contract.
