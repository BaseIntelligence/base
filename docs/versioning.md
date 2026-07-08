# Versioning Policy

BASE releases start at `3.0.0` and follow Semantic Versioning (`MAJOR.MINOR.PATCH`) for the application, the Python package, and release image tags.

## Sources Of Truth

Update these together for every release:

- `pyproject.toml`: Python package version, without a leading `v`.
- `uv.lock`: editable `base` package entry, without a leading `v`.
- `.github/workflows/ci.yml`: GHCR tag policy.

For the `3.0.4` release the Python package version is `3.0.4` and the Git release tag is `v3.0.4`.

## SemVer Rules

- `MAJOR`: breaking public API, CLI, config, environment variable, Docker runtime, database migration, deployment, or validator behavior changes.
- `MINOR`: backward-compatible features.
- `PATCH`: backward-compatible fixes.
- Released versions are immutable; fix forward with a new version.
- Python package versions stay PEP 440-compatible, so they omit the Git tag's leading `v`.

## GitHub And GHCR Tags

Use Git tags with a leading `v` (such as `v3.0.4`) for release events. The GitHub Actions metadata policy publishes canonical GHCR image tags from the tag event using:

```text
type=semver,pattern={{version}}
type=semver,pattern={{raw}}
type=sha,prefix=sha-
```

So a `v3.0.4` tag publishes the canonical `3.0.4` image tag, the compatibility `v3.0.4` tag, and a traceable `sha-<commit>` tag. Branch builds publish a mutable `main` tag, and `main` also publishes `latest`; those mutable tags feed the supervisor image-updater auto-update channel on first-party deployments.

Pull requests build images with `push: false`. GHCR publication happens only from trusted events: `main`, `v*.*.*` tags, or a manual `workflow_dispatch` with `confirm_publish` set to `true`.

## GitHub Releases

Pushing a `v*.*.*` tag creates a GitHub Release only after CI validation and both GHCR publish jobs succeed. Branch pushes and manual `workflow_dispatch` runs can publish images under the trusted-event rules above but do not create a GitHub Release.

Release descriptions combine GitHub-generated release notes with a maintained body listing the published `base` and `base-master` GHCR tags (canonical SemVer, compatibility `v` tag, traceable `sha-<commit>` tag) plus a note that production should pin the SemVer tag and immutable digest. Tags containing a hyphen (such as `v3.1.0-rc.1`) are marked prereleases; stable tags are marked as the latest release.

## Production Image Policy

Pinned production references must use a SemVer image tag plus a digest:

```text
ghcr.io/baseintelligence/base:3.0.4@sha256:<64-hex-digest>
```

The digest is the immutable deployment selector; the tag provides human-readable release context. Production policy accepts digest-pinned `latest` only for the autonomous update channel and rejects untagged references, missing digests, non-SemVer non-`latest` tags, and mutable auto-update in pinned production mode.

Mutable tags such as `latest` and `main` are allowed for the default supervisor auto-update mode. In that mode the manager runs the master admin, proxy, broker, and challenge services from `ghcr.io/baseintelligence/base-master:latest`, and the image-updater / challenge-image-updater loops resolve the public GHCR tag digest and roll the Swarm services to `tag@sha256:<digest>` only when a mutable tag moves. The on-chain submitter deploys from `ghcr.io/baseintelligence/base:latest`; it fetches master-computed weights and performs the final Bittensor submission. The updaters use anonymous GHCR digest checks (no pull secret needed while packages are public). To roll back or freeze, disable mutable auto-update and pin SemVer plus digest.

## Release Execution Boundary

Do not create Git tags, GitHub releases, GHCR packages, or real-node rollouts unless the operator explicitly confirms that external side effect. Local validation and `push: false` Docker builds are safe pre-release checks.
