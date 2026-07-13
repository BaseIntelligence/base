# Versioning Policy

BASE releases start at `3.0.0` and follow Semantic Versioning (`MAJOR.MINOR.PATCH`) for the application, the Python package, and release image tags.

## Sources Of Truth

Update these together for every release:

- `pyproject.toml`: Python package version, without a leading `v`.
- `uv.lock`: editable `base` package entry, without a leading `v`.
- `.github/workflows/ci.yml`: GHCR tag policy.

For the `3.1.2` SDK release the Python package version is `3.1.2` and the Git release tag is `v3.1.2`.

## SemVer Rules

- `MAJOR`: breaking public API, CLI, config, environment variable, Docker runtime, database migration, deployment, or validator behavior changes.
- `MINOR`: backward-compatible features.
- `PATCH`: backward-compatible fixes.
- Released versions are immutable; fix forward with a new version.
- Python package versions stay PEP 440-compatible, so they omit the Git tag's leading `v`.

## GitHub And GHCR Tags

Use Git tags with a leading `v` (such as `v3.1.2`) for release events. The GitHub Actions metadata policy publishes canonical GHCR image tags from the tag event using:

```text
type=semver,pattern={{version}}
type=semver,pattern={{raw}}
type=sha,prefix=sha-
```

So a `v3.1.2` tag publishes the canonical `3.1.2` image tag, the compatibility `v3.1.2` tag, and a traceable `sha-<commit>` tag. Branch builds may publish a mutable `main` / `latest` tag for development convenience; those tags are **not** production selectors by themselves.

Pull requests build images with `push: false`. GHCR publication happens only from trusted events: `main`, `v*.*.*` tags, or a manual `workflow_dispatch` with `confirm_publish` set to `true`.

## GitHub Releases

Pushing a `v*.*.*` tag creates a GitHub Release only after CI validation and both GHCR publish jobs succeed. Branch pushes and manual `workflow_dispatch` runs can publish images under the trusted-event rules above but do not create a GitHub Release.

Release descriptions combine GitHub-generated release notes with a maintained body listing the published `base` / `base-master` / runtime GHCR tags (canonical SemVer, compatibility `v` tag, traceable `sha-<commit>` tag) plus a note that production must pin an **immutable digest**. Tags containing a hyphen (such as `v3.1.0-rc.1`) are marked prereleases; stable tags are marked as the latest release.

## Production Image Policy

Pinned production references must use repository plus digest (optionally with a human Tag context):

```text
ghcr.io/baseintelligence/base-master@sha256:<64-hex-digest>
# or repository:tag@sha256:<64-hex-digest>
```

The digest is the immutable deployment selector. Production policy rejects untagged
references used as the sole pin, missing digests on production targets, and any
auto-update that only mutates a floating `latest` without digest verification.

## Auto-update policy (Compose watcher)

Supported challenge auto-update is the **master-resident Compose challenge
watcher**, not Swarm service mutation of `latest`:

1. Desired image is an approved **digest-pinned** reference.
2. Watcher records current vs desired digest and durable rollout phase/intent.
3. Controlled `docker pull` of the desired pin.
4. Targeted recreate of only the affected long-lived Compose service inside the
   project boundary.
5. Health and version verification before commit.
6. On failure: restore the previous digest, record bounded backoff, and resume
   safely after master restart.

Independent **validators** now auto-update their runtime image by default.
`install-validator.sh` enables a host-side systemd timer that tracks
`ghcr.io/baseintelligence/base-validator-runtime:latest`, always applies as
`repository@sha256:<digest>` (never bare `:latest` as the compose runtime
selector), and recreates only the agent service with LKG rollback, hold, and
bounded backoff. Image auto-update remains host-side; shipping Compose may also
mount host `docker.sock` into the agent for later challenges-on-validator prep.
Opt out with `--no-auto-update` or freeze with `BASE_VALIDATOR_IMAGE_UPDATE_HOLD=1`.

Master application images for the Compose master project remain operator-
driven (reinstall/recreate with new pins). There is no supported
“always follow mutable `latest` via Swarm image updater” path for new installs.

To freeze challenges, stop advancing desired digests (or disable the watcher
interval) and leave the running pin in place. To roll back, set the approved pin
back to the previous digest and let the watcher (or a controlled recreate) apply
it after verify.

## Release Execution Boundary

Do not create Git tags, GitHub releases, GHCR packages, or real-node rollouts unless the operator explicitly confirms that external side effect. Local validation and `push: false` Docker builds are safe pre-release checks.

## Historical note

Documentation or tools under `deploy/swarm/` that describe rolling Swarm services
to `tag@sha256` whenever a mutable tag moves are **historical**. They are not the
shipping auto-update mechanism for Compose topology installs.
