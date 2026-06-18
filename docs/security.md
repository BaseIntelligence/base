# Security Model

![Platform Banner](../assets/banner.jpg)

## Isolation Rules

* The shared control-plane PostgreSQL is available only to the master or validator control-plane process that owns that deployment. Its URL comes from `PLATFORM_DATABASE_URL` or a Docker secret.
* Challenges never receive master, validator, or central control-plane PostgreSQL credentials.
* Each challenge gets only its own SQLite database on its `/data` Swarm volume (`CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3`).
* The submitter never receives master DB credentials.
* Internal challenge calls require per-challenge shared tokens.
* Public proxy strips sensitive headers.
* Public proxy blocks internal challenge paths.
* Agent Challenge env and launch proxy routes preserve only `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp` for signed miner actions.

## Production Policy Boundaries

The production boundary is stricter than local development:

* Dev, test, and local runs may use SQLite for master state. Production control-plane state must use PostgreSQL loaded from a Docker secret or an explicit `PLATFORM_DATABASE_URL`; SQLite is rejected for control-plane state.
* Challenge runtime state is always SQLite on the challenge `/data` Swarm volume. Challenges never receive control-plane Postgres credentials.
* Dev and local challenge images may be local, mutable, or tagged `latest` while iterating. Production images must include a tag and a `sha256` digest; SemVer tags are for pinned releases, while `latest@sha256:<digest>` is reserved for the autonomous update channel, starting with Platform `3.0.0` for release images. Production rejects untagged references and missing digests; `latest` is accepted only when digest-pinned for the autonomous update channel.
* Production image allowlists must be scoped to a registry and namespace such as `ghcr.io/platformnetwork/`. Broad prefixes such as `platformnetwork/` are development-only.

## Swarm Runtime Boundary

First-party deployments use Docker Swarm rolling service updates. Challenge services run on the manager node (`node.role==manager`); broker-dispatched evaluation jobs run on worker nodes constrained by `node.labels.platform.workload==cpu` or `==gpu`. Broker-created challenge jobs must not receive the host Docker socket.

Broker GPU placement is expressed only through Swarm node labels and generic resources. A positive broker `gpu_count` becomes `--generic-resource NVIDIA-GPU=<N>` on a job constrained to `node.labels.platform.workload==gpu`; the name `NVIDIA-GPU` is case-sensitive and must match the worker `daemon.json` advertisement. Omitted or `None` stays CPU-only. Challenge metadata, labels, environment values, and device IDs are not placement semantics. Network isolation uses encrypted overlay networks created with MTU 1450; a job requesting `network: none` is attached to a dedicated internal (no external routes) encrypted overlay because Swarm services cannot attach to the predefined `none` network.

The control-plane database credential is written only into a Docker secret and must not be printed in stdout, stderr, service definitions, docs evidence, or support logs. Challenge services receive only per-challenge runtime secrets. The challenge `/data` Swarm volume is retained by default when a challenge is removed. Manual deletion is destructive and must be treated as an explicit purge.

### Privileged escape hatch

A Swarm service cannot run `--privileged` or `--gpus`, so `docker service create` never emits them. Challenges that legitimately need a privileged Docker-in-Docker job use the capability-gated escape hatch: instead of a Swarm service, the broker runs the job as a direct local `docker run` on a worker node. The escape hatch is the only path that grants privilege, it is gated per challenge, and the DinD container owns its own `/var/lib/docker` volume rather than the host Docker socket.

## PID and Swap Boundary

Swarm service resources map CPU and memory to `--limit-cpu` and `--limit-memory`, and PID ceilings to `--limit-pids`. `docker service create` does not support `--memory-swap` or `--security-opt`, so swap limits are not emitted and `no-new-privileges` is enforced daemon-wide through `daemon.json` rather than per service. If production needs swap ceilings, enforce them with daemon or cgroup configuration and document that policy with the node runbook.

## Broker Archive and Cleanup Security

Broker archive uploads are treated as untrusted input. The Swarm broker path rejects absolute paths, parent traversal, links, and device members before extraction, and malformed broker images are rejected before any service is created.

Broker job cleanup is two-layered. The broker `/v1/docker/cleanup` path removes the Swarm service (`docker service rm`) and releases the workload and GPU ledger entries on success and failure. The manager-only supervisor timeout-reaper independently reaps jobs that exceed their timeout, so a crashed or unreachable challenge cannot leak long-running services. Evidence should prove cleanup behavior without storing archive payloads, bearer credentials, private keys, or credentialed database URLs.

## Secrets

Admin tokens, challenge tokens, the control-plane database URL, registry credentials, and wallet material must come from files, environment variables, or Docker secrets. Swarm secrets are mounted inside containers at `/run/secrets/platform/<name>`, and value-bearing secrets reach `docker secret create` via stdin, never as argv. Don't store clear text secrets in registry metadata responses, docs, or evidence files.

Agent Challenge miner env values are per-submission secrets owned by the challenge, not by Platform registry. They are master-validator scoped, encrypted at rest by Agent Challenge, injected into Harbor/Terminal-Bench runtime, and cannot be retrieved after submission. Platform proxy forwards the request body to the challenge but must not parse, persist, log, registry-serialize, or evidence-capture submitted env values. Public responses can expose metadata only: env keys, count, empty confirmation, lock state, and timestamps.

## Failure Behavior

If a challenge fails health checks or `get_weights`, its contribution is zero for that epoch. The master doesn't auto-disable it.

For public challenge requests, transport failures at ingress, Platform proxy, or challenge service discovery become safe 502 responses. Challenge-origin non-2xx responses should pass through when they are safe. User interfaces should render unavailable copy and must not display raw text such as `Platform request failed with status 502`.
