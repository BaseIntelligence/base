# Security Model

![BASE Banner](../assets/banner.jpg)

## Isolation Rules

* The shared control-plane PostgreSQL is reachable only by the master process that owns the Compose project; its URL is provided as a file-backed secret (or explicitly documented env for non-production). Challenges never receive it.
* Challenges never receive master, validator, or central control-plane PostgreSQL credentials. Shipping embed path stores each challenge SQLite under the master volume (`/var/lib/base/challenges/{prism,agent-challenge}`); emergency dual-run may still use a dedicated `/data` volume. Master is the sole writer; **no multi-writer SQLite** across containers.
* Validators never receive master DB credentials; each validator uses independent Compose project credentials and its own wallet when submitting on-chain. Shipping validators are **weight-only** and never write submissions/leaderboards.
* Internal challenge calls require per-challenge shared tokens mounted as secret files.
* The public proxy strips sensitive headers and blocks internal challenge paths. Public prefixes `/challenges/prism` and `/challenges/agent-challenge` are unchanged.
* Agent Challenge env and launch proxy routes preserve only `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp` for signed miner actions.

## Production Policy Boundaries

The production boundary is stricter than local development:

* Dev, test, and local runs may use SQLite for master state. Production control-plane state must use PostgreSQL from a secret file (preferred) or an explicit `BASE_DATABASE_URL`; SQLite is rejected.
* Challenge runtime state is always SQLite on the challenge `/data` volume.
* Dev and local challenge images may be local builds for disposable tests. Production images must be **digest-pinned** (`repository@sha256:<64 hex>`). Untagged references and missing digests are rejected for production installs.
* Production image allowlists must be scoped to a registry and namespace such as `ghcr.io/baseintelligence/`. Broad prefixes such as `baseintelligence/` are development-only.

## Compose Runtime Boundary

First-party deployments use Docker Compose on a single host:

* Master project: `base-master-validator` + `master-postgres` only. Challenges are **embedded** as localhost ASGI inside the master image (`127.0.0.1:18080` Prism, `127.0.0.1:18081` agent-challenge). There is **no** shipping `challenge-*` Compose service.
* Networks: internal `db` (master + Postgres only), internal `app` (available; challenges bind loopback inside master), and a non-internal network solely so the master public API can bind a host port.
* Challenges never join `db` and never receive the Docker socket as separate peers. The master application may mount the Docker socket **read-only** solely to manage its own Compose project (watcher / adoption); embedded challenges have no separate socket mount.
* Validators run in **independent** Compose projects with their own networks, volumes, and identities (weight-only default vs `https://chain.joinbase.ai`).
* Master **never** constructs or invokes `set_weights`. Validators submit with their own wallets when gated.
* Base and Prism do **not** create short-lived evaluator containers. Evaluation is external/long-lived (agent-challenge Phala TEE when enabled, or miner-funded workers) and is verified/ingested rather than lifecycle-managed as `--rm` jobs by the application. Prism product has no TEE package (provider trust + IMAGE_PIN).
* Docker Swarm (`docker service`, `docker stack`, Swarm secrets, overlays, placement constraints, join tokens) is **not** a supported target for new installs.

The control-plane database credential must never appear in stdout, stderr, rendered Compose manifests, docs evidence, or support logs. Challenges receive only per-challenge runtime secrets (file-backed tokens). Master volume challenge paths are retained by default with master backup/teardown; manual deletion is an explicit, destructive purge.

### Privileged and GPU evaluation

Application paths must not grant challenges host Docker privilege or raw GPU
device mounts through the Compose target topology. Heavy evaluation that needs
special hardware lives in external worker/TEE boundaries with their own trust
model (see miner worker plane and Prism TEE docs). Capability-gated historical
DinD escape hatches on multi-host Swarm fabric are not reintroduced as a required
Compose operator path.

## PID and Resource Boundary

Compose `deploy.resources` / container `cpus` / `mem_limit` / `pids_limit`
fields (as used by the shipping manifests and orchestrator) enforce operator-set
ceilings where the engine supports them. Prefer daemon-wide `no-new-privileges`
and documented host cgroup policy when a finer node-wide swap policy is required.

## Evaluation Cleanup Security

Since the Compose target path does not create ephemeral evaluator Swarm services,
cleanup is project-scoped:

* Teardown scripts and `docker compose down` affect only the named project.
* Watcher failure paths roll images back rather than leaving half-applied pins
  without durable intent.
* Evidence should prove teardown without capturing secret file contents,
  private keys, or credentialed database URLs.

Historical broker archive extraction hardening (path traversal, link rejection)
remains relevant for any residual broker surface, but that surface is not a
required greenfield Swarm deploy.

## Secrets

Admin tokens, challenge tokens, the control-plane database password/URL pieces,
registry credentials, and wallet material must come from **files** (mode `0600`,
parent directories `0700`) or tightly controlled environment injection that never
embeds values in Compose YAML. Prefer `*_FILE` mounts read-only into containers.
Never store clear-text secrets in registry metadata responses, docs, or evidence files.

Agent Challenge miner env values are per-submission secrets owned by the challenge,
not by the BASE registry. They are master-validator scoped, encrypted at rest by
Agent Challenge, injected into the Harbor/Terminal-Bench runtime, and cannot be
retrieved after submission. The BASE proxy forwards the request body to the
challenge but must not parse, persist, log, registry-serialize, or evidence-capture
submitted env values. Public responses expose metadata only: env keys, count, empty
confirmation, lock state, and timestamps.

## Failure Behavior

If a challenge fails health checks or raw-weight publication for an epoch, its
contribution is handled by master aggregation policy for that epoch; operators
should inspect watcher and registry status rather than assuming silent Swarm
service self-heal.

For public challenge requests, transport failures at ingress, the BASE proxy, or
challenge service discovery become safe 502 responses. Challenge-origin non-2xx
responses pass through when safe. User interfaces should render unavailable copy
and must not display raw text such as `BASE request failed with status 502`.

## Removed surfaces

* LLM gateway services, routes, tokens, and provider clients
* Swarm install path for greenfield hosts (`install-swarm.sh`, Swarm secrets, overlays)
* Application-launched evaluator containers

## Phala attestation residual risk

BASE ships fail-closed Phala-tier verification helpers (schema bounds, measurement allowlist, `dcap-qvl` temp-file quote verify, park-vs-reject on verifier outage). Product posture is **cryptographically-anchored trust-but-audit**, not a claim of absolute TEE safety.

Residual risks operators should plan for:

* Hardware / microarchitectural TEE residual risk (including published TDX class attacks such as TEE.fail-style research) is not eliminated by software allowlists.
* Intel PCS / collateral freshness and `dcap-qvl` availability affect park vs reject; a parked result is intentionally not accepted.
* Challenge-owned score acceptance, RA-TLS key release, and CVM image pinning remain agent-challenge responsibilities (available after PR merge for cross-repo docs). BASE proxy must never expose those private challenge routes on the public edge.
* Flag-off (`agent_challenge_attested_routes_enabled=false`) preserves the legacy non-Phala public surfaces; operators enabling the flag must deploy the matching agent-challenge attested mode rather than mixing topologies.

* Public proxy strips sensitive headers (and, for agent-challenge attested mode, strips attested trust-shaped headers and prefixes).
