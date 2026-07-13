# Architecture

![BASE Banner](../assets/banner.jpg)

BASE runs as a **single-host Docker Compose** topology. Compose is the only supported
shipping runtime for new installs. There is no Helm chart, no Kubernetes manifests,
and no `runtime.backend` selector that switches to Swarm: the target backend is Compose.

Historical `deploy/swarm/` material is retained only as an unsupported reference. Do not
use `install-swarm.sh`, `docker service`, or `docker stack` for greenfield installs.

## Coordination flow

```mermaid
flowchart LR
    M[Miners] -->|submit| P["BASE master proxy"]
    P --> C[Challenge service long-lived]
    C -->|raw-weight push| G[Master aggregation]
    V[Validators independent Compose] -->|register / pull / result| CP[Coordination plane]
    G -->|GET /v1/weights/latest| V
    V -->|set_weights own wallet| BT[Bittensor]
```

Miners reach challenges through the master public proxy. Each challenge owns scoring and
state, then **pushes** authenticated raw hotkey weights to the master. The master
persists snapshots, aggregates a final vector, and serves it. **Validators never compute
canonical aggregation**; each independent validator fetches the master vector and submits
it on-chain with its own wallet. The master **never** constructs or invokes `set_weights`.

There is **no LLM gateway** in the target path. Challenge admission and scoring belong to
each challenge service (Prism is deterministic). Application code does **not** launch
evaluator containers; external long-lived TEE evaluation is verified and ingested, not
orchestrated as --rm jobs by Base or Prism.

## Master Compose project

The master project (`deploy/compose/docker-compose.yml`, installer
`deploy/compose/install-master.sh`) hosts:

| Service | Role |
| --- | --- |
| `base-master-validator` | Public proxy, coordination plane, raw-weight ingress, aggregation, health/version, **digest-aware challenge watcher** |
| `master-postgres` | Durable control-plane PostgreSQL (private network only) |
| one `challenge-<slug>` | Long-lived combined challenge service per active challenge |

Exact cardinality is one application container, one PostgreSQL container, and one
long-lived container per active challenge. There is no gateway sidecar, no challenge
PostgreSQL, no evaluator service, and no Swarm broker overlay in this topology.

Master config and secrets are host files (mode `0600`, parent dirs `0700`) bind-mounted
read-only. Compose manifests never embed secret values. The control-plane database URL is
private to the master process and never reaches challenge containers.

Networks:

- `db` (internal): master + PostgreSQL only; no host publication of `5432`.
- `app` (internal): master + challenge services.
- `public` (non-internal): master host API only (operators typically bind loopback in
  the `3100-3199` test range; production should put a reverse proxy in front).

## Independent validators

Each validator is an **independent Compose project**
(`deploy/compose/docker-compose.validator.yml`, installer
`deploy/compose/install-validator.sh`). Validators register and heartbeat with the
master, pull assignments, report results, fetch the final weight vector, and may submit
on-chain with their own hotkey. They never receive master PostgreSQL credentials,
challenge volumes, Docker socket access, aggregation controls, or challenge lifecycle
operators. Teardown of one validator project does not affect another or the master.

## Challenge isolation

Each active challenge is a **long-lived Compose service** with its own OCI image (digest
pin), internal shared token, public routes behind the proxy, membership on the private
`app` network only, and a named `/data` volume for SQLite and artifacts.

Challenge state is SQLite on that volume
(`sqlite+aiosqlite:////data/challenge.sqlite3`). BASE provisions no Postgres server per
challenge; each challenge owns its `/data` volume and never receives a control-plane
database credential. Volumes are retained when a challenge service is stopped or
removed; purge is an explicit operator action on the named volume.

## Digest-aware auto-update (watcher)

Auto-update of challenges is **not** Swarm service mutation of a mutable `latest` tag.
It is the **master-resident Compose challenge watcher** running inside
`base-master-validator`:

1. Resolve an approved **immutable** image reference (`repository@sha256:<64 hex>`).
2. Record current vs desired digest and durable rollout intent.
3. Controlled pull of the desired image.
4. Targeted recreate of only the affected Compose service (project-scoped).
5. Health and version verify.
6. On failure, restore the previous digest with **bounded backoff**; durable state
   survives master restart.

The watcher never creates evaluator containers, never calls `docker service` / Swarm
APIs, and only mutates services inside the configured Compose project boundary.

Operator install and deeper cardinality rules: [Compose-only deployment](compose.md) and
[Deploy from scratch](deploy.md).

## Weight protocol

1. A challenge computes a raw hotkey-weight snapshot.
2. It pushes a versioned payload to the master's private authenticated ingress.
3. The master validates challenge binding, rejects malformed / replayed / stale payloads,
   and persists the snapshot.
4. Duplicate deliveries of the same epoch/revision are idempotent.
5. The master normalizes, applies emission shares, maps hotkeys to UIDs, applies
   burn/zero-miner policy, and serves a final vector.
6. Validators fetch that vector and submit independently.

## Out of scope / unsupported

- Docker Swarm for new installs (`deploy/swarm/`, `install-swarm.sh`, overlays, Swarm
  secrets, replicated jobs, placement constraints, `docker service` / `docker stack`)
- LLM gateway services, tokens, routes, and provider clients
- Application-launched evaluator containers (`docker run`, `docker compose run` jobs)
- Helm / Kubernetes
- Per-challenge Postgres servers managed by BASE
- Automated destructive challenge purge without explicit operator action

## Agent Challenge Phala Intel TDX path

Agent Challenge attestation is **separate from the PRISM miner-funded GPU worker plane**. BASE owns the shared proof, proxy, and assignment surfaces below; end-to-end self-deploy review→eval, RA-TLS key release, and score acceptance live in the agent-challenge service when its own attestation flags are on (cross-repo challenge docs are available after PR merge).

### Flag-off vs flag-on

| Surface | Flag off (default) | Flag on |
|---------|--------------------|---------|
| Master setting | `master.agent_challenge_attested_routes_enabled=false` | `master.agent_challenge_attested_routes_enabled=true` |
| Public proxy | Legacy signed submission / env / launch passthrough | Fail-closed **allowlist** of review/eval + status/SSE + benchmark metadata only |
| Evaluation ownership | Legacy R=1 `own_runner` on validators (reassign on failure, never multi-replica) | Full attested mode: **one miner-funded external eval (R=1)**; BASE creates **zero** agent-challenge validator assignment, retry, replica, reconciliation, audit, or fold rows for those units |
| Phala verifier | Not required on agent-challenge results | Challenge/operators use BASE Phala-tier schema + quote helpers; BASE does not invent challenge-side score policy |

Legacy path is byte-identical when the master attested-routes flag stays off. PRISM worker-plane replication (default R=2, `compute.replication_factor`) is unchanged either way.

```mermaid
flowchart TB
  M[Miner] --> P[Public BASE proxy]
  P -->|allowlisted review/eval only when flag on| AC[agent-challenge service]
  AC -->|miner-funded external Eval R=1| TEE[Phala Intel TDX CVM]
  TEE -->|EvalExecutionProof tier phala-tdx| AC
  AC -->|internal weights / control plane only| MSTR[Master]
  V[Validators] -.->|sampled replay audit only when scheduled| AC
```

### Public vs private challenge proxy boundary

Public clients reach challenges only at `/challenges/{slug}/...`. The proxy always blocks:

- `/internal/*`, `/health`, `/version`
- Generic benchmark-execution-shaped paths (for example `/benchmark-executions` and benchmark `run` / `execute` / `launch` leaves)

With **attested routes enabled**, agent-challenge is additionally **allowlist fail-closed** (`src/base/master/app_proxy.py`): only the exact signed review/eval rows, signed `POST /submissions`, `GET .../status` and `GET .../events`, and `GET /benchmarks/tasks` are forwardable. Capability, assignment, evidence, key-release, direct result ingestion, results aliases, env/launch (legacy), wrong methods, neighboring paths, and non-canonical raw path bytes are denied **locally** (404 before any upstream call). Private routes must never fall through the public proxy.

On allowlisted agent-challenge routes the proxy:

- Preserves miner signature headers `X-Hotkey`, `X-Signature`, `X-Nonce`, `X-Timestamp` where the miner signs the challenge-local path
- Strips hop-by-hop, internal, and attested trust-shaped headers/prefixes (for example `x-base-*`, `x-attestation-*`, `x-ra-tls-*`, client-IP spoofs) so edge clients cannot inject trust

### ExecutionProof Phala tier (BASE schema)

Every worker/eval proof envelope is the shared `ExecutionProof` model in `src/base/schemas/worker.py`:

- Integer tiers **0 / 1 / 2** remain the PRISM worker-plane tiers (manifest + sr25519; image digest; optional in-guest attestation).
- Phala Intel TDX uses string tier **`phala-tdx`** (`PHALA_TDX_TIER`), with an `attestation` block (`PhalaAttestation` / strict `EvalPhalaAttestation`).
- Canonical **Eval** wire equals schema-closed `EvalExecutionProof` (version 1, tier `phala-tdx`, digest-pinned `image_digest`, empty worker-signature placeholders for validator rebind, no extra fields).
- Bound examples (fail closed at parse): TDX quote ≤ 64 KiB, event log ≤ 4096 entries / 2 MiB, **`vm_config` ≤ 256 KiB** (`EVAL_MAX_VM_CONFIG_BYTES`), closed `vm_config` fields `{vcpu, memory_mb, os_image_hash}`.
- Measurement registers use fixed hex widths; `report_data` is 64-byte (128 hex) left-aligned binding with domain tag `base-agent-challenge-v1` (`src/base/worker/proof.py`).

### Quote verify and park vs reject

BASE quote helpers (`src/base/worker/phala_quote.py`, `phala_verify.py`, `proof.py`) verify Phala-tier proofs:

1. **Tier-0** worker (or rebound) signature over `sha256("{manifest_sha256}:{unit_id}")` always required.
2. Quote structure + RTMR3 event-log replay from the hardware-signed TD report (compose bound by content, not trusted by value).
3. Cryptographic DCAP verification via the **`dcap-qvl`** CLI: quote bytes are written to a **temp file** and passed by path (not as an inline argv hex body).
4. Measurement must match a non-empty validator **allowlist** (fail-closed when empty or unloaded).
5. Nonce freshness via validator-issued state; acceptable TCB default includes `UpToDate`.

Outcomes:

- Cryptographic / structure failure → **reject**
- Transient `dcap-qvl` missing, timeout, exit-0 non-JSON, or missing TCB status → **park** (`VerifierUnavailableError`): never accept, never permanent fraud-reject

BASE does not claim perfect hardware trust. Treat Phala as **cryptographically-anchored trust-but-audit**; residual TEE and collateral risks remain (see [Security](security.md)).

### Replay audit seam

Sampled validator replay audits for agent-challenge use the labelled transport in `src/base/master/replay_audit.py` (`agent-challenge.replay-audit.v1`). Requests carry the complete immutable Eval plan; plan digests and trial scores are validated on the BASE wire before forward or accept. This is not ordinary multi-replica worker reconciliation.
