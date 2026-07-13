# Miner Worker Deployment Guide (PRISM Compute Plane)

Deploy a **miner-funded GPU worker** so your hotkey stays eligible to submit to PRISM and
so PRISM's heavy evaluation runs on GPU capacity **you** pay for (rented on Lium or Targon,
or your own hardware) instead of on BASE validators.

This guide covers, for both **Lium** and **Targon**:

- [What the worker plane is](#what-the-worker-plane-is)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Publishing the worker image](#publishing-the-worker-image)
- [Deploy on Lium](#deploy-on-lium)
- [Deploy on Targon](#deploy-on-targon)
- [Deploy locally / on your own hardware](#deploy-locally--on-your-own-hardware)
- [Costs and price guidance](#costs-and-price-guidance)
- [Monitoring your fleet](#monitoring-your-fleet)
- [Troubleshooting](#troubleshooting)

> The whole (PRISM) worker plane is gated behind a feature flag
> (`compute.worker_plane_enabled` on the master, `worker_plane` in the prism config). With
> the flag off, nothing here applies and the subnet behaves exactly as before.
>
> **This guide is PRISM-only.** Agent Challenge Phala Intel TDX attestation (other
> challenge, separate flags, R=1 external eval, public-proxy allowlist) is **not** the
> miner GPU worker plane. See [Architecture: Agent Challenge Phala path](../architecture.md#agent-challenge-phala-intel-tdx-path)
> and the agent-challenge operator docs (available after PR merge).

---

## What the worker plane is

- **`base worker agent`** mirrors the validator agent loop
  (register → heartbeat → pull → execute → post), running inside a GPU instance you fund.
- **Signed enrollment** binds your **miner hotkey** to a **worker keypair**: the miner signs
  `worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}` (sr25519), and the master verifies it
  against the metagraph before the worker can pull work.
- **Admission rule**: when the master enforces it, you need **≥1 active worker bound to your
  hotkey** to submit to PRISM; a submission from a hotkey with no active worker is rejected with
  HTTP `403 NO_ACTIVE_WORKER`. Deploying a worker unlocks submitting.
- **Anti-collusion**: a worker never evaluates a submission from its own owner; each unit is
  replicated across **2 distinct owners** (R=2) and reconciled by comparing manifest hashes.
- **Proof tiers** attached to every result (see [Costs and price guidance](#costs-and-price-guidance)
  for how tier affects audit rate):
  - **Tier 0** (mandatory, all backends): deterministic manifest hash + your worker's sr25519
    signature. This is the source of truth for reconciliation and audits.
  - **Tier 1**: the evaluator image digest matches the pinned worker/evaluator digest. On Lium
    this is cross-checkable against the signed `GET /watchtower/digest`.
  - **Tier 2**: in-guest TDX + nvtrust attestation. **Gated OFF on Targon today** (Targon
    exposes no consumer-facing attestation surface), so Targon proofs carry tier ≤ 1. Tier 0 is
    always available; tier 1 is available on both providers.

Your provider API key (`LIUM_API_KEY` / `TARGON_API_KEY`) is used **only** to authenticate
your own provider calls from the CLI/agent. It is **never** sent to the master, never written
into the pod env, and never logged.

---

## Prerequisites

- A **registered miner hotkey** on the subnet (present in the metagraph). The worker binding is
  signed by this hotkey.
- The `base` CLI. In a dev checkout run it through `uv`:

  ```bash
  uv run base worker --help
  ```

  Expected: a command group listing `agent`, `deploy`, and `status`:

  ```text
  Usage: base worker [OPTIONS] COMMAND [ARGS]...

   Deploy and manage miner-funded GPU worker agents

  ╭─ Commands ─────────────────────────────────────────────────────────────────╮
  │ agent   Run the miner-funded GPU worker agent loop.                         │
  │ deploy  Deploy a worker agent locally or onto a rented provider instance.   │
  │ status  Render the worker fleet from the master's ``GET /v1/workers``.      │
  ╰────────────────────────────────────────────────────────────────────────────╯
  ```

  > `base worker` (the miner-funded GPU worker plane) is a distinct, top-level group from any
  > historical `base master worker` Swarm node-management commands. Swarm is not the supported
  > base install path; the groups still do not collide.

- A **provider account with credit**:
  - **Lium**: an API key (`LIUM_API_KEY`) and a registered SSH key (Lium requires an SSH public
    key on every rent, even for non-interactive workloads).
  - **Targon**: an API key (`TARGON_API_KEY`) and enough dashboard credit (Targon does **not**
    expose balance via API; see [Troubleshooting](#troubleshooting)).
- A reachable BASE **master coordination-plane URL** (`worker.agent.master_url` in your config).

Inspect the deploy options before running anything:

```bash
uv run base worker deploy --help
```

Expected (abridged): `--provider` is **required** and accepts `lium | targon | local`, plus a
`--max-price` bound and a `--config` path:

```text
Usage: base worker deploy [OPTIONS]

╭─ Options ──────────────────────────────────────────────────────────────────╮
│ *  --provider    TEXT   Where to run the worker agent: lium | targon | local.│
│    --max-price   FLOAT  Max price per GPU/hour bounding provider offer        │
│                         selection.                                            │
│    --config      PATH   [default: config/worker.example.yaml]                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Configuration

Start from [`config/worker.example.yaml`](../../config/worker.example.yaml) and set at least:

```yaml
compute:
  worker_plane_enabled: true

worker:
  agent:
    master_url: https://<your-master-host>   # required: the coordination plane
    broker_url: http://127.0.0.1:8082         # the worker's OWN local Docker broker
    capabilities:
      - gpu
  deploy:
    provider: lium            # local | lium | targon (also selectable via --provider)
    gpu_count: 1
    max_price_per_hour: 1.50  # cost cap per GPU/hour (also selectable via --max-price)
    max_lifetime_hours: 1.0   # bounded pod lifetime (keep it short)
    # REQUIRED for a lium/targon deploy: a PUBLICLY-pullable, digest-pinned worker
    # image. A provider deploy refuses to run without both (see "Publishing the
    # worker image"). Not needed for provider: local.
    image: ghcr.io/<your-public-namespace>/base-worker
    image_digest: sha256:<64 hex>         # the immutable pin
    image_tag: v1                          # informational-only (see note below)
    startup_commands: tail -f /dev/null   # MUST be metachar-free (see Troubleshooting)
    ssh_public_key_file: /path/to/your_key.pub   # Lium requires an SSH key
    ssh_key_name: my-worker-key
  identity:
    # The WORKER keypair signs coordination requests + ExecutionProofs.
    key_uri: null            # e.g. an sr25519 URI, a mnemonic, or a wallet; else falls back to network.wallet
    # The MINER binding: EITHER a miner key that signs the binding at deploy time...
    miner_key_uri: null
    # ...OR a pre-signed binding (for a pod that never holds the miner key):
    miner_hotkey: null
    binding_signature: null
    binding_nonce: null
```

Every field is overridable by env with the `BASE_` prefix and `__` nesting, e.g.
`BASE_WORKER__AGENT__MASTER_URL=...` or `BASE_COMPUTE__WORKER_PLANE_ENABLED=true`.

**Identity note.** The worker needs a worker keypair (falls back to `network.wallet` when unset)
and a miner-signed binding. On a rented pod that must never hold your miner key, sign the binding
on your own machine and pass the pre-signed `miner_hotkey` + `binding_signature` + `binding_nonce`
instead of a miner key. The CLI does this for you when a miner key is configured locally.

---

## Publishing the worker image

A `lium`/`targon` deploy runs a **worker image** you must publish yourself, pinned by digest.
`base worker deploy --provider lium|targon` **refuses to run** unless `worker.deploy.image` **and**
`worker.deploy.image_digest` are set (env `BASE_WORKER__DEPLOY__IMAGE` /
`BASE_WORKER__DEPLOY__IMAGE_DIGEST`); there is deliberately no baked-in default image.

> ⚠️ **The worker image must be PUBLICLY pullable.** A private GHCR image makes Lium pod creation
> fail with `CREATION_FAILED` because the rented executor host cannot pull it, so the deploy will
> not silently pin a placeholder.

> 🚧 **Publishing is a release/operator action.** Pushing to a public registry needs push
> credentials you may not have as a miner — coordinate with the subnet operators for the canonical
> published image + digest.

Build, publish, and pin `docker/Dockerfile.worker` (a slim `python:3.12-slim` image carrying the
`base` worker agent + docker CLI):

```bash
# 1. Build from the base repo root (the Dockerfile COPYs the source in).
docker build -f docker/Dockerfile.worker -t ghcr.io/<your-public-namespace>/base-worker:v1 .

# 2. Log in and push to a PUBLIC registry namespace (make the package public in the
#    registry UI so an unauthenticated provider host can pull it).
echo "$GHCR_PAT" | docker login ghcr.io -u <your-username> --password-stdin
docker push ghcr.io/<your-public-namespace>/base-worker:v1

# 3. Read back the immutable digest the registry assigned, and pin THAT.
docker buildx imagetools inspect ghcr.io/<your-public-namespace>/base-worker:v1 \
  --format '{{println .Manifest.Digest}}'
# -> sha256:<64 hex>
```

Then set the image + digest in your config (or via env), e.g.:

```yaml
worker:
  deploy:
    image: ghcr.io/<your-public-namespace>/base-worker
    image_digest: sha256:<64 hex>   # the value printed above
    image_tag: v1                    # informational-only (see note below)
```

> **`image_tag` is informational-only.** A `lium`/`targon` deploy provisions the worker
> **by digest** (`image` + `image_digest`); `worker.deploy.image_tag` is **not consumed** by
> the deploy path and does **not** affect which image bytes run. Keep it only as a
> human-readable note of the tag the digest was published under. What earns **tier 1** is the
> digest pin, not the tag.

**Verify it is actually pullable before deploying.** From a clean host with no registry auth:

```bash
docker pull ghcr.io/<your-public-namespace>/base-worker@sha256:<64 hex>
```

If that pull fails unauthenticated, the image is not public and a Lium deploy will
`CREATION_FAILED`. Digest-pinning (`@sha256:...`) is also what earns **tier 1** proofs: the running
image bytes are provably the ones you published (cross-checkable on Lium against
`GET /watchtower/digest`).

---

## Deploy on Lium

1. Export **your** Lium API key (never commit it):

   ```bash
   export LIUM_API_KEY=<your-lium-api-key>
   ```

2. Deploy, bounding the price per GPU/hour:

   ```bash
   uv run base worker deploy --provider lium --max-price 1.50
   ```

   The CLI plans offline, then rents once: it selects the cheapest **suitable** offer at or below
   `--max-price` (preferring an exact GPU-count executor; a partial slice of a multi-GPU node is
   rejected by Lium), ensures the worker template and SSH key exist (idempotent), rents with a
   bounded `termination_hours` (from `max_lifetime_hours`), and boots the image, which enrolls with
   the master under your signed binding.

   ```text
   Selected lium offer <executor-id> (NVIDIA A100-SXM4-80GB x1) @ 0.52/GPU/hr
   Provisioned lium instance <pod-id> (status=PENDING); the worker enrolls with the master on boot
   ```

3. Wait for the pod to reach `RUNNING` (a minute or two; up to ~15 min for a cold image pull) and
   confirm enrollment with [`base worker status`](#monitoring-your-fleet).

> **Cost guardrails are enforced.** Deploy never issues an unbounded rent (a bounded
> `termination_hours` is always set) and refuses an over-cap offer before renting. Confirm the pod
> is gone when you are done (see Troubleshooting → leftover pods).

---

## Deploy on Targon

1. Export **your** Targon API key:

   ```bash
   export TARGON_API_KEY=<your-targon-api-key>
   ```

2. Deploy:

   ```bash
   uv run base worker deploy --provider targon --max-price 3.50
   ```

   The CLI lists the live Targon inventory (`GET /inventory?type=rental&gpu=true`), selects a
   GPU shape within `--max-price` by `cost_per_hour`, and deploys the worker app; the worker
   enrolls with the master on boot, exactly as on Lium.

3. Confirm enrollment with [`base worker status`](#monitoring-your-fleet).

> **Targon has no balance API.** The CLI cannot pre-check your credit. If a deploy fails with an
> insufficient-credit error, it is surfaced as a distinct typed error and is **not retried** —
> top up in the Targon web dashboard and retry. See [Troubleshooting](#troubleshooting).

---

## Deploy locally / on your own hardware

If you already run a GPU box (or want to test against a local master), no provider key is needed:

```bash
uv run base worker deploy --provider local
```

This starts a worker agent process on this host pointed at `worker.agent.master_url`, then polls
`GET /v1/workers` until the worker reaches `active` (bounded by `deploy.ready_timeout_seconds`,
default 60s). On success it prints:

```text
Started worker agent process pid=<pid> (provider=local)
Worker <worker-id> active (pubkey=<worker-pubkey>, owner=<miner-hotkey>, provider=local)
```

To run the long-lived agent loop directly (e.g. under a supervisor), use:

```bash
uv run base worker agent --config config/worker.example.yaml
```

---

## Costs and price guidance

**You pay for the instance.** The subnet does not reimburse GPU spend; running a worker is the
price of being submission-eligible and contributing compute.

- **Bound every deploy** with `--max-price` (per GPU/hour) or `worker.deploy.max_price_per_hour`;
  an all-over-cap situation fails with a clear "no offer within budget" error and provisions nothing.
- **Keep pod lifetime short.** `worker.deploy.max_lifetime_hours` maps to the provider's bounded
  termination window; keep it small (1–2h) and redeploy as needed.
- **Prefer single-GPU executors.** On Lium, `gpu_count=1` is the common case, and deploy planning
  prefers an executor whose GPU count matches your request.

**Lium price signal** (real USD, checked live at selection): prices are per GPU/hour — prefer
offers **< $1.50/GPU/hr** (observed: A100-SXM4-80GB ~$0.52/GPU/hr, RTX-class under $1/GPU/hr; a
short rent → run → delete cycle costs ~**$0.004**). Check balance with Lium's `GET /users/me`.

**Targon price signal** (from live inventory `cost_per_hour`): H100 ×1 ≈ **$2.50/hr**, H200 ×1 ≈
**$3.29/hr**, B200 ×1 ≈ **$5.30/hr** (multi-GPU shapes scale up; confidential-compute from
~$2.50/hr). No balance endpoint — track spend in the dashboard.

**Proof tier affects audit load, not price.** Lower-assurance proofs are replayed more often
(tier 0 ≈ 10%, tier 1 ≈ 5%, tier 2 ≈ 2%). A worker whose image digest matches the pinned evaluator
digest earns tier 1 and is audited less.

---

## Monitoring your fleet

Render the fleet the master knows about:

```bash
uv run base worker status
```

Output is one row per worker with its status, owner, provider, fault count, and last-seen time:

```text
WORKER_ID            OWNER                PROVIDER   STATUS   FAULTS LAST_SEEN
worker-abc123        5F...minerhotkey     lium       active   0      2026-07-07T14:30:00+00:00
```

Filter to just the **active** workers of one hotkey (this is the same query the admission rule
uses):

```bash
uv run base worker status --hotkey <your-miner-hotkey>
```

**Status lifecycle:** `pending → active → stale → retired`.

- `pending`: registered, awaiting its first heartbeat.
- `active`: verified binding **and** a heartbeat within the freshness window
  (`compute.worker_heartbeat_ttl_seconds`, default 120s). Only `active` workers get assignments and
  satisfy the admission rule.
- `stale`: no heartbeat within the TTL; stops receiving new units, returns to `active` on the next
  heartbeat.
- `retired`: terminal — re-enroll a fresh worker instead.

`FAULTS` counts audit-attributed faults (a worker whose manifest diverged from an authoritative
validator replay). The same fleet state is JSON at `GET /v1/workers`, field-for-field with
`base worker status`.

---

## Troubleshooting

**`provider '<lium|targon>' requires the <LIUM_API_KEY|TARGON_API_KEY> environment variable`
(exit 2).** The provider key is unset/blank; the CLI refuses **before** any provider or master
call, so nothing was provisioned. Export the key and retry.

**`unsupported provider '<x>'; expected one of local, lium, targon` (exit 2).** Use exactly one of
`local`, `lium`, or `targon`.

**`no rentable offer within budget (...); nothing was provisioned` (exit 1).** Every offer is above
your cap. Raise `--max-price` (or `worker.deploy.max_price_per_hour`) or retry when cheaper capacity
appears. Nothing was rented.

**`provider '<lium|targon>' deploy requires an explicit worker image ...` (exit 1).**
`worker.deploy.image` and/or `worker.deploy.image_digest` are unset. A provider deploy is
**fail-fast** (no provider call) without a **PUBLICLY-pullable, digest-pinned** image. Set both (or
`BASE_WORKER__DEPLOY__IMAGE` / `BASE_WORKER__DEPLOY__IMAGE_DIGEST`); the digest must be
`sha256:<64 hex>`. See [Publishing the worker image](#publishing-the-worker-image).

**Lium pod stuck at / fails with `CREATION_FAILED`.** The executor host could not pull your image —
almost always because it is in a **private** namespace. Publish it **publicly**, confirm an
unauthenticated `docker pull <image>@<digest>` succeeds, then redeploy.

**Lium: `403 Request blocked` when deploying.** Lium's WAF rejects any request body containing a
**loopback URL**. The CLI now omits loopback coordination URLs (master/broker/gateway) from the
`POST /templates` body automatically, and the agent resolves the master URL at runtime (a loopback
master is reachable via a reverse SSH tunnel). Set `worker.agent.master_url` to a **public** URL for
a normal remote deploy; keep the loopback default only for a local/tunnelled master.

**Targon balance is invisible / deploy fails on credits.** Targon exposes no balance/billing
endpoint — the value only lives in the web dashboard. An insufficient-credit error is raised as a
distinct typed error and is **not retried**; top up in the dashboard and retry (do not loop).

**Lium: `Malicious startup command detected ...` (HTTP 400).** Lium rejects template
`startup_commands` containing shell metacharacters (`&&`, `;`, `|`, quotes, …). Keep it a single
metachar-free command — the default `tail -f /dev/null` works (Lium's pod agent provides SSH
independently).

**Lium: renting a single GPU on a multi-GPU node returns HTTP 400.** Renting a partial slice is
rejected; prefer an executor whose GPU count equals your request (planning already prefers an
exact-count executor and falls back to the next-cheapest suitable offer).

**Lium requires an SSH key even for non-interactive workers.** Set
`worker.deploy.ssh_public_key_file` (or `ssh_public_key`) and, if needed, `ssh_key_name`, and
register the key with your Lium account first.

**Pod is slow to reach `RUNNING`.** Provisioning takes minutes; a cold image pull can take up to
~15 min. Poll `base worker status` until the worker shows `active`.

**`worker <pubkey> did not reach active within <N>s` (local deploy).** The agent never
enrolled/heartbeated in time. Check that `worker.agent.master_url` is reachable,
`compute.worker_plane_enabled` is on at the master, the miner hotkey is in the metagraph, and the
binding is valid; raise `worker.deploy.ready_timeout_seconds` for slow networks.

**Worker shows `stale`.** Heartbeats stopped longer than `compute.worker_heartbeat_ttl_seconds`
(default 120s). It returns to `active` on the next heartbeat; if not, check the agent process and
its connectivity.

**Submission rejected with `403 NO_ACTIVE_WORKER`.** The admission rule requires ≥1 **active**
worker bound to your hotkey. Deploy one, wait for `base worker status --hotkey <your-hotkey>` to
show it `active`, then resubmit.

**Leftover pod after a test.** Deploy always sets a bounded lifetime, but still confirm nothing is
left running — on Lium, `GET /pods` should return an empty list for your account.

> **Never commit or log your provider key.** Keep `LIUM_API_KEY` / `TARGON_API_KEY` in your
> environment only; the CLI never forwards them to the master and never writes them into a pod.
