# AGENTS.md — BASE control plane

Short contract for agents and operators. Prefer linking over restating runbooks.

## Monorepo map

| Path | Role |
|------|------|
| `bins/` | Runnable processes (validator, gateway, updater, challenges, …) |
| `crates/` | Libraries shared by binaries |
| `xtask/` | Repo gates and maintenance tasks |
| `deploy/` | Compose matrix, Terraform, pins, secrets helpers, remote deploy |
| `docs/` | Architecture, frozen specs, runbooks, completeness |
| `config/` | Shared non-secret configuration |

Working branch: **`main`**. Prod ships from annotated tags `v*.*.*` cut on `main`.

## Non-negotiables

- **Digest-only images** in deploy paths — no floating tags in prod pins/compose.
- **Secrets** via age + files under `deploy/env/` / `deploy/secrets/` — never baked into images or cloud-init.
- **Gateway runs on master only** (`--profile master` / `role-master.yml`). Validators point at the master gateway over VPC.
- **`evil-gateway` is test-only** — never enable on prod hosts; assert with `deploy/scripts/assert-evil-gateway-not-default.sh`.
- Platform is **DigitalOcean Droplets + Docker Compose**, not App Platform / DOKS.

## Wallet / key roles (do not conflate)

| Key | Who | Needed for |
|-----|-----|------------|
| `gateway_sk` | Gateway | Bundle **seal** signatures (`POST /v1/admin/seal`) |
| `prism_sk` / `design_sk` / `bounty_sk` | Challenge / smoke | Signed leaves (`POST /v1/weights/raw`); pubs must match trust root |
| Gateway owner wallet + `BASE_GATEWAY_REQUIRE_OWNER` | Gateway | Master-only **identity** check (live/prod). **Not** required to seal or serve `/v1/weights/latest` |
| Validator wallet | Validator | On-chain weight **submit** only — validators *fetch* sealed weights; they do not need a gateway wallet |

`GET /v1/weights/latest` is **fail-closed**: with no sealed bundle (or decode error) the gateway serves a **burn vector** (uid 0 = 100%, `sealed: false`) rather than 404. A missing gateway wallet is unrelated.

## Challenge public docs (miner-facing repos)

Each live challenge has a **separate public GitHub repo** for miners. Those repos must contain **only** human miner documentation plus example / test harness code — **never** control-plane, gateway, validator, or orchestrator source. Public repos use a `docs/` layout (hero README + banner under `assets/`).

| Challenge | Public repo | Role |
|-----------|-------------|------|
| Design | [`BaseIntelligence/design-challenge`](https://github.com/BaseIntelligence/design-challenge) | Miner docs + baseline harness |
| Prism | [`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism) | Miner docs + recipe examples (publish / keep in sync; no control-plane code) |
| Bounty | *(public repo follow-up)* | Miner docs mirror in [`docs/external-miner/bounty.md`](docs/external-miner/bounty.md) until published |

Monorepo mirror for CI and operators: [`docs/external-miner/`](docs/external-miner/). Frozen contracts stay in this repo (`docs/DESIGN_CHALLENGE.md`, `docs/PRISM.md`, `docs/BOUNTY_CHALLENGE.md`, …).

**When a challenge product or public API changes**, agents **must** update:

1. The challenge’s public miner repo (README / examples), and
2. [`docs/external-miner/`](docs/external-miner/) in this monorepo as needed.

Do not leave miner-facing docs stale after shipping API, quota, round, or scoring changes.

## Challenge verification (mandatory path coverage)

When verifying a challenge (local-e2e, staging, or focused tests), **simulate a submission** end-to-end — do not stop at process healthz. Challenges evaluate on **master only**; the validator has **no challenge exec** (fetch sealed weights only).

1. Happy-path harness / intake POST (or equivalent) through the challenge service on master.
2. Edge / failure probes: bad harness, sanitize reject, quota, wrong routes/auth.
3. **Design — baseline:** submit the reference agent at [`docs/external-miner/examples/design-baseline/`](docs/external-miner/examples/design-baseline/) (`agent.py` + `pyproject.toml`). After `POST /v1/harness`, poll `GET /v1/runs/{id}` + `/events` + `/logs?since=` until `awaiting_admin` / terminal; assert `GET /v1/runs/{id}/pages` lists `index.html`, `pricing.html`, `components.html` and `GET /v1/view/{run_id}/{page}` returns **200**; probe `GET /v1/stats` and `GET /v1/dashboard`.
4. **Design — cheat:** submit a malicious/copy harness; expect agentic `cheat`/`suspicious` → `Score(0)` (not admin-eligible). Poll events/logs the same way.
5. **Design — admin winners:** with operator bearer (`deploy/secrets/design/annotator_tokens`), `GET /v1/admin/rounds/{id}/candidates` then `POST /v1/admin/rounds/{id}/winners` with 1 or 2 clean harness ids (`SCORE_MAX` or `SCORE_MAX/2`).
6. Leaf emission → `POST /v1/weights/raw` → seal → `GET /v1/weights/latest` with **`sealed: true`** (burn fallback alone is not a real seal).

**Never host Sim in staging/prod** — Docker sandbox only there. `SimSandbox` / `BASE_ALLOW_HOST_SIM=1` is CI/local opt-in only; do **not** treat stub pages (`sim-install-ok` / `sim-run-ok` without executing `agent.py`) as proof. Prefer `DESIGN_FORCE_SIM=false` + OpenRouter when `deploy/secrets/openrouter/api_key` is present.

Local smoke automates the weights seal step via `weights-smoke` inside `./deploy/scripts/local-e2e.sh --smoke` (see [`deploy/AGENTS.md`](deploy/AGENTS.md) and [`docs/runbooks/local-testnet-e2e.md`](docs/runbooks/local-testnet-e2e.md)).

## Required gates (before merge)

Match CI (`.github/workflows/ci.yml`):

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- tests + `cargo deny`
- `cargo run -p xtask -- loc-cap`
- `cargo run -p xtask -- consensus-lint`
- `cargo run -p xtask -- spec-check`
- `cargo run -p xtask -- design-check`
- `cargo run -p xtask -- external-docs-check`

## Where to read what

| Need | Start here |
|------|------------|
| System map / process topology | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Deploy / Compose / DO topology | [`deploy/README.md`](deploy/README.md) + [`deploy/AGENTS.md`](deploy/AGENTS.md) |
| **Local full-subnet test** (master+gateway+validator on testnet 541 + tunnel) | [`docs/runbooks/local-testnet-e2e.md`](docs/runbooks/local-testnet-e2e.md) · [`deploy/AGENTS.md`](deploy/AGENTS.md) § Local testnet E2E · `./deploy/scripts/local-e2e.sh --help` |
| Doc authority vs evidence | [`docs/AGENTS.md`](docs/AGENTS.md) |
| Component status | [`docs/COMPLETENESS.md`](docs/COMPLETENESS.md) |
| Frozen contracts | [`docs/BUNDLE_SPEC.md`](docs/BUNDLE_SPEC.md), [`docs/DESIGN_CHALLENGE.md`](docs/DESIGN_CHALLENGE.md), [`docs/PRISM.md`](docs/PRISM.md) |
| Miner HTTP submit | [`docs/external-miner/`](docs/external-miner/) · public: [design-challenge](https://github.com/BaseIntelligence/design-challenge), [prism](https://github.com/BaseIntelligence/prism) · bounty mirror: [`bounty.md`](docs/external-miner/bounty.md) |
| Threat / operator checklist | [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), [`docs/OPERATOR_SECURITY.md`](docs/OPERATOR_SECURITY.md) |

## Do not commit

- `deploy/env/*.env` (materialized secrets)
- `deploy/secrets/**` (except documented `README.md` placeholders)
- `deploy/terraform/*.tfstate*` / `terraform.tfvars` / local `.terraform/`
- Age identities, wallets, `receipt_sk`, `*.pem` / `*.key` / `*.age`
- Treating `docs/evidence/` or `docs/spikes/` as product code or normative spec
