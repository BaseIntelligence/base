# AGENTS.md — docs

How to treat documentation in this repo.

## Normative vs non-normative

| Kind | Paths | Treat as |
|------|-------|----------|
| **Normative** | `ARCHITECTURE.md`, frozen specs (`BUNDLE_SPEC.md`, `DESIGN_CHALLENGE.md`, `PRISM.md`, …), `THREAT_MODEL.md`, `OPERATOR_SECURITY.md`, `COMPLETENESS.md`, `runbooks/`, `external-miner/` | Source of truth for contracts, ops, and status |
| **Non-normative** | `evidence/`, `spikes/` | Historical ops notes / experiments. **Do not** implement against them as spec; **do not** delete in cleanup passes without an explicit ops decision |

When a spike or evidence report conflicts with a frozen spec or runbook, the normative doc wins.

## Runbook index

| Runbook | Use when |
|---------|----------|
| [`runbooks/promote-rollback-restore.md`](runbooks/promote-rollback-restore.md) | Digest promote, rollback, Postgres backup/restore |
| [`runbooks/local-testnet-e2e.md`](runbooks/local-testnet-e2e.md) | Local laptop/VM full subnet stack on testnet 541 + ephemeral gateway tunnel |
| [`runbooks/staging-testnet-e2e.md`](runbooks/staging-testnet-e2e.md) | Staging droplet testnet end-to-end validation |
| [`runbooks/trust-root-rotation.md`](runbooks/trust-root-rotation.md) | Trust-root key rotation |
| [`runbooks/gateway-failover.md`](runbooks/gateway-failover.md) | Gateway kill/restart / failover checks |
| [`runbooks/measurement-repin-socket-proxy.md`](runbooks/measurement-repin-socket-proxy.md) | Socket-proxy measurement re-pin |
| [`runbooks/design-enable-and-emission.md`](runbooks/design-enable-and-emission.md) | Design keygen + emission unlock |
| [`runbooks/prism-enable-lium-and-emission.md`](runbooks/prism-enable-lium-and-emission.md) | Prism Lium + emission |

Deploy topology and CI lanes: [`../deploy/README.md`](../deploy/README.md) and [`../deploy/AGENTS.md`](../deploy/AGENTS.md).  
Repo-wide agent contract: [`../AGENTS.md`](../AGENTS.md).

## Challenge / local E2E verification

When updating challenge or local-subnet docs/runbooks, keep these invariants:

- **Master-only eval** — design/prism challenge services run on master; validator has **no challenge exec** (fetch sealed weights only).
- **Simulate submissions** — submit baseline **and** a cheat fixture through the challenge service; poll `/events` + `/logs`; do not treat `/health` alone as proof.
- **Design admin winners** — after clean `awaiting_admin`, operator bearer awards 1|2 winners; then leaf → seal path.
- **No host Sim in staging/prod** — Docker only; `SimSandbox` / `BASE_ALLOW_HOST_SIM` are CI/local opt-in.
- **Seal path** — `POST /v1/weights/raw` → seal → `GET /v1/weights/latest` ≠ 404. That path needs `challenge_sk` + `gateway_sk`, **not** a gateway owner wallet. Validator wallets are for on-chain submit only.
- Normative local procedure: [`runbooks/local-testnet-e2e.md`](runbooks/local-testnet-e2e.md). Repo contract: [`../AGENTS.md`](../AGENTS.md) § Challenge verification.
