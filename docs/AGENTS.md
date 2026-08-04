# AGENTS.md — docs

How to treat documentation in this repo.

## Normative vs non-normative

| Kind | Paths | Treat as |
|------|-------|----------|
| **Normative** | `ARCHITECTURE.md`, frozen specs (`BUNDLE_SPEC.md`, `AGENT_CHALLENGE.md`, …), `THREAT_MODEL.md`, `OPERATOR_SECURITY.md`, `COMPLETENESS.md`, `runbooks/`, `external-miner/` | Source of truth for contracts, ops, and status |
| **Non-normative** | `evidence/`, `spikes/` | Historical ops notes / experiments. **Do not** implement against them as spec; **do not** delete in cleanup passes without an explicit ops decision |

When a spike or evidence report conflicts with a frozen spec or runbook, the normative doc wins.

## Runbook index

| Runbook | Use when |
|---------|----------|
| [`runbooks/promote-rollback-restore.md`](runbooks/promote-rollback-restore.md) | Digest promote, rollback, Postgres backup/restore |
| [`runbooks/staging-testnet-e2e.md`](runbooks/staging-testnet-e2e.md) | Staging testnet end-to-end validation |
| [`runbooks/trust-root-rotation.md`](runbooks/trust-root-rotation.md) | Trust-root key rotation |
| [`runbooks/gateway-failover.md`](runbooks/gateway-failover.md) | Gateway kill/restart / failover checks |
| [`runbooks/measurement-repin-socket-proxy.md`](runbooks/measurement-repin-socket-proxy.md) | Socket-proxy measurement re-pin |
| [`runbooks/hypertraining-enable-real-and-emission.md`](runbooks/hypertraining-enable-real-and-emission.md) | Hypertraining real backend + emission |
| [`runbooks/prism-enable-lium-and-emission.md`](runbooks/prism-enable-lium-and-emission.md) | Prism Lium + emission |

Deploy topology and CI lanes: [`../deploy/README.md`](../deploy/README.md) and [`../deploy/AGENTS.md`](../deploy/AGENTS.md).  
Repo-wide agent contract: [`../AGENTS.md`](../AGENTS.md).
