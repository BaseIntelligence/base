# Agent Challenge Documentation

Agent Challenge rewards miners for building software engineering agents for the BASE subnet.
**Production evaluation is miner self-deploy on Phala Cloud Intel TDX CVMs** (attested review, then
attested eval). Review and eval guests run on Phala even when the validator host has no local TDX.
The validator is the trust root for dual measurement allowlists, RA-TLS golden AES-256 key release,
and score acceptance. Challenge logic lives here; [BASE](https://github.com/BaseIntelligence/base)
is the cross-repo hub (proxy, registry, proofs, R=1).

Start with the [project README](../README.md) for positioning, then use the audience table below.

## By audience

### Miners

| Guide | Contents |
| --- | --- |
| [Getting started](miner/getting-started.md) | **Day-1:** joinbase URLs, dashboard and/or `submit_agent.py`, Troubleshooting |
| [Miner hub](miner/README.md) | Reference: expectations, signing, scored path, BASE routes |
| [Submit agent](miner/submit-agent.md) | Package and sign the ZIP submission (A→Z) |
| [Self-deploy (how-to advanced)](miner/self-deploy.md) | Phala TDX review/eval CLI, encrypted_env, RESULT post, money, teardown |
| [Attestation TEE (concepts)](miner/attestation-tee.md) | Intel TDX, dual images, report_data domains, GetTlsKey, RA-TLS, trust-but-audit |

### Validators / operators

| Guide | Contents |
| --- | --- |
| [Validator hub](validator/README.md) | Role model: allowlist, key-release, operator controls (not scored-job deployer) |
| [Operator self-deploy](validator/self-deploy.md) | Production flags ON, dual allowlists, KR 8701 RA-TLS, CA roles, quote acceptance |

### Developers / integrators

| Guide | Contents |
| --- | --- |
| [Architecture](architecture.md) | End-to-end mermaid flows and trust domains |
| [Evaluation](evaluation.md) | Lifecycle, prepare/deploy/KR/score gate, status vocabulary, scoring |
| [Security](security.md) | Residual TEE.fail / pin-drift / provider risk, isolation, secrets |
| [Frontend API contract](frontend-api-contract.md) | Public routes, fields, 502 handling |
| [Behavior ledger](behavior-ledger.md) | Intentional code-truthful observations for maintainers |

## Production vs offline

| Mode | Flags | Scored path |
| --- | --- | --- |
| **Production** | `phala_attestation_enabled` and `attested_review_enabled` both ON | Miner self-deploy: review CVM then eval CVM; GetTlsKey + RA-TLS key release; direct attested RESULT |
| **Offline / compat** | Flags OFF (or mixed closed) | Local and CI without Phala; not production scoring |

Validators do **not** deploy production score jobs for miners. Broker `list_pending_work_units` style
execution is legacy relative to the production TEE path.

## Related root pointers

- [Project README](../README.md)
- License: Apache-2.0
