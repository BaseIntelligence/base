# AGENT_CHALLENGE checklist (task 9)

Maps each required pin from plan item 9 to a section heading in
[`AGENT_CHALLENGE.md`](./AGENT_CHALLENGE.md).

CI: `cargo run -p xtask -- agent-challenge-check` fails if any marker is missing
from `AGENT_CHALLENGE.md`.

| Pin | Requirement (plan task 9) | Section heading | Anchor marker (must appear in spec) |
|-----|---------------------------|-----------------|-------------------------------------|
| (T) | What runs where (miner CVM vs challenge vs validator) | ## 1. What runs where (topology) | `## 1. What runs where (topology)` |
| (P) | Request/response protocol challenge↔miner CVM | ## 4. Challenge ↔ miner CVM protocol | `## 4. Challenge ↔ miner CVM protocol` |
| (S) | Score meaning + concrete scoring rule | ## 5. Score meaning and scoring rule (`challenge_scoring_version = 1`) | `## 5. Score meaning and scoring rule` |
| (K) | Key custody for challenge signing key | ## 6. Key custody (challenge signing key) | `## 6. Key custody (challenge signing key)` |
| (D) | Declared participant set + NoScore reasons (D24) | ## 7. Declared participant set and `NoScore` reasons (D24) | `## 7. Declared participant set and` |
| (C) | Compose services, ports, image contract (tasks 37/38) | ## 9. Compose services, ports, image contract (tasks 37/38) | `## 9. Compose services, ports, image contract` |
| (A) | Verified attestation precondition stated explicitly | ## 3. Attestation precondition (explicit) | `## 3. Attestation precondition (explicit)` |

## Extra pins verified by agent-challenge-check

| Pin | Marker substring required in AGENT_CHALLENGE.md |
|-----|--------------------------------------------------|
| bundle_protocol_version | `protocol_version = 1` |
| challenge_id | `agent-v1` |
| scoring_version | `challenge_scoring_version` |
| SCORE_MAX | `1_000_000` |
| SOFT_MS | `SOFT_MS` |
| HARD_MS | `HARD_MS` |
| fixture_task_id | `4a590b2abf87da6bccd97d8fbe5d2e774bdbda3ad421119688010537be2b31ec` |
| fixture_answer | `83180b08e05630496531a158d174ce69ba857d854d8692087947706c159a487c` |
| attestation_precondition | `precondition for emitting` |
| NoScore_attestation | `AttestationNotVerified` |
| D24_silence | `Silence is a bug` |
| compose_agent_port | `8080` |
| compose_challenge_port | `8090` |
| image_agent | `ghcr.io/baseintelligence/gbase-agent` |
| no_latest | `:latest` |
| BUNDLE_SPEC_link | `BUNDLE_SPEC.md` |
| D10_report_data | `gbase-attest-v1` |
| rawweight_domain | `gbase-rawweight-v1` |

## Maintenance

When editing `AGENT_CHALLENGE.md` headings, update this table and keep the
markers so `xtask agent-challenge-check` stays green.
