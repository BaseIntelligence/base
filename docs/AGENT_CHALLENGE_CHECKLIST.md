# AGENT_CHALLENGE checklist (task 9 / task 16 re-freeze)

Maps each required pin from plan item 9 (topology / protocol / score / keys / D24 /
compose / attestation) to a section heading in
[`AGENT_CHALLENGE.md`](./AGENT_CHALLENGE.md).

CI: `cargo run -p xtask -- agent-challenge-check` fails if any marker is missing
from `AGENT_CHALLENGE.md`.

**scoring_version 2 pin delta (task 16):** live pins track pack domains, `model.patch`,
`ChallengeInternal`, work receipt, and v2 fixture digests. Historical v1
`SOFT_MS`/`HARD_MS` latency pins are **removed** (may appear only as retired prose).

| Pin | Requirement (plan task 9) | Section heading | Anchor marker (must appear in spec) |
|-----|---------------------------|-----------------|-------------------------------------|
| (T) | What runs where (miner CVM vs challenge vs validator) | ## 1. What runs where (topology) | `## 1. What runs where (topology)` |
| (P) | Request/response protocol challenge↔miner CVM | ## 4. Challenge ↔ miner CVM protocol | `## 4. Challenge ↔ miner CVM protocol` |
| (S) | Score meaning + concrete scoring rule | ## 5. Score meaning and scoring rule (`challenge_scoring_version = 2`) | `## 5. Score meaning and scoring rule` |
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
| scoring_version_2 | `u16 = 2` |
| SCORE_MAX | `1_000_000` |
| task_id_v2_domain | `gbase-agent-task-id-v2` |
| task_blob_v2_domain | `gbase-agent-task-blob-v2` |
| answer_v2_domain | `gbase-agent-answer-v2` |
| pack_select_domain | `gbase-agent-pack-select-v1` |
| work_receipt_domain | `gbase-agent-work-receipt-v1` |
| model_patch | `model.patch` |
| ChallengeInternal | `ChallengeInternal` |
| HarborVerifier | `HarborVerifier` |
| fixture_task_id_v2 | `b1c18e56abe993e20e8dadcb72c7a7cadee8975e5741d15d1acb37f5ea367644` |
| fixture_answer_v2 | `703b806158d655e5d37a5b45e3cbdf1e04735517805377199d108ae2a45ead5d` |
| attestation_precondition | `precondition for emitting` |
| NoScore_attestation | `AttestationNotVerified` |
| D24_silence | `Silence is a bug` |
| compose_agent_port | `8080` |
| compose_challenge_port | `8090` |
| image_agent | `ghcr.io/baseintelligence/base/gbase-agent` |
| no_latest | `:latest` |
| socket_proxy | `socket-proxy` |
| BUNDLE_SPEC_link | `BUNDLE_SPEC.md` |
| D10_report_data | `gbase-attest-v1` |
| rawweight_domain | `gbase-rawweight-v1` |
| PARTICIPATION_FLOOR | `PARTICIPATION_FLOOR` |
| F7_parked | `NoScore(AttestationNotVerified)` |
| no_score_in_cvm | `NO challenge signing key` |
| park_no_credit | `Park grants` |
| agent_egress_open | `Default: OPEN` |
| metis_b6_residual | `Metis B6 residual risk` |
| model_key_q3a | `Q3=A` |

## Maintenance

When editing `AGENT_CHALLENGE.md` headings, update this table and keep the
markers so `xtask agent-challenge-check` stays green. Pin deltas for scoring
version bumps land in the same commit as the spec rewrite.