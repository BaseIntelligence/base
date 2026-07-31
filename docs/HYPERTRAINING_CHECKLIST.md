# HYPERTRAINING checklist (task 17 freeze)

Maps each required pin from plan item 3 / xtask `hypertraining-check`
to a section heading in [`HYPERTRAINING.md`](./HYPERTRAINING.md).

CI: `cargo run -p xtask -- hypertraining-check` fails if any
marker is missing from `HYPERTRAINING.md`.

**Status:** FROZEN with the task 17 check binary. Pins below are the contract
for the freeze doc. Do not weaken pins without bumping
`challenge_scoring_version` when score-affecting.

| Pin | Requirement | Section heading | Anchor marker (must appear in spec) |
|-----|-------------|-----------------|-------------------------------------|
| (T) | Topology: miner owns train fork, validator owns measure | ## 1. What runs where (topology) | `## 1. What runs where (topology)` |
| (I) | Identifiers and versions | ## 2. Identifiers and versions | `## 2. Identifiers and versions` |
| (E) | Emission 0 bps posture | ## 2. Identifiers and versions | `emission_share_bps = 0` |
| (A) | Attestation profile-aware precondition | ## 3. Attestation precondition (profile-aware) | `## 3. Attestation precondition` |
| (P) | Miner submit protocol | ## 4. Miner submit protocol (challenge ↔ miner) | `## 4. Miner submit protocol` |
| (S) | Sealed surface summary | ## 5. Sealed surface summary | `## 5. Sealed surface summary` |
| (G) | Three guards summary | ## 6. Three guards summary | `## 6. Three guards summary` |
| (R) | Score meaning + integer map scoring_version 1 | ## 7. Score meaning and scoring rule (`challenge_scoring_version = 1`) | `## 7. Score meaning and scoring rule` |
| (K) | Key custody | ## 8. Key custody (challenge signing key) | `## 8. Key custody (challenge signing key)` |
| (D) | D24 participant set + NoScore | ## 9. Declared participant set and `NoScore` reasons (D24) | `## 9. Declared participant set and` |
| (L) | Leaf emission + gateway | ## 10. Leaf emission and gateway POST | `## 10. Leaf emission and gateway POST` |
| (C) | Compose ports / no Harbor | ## 11. Compose services, ports, image contract | `## 11. Compose services, ports, image contract` |
| (B) | ClusterBackend Sim now / Real later | ## 13. ClusterBackend contract | `## 13. ClusterBackend contract` |

## Extra pins verified by hypertraining-check

| Pin | Marker substring required in HYPERTRAINING.md |
|-----|-----------------------------------------------|
| challenge_id | `hypertraining` |
| challenge_id_field | `challenge_id` |
| scoring_version | `challenge_scoring_version` |
| scoring_version_1 | `u16 = 1` |
| bundle_protocol_version | `protocol_version = 1` |
| emission_zero | `emission_share_bps = 0` |
| agent_v1_bps | `10000` |
| SCORE_MAX | `1_000_000` |
| rawweight_domain | `base-rawweight-v1` |
| BUNDLE_SPEC_link | `BUNDLE_SPEC.md` |
| design_source | `challenge-training-fork.md` |
| sim_backend | `SimBackend` |
| real_backend_deferred | `RealBackend` |
| not_live_b300 | `Not live` |
| kernel_kappa | `κ = 2` |
| guard_1 | `Guard 1` |
| guard_2 | `Guard 2` |
| guard_3 | `Guard 3` |
| screen_k | `K=3` |
| promotion_k | `K=5` |
| binary_reject | `> 0.85` |
| D24_silence | `Silence is a bug` |
| no_llm_gate | `LLM is never a gate` |
| no_cuda_sandbox_security | `CUDA / container sandbox as security boundary` |
| no_harbor | `Harbor` |
| no_aws_required | `AWS` |
| compose_port | `8091` |
| te_pin | `2.18.0+e7c550c5` |
| mlm_commit | `cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54` |
| allowlist | `megatron/core/fusions/**` |
| denylist | `megatron/core/datasets/**` |
| marginal_delta | `Δ(candidate)` |
| branding_base | `product name is **base**` |
| task_id_domain | `base-hypertraining-task-id-v1` |
| task_blob_domain | `base-hypertraining-task-blob-v1` |
| answer_domain | `base-hypertraining-answer-v1` |
| receipt_domain | `base-hypertraining-receipt-v1` |

## Maintenance

When editing `HYPERTRAINING.md` headings, update this table and keep the
markers so `xtask hypertraining-check` stays green.

Do **not** rewrite [`AGENT_CHALLENGE.md`](./AGENT_CHALLENGE.md) or its checklist
for hypertraining work.
