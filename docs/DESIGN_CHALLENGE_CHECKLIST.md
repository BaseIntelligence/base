# DESIGN_CHALLENGE checklist

Maps each required pin from the design-challenge freeze to a section heading in
[`DESIGN_CHALLENGE.md`](./DESIGN_CHALLENGE.md).

CI: `cargo run -p xtask -- design-check` fails if any marker is missing from
`DESIGN_CHALLENGE.md`.

**Status:** FROZEN with the `design-check` binary. Do not weaken score-affecting
pins without bumping `challenge_scoring_version`.

| Pin | Requirement | Section heading | Anchor marker (must appear in spec) |
|-----|-------------|-----------------|-------------------------------------|
| (T) | Topology: master-only challenge + sandbox + egress; validator no exec | ## 1. What runs where (topology) | `## 1. What runs where (topology)` |
| (I) | Identifiers and versions | ## 2. Identifiers and versions | `## 2. Identifiers and versions` |
| (E) | Emission 0 bps (prism 100%) | ## 2. Identifiers and versions | `emission_share_bps = 0` |
| (H) | Miner harness contract (`agent.py` / `pyproject.toml`) | ## 3. Miner harness contract | `## 3. Miner harness contract` |
| (S) | Sandbox hardening | ## 4. Sandbox hardening | `## 4. Sandbox hardening` |
| (Z) | Sanitize rules | ## 5. Sanitize rules | `## 5. Sanitize rules` |
| (V) | Viewer headers / CSP sandbox | ## 6. Viewer headers and CSP | `## 6. Viewer headers and CSP` |
| (R) | Rounds 8_640s (10/day) + bank_v1 auto + split manual/scheduled quota | ## 7. Rounds and quotas | `## 7. Rounds and quotas` |
| (L) | Admin winners 1\|2 + rolling 10-round points share + AgenticReview | ## 8. Admin winners + agentic anti-cheat | `## 8. Admin winners + agentic anti-cheat` |
| (X) | Elimination bottom 20% + 10-round cooldown | ## 9. Elimination | `## 9. Elimination` |
| (D) | D24 exact-E participant set | ## 10. Declared participant set and `NoScore` reasons (D24) | `## 10. Declared participant set and` |
| (K) | Key custody | ## 11. Key custody (challenge signing key) | `## 11. Key custody (challenge signing key)` |
| (C) | Compose ports / image contract | ## 12. Compose services, ports, image contract | `## 12. Compose services, ports, image contract` |
| (A) | HTTP API surface | ## 13. HTTP API surface | `## 13. HTTP API surface` |

## Extra pins verified by design-check

| Pin | Marker substring required in DESIGN_CHALLENGE.md |
|-----|--------------------------------------------------|
| challenge_id | `design` |
| challenge_id_field | `challenge_id` |
| scoring_version | `challenge_scoring_version` |
| scoring_version_3 | `u16 = 3` |
| bundle_protocol_version | `protocol_version = 1` |
| emission_share | `emission_share_bps = 0` |
| bps_sum | `10000` |
| SCORE_MAX | `1_000_000` |
| compose_port | `8093` |
| round_secs | `8_640` |
| round_id_floor | `floor(unix_secs / 8640)` |
| rounds_per_day | `10 rounds` |
| agent_run_timeout | `AGENT_RUN_TIMEOUT_SECS = 1_800` |
| scoring_window | `SCORING_WINDOW_ROUNDS = 10` |
| daily_quota | `MANUAL_DAILY_RUN_QUOTA = 10` |
| scheduled_quota | `DESIGN_SCHEDULED_DAILY_RUN_CAP` |
| selfsim_excluded | `other hotkeys' and same-coldkey prior art only` |
| prompts_per_round | `1 prompt` |
| unscored_epoch_limit | `UNSCORED_EPOCH_LIMIT = 5` |
| admin_reject_route | `/v1/admin/rounds/{id}/reject` |
| metagraph_cache_ttl | `15m` |
| bank_v1 | `bank_v1.json` |
| agent_py | `agent.py` |
| pyproject | `pyproject.toml` |
| zip_submit | `application/zip` |
| env_vars | `env_vars` |
| required_pages | `index.html` |
| pricing_page | `pricing.html` |
| components_page | `components.html` |
| readonly_rootfs | `ReadonlyRootfs` |
| cap_drop | `CapDrop` |
| no_new_privileges | `no-new-privileges:true` |
| network_mode | `design-sandbox-egress` |
| name_prefix | `base-design-` |
| csp_sandbox | `sandbox; default-src 'none'` |
| html_never_served | `Produced HTML is never served` |
| agentic_review_stage | `AgenticReview` |
| admin_winners_route | `/v1/admin/rounds/{id}/winners` |
| admin_not_gateway | `not exposed via gateway` |
| window_share | `SCORE_MAX × p / total_window_points` |
| allowed_inspiration | `Mobbin` |
| master_only | `master-only` |
| no_prompt_validation | `no human prompt validation` |
| elimination_bps | `ELIMINATION_BOTTOM_BPS = 2000` |
| cooldown_rounds | `ELIMINATION_COOLDOWN_ROUNDS = 10` |
| bottom_20 | `bottom 20%` |
| D24_silence | `Silence is a bug` |
| exact_E | `emit_signed_leaf_set` |
| no_phala | `no Phala/CVM` |
| BUNDLE_SPEC_link | `BUNDLE_SPEC.md` |
| rawweight_domain | `base-rawweight-v1` |
| round_domain | `base-design-round-id-v1` |
| submission_domain | `base-design-submission-v1` |
| pair_domain | `base-design-pair-id-v1` |
| sim_sandbox | `SimSandbox` |
| ChallengeInternal | `ChallengeInternal` |
| NotAttempted | `NotAttempted` |
| run_logs_route | `/v1/runs/{id}/logs` |
| stats_route | `/v1/stats` |
| dashboard_route | `/v1/dashboard` |
| miners_route | `/v1/miners/{hotkey}` |
| poll_hint | `poll_hint_ms` |

## Maintenance

When editing `DESIGN_CHALLENGE.md` headings, update this table and keep the
markers so `xtask design-check` stays green.

Do **not** reintroduce agent-v1 / hypertraining / Phala CVM miner paths into this
freeze doc.
