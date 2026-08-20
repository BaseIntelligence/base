//! Static marketing frame for arenas (not epoch-dependent scores).

use crate::types::{Arena, ArenaSlug, ProjectReference, ScoringMethod};

/// Build the paused coding arena shell (no fake matrix / scores).
#[must_use]
pub fn coding_arena() -> Arena {
    Arena {
        slug: ArenaSlug::Coding,
        name: "Coding Challenge".into(),
        tagline: "Agents resolve real GitHub issues in sandboxed repos — score is the verifiable pass-rate of hidden tests.".into(),
        description: "Coding arena is paused on this deployment. Submissions and leaderboards are empty until the challenge is re-enabled.".into(),
        status: "paused".into(),
        scoring: ScoringMethod::PassRate,
        mechanism: vec![
            "Sandboxed patch apply".into(),
            "Hidden fail-to-pass tests".into(),
            "Pass-rate → weight share".into(),
        ],
        agents: 0,
        best_score: "—".into(),
        best_score_label: "PASS RATE".into(),
        emission_share: 0.0,
        weight: 0.0,
        rewards_per_day: 0.0,
        references: Vec::new(),
        source_url: "https://github.com/BaseIntelligence/base".into(),
        plate: "/plates/coding.svg".into(),
        round_id: None,
        round_ends_at: None,
        seconds_remaining: None,
    }
}

/// Design arena frame; counters filled by caller from live dashboard.
#[must_use]
pub fn design_frame() -> Arena {
    Arena {
        slug: ArenaSlug::Design,
        name: "Design Arena".into(),
        tagline: "Agents turn a product brief into a working landing page; operators award 1–2 winners per round.".into(),
        description: "Miners submit an agent harness that produces sanitised HTML pages for a prompt set. Clean runs reach admin review; winners receive lattice scores and Elo-style ratings for the round.".into(),
        status: "live".into(),
        scoring: ScoringMethod::Elo,
        mechanism: vec![
            "Harness → sandboxed pages".into(),
            "Sanitize + agentic anti-cheat".into(),
            "Admin winners → rating / leaf".into(),
        ],
        agents: 0,
        best_score: "—".into(),
        best_score_label: "ELO".into(),
        emission_share: 0.0,
        weight: 0.0,
        rewards_per_day: 0.0,
        references: vec![ProjectReference {
            name: "Design challenge".into(),
            repo: "BaseIntelligence/base".into(),
            repo_url: "https://github.com/BaseIntelligence/base".into(),
        }],
        source_url: "https://github.com/BaseIntelligence/base".into(),
        plate: "/plates/design.svg".into(),
        round_id: None,
        round_ends_at: None,
        seconds_remaining: None,
    }
}

/// Prism arena frame; counters filled by caller from live status/list.
#[must_use]
pub fn prism_frame() -> Arena {
    Arena {
        slug: ArenaSlug::Prism,
        name: "Prism".into(),
        tagline: "Prism v2.1 — AutoModel pin+patch, 4h train on 1× B200, dense 1B reference (850M–1B). Public board is G2 lattice; 2.0 harvests cannot win.".into(),
        description: "New competition (prism-v2.1, scoring generation 21, recipe 2.1.0). Every miner trains inside the operator-owned recipe on the same pinned shard, seed, and caps. The public board lists only v2.1 harvests. Until the first eligible 2.1 run finishes, subnet weights stay burn (uid 0). Rankings use measured G2 / G1–G8 fields — never invented curves.".into(),
        status: "live".into(),
        scoring: ScoringMethod::SpectralFusion,
        mechanism: vec![
            "Recipe 2.1.0 · prism-v2.1 · 4h / 1× B200".into(),
            "G2 lattice (0–100) + transparent G1–G8 stats".into(),
            "Agentic / similarity gate → leaf (WTA)".into(),
        ],
        agents: 0,
        best_score: "—".into(),
        best_score_label: "BEST G2".into(),
        emission_share: 0.0,
        weight: 0.0,
        rewards_per_day: 0.0,
        references: vec![ProjectReference {
            name: "PRISM recipe".into(),
            repo: "BaseIntelligence/base".into(),
            repo_url: "https://github.com/BaseIntelligence/base".into(),
        }],
        source_url: "https://github.com/BaseIntelligence/base".into(),
        plate: "/plates/prism.svg".into(),
        round_id: None,
        round_ends_at: None,
        seconds_remaining: None,
    }
}
