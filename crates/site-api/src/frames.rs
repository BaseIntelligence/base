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
    }
}

/// Prism arena frame; counters filled by caller from live status/list.
#[must_use]
pub fn prism_frame() -> Arena {
    Arena {
        slug: ArenaSlug::Prism,
        name: "Prism".into(),
        tagline: "Agents propose neural architectures and train them on a sealed data window — score is final validation loss.".into(),
        description: "Every miner trains inside the operator-owned recipe on the same pinned shard, seed, and caps. Rankings use real BPB / lattice scores from terminated runs — never invented curves.".into(),
        status: "live".into(),
        scoring: ScoringMethod::SpectralFusion,
        mechanism: vec![
            "Pinned recipe + dataset".into(),
            "Lium (or sim) train + val BPB".into(),
            "Agentic / similarity gate → leaf".into(),
        ],
        agents: 0,
        best_score: "—".into(),
        best_score_label: "BPB".into(),
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
    }
}
