//! `prism-registry` — architecture competition scoring + top-model publish.
//!
//! Two pieces, both master-side:
//!
//! - [`competition_scores`] — per-epoch emission math for the architecture
//!   competition (SCORE_MAX lattice preserved; exact rule documented in
//!   `docs/PRISM.md` § Architecture competition).
//! - [`TopModelPublisher`] — publishes each new global-best **lattice score**
//!   model (G2 benches under `scoring_version` 4 — never min-bpb alone) to
//!   the public `BaseIntelligence/prism` GitHub repo under `top-model/`, via a
//!   token read from a deploy secret file (`PRISM_TOPMODEL_GITHUB_TOKEN_FILE`;
//!   graceful no-op when absent).
//! - [`HfTopModelPublisher`] — same trigger, commits a reloadable custom-arch
//!   pack to HuggingFace (`PRISM_TOPMODEL_HF_TOKEN_FILE`; default repo
//!   `BaseIntelligence/top-prism-architecture`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

mod hf;
mod hooks;
mod publish;
mod weights;

// Emission competition math lives in the `prism-competition` sibling crate
// (LOC cap); re-exported here so callers keep the historical import path.
pub use hf::HfTopModelPublisher;
pub use hooks::post_score_hooks;
pub use prism_competition::{
    apply_emission, apply_emission_with, apply_owner_split, apply_significance, apply_top3_decay,
    apply_wta, competition_scores, contamination, emission_leaves, emission_leaves_with, evidence,
    frontier, owner_split_bps_from_env, paired, paired_evidence, paired_test, plan_emission, sig,
    sig_context, AxisScore, Direction, EliteArchive, EmissionMode, EmissionPlan, ExampleSeries,
    PairedInput, PairedOutcome, PairedRefusal, RunEvidence, SigContext, BAND_BPS, CHAMPION_BPS,
    CHAMPION_FLOOR_BPS, DEADZONE, DISPLACEMENT_METRICS, EXPLORE_POOL_BPS, MAX_EXPLORE_SLOTS,
    MIN_DECIDED, MIN_WIN_RATE_BPS, OWNER_ARCH_CREDIT_ENABLED, TOP3_DECAY_BPS,
};
pub use publish::{
    require_topmodel_weights, TopModelPublisher, TopModelRequest, TOPMODEL_REPO_PATH,
};
pub use weights::{CheckpointMeta, TOPMODEL_RELEASE_TAG};
