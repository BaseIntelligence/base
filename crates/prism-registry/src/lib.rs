//! `prism-registry` — architecture competition scoring + top-model publish.
//!
//! Two pieces, both master-side:
//!
//! - [`competition_scores`] — per-epoch emission math for the architecture
//!   competition (SCORE_MAX lattice preserved; exact rule documented in
//!   `docs/PRISM.md` § Architecture competition).
//! - [`TopModelPublisher`] — publishes each new global-best bpb model to the
//!   public `BaseIntelligence/prism` GitHub repo under `top-model/`, via a
//!   token read from a deploy secret file (`PRISM_TOPMODEL_GITHUB_TOKEN_FILE`;
//!   graceful no-op when absent).
//! - [`HfTopModelPublisher`] — same trigger, commits sources to HuggingFace
//!   (`PRISM_TOPMODEL_HF_TOKEN_FILE`; default repo
//!   `BaseIntelligence/prism-top-model`).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::module_name_repetitions)]

mod competition;
mod hf;
mod hooks;
mod publish;
mod weights;

pub use competition::{apply_wta, competition_scores, OWNER_ARCH_CREDIT_ENABLED};
pub use hf::HfTopModelPublisher;
pub use hooks::post_score_hooks;
pub use publish::{TopModelPublisher, TopModelRequest, TOPMODEL_REPO_PATH};
pub use weights::{CheckpointMeta, TOPMODEL_RELEASE_TAG};
