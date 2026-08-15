//! PRISM recipe 2.0 — `AutoModel` pin metadata, fail-closed patch apply, diffstat,
//! and ZIP intake materialisation (`.prism/` diff artifacts for review APIs).
//!
//! # Typical flow (callers)
//!
//! 1. Resolve pin id → [`AutomodelPin`] ([`AUTOMODEL_PIN`] or [`FIXTURE_PIN`]).
//! 2. Optionally [`verify_pin_tree`] against a staged checkout.
//! 3. [`intake_automodel_zip`] / [`materialize`] with miner ZIP members.
//! 4. Persist [`MaterializedAutomodel`] files as `tree_blob`; serve
//!    [`diff_from_files`] on `GET /v1/submissions/{id}/diff`.
//!
//! CI uses the tiny tree under `fixtures/automodel-pin/` plus
//! `fixtures/patches/happy.patch` — never a full NVIDIA clone.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

mod apply;
mod classify;
mod env;
mod intake;
mod pin;
mod review;

pub use apply::{
    apply_patch, apply_patch_rust, validate_patch, ApplyError, ApplyResult, MAX_PATCH_BYTES,
    MAX_PATCH_FILES,
};
pub use classify::{
    classify_path, diffstat, DiffStat, DiffStatEntry, PathClass, ARCH_ROOTS, DATA_ROOTS,
    TRAINER_ROOTS,
};
pub use env::{
    DEFAULT_TRAIN_ENTRY, ENV_APPLIED_DIR, ENV_BASE, ENV_FIXTURE, ENV_FIXTURE_ROOT, ENV_PATCH_PATH,
    ENV_PIN_DIR, ENV_PRISM_TOML, INTAKE_TREE_BLOB_NOTE, POD_APPLIED_DIRNAME, POD_STAGE_MANIFEST,
};
pub use intake::{
    allow_fixture_intake, allow_legacy_intake, diff_from_files, expand_automodel,
    expand_tree_blob_for_pod, extract_automodel_zip, fixture_automodel_zip, intake_automodel_zip,
    materialize, pack_tree_blob, pin_checkout_dir, resolve_pin, submission_id_for_patch,
    zip_is_automodel_layout, AutomodelMembers, ExpandedAutomodel, IntakeError,
    MaterializedAutomodel, ALLOW_LEGACY_ENV, MEMBER_BASE, MEMBER_PATCH, MEMBER_PYPROJECT,
    MEMBER_REQUIREMENTS, MEMBER_TOML, META_BASE, META_DIFFSTAT, META_PATCH,
};
pub use pin::{
    fixture_happy_patch_path, fixture_pin_dir, tree_content_sha256, verify_pin_tree, AutomodelPin,
    AUTOMODEL_CONTENT_SHA256, AUTOMODEL_GIT_COMMIT, AUTOMODEL_GIT_TAG, AUTOMODEL_GIT_URL,
    AUTOMODEL_PIN, AUTOMODEL_PIN_ID, AUTOMODEL_RECIPE_VERSION, FIXTURE_CONTENT_SHA256, FIXTURE_PIN,
    FIXTURE_PIN_ID,
};
pub use review::{
    delta_surface_from_files, delta_surface_from_materialized, files_have_automodel_diff,
    materialize_delta_review, patch_sha256_hex, patch_text_from_tree_blob, DeltaReviewSurface,
    REVIEW_BRIEF_REL, REVIEW_DIFFSTAT_REL, REVIEW_PATCH_REL,
};
