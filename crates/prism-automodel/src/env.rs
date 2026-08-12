//! Shared env / pod layout names for intake ↔ harness coordination.
//!
//! Master (intake) materialises the pin + applied tree, then either:
//! - sets these env vars for a host/Sim harness run, or
//! - stages the same paths onto the Lium pod before `main.py` starts.
//!
//! The Python harness (`prismlib.automodel`) reads the same names.

/// Clean `AutoModel` pin checkout (content-sha verified when pin has a sha).
pub const ENV_PIN_DIR: &str = "PRISM_AUTOMODEL_PIN_DIR";

/// Already-applied tree (pin + miner patch). Preferred when intake applied on master.
pub const ENV_APPLIED_DIR: &str = "PRISM_AUTOMODEL_APPLIED_DIR";

/// Path to `automodel.patch` bytes (used when harness applies onto [`ENV_PIN_DIR`]).
pub const ENV_PATCH_PATH: &str = "PRISM_AUTOMODEL_PATCH_PATH";

/// Pin id echoed from `automodel.base` (`automodel@v0.5.0` / `automodel@fixture-v1`).
pub const ENV_BASE: &str = "PRISM_AUTOMODEL_BASE";

/// Optional path to miner `prism.toml` (entry / `model_config` knobs).
pub const ENV_PRISM_TOML: &str = "PRISM_AUTOMODEL_PRISM_TOML";

/// When `1`/`true`: CI/Sim fixture pin + happy patch. **Not** proof of a real `AutoModel` train
/// (same spirit as `DESIGN_FORCE_SIM` / host-sim stubs).
pub const ENV_FIXTURE: &str = "PRISM_AUTOMODEL_FIXTURE";

/// Override root containing `automodel-pin/` + `patches/happy.patch` (defaults to crate fixtures).
pub const ENV_FIXTURE_ROOT: &str = "PRISM_AUTOMODEL_FIXTURE_ROOT";

/// Pod-relative directory under `$PRISM_WORKDIR` where the harness stages the applied tree.
pub const POD_APPLIED_DIRNAME: &str = "automodel";

/// Stage manifest filename written under `$PRISM_WORKDIR`.
pub const POD_STAGE_MANIFEST: &str = ".prism_automodel.json";

/// Default train entry under the applied tree when `prism.toml` omits `entry`.
pub const DEFAULT_TRAIN_ENTRY: &str = "nemo_automodel/recipes/llm/train_ft.py";

/// Intake also packs [`crate::MaterializedAutomodel`] as `tree_blob` under
/// `submission/` (see `prism_tree::POD_SUBMISSION_DIR`). The harness copies
/// that tree into [`POD_APPLIED_DIRNAME`] when env paths are unset.
pub const INTAKE_TREE_BLOB_NOTE: &str =
    "tree_blob → $PRISM_WORKDIR/submission/ (nemo_automodel/ + .prism/); \
     harness stages → $PRISM_WORKDIR/automodel/";

/// Serialize `PRISM_AUTOMODEL_FIXTURE` mutations across parallel unit tests.
#[cfg(test)]
pub static FIXTURE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with [`ENV_FIXTURE`] forced on/off (test-only; process-wide lock).
#[cfg(test)]
pub fn with_fixture_env(on: bool, f: impl FnOnce()) {
    let _g = FIXTURE_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if on {
        std::env::set_var(ENV_FIXTURE, "1");
    } else {
        std::env::remove_var(ENV_FIXTURE);
    }
    f();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_names_stable_for_harness() {
        assert_eq!(ENV_PIN_DIR, "PRISM_AUTOMODEL_PIN_DIR");
        assert_eq!(ENV_APPLIED_DIR, "PRISM_AUTOMODEL_APPLIED_DIR");
        assert_eq!(ENV_FIXTURE, "PRISM_AUTOMODEL_FIXTURE");
        assert_eq!(
            DEFAULT_TRAIN_ENTRY,
            "nemo_automodel/recipes/llm/train_ft.py"
        );
        assert!(INTAKE_TREE_BLOB_NOTE.contains("submission/"));
        assert!(INTAKE_TREE_BLOB_NOTE.contains(POD_APPLIED_DIRNAME));
    }
}
