//! Lium GPU rental client for PRISM master-centralized eval.
//!
//! **No Phala CVM.** Live GPU eval is **miner-funded by default**: intake takes
//! `X-Lium-Api-Key` and rents via that key ([`PayerKeyVault`]). The operator
//! may still hold `LIUM_API_KEY` for Sim fallback / opt-in operator billing
//! (`PRISM_ALLOW_OPERATOR_LIUM=1`). Cost guardrails refuse unbounded lifetime /
//! price **before** any rent call. Every provision path terminates + verifies
//! on failure.
//!
//! # Backends
//!
//! * [`LiumClient`] — real HTTPS to `https://lium.io/api` (`X-API-Key`) + SSH exec.
//! * [`SimLiumBackend`] — offline deterministic metrics for CI (no network).
//!
//! # Secrets
//!
//! The API key is never logged, never placed in `Debug`/`Display`, never sent to
//! the pod environment. SSH private key path only; key material never logged.

#![forbid(unsafe_code)]
// Pedantic noise matching hypertraining-eval / thin HTTP wrappers.
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_fields_in_debug)] // api_key intentionally redacted
#![allow(clippy::too_many_lines)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::duration_suboptimal_units)]
#![allow(clippy::manual_clamp)]

mod artifacts;
mod client;
mod payer;
mod sim;
mod ssh;

pub use artifacts::harvest_checkpoint_ssh;
pub use client::LiumClient;
pub use payer::{
    allow_operator_lium_fallback, normalize_lium_api_key, require_miner_lium, PayerBackendFactory,
    PayerKeyVault, LIUM_API_KEY_HEADER,
};
pub use prism_artifacts::{
    artifact_dir_for, artifact_root, checkpoint_path_for, ensure_artifact_root,
    write_sim_checkpoint, MAX_CHECKPOINT_BYTES, POD_WORKDIR,
};
pub use sim::SimLiumBackend;
pub use ssh::{parse_ssh_target, resolve_private_key, truncate_tail, SshTarget};
// The data contract lives in `prism-lium-types` (per-crate LOC cap); it is
// re-exported wholesale so `prism_lium::…` stays the single import path.
pub use prism_lium_types::{
    CostGuardrailError, EvalReceipt, EvalTelemetry, GpuPreference, Instance, InstanceSpec,
    LiumError, LiumSshConfig, NoScoreGate, Offer, ProbePoint, RemoteExecResult, TelemetryPoint,
};

use async_trait::async_trait;

/// Master-side eval job backend (Sim or Real Lium).
#[async_trait]
pub trait EvalJobBackend: Send + Sync {
    /// List rentable offers (filtered by max price when set).
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError>;

    /// Provision under cost guardrails; fail-closed cleanup on error.
    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError>;

    /// Terminate (idempotent).
    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError>;

    /// True when the instance is absent from the provider.
    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError>;

    /// Sealed eval on the instance (`tree_blob` = v3 staged tree, else `None`).
    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError>;

    async fn harvest_logs(&self, _instance_id: &str) -> Result<String, LiumError> {
        Ok(String::new())
    }

    async fn harvest_artifacts(
        &self,
        _instance_id: &str,
        _dest_dir: &std::path::Path,
        _seed: &[u8],
        _n_params: Option<u64>,
    ) -> Result<std::path::PathBuf, LiumError> {
        Err(LiumError::Exec(
            "artifact harvest not supported on this backend".into(),
        ))
    }
}

/// Tail bytes retained for harness stderr / harvested logs in error_detail.
pub const HARNESS_LOG_RETAIN_BYTES: usize = 32_768;
/// Default Lium API base URL.
pub const LIUM_API_BASE_URL: &str = "https://lium.io/api";
/// Floor for `max_lifetime_hours` (Lium `termination_hours` is 1h granularity).
pub const MIN_LIFETIME_HOURS: f64 = 1.0;

/// Serializes tests that mutate `PRISM_EVAL_ASSETS_DIR` (client + sim).
#[cfg(test)]
pub(crate) static ASSETS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifetime_floor_is_one_hour() {
        let floor = MIN_LIFETIME_HOURS;
        assert!((floor - 1.0).abs() < f64::EPSILON || floor > 1.0);
    }
}
