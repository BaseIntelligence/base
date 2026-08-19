//! Lium GPU rental client for PRISM master-centralized eval.
//!
//! Miner-funded by default (`prism-lium-payer`); optional operator
//! `LIUM_API_KEY` when `PRISM_ALLOW_OPERATOR_LIUM=1`. Guardrails refuse
//! unbounded lifetime/price before rent; provision fails closed.
//! [`LiumClient`] talks to `https://lium.io/api`; [`SimLiumBackend`] is
//! offline CI. API keys and SSH material are never logged.

#![forbid(unsafe_code)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::missing_errors_doc,
    clippy::missing_fields_in_debug,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::redundant_closure_for_method_calls,
    clippy::duration_suboptimal_units,
    clippy::manual_clamp
)]

mod artifacts;
mod client;
mod sim;
mod ssh;

pub use artifacts::harvest_checkpoint_ssh;
pub use client::LiumClient;
pub use prism_artifacts::{
    artifact_dir_for, artifact_root, checkpoint_path_for, ensure_artifact_root,
    write_sim_checkpoint, MAX_CHECKPOINT_BYTES, POD_WORKDIR,
};
pub use prism_lium_harness::{
    classify_log, parse_harness_probe, parse_metrics_output, HarnessProbe, HarnessProgress,
    HARNESS_ABSENT, HARNESS_HARVEST_CMD, TRAIN_DONE_MARKER,
};
pub use sim::SimLiumBackend;
pub use ssh::{parse_ssh_target, resolve_private_key, truncate_tail, SshTarget};
// The data contract lives in `prism-lium-types` (per-crate LOC cap); it is
// re-exported wholesale so `prism_lium::…` stays the single import path.
pub use prism_lium_types::{
    effective_gpu_count, gpu_count_from_label, pod_gpu_count_from_env, pod_gpu_preference_from_env,
    CostGuardrailError, EvalReceipt, EvalTelemetry, GpuPreference, Instance, InstanceSpec,
    LiumError, LiumSshConfig, NoScoreGate, Offer, ProbePoint, RemoteExecResult, TelemetryPoint,
    DEFAULT_MAX_PRICE_PER_HOUR, DEFAULT_POD_GPU_COUNT,
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

    #[rustfmt::skip]
    async fn harvest_logs(&self, _: &str) -> Result<String, LiumError> { Ok(String::new()) }
    #[rustfmt::skip]
    async fn harvest_artifacts(&self, _: &str, _: &std::path::Path, _: &[u8], _: Option<u64>) -> Result<std::path::PathBuf, LiumError> {
        Err(LiumError::Exec("artifact harvest not supported on this backend".into()))
    }
    #[rustfmt::skip]
    async fn instance_running(&self, _: &str) -> Result<bool, LiumError> { Ok(false) }
    /// Reattach to a detached harness (no re-upload). `HARNESS_ABSENT` → fresh exec.
    #[rustfmt::skip]
    async fn resume_eval(&self, instance_id: &str) -> Result<RemoteExecResult, LiumError> {
        Err(LiumError::Exec(format!("{HARNESS_ABSENT}: resume unsupported on this backend ({instance_id})")))
    }
}

/// Tail bytes for harness stderr snippets (not the metrics sidecar).
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
