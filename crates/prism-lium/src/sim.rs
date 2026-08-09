//! Offline Sim Lium backend — deterministic metrics, no network.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::{EvalJobBackend, MIN_LIFETIME_HOURS};
use prism_lium_types::{CostGuardrailError, LiumError};
use prism_lium_types::{Instance, InstanceSpec, Offer, RemoteExecResult};

/// In-memory sim backend for CI / e2e without billable Lium.
pub struct SimLiumBackend {
    pods: Mutex<HashMap<String, bool>>,
}

impl Default for SimLiumBackend {
    fn default() -> Self {
        Self {
            pods: Mutex::new(HashMap::new()),
        }
    }
}

impl SimLiumBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn validate_spec(spec: &InstanceSpec) -> Result<(), CostGuardrailError> {
        if spec.max_lifetime_hours <= 0.0 {
            return Err(CostGuardrailError::LifetimeMissing);
        }
        if spec.max_lifetime_hours < MIN_LIFETIME_HOURS {
            return Err(CostGuardrailError::LifetimeBelowFloor);
        }
        if spec.max_price_per_hour <= 0.0 {
            return Err(CostGuardrailError::PriceMissing);
        }
        Ok(())
    }
}

#[async_trait]
impl EvalJobBackend for SimLiumBackend {
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        let mut offers = vec![
            Offer {
                id: "sim-blackwell".into(),
                gpu_type: "NVIDIA RTX BLACKWELL B200".into(),
                gpu_count: 1,
                price_per_hour: 1.2,
                provider: "sim".into(),
            },
            Offer {
                id: "sim-a100".into(),
                gpu_type: "NVIDIA A100-SXM4-80GB".into(),
                gpu_count: 1,
                price_per_hour: 0.5,
                provider: "sim".into(),
            },
        ];
        if let Some(max) = max_price_per_hour {
            offers.retain(|o| o.price_per_hour <= max);
        }
        Ok(offers)
    }

    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError> {
        Self::validate_spec(spec)?;
        let offers = self.list_offers(Some(spec.max_price_per_hour)).await?;
        if offers.is_empty() {
            return Err(CostGuardrailError::NoCapacity.into());
        }
        let id = format!("sim-pod-{}", spec.name);
        self.pods
            .lock()
            .map_err(|_| LiumError::Api("sim lock poisoned".into()))?
            .insert(id.clone(), true);
        Ok(Instance {
            id,
            status: "RUNNING".into(),
            provider: "sim".into(),
            gpu_type: Some("SIM".into()),
            ssh_connect_cmd: None,
        })
    }

    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError> {
        self.pods
            .lock()
            .map_err(|_| LiumError::Api("sim lock poisoned".into()))?
            .remove(instance_id);
        Ok(())
    }

    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError> {
        let map = self
            .pods
            .lock()
            .map_err(|_| LiumError::Api("sim lock poisoned".into()))?;
        Ok(!map.contains_key(instance_id))
    }

    async fn exec_eval(
        &self,
        _instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        // Deterministic fake BPB from code hash: lower is better quality for PRISM.
        let mut h = Sha256::new();
        h.update(architecture_py.as_bytes());
        h.update(training_py.as_bytes());
        if let Some(b) = tree_blob {
            h.update(b);
        }
        let dig = h.finalize();
        let n = u64::from_be_bytes(dig[0..8].try_into().unwrap_or([0; 8]));
        // Map to BPB in (1.0, 5.0)
        let bpb = 1.0 + (n as f64 % 4000.0) / 1000.0;
        // Synthetic decaying loss series so sim e2e exercises the same
        // telemetry plumbing (store rows, site-api series) as real Lium runs.
        let start = bpb + 2.0;
        let loss_series = (1..=5u64)
            .map(|step| prism_lium_types::TelemetryPoint {
                step,
                loss: start - (start - bpb) * (step as f64 / 5.0),
                grad_norm: Some(1.0 / step as f64),
                at_secs: Some(step as f64 * 2.0),
                layer_stats: None,
            })
            .collect();
        // Optional fake-assets mode (E5): when the operator eval-assets env
        // is set master-side, the sim mirrors the two-phase outcome
        // (private tier) so staging control-flow is testable offline.
        // Default stays v1-shaped.
        let fake_assets = std::env::var("PRISM_EVAL_ASSETS_DIR")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        let note_suffix = ["", " (fake private assets staged)"][usize::from(fake_assets)];
        Ok(RemoteExecResult {
            bpb,
            // Budget-plausible shape (not a stub): a 12M model early-stopping
            // ~13 min into a capped run, ~1M tokens (2048 rows × 512 ctx).
            // Reviewers (LLM or human) must see internally consistent metrics.
            tokens_seen: 1_048_576,
            wall_clock_seconds: 780.0,
            gpu_type: Some("SIM".into()),
            notes: format!("sim-eval{note_suffix}"),
            n_params: Some(12_000_000),
            val_rows: Some(256),
            telemetry: Some(prism_lium_types::EvalTelemetry {
                loss_series,
                finish_reason: Some("finish_evaluation".into()),
                report_count: 5,
            }),
            // Sim stays v1-shaped: v2 fields are exercised by the receipt
            // parsing tests and real Lium runs, not the offline backend.
            metrics_version: None,
            tokens_seen_source: None,
            probe_curve: None,
            pod_manifest: None,
            netns: None,
            harness_files_sha256: None,
            eval_tier: fake_assets.then_some("private".into()),
            extra: std::collections::BTreeMap::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn sim_lifecycle_finally_terminate() {
        let b = SimLiumBackend::new();
        let spec = InstanceSpec {
            name: "t1".into(),
            max_lifetime_hours: 1.0,
            max_price_per_hour: 2.0,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec![],
            ssh_key_name: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        };
        let inst = b.provision(&spec).await.unwrap();
        assert!(!b.verify_terminated(&inst.id).await.unwrap());
        let r = b
            .exec_eval(
                &inst.id,
                "def build_model(c): pass",
                "def train(m,c): pass",
                None,
            )
            .await
            .unwrap();
        assert!(r.bpb > 1.0);
        b.terminate(&inst.id).await.unwrap();
        assert!(b.verify_terminated(&inst.id).await.unwrap());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)] // env mutation must be serialized across the awaits
    async fn sim_fake_assets_mode_mirrors_private_tier() {
        let _guard = crate::ASSETS_ENV_LOCK.lock().unwrap();
        let b = SimLiumBackend::new();
        std::env::remove_var("PRISM_EVAL_ASSETS_DIR");
        let r = b
            .exec_eval(
                "pod",
                "def build_model(c): pass",
                "def train(m,c): pass",
                None,
            )
            .await
            .unwrap();
        assert!(r.eval_tier.is_none());
        assert_eq!(r.notes, "sim-eval");
        std::env::set_var("PRISM_EVAL_ASSETS_DIR", "/tmp/sim-assets");
        let r = b
            .exec_eval(
                "pod",
                "def build_model(c): pass",
                "def train(m,c): pass",
                None,
            )
            .await
            .unwrap();
        assert_eq!(r.eval_tier.as_deref(), Some("private"));
        assert!(r.notes.contains("fake private assets"));
        std::env::remove_var("PRISM_EVAL_ASSETS_DIR");
    }

    #[tokio::test]
    async fn sim_refuses_subhour_lifetime() {
        let b = SimLiumBackend::new();
        let spec = InstanceSpec {
            name: "t".into(),
            max_lifetime_hours: 0.5,
            max_price_per_hour: 1.0,
            gpu_count: 1,
            image_digest: None,
            ssh_public_keys: vec![],
            ssh_key_name: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        };
        let e = b.provision(&spec).await.unwrap_err();
        assert!(matches!(
            e,
            LiumError::Cost(CostGuardrailError::LifetimeBelowFloor)
        ));
    }
}
