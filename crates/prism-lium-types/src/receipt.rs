//! Eval receipt + fail-closed integrity gates (no Phala CVM).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::LiumError;

/// Master-owned receipt for one Lium (or sim) eval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalReceipt {
    /// `lium` or `sim`.
    pub provider: String,
    /// Pod / instance id.
    pub pod_id: String,
    /// Image digest pin (empty for pure sim).
    pub image_digest: String,
    /// Hash of architecture+training sources.
    pub submission_hash: String,
    /// Hash of metrics payload.
    pub metrics_hash: String,
    /// Whether terminate was verified.
    pub termination_verified: bool,
}

impl EvalReceipt {
    /// SHA-256 hex of concatenated architecture + training sources.
    #[must_use]
    pub fn hash_submission(architecture_py: &str, training_py: &str) -> String {
        let mut h = Sha256::new();
        h.update(architecture_py.as_bytes());
        h.update([0xff]);
        h.update(training_py.as_bytes());
        hex::encode(h.finalize())
    }

    /// SHA-256 hex of a metrics JSON-ish string.
    #[must_use]
    pub fn hash_metrics_bytes(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        hex::encode(h.finalize())
    }
}

/// Fail-closed gates that force NoScore when integrity fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoScoreGate;

impl NoScoreGate {
    /// Validate receipt; Err → caller emits NoScore.
    ///
    /// # Errors
    /// Missing digest (when required), terminate not verified, empty submission/metrics hash.
    pub fn check(receipt: &EvalReceipt, require_image_digest: bool) -> Result<(), LiumError> {
        if receipt.submission_hash.is_empty() {
            return Err(LiumError::Integrity("empty submission_hash".into()));
        }
        if receipt.metrics_hash.is_empty() {
            return Err(LiumError::Integrity("empty metrics_hash".into()));
        }
        if require_image_digest && receipt.image_digest.is_empty() {
            return Err(LiumError::Integrity("missing image_digest pin".into()));
        }
        if !receipt.termination_verified {
            return Err(LiumError::Integrity(
                "termination_verified=false (orphan / leak risk)".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_rejects_unverified_terminate() {
        let r = EvalReceipt {
            provider: "sim".into(),
            pod_id: "p1".into(),
            image_digest: String::new(),
            submission_hash: "aa".into(),
            metrics_hash: "bb".into(),
            termination_verified: false,
        };
        assert!(NoScoreGate::check(&r, false).is_err());
    }

    #[test]
    fn gate_accepts_sim_without_digest() {
        let r = EvalReceipt {
            provider: "sim".into(),
            pod_id: "p1".into(),
            image_digest: String::new(),
            submission_hash: "aa".into(),
            metrics_hash: "bb".into(),
            termination_verified: true,
        };
        assert!(NoScoreGate::check(&r, false).is_ok());
    }

    #[test]
    fn submission_hash_stable() {
        let a = EvalReceipt::hash_submission("arch", "train");
        let b = EvalReceipt::hash_submission("arch", "train");
        let c = EvalReceipt::hash_submission("arch", "train2");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
