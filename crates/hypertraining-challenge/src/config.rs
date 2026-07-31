//! Orchestrator configuration.

/// Runtime knobs for the hypertraining challenge service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypertrainingConfig {
    /// When `true` (production default), missing/non-verified attestation yields
    /// `NoScore(AttestationNotVerified)` before any Score is emitted.
    ///
    /// Sim / offline tests set this to `false` so fixture miners can score without
    /// a live attestation control plane.
    pub require_attestation: bool,
}

impl Default for HypertrainingConfig {
    fn default() -> Self {
        Self {
            require_attestation: true,
        }
    }
}

impl HypertrainingConfig {
    /// Production profile: attestation required.
    #[must_use]
    pub const fn production() -> Self {
        Self {
            require_attestation: true,
        }
    }

    /// Sim / unit-test profile: attestation gate off.
    #[must_use]
    pub const fn sim() -> Self {
        Self {
            require_attestation: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_requires_attestation() {
        assert!(HypertrainingConfig::default().require_attestation);
        assert!(HypertrainingConfig::production().require_attestation);
        assert!(!HypertrainingConfig::sim().require_attestation);
    }
}
