//! Pluggable per-challenge funding policy.

use async_trait::async_trait;

use crate::error::FundingError;
use crate::types::FundingEconomics;

/// Challenge-specific eligibility + economics.
#[async_trait]
pub trait ChallengeFundingPolicy: Send + Sync {
    /// Stable challenge id (`prism`, `design`, …).
    fn challenge_id(&self) -> &str;

    /// Quote economics.
    fn economics(&self) -> FundingEconomics;

    /// Reject ineligible hotkeys before issuing a quote.
    async fn ensure_eligible(&self, hotkey: &str) -> Result<(), FundingError>;

    /// When true, at most one unspent/spent credit history blocks new quotes
    /// after a credit was already granted (one-funding-per-hotkey).
    fn one_funding_per_hotkey(&self) -> bool {
        true
    }
}

/// Always-eligible policy for tests / Design scaffolding.
#[derive(Debug, Clone)]
pub struct OpenFundingPolicy {
    /// Challenge id.
    pub challenge_id: String,
    /// Economics.
    pub economics: FundingEconomics,
}

#[async_trait]
impl ChallengeFundingPolicy for OpenFundingPolicy {
    fn challenge_id(&self) -> &str {
        &self.challenge_id
    }

    fn economics(&self) -> FundingEconomics {
        self.economics
    }

    async fn ensure_eligible(&self, _hotkey: &str) -> Result<(), FundingError> {
        Ok(())
    }
}

/// Hotkey membership + prior-submission checks injected by the challenge.
#[async_trait]
pub trait EligibilityChecker: Send + Sync {
    /// Return `Ok(())` when the hotkey may receive funding.
    async fn ensure_eligible(&self, hotkey: &str) -> Result<(), FundingError>;
}

/// Prism policy: economics defaults + injected eligibility.
#[derive(Clone)]
pub struct PrismFundingPolicy {
    economics: FundingEconomics,
    eligibility: std::sync::Arc<dyn EligibilityChecker>,
}

impl PrismFundingPolicy {
    /// Prism defaults with a custom eligibility checker.
    #[must_use]
    pub fn new(eligibility: std::sync::Arc<dyn EligibilityChecker>) -> Self {
        Self {
            economics: FundingEconomics::prism_default(),
            eligibility,
        }
    }

    /// Override economics (env knobs).
    #[must_use]
    pub fn with_economics(mut self, economics: FundingEconomics) -> Self {
        self.economics = economics;
        self
    }
}

#[async_trait]
impl ChallengeFundingPolicy for PrismFundingPolicy {
    fn challenge_id(&self) -> &'static str {
        "prism"
    }

    fn economics(&self) -> FundingEconomics {
        self.economics
    }

    async fn ensure_eligible(&self, hotkey: &str) -> Result<(), FundingError> {
        self.eligibility.ensure_eligible(hotkey).await
    }

    fn one_funding_per_hotkey(&self) -> bool {
        true
    }
}

/// Test checker: allowlist of hotkeys.
#[derive(Debug, Default)]
pub struct AllowlistEligibility {
    /// Allowed hotkeys.
    pub allowed: std::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl EligibilityChecker for AllowlistEligibility {
    async fn ensure_eligible(&self, hotkey: &str) -> Result<(), FundingError> {
        let g = self
            .allowed
            .lock()
            .map_err(|_| FundingError::Store("eligibility lock".into()))?;
        if g.iter().any(|h| h == hotkey) {
            Ok(())
        } else {
            Err(FundingError::Ineligible(
                "hotkey not registered / not eligible".into(),
            ))
        }
    }
}
