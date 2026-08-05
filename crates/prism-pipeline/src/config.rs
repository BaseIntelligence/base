//! Orchestrator configuration.

/// Runtime knobs for the PRISM challenge service.
#[derive(Debug, Clone, PartialEq)]
pub struct PrismConfig {
    /// When true, real Lium path requires image digest pin on receipt.
    pub require_image_digest: bool,
    /// Max USD/hour for Lium offers.
    pub max_price_per_hour: f64,
    /// Max lifetime hours (≥ 1).
    pub max_lifetime_hours: f64,
    /// Global concurrent Lium evals (v1 = 1).
    pub max_concurrent_evals: u32,
    /// SSH public keys for Real Lium rent (empty for Sim).
    pub ssh_public_keys: Vec<String>,
    /// Optional image digest pin recorded on receipts.
    pub default_image_digest: Option<String>,
    /// Optional preferred Lium offer id (skip cheapest selection).
    pub preferred_offer_id: Option<String>,
    /// Optional Lium template id.
    pub template_id: Option<String>,
    /// Optional template name (default prism-mission-e2e for live).
    pub template_name: Option<String>,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self::sim()
    }
}

impl PrismConfig {
    /// Production profile (live Lium with digest pin when available).
    #[must_use]
    pub fn production() -> Self {
        Self {
            require_image_digest: false, // pin when operator supplies digest
            max_price_per_hour: 2.5,
            max_lifetime_hours: 2.0,
            max_concurrent_evals: 1,
            ssh_public_keys: vec![],
            default_image_digest: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: Some("prism-mission-e2e".into()),
        }
    }

    /// Live smoke profile (billable Lium, no image digest required).
    #[must_use]
    pub fn live_smoke() -> Self {
        Self {
            require_image_digest: false,
            max_price_per_hour: 1.5,
            max_lifetime_hours: 1.0,
            max_concurrent_evals: 1,
            ssh_public_keys: vec![],
            default_image_digest: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: Some("prism-mission-e2e".into()),
        }
    }

    /// Sim / unit-test profile.
    #[must_use]
    pub fn sim() -> Self {
        Self {
            require_image_digest: false,
            max_price_per_hour: 2.5,
            max_lifetime_hours: 1.0,
            max_concurrent_evals: 1,
            ssh_public_keys: Vec::new(),
            default_image_digest: None,
            preferred_offer_id: None,
            template_id: None,
            template_name: None,
        }
    }

    /// Attach SSH public keys for Real rent.
    #[must_use]
    pub fn with_ssh_public_keys(mut self, keys: Vec<String>) -> Self {
        self.ssh_public_keys = keys;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles() {
        assert!(!PrismConfig::sim().require_image_digest);
        assert!(!PrismConfig::production().require_image_digest);
        assert!(PrismConfig::default().max_lifetime_hours >= 1.0);
        assert!(PrismConfig::live_smoke().max_price_per_hour > 0.0);
    }
}
