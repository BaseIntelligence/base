//! Agent container network egress posture (scoring_version 2).
//!
//! # Locked default
//!
//! **OPEN** — no network egress lockdown in the agent environment container.
//! Stripping protects **grading-channel integrity** (held-out tests/solution never
//! reach the miner), not miner honesty. Score honesty remains a D19 disclaimer;
//! public PR leakage and open egress are explicitly out of scope for attestation.
//!
//! An allowlisted egress proxy alternative exists as a config variant but is
//! **OFF by default** and must not be implied as the v2 baseline.

use serde::{Deserialize, Serialize};

/// How the pack environment container may reach the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentEgressPosture {
    /// Default v2: container networking left enabled (`NetworkDisabled=false`).
    Open,
    /// Optional future posture: traffic only via an allowlisted proxy.
    /// Not enabled by default; selecting it does not change the documented default.
    AllowlistedProxy,
}

/// Documented and tested default for agent-v1 / scoring_version 2.
pub const DEFAULT_AGENT_EGRESS_POSTURE: AgentEgressPosture = AgentEgressPosture::Open;

/// Spec / log label for the default posture (stable string for gates and tests).
pub const EGRESS_POSTURE_OPEN_LABEL: &str = "open";

impl AgentEgressPosture {
    /// Stable lowercase label (`open` / `allowlisted_proxy`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => EGRESS_POSTURE_OPEN_LABEL,
            Self::AllowlistedProxy => "allowlisted_proxy",
        }
    }

    /// Docker `NetworkDisabled` flag for this posture.
    ///
    /// `Open` and the default path leave networking **enabled** (`false`).
    /// `AllowlistedProxy` also keeps the stack interface up (proxy is in-path);
    /// full lockdown is intentionally not the v2 default.
    #[must_use]
    pub const fn network_disabled(self) -> bool {
        match self {
            Self::Open | Self::AllowlistedProxy => false,
        }
    }

    /// True when this is the locked v2 default.
    #[must_use]
    pub const fn is_default(self) -> bool {
        matches!(self, Self::Open)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_egress_is_open_and_not_network_disabled() {
        assert_eq!(DEFAULT_AGENT_EGRESS_POSTURE, AgentEgressPosture::Open);
        assert_eq!(DEFAULT_AGENT_EGRESS_POSTURE.as_str(), "open");
        assert!(!DEFAULT_AGENT_EGRESS_POSTURE.network_disabled());
        assert!(DEFAULT_AGENT_EGRESS_POSTURE.is_default());
        assert!(!AgentEgressPosture::AllowlistedProxy.network_disabled());
    }
}
