//! Graduated sanctions (brief §12.7).

/// Graduated anti-noise sanction (distinct from honest failure).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sanction {
    /// High similarity, no measure / no penalty stake.
    SilentReject,
    /// High-ish similarity: escalate promotion K (and attribution may be required later).
    EscalateK {
        /// Promotion K to use.
        k: u32,
    },
    /// Source rewritten, binary identical — demonstrated intent (slash stub flag).
    SlashIntent,
    /// Fingerprint already submitted by miner within cooldown.
    DedupeReject {
        /// Last accepted segment for this fingerprint.
        last_segment: u64,
        /// Cooldown window N.
        window_n: u64,
    },
    /// Allowed to measure with base or table K (no sanction).
    None {
        /// Promotion K from the similarity table.
        k: u32,
    },
}

impl Sanction {
    /// Whether measurement / promotion should proceed.
    #[must_use]
    pub const fn allows_measure(self) -> bool {
        matches!(self, Self::None { .. } | Self::EscalateK { .. })
    }

    /// Whether this is a slash-intent flag (stake slash is orchestrator-owned).
    #[must_use]
    pub const fn is_slash_intent(self) -> bool {
        matches!(self, Self::SlashIntent)
    }
}

/// Source similarity below this with identical binary → [`Sanction::SlashIntent`].
pub const SLASH_SOURCE_SIM_MAX: f64 = 0.50;

/// Decide sanction from level scores and optional claimed gain.
///
/// - Binary sim `> 0.85` → [`Sanction::SilentReject`] (no measure).
/// - Binary identical (sim == 1.0) and source sim `< SLASH_SOURCE_SIM_MAX` → [`Sanction::SlashIntent`].
/// - Otherwise map K table; mid/high bands that still measure use [`Sanction::EscalateK`] when
///   `k > K_BASE`, else [`Sanction::None`].
#[must_use]
pub fn decide_sanction(source_sim: f64, binary_sim: f64, k_base: u32) -> Sanction {
    use crate::k_table::{k_for_binary_similarity, KBySim, K_BASE};

    // Demonstrated intent: rewritten source, bit-identical normalized binary.
    if binary_sim >= 1.0 - f64::EPSILON && source_sim < SLASH_SOURCE_SIM_MAX {
        return Sanction::SlashIntent;
    }

    match k_for_binary_similarity(binary_sim) {
        KBySim::AutoReject { .. } => Sanction::SilentReject,
        KBySim::Measure { k } => {
            let _ = k_base;
            if k > K_BASE {
                Sanction::EscalateK { k }
            } else {
                Sanction::None { k }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slash_when_source_differs_binary_same() {
        let s = decide_sanction(0.1, 1.0, 5);
        assert_eq!(s, Sanction::SlashIntent);
        assert!(!s.allows_measure());
        assert!(s.is_slash_intent());
    }

    #[test]
    fn silent_reject_above_threshold() {
        assert_eq!(decide_sanction(0.9, 0.90, 5), Sanction::SilentReject);
    }

    #[test]
    fn novel_gets_base_k() {
        assert_eq!(decide_sanction(0.0, 0.1, 5), Sanction::None { k: 5 });
    }
}
