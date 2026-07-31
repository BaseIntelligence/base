//! Fingerprint dedupe: same normalized binary fingerprint per miner for N segments.

use std::collections::BTreeMap;

use crate::error::AntinoisError;

/// Default cooldown window in segments when not configured.
pub const DEFAULT_DEDUPE_SEGMENTS: u64 = 16;

/// In-memory per-miner fingerprint → last-seen segment index.
#[derive(Debug, Default, Clone)]
pub struct FingerprintDedupe {
    /// `miner_id` → (fingerprint hex → last segment index).
    by_miner: BTreeMap<String, BTreeMap<String, u64>>,
    /// Cooldown window N (segments).
    window_n: u64,
}

impl FingerprintDedupe {
    /// Create a store with cooldown window `n` segments (`n > 0`).
    ///
    /// # Errors
    /// [`AntinoisError::InvalidDedupeWindow`] when `n == 0`.
    pub fn new(window_n: u64) -> Result<Self, AntinoisError> {
        if window_n == 0 {
            return Err(AntinoisError::InvalidDedupeWindow);
        }
        Ok(Self {
            by_miner: BTreeMap::new(),
            window_n,
        })
    }

    /// Cooldown window N.
    #[must_use]
    pub fn window_n(&self) -> u64 {
        self.window_n
    }

    /// Check whether `(miner, fingerprint)` is still in cooldown at `segment_index`.
    ///
    /// On allow, records the fingerprint at this segment. On reject, leaves store unchanged.
    ///
    /// # Errors
    /// [`AntinoisError::EmptyMinerId`] when `miner_id` is empty.
    pub fn check_and_record(
        &mut self,
        miner_id: &str,
        fingerprint_hex: &str,
        segment_index: u64,
    ) -> Result<DedupeOutcome, AntinoisError> {
        if miner_id.is_empty() {
            return Err(AntinoisError::EmptyMinerId);
        }
        let miner = self.by_miner.entry(miner_id.to_owned()).or_default();
        if let Some(&last) = miner.get(fingerprint_hex) {
            let elapsed = segment_index.saturating_sub(last);
            // Same segment or within N segments after last → reject.
            // "rejected for N segments" means last, last+1, ..., last+N-1 blocked if we
            // count N remaining; plan: "same fingerprint per miner for N segments".
            // Interpret: if segment_index - last < N, reject (including re-submit same segment).
            if elapsed < self.window_n {
                return Ok(DedupeOutcome::Rejected {
                    last_segment: last,
                    window_n: self.window_n,
                });
            }
        }
        miner.insert(fingerprint_hex.to_owned(), segment_index);
        Ok(DedupeOutcome::Allowed)
    }
}

/// Result of a dedupe check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupeOutcome {
    /// First time or outside cooldown — recorded.
    Allowed,
    /// Same fingerprint still within N segments for this miner.
    Rejected {
        /// Segment index when fingerprint was last accepted.
        last_segment: u64,
        /// Configured window N.
        window_n: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn second_submit_within_n_rejected() {
        let mut d = FingerprintDedupe::new(3).expect("n");
        assert_eq!(
            d.check_and_record("m1", "aa", 10).expect("ok"),
            DedupeOutcome::Allowed
        );
        assert!(matches!(
            d.check_and_record("m1", "aa", 11).expect("ok"),
            DedupeOutcome::Rejected { .. }
        ));
        assert!(matches!(
            d.check_and_record("m1", "aa", 12).expect("ok"),
            DedupeOutcome::Rejected { .. }
        ));
        // elapsed = 13-10 = 3 >= 3 → allowed
        assert_eq!(
            d.check_and_record("m1", "aa", 13).expect("ok"),
            DedupeOutcome::Allowed
        );
    }

    #[test]
    fn different_miners_independent() {
        let mut d = FingerprintDedupe::new(5).expect("n");
        assert_eq!(
            d.check_and_record("a", "fp", 1).expect("ok"),
            DedupeOutcome::Allowed
        );
        assert_eq!(
            d.check_and_record("b", "fp", 1).expect("ok"),
            DedupeOutcome::Allowed
        );
    }
}
