//! Marginal wall-clock gain Δ over the current champion (brief §11.1).
//!
//! ```text
//! Δ(candidate) = T_champion − T_candidate     (saved compute time, ms)
//! pay ∝ max(Δ, 0)  iff guards 1–3 passed
//! ```

/// Inputs for a single marginal-pay decision (integer ms only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PayInputs {
    /// Champion segment wall-clock (ms).
    pub t_champ_ms: u64,
    /// Candidate segment wall-clock (ms).
    pub t_cand_ms: u64,
    /// True only when guards 1–3 all passed and promotion rules allow pay.
    pub guards_passed: bool,
}

/// Signed raw Δ in ms: `T_champ − T_cand` (may be negative when candidate is slower).
#[must_use]
pub fn raw_delta_ms(t_champ_ms: u64, t_cand_ms: u64) -> i128 {
    i128::from(t_champ_ms) - i128::from(t_cand_ms)
}

/// Payable Δ in ms: `max(T_champ − T_cand, 0)` when `guards_passed`, else `0`.
///
/// Resubmitting the champion unchanged → Δ = 0. Slower candidate → 0.
/// Guards failed → 0 even if wall-clock improved.
#[must_use]
pub fn payable_delta_ms(inputs: &PayInputs) -> u64 {
    if !inputs.guards_passed {
        return 0;
    }
    inputs.t_champ_ms.saturating_sub(inputs.t_cand_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_delta_positive_when_cand_faster() {
        assert_eq!(raw_delta_ms(1_000, 800), 200);
    }

    #[test]
    fn raw_delta_negative_when_cand_slower() {
        assert_eq!(raw_delta_ms(1_000, 1_200), -200);
    }

    #[test]
    fn payable_zero_when_equal() {
        let p = PayInputs {
            t_champ_ms: 5_000,
            t_cand_ms: 5_000,
            guards_passed: true,
        };
        assert_eq!(payable_delta_ms(&p), 0);
    }

    #[test]
    fn payable_zero_when_slower() {
        let p = PayInputs {
            t_champ_ms: 5_000,
            t_cand_ms: 6_000,
            guards_passed: true,
        };
        assert_eq!(payable_delta_ms(&p), 0);
    }

    #[test]
    fn payable_zero_when_guards_fail_even_if_faster() {
        let p = PayInputs {
            t_champ_ms: 5_000,
            t_cand_ms: 1_000,
            guards_passed: false,
        };
        assert_eq!(payable_delta_ms(&p), 0);
    }

    #[test]
    fn payable_positive_when_faster_and_guards_ok() {
        let p = PayInputs {
            t_champ_ms: 5_000,
            t_cand_ms: 4_000,
            guards_passed: true,
        };
        assert_eq!(payable_delta_ms(&p), 1_000);
    }
}
