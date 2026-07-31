//! Benjamini–Hochberg FDR control across challenger p-values.

use crate::error::PromoError;

/// Default α for promotion duels (brief §9 / pins).
pub const DEFAULT_ALPHA: f64 = crate::state::ALPHA;

/// Decide which hypotheses are rejected under BH at level `alpha`.
///
/// Input `p_values[i]` is the raw one-sided paired-test p for challenger `i`.
/// Returns a `bool` per input index: `true` means reject H0 (statistically
/// significant after FDR control).
///
/// Procedure (Benjamini & Hochberg 1995):
/// 1. Sort p ascending with original indices.
/// 2. Find largest rank `k` (1-based) with `p_(k) <= (k/m) * alpha`.
/// 3. Reject all hypotheses with rank ≤ k.
///
/// Empty input → empty output. Invalid p → [`PromoError::InvalidPValue`].
pub fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> Result<Vec<bool>, PromoError> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(PromoError::InvalidPValue(format!("alpha={alpha}")));
    }
    let m = p_values.len();
    if m == 0 {
        return Ok(Vec::new());
    }
    for (i, &p) in p_values.iter().enumerate() {
        if !(0.0..=1.0).contains(&p) || p.is_nan() {
            return Err(PromoError::InvalidPValue(format!("idx={i} p={p}")));
        }
    }

    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        p_values[a]
            .partial_cmp(&p_values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // largest k (1-based) with p_(k) <= (k/m)*alpha
    let mut max_k = 0_usize;
    for (rank0, &idx) in order.iter().enumerate() {
        let k = rank0 + 1;
        #[allow(clippy::cast_precision_loss)] // cohort m and rank k are tiny (<< 2^52)
        let threshold = (k as f64 / m as f64) * alpha;
        if p_values[idx] <= threshold {
            max_k = k;
        }
    }

    let mut out = vec![false; m];
    for (rank0, &idx) in order.iter().enumerate() {
        let k = rank0 + 1;
        if k <= max_k {
            out[idx] = true;
        }
    }
    Ok(out)
}

/// Convenience: BH at [`DEFAULT_ALPHA`].
pub fn benjamini_hochberg_default(p_values: &[f64]) -> Result<Vec<bool>, PromoError> {
    benjamini_hochberg(p_values, DEFAULT_ALPHA)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_cohort_returns_empty() {
        let r = benjamini_hochberg(&[], 0.05).expect("ok");
        assert!(r.is_empty());
    }

    #[test]
    fn single_significant_p_rejected_h0() {
        // p=0.01 <= (1/1)*0.05
        let r = benjamini_hochberg(&[0.01], 0.05).expect("ok");
        assert_eq!(r, vec![true]);
    }

    #[test]
    fn single_large_p_not_significant() {
        let r = benjamini_hochberg(&[0.20], 0.05).expect("ok");
        assert_eq!(r, vec![false]);
    }

    #[test]
    fn classic_two_challenger_bh() {
        // m=2, alpha=0.05
        // sorted: 0.01, 0.04
        // k=1: 0.01 <= 0.025 → ok
        // k=2: 0.04 <= 0.05 → ok → max_k=2 both true
        let r = benjamini_hochberg(&[0.04, 0.01], 0.05).expect("ok");
        assert_eq!(r, vec![true, true]);
    }

    #[test]
    fn only_smallest_survives_when_second_too_large() {
        // m=2: p=0.01 and p=0.04 with alpha=0.01
        // k=1: 0.01 <= 0.005? no
        // wait with alpha=0.05:
        // k=1: 0.01 <= 0.025 yes
        // k=2: 0.049 <= 0.05 yes → both
        // Use p=0.04 and p=0.049 with alpha=0.05:
        // sorted 0.04, 0.049
        // k=1: 0.04 <= 0.025? NO
        // k=2: 0.049 <= 0.05 yes → max_k=2 both true (BH step-up)
        //
        // For only-smallest: p=[0.01, 0.04], alpha=0.05 both pass.
        // p=[0.01, 0.20]: k=1 0.01<=0.025 yes; k=2 0.20<=0.05 no → max_k=1
        // only index of 0.01 is true
        let r = benjamini_hochberg(&[0.20, 0.01], 0.05).expect("ok");
        assert_eq!(r, vec![false, true]);
    }

    #[test]
    fn invalid_p_errors() {
        let err = benjamini_hochberg(&[1.5], 0.05).expect_err("bad p");
        assert!(matches!(err, PromoError::InvalidPValue(_)));
    }

    #[test]
    fn boundary_p_equals_threshold() {
        // m=1, p=0.05, alpha=0.05 → 0.05 <= 0.05 → true
        let r = benjamini_hochberg(&[0.05], 0.05).expect("ok");
        assert_eq!(r, vec![true]);
    }
}
