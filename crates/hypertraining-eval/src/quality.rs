//! Guard 2: paired one-sided non-inferiority test on continuous val loss.
//!
//! # Internal `f64`
//!
//! Mean, sample sd, and Student-t critical comparison use `f64` **only inside
//! this module**. Public inputs/outputs stay on [`crate::LossMicro`]. This is
//! not a leaf score path (`hypertraining-pay` maps rewards to integer scores).

use crate::epsilon::{compute_epsilon_micro, resolve_sigma_d_micro, EpsilonParams};
use crate::error::EvalError;
use crate::types::{EvalRun, LossMicro};

/// Significance level `α = 0.05` (brief §8.2 / plan pins).
pub const ALPHA: f64 = 0.05;

/// Report from the quality non-inferiority test.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Whether `H0` was rejected (candidate not significantly worse).
    pub quality_ok: bool,
    /// Paired differences `d_i = L_champ − L_cand` (micro).
    pub differences_micro: Vec<LossMicro>,
    /// Mean of `d_i` (micro, truncated toward zero from `f64`).
    pub mean_d_micro: LossMicro,
    /// `ε` used (micro).
    pub epsilon_micro: LossMicro,
    /// `σ̂_d` used for `ε` (micro).
    pub sigma_d_micro: LossMicro,
    /// One-sided t statistic (internal `f64`; documented).
    pub t_stat: f64,
    /// Critical value `t_{df, 1−α}` (internal `f64`).
    pub t_critical: f64,
    /// Degrees of freedom `n − 1`.
    pub df: usize,
}

/// Build paired differences `d_i = L_champ − L_cand` with seed alignment checks.
///
/// # Errors
///
/// Returns [`EvalError::PairedLengthMismatch`], [`EvalError::InsufficientPairs`],
/// or [`EvalError::SeedMismatch`] when the paired contract is violated.
pub fn paired_differences(
    champ_runs: &[EvalRun],
    cand_runs: &[EvalRun],
) -> Result<Vec<LossMicro>, EvalError> {
    if champ_runs.len() != cand_runs.len() {
        return Err(EvalError::PairedLengthMismatch {
            champ: champ_runs.len(),
            cand: cand_runs.len(),
        });
    }
    if champ_runs.len() < 2 {
        return Err(EvalError::InsufficientPairs {
            got: champ_runs.len(),
        });
    }
    let mut out = Vec::with_capacity(champ_runs.len());
    for (i, (c, k)) in champ_runs.iter().zip(cand_runs.iter()).enumerate() {
        if c.seed != k.seed {
            return Err(EvalError::SeedMismatch {
                index: i,
                champ_seed: c.seed,
                cand_seed: k.seed,
            });
        }
        out.push(c.val_loss_micro.saturating_sub(k.val_loss_micro));
    }
    Ok(out)
}

/// One-sided paired t-test of `H0: E[d] ≤ −ε` at `α = 0.05`.
///
/// Reject `H0` (`quality_ok`) when `t = (d̄ + ε) / (s/√n) > t_{n−1, 1−α}`.
///
/// # Errors
///
/// Propagates [`paired_differences`] contract errors.
pub fn quality_non_inferiority(
    champ_runs: &[EvalRun],
    cand_runs: &[EvalRun],
    params: &EpsilonParams,
) -> Result<QualityReport, EvalError> {
    let diffs = paired_differences(champ_runs, cand_runs)?;
    let n = diffs.len();
    let df = n - 1;

    let mean_champ = mean_i64(champ_runs.iter().map(|r| r.val_loss_micro));
    let sample_sd = sample_sd_i64(&diffs);
    let sigma_for_eps = match params.calibrated_sigma_d_micro {
        Some(s) => s.abs(),
        None => resolve_sigma_d_micro(mean_champ, params),
    };
    let epsilon = compute_epsilon_micro(mean_champ, sigma_for_eps, params);

    let mean_d = mean_i64(diffs.iter().copied());
    let mean_d_f = mean_d as f64;
    let eps_f = epsilon as f64;
    let s = sample_sd.max(1e-12);
    let se = s / (n as f64).sqrt();
    let t_stat = (mean_d_f + eps_f) / se;
    let t_critical = t_critical_one_sided_05(df);
    let quality_ok = t_stat > t_critical;

    Ok(QualityReport {
        quality_ok,
        differences_micro: diffs,
        mean_d_micro: mean_d,
        epsilon_micro: epsilon,
        sigma_d_micro: sigma_for_eps,
        t_stat,
        t_critical,
        df,
    })
}

fn mean_i64(xs: impl Iterator<Item = i64>) -> i64 {
    let mut sum = 0_i128;
    let mut n = 0_i128;
    for x in xs {
        sum += i128::from(x);
        n += 1;
    }
    if n == 0 {
        return 0;
    }
    i64::try_from(sum / n).unwrap_or(0)
}

fn sample_sd_i64(xs: &[i64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let var = xs
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (n as f64 - 1.0);
    var.sqrt()
}

fn t_critical_one_sided_05(df: usize) -> f64 {
    match df {
        0 => f64::INFINITY,
        1 => 6.313_8,
        2 => 2.920_0,
        3 => 2.353_4,
        4 => 2.131_8,
        5 => 2.015_0,
        6 => 1.943_2,
        7 => 1.894_6,
        8 => 1.859_5,
        9 => 1.833_1,
        10 => 1.812_5,
        15 => 1.753_1,
        20 => 1.724_7,
        30 => 1.697_3,
        40 => 1.683_9,
        60 => 1.670_6,
        120 => 1.657_7,
        _ if df > 120 => 1.644_9,
        d => {
            let lo = match d {
                11..=15 => (10, 1.812_5, 15, 1.753_1),
                16..=20 => (15, 1.753_1, 20, 1.724_7),
                21..=30 => (20, 1.724_7, 30, 1.697_3),
                31..=40 => (30, 1.697_3, 40, 1.683_9),
                41..=60 => (40, 1.683_9, 60, 1.670_6),
                _ => (60, 1.670_6, 120, 1.657_7),
            };
            let (d0, t0, d1, t1) = lo;
            let w = (d - d0) as f64 / (d1 - d0) as f64;
            t0 + w * (t1 - t0)
        }
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn t_crit_k5_df4_near_2_13() {
        let t = t_critical_one_sided_05(4);
        assert!((t - 2.131_8).abs() < 1e-6);
    }

    #[test]
    fn equal_losses_reject_h0() {
        let champ: Vec<_> = (0..5)
            .map(|i| EvalRun {
                seed: i,
                val_loss_micro: 1_500_000,
            })
            .collect();
        let cand = champ.clone();
        let r = quality_non_inferiority(&champ, &cand, &EpsilonParams::must_calibrate_defaults())
            .expect("ok");
        assert!(r.quality_ok, "t={} crit={}", r.t_stat, r.t_critical);
        assert_eq!(r.mean_d_micro, 0);
    }
}
