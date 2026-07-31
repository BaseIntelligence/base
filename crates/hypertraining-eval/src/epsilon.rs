//! `ε` threshold for Guard 2 non-inferiority (brief §8.2).

use crate::types::LossMicro;

/// Relative loss budget: **0.25%** of champion mean loss `L` (DeepSeek-V3 FP8/BF16 band).
pub const LOSS_REL_BUDGET_BPS: u32 = 25;

/// Coefficient on `σ̂_d`: **0.5** → `half_sigma = sigma_d / 2`.
pub const SIGMA_COEFF_NUM: u32 = 1;
/// Denominator paired with [`SIGMA_COEFF_NUM`] (default 2 → half sigma).
pub const SIGMA_COEFF_DEN: u32 = 2;

/// `MUST_CALIBRATE`: default relative `σ̂_d` as basis points of `L` when calibration
/// (§9.1, K=10) has not run. Placeholder until hardware calibration publishes `σ̂`.
pub const DEFAULT_SIGMA_D_REL_BPS: u32 = 50;

/// `MUST_CALIBRATE`: absolute floor on default `σ̂_d` (micro-loss units).
pub const DEFAULT_SIGMA_D_ABS_MICRO: LossMicro = 1_000;

/// Human-readable pin for operators / freeze docs.
pub const MUST_CALIBRATE_NOTE: &str =
    "MUST_CALIBRATE: σ̂_d defaults (DEFAULT_SIGMA_D_REL_BPS / DEFAULT_SIGMA_D_ABS_MICRO) \
     are placeholders until §9.1 calibration on target hardware publishes σ̂_d. \
     Replace via EpsilonParams::calibrated_sigma_d_micro after calibration.";

/// Parameters for `ε = min(0.25%·L, 0.5·σ̂_d)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpsilonParams {
    /// Relative budget in basis points of `L` (default 25 = 0.25%).
    pub loss_rel_budget_bps: u32,
    /// Numerator of coefficient on `σ̂_d` (default 1).
    pub sigma_coeff_num: u32,
    /// Denominator of coefficient on `σ̂_d` (default 2 → 0.5).
    pub sigma_coeff_den: u32,
    /// When `Some`, use this calibrated `σ̂_d` (micro). When `None`, derive
    /// `MUST_CALIBRATE` default from `L`.
    pub calibrated_sigma_d_micro: Option<LossMicro>,
}

impl EpsilonParams {
    /// Defaults with `MUST_CALIBRATE` `σ̂_d` placeholders (no hardware calibration yet).
    #[must_use]
    pub const fn must_calibrate_defaults() -> Self {
        Self {
            loss_rel_budget_bps: LOSS_REL_BUDGET_BPS,
            sigma_coeff_num: SIGMA_COEFF_NUM,
            sigma_coeff_den: SIGMA_COEFF_DEN,
            calibrated_sigma_d_micro: None,
        }
    }

    /// Use a published calibration `σ̂_d` (micro-loss).
    #[must_use]
    pub const fn with_calibrated_sigma(sigma_d_micro: LossMicro) -> Self {
        Self {
            loss_rel_budget_bps: LOSS_REL_BUDGET_BPS,
            sigma_coeff_num: SIGMA_COEFF_NUM,
            sigma_coeff_den: SIGMA_COEFF_DEN,
            calibrated_sigma_d_micro: Some(sigma_d_micro),
        }
    }
}

impl Default for EpsilonParams {
    fn default() -> Self {
        Self::must_calibrate_defaults()
    }
}

/// Resolve `σ̂_d`: calibrated value or `MUST_CALIBRATE` default from `L`.
#[must_use]
pub fn resolve_sigma_d_micro(mean_champ_loss: LossMicro, params: &EpsilonParams) -> LossMicro {
    if let Some(s) = params.calibrated_sigma_d_micro {
        return s.abs();
    }
    let l = mean_champ_loss.abs();
    let rel = (l.saturating_mul(i64::from(DEFAULT_SIGMA_D_REL_BPS))) / 10_000;
    rel.max(DEFAULT_SIGMA_D_ABS_MICRO)
}

/// `ε = min(rel_budget · L, sigma_coeff · σ̂_d)` in micro-loss (non-negative).
#[must_use]
pub fn compute_epsilon_micro(
    mean_champ_loss: LossMicro,
    sigma_d_micro: LossMicro,
    params: &EpsilonParams,
) -> LossMicro {
    let l = mean_champ_loss.abs();
    let rel = (l.saturating_mul(i64::from(params.loss_rel_budget_bps))) / 10_000;
    let den = i64::from(params.sigma_coeff_den.max(1));
    let half_sigma = (sigma_d_micro.abs().saturating_mul(i64::from(params.sigma_coeff_num))) / den;
    rel.min(half_sigma).max(0)
}
