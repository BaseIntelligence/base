//! Prism v3 composite scoring — the weighted-geometric aggregate over the
//! G1–G8 evaluation battery, implementing
//! `docs/spikes/prism-v3/research/12-score-aggregation.md` §7 steps 0–6:
//!
//! 0. anchor sets are pre-registered constants (see `prism-recipe` `anchors`
//!    module; this crate consumes the same canonical JSON schema),
//! 1. per-metric fixed-anchor normalization clipped to `[0, 1]`,
//! 2. two-level means (within sub-metric, then across) → group scores,
//!    then the step-2.5 mirror-gap penalty `max(0, (x_public − x_mirror) −
//!    τ_m)` deducted from G2/G4 pre-composite,
//! 3. lexicographic gates (g3 ≥ 0.25, g8 ≥ 0.5, budgets, CI half-width ≤ δ),
//! 4. weighted geometric mean `C = ∏ g_k^{w_k}`,
//! 5. clustered bootstrap (B ≥ 1000) recomputing the entire pipeline per
//!    resample → `SE(C)` and per-axis percentile CIs,
//! 6. `lattice = round(SCORE_MAX × max(0, C − 1.645·SE))`.
//!
//! Decoupling note: the anchor-set registry lives in `prism-recipe`
//! (versioned, hash-committed); this module parses the same canonical JSON
//! into its own runtime view ([`CompositeAnchors::from_json`]) so the
//! pipeline never hardcodes per-metric normalization.

use std::collections::BTreeMap;

use prism_challenge_task::SCORE_MAX;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Number of composite groups (G1–G8).
pub const N_GROUPS: usize = 8;

/// Group keys in canonical order.
pub const GROUP_KEYS: [&str; N_GROUPS] = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8"];

/// Clip to `[0, 1]`; non-finite values fail closed to 0.
fn clip01(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Default relative weight of a sub-metric inside its group (equal mean).
fn default_metric_weight() -> f64 {
    1.0
}

/// One group's sub-metric: normalization descriptor plus an optional
/// relative weight (G5 uses unequal internal weights; others default to 1).
///
/// Extra JSON fields from the anchor set (`status`, `note`) are accepted and
/// ignored so this view stays a pure scoring consumer of the same file
/// `prism-recipe` embeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSpec {
    /// Normalization descriptor (step 1).
    #[serde(flatten)]
    pub norm: NormDesc,
    /// Relative weight within the group; group mean is weight-normalized.
    #[serde(default = "default_metric_weight")]
    pub weight: f64,
    /// Provenance marker from the anchor JSON (`"placeholder"`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Free-form note from the anchor JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Per-metric normalization descriptor (step 1). Mirrors the `kind`-tagged
/// JSON schema of the anchor set.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NormDesc {
    /// Accuracy-like: `clip01((x - chance) / (1 - chance))`.
    Accuracy {
        /// Chance floor (e.g. `1/#choices`; 0 for generative probes).
        chance: f64,
    },
    /// bpb per domain, log-ratio between fixed anchors:
    /// `clip01((ln chance − ln x) / (ln chance − ln reference))`.
    BpbLogRatio {
        /// Chance-level bpb (`log2|V|` scaled to bytes).
        chance: f64,
        /// Pre-registered reference-architecture bpb.
        reference: f64,
    },
    /// Efficiency, capped log-ratio to the reference recipe:
    /// `clip01(ln(x / reference) / ln(cap / reference))`. `cap < reference`
    /// encodes lower-better metrics (both logs negative).
    EfficiencyLogRatio {
        /// Reference operating point (score 0).
        reference: f64,
        /// Plausible extreme (score 1); below `reference` if lower-better.
        cap: f64,
    },
    /// Stability: already a bounded construction; `clip01(x)`.
    StabilityBounded,
}

impl NormDesc {
    /// Normalize a raw metric value into `[0, 1]` (fixed anchors, step 1).
    #[must_use]
    pub fn normalize(&self, x: f64) -> f64 {
        if !x.is_finite() {
            return 0.0;
        }
        match *self {
            Self::Accuracy { chance } => {
                let span = 1.0 - chance;
                if span <= 0.0 {
                    return 0.0;
                }
                clip01((x - chance) / span)
            }
            Self::BpbLogRatio { chance, reference } => {
                let denom = chance.ln() - reference.ln();
                if x <= 0.0 || denom <= 0.0 {
                    return 0.0;
                }
                clip01((chance.ln() - x.ln()) / denom)
            }
            Self::EfficiencyLogRatio { reference, cap } => {
                let denom = (cap / reference).ln();
                if x <= 0.0 || reference <= 0.0 || denom.abs() < f64::EPSILON {
                    return 0.0;
                }
                clip01((x / reference).ln() / denom)
            }
            Self::StabilityBounded => clip01(x),
        }
    }
}

/// One group's weight + sub-metric anchors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupSpec {
    /// Composite weight `w_k` (Σ over groups = 1).
    pub weight: f64,
    /// Sub-metric anchors keyed `org.<group>.<name>`.
    pub metrics: BTreeMap<String, MetricSpec>,
}

/// Lexicographic gate thresholds (step 3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GateThresholds {
    /// Minimum G3 (recall probes) group score.
    pub g3_min: f64,
    /// Minimum G8 (stability) group score.
    pub g8_min: f64,
    /// Max per-axis clustered 95% CI half-width δ.
    pub ci_half_width_delta: f64,
    /// Fixed-recipe parameter budget.
    pub max_params: u64,
    /// Fixed-recipe wall-clock budget in seconds (6h = 21600).
    pub max_wall_s: f64,
}

/// Mirror-gap penalty parameters (step 2.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirrorParams {
    /// Tolerance τ_m: gaps up to this are free.
    pub tau_m: f64,
    /// Groups the penalty applies to (`["g2", "g4"]`).
    pub groups: Vec<String>,
}

/// Clustered-bootstrap parameters (steps 5–6).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapParams {
    /// Resample count B (≥ 1000).
    pub b: u32,
    /// One-sided 95% LCB z multiplier (1.645).
    pub lcb_z: f64,
}

/// Runtime view of a pre-registered anchor set (parsed from the same
/// canonical JSON `prism-recipe` embeds and hash-commits).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeAnchors {
    /// Anchor-set version.
    pub version: u16,
    /// Groups keyed `g1`..`g8`.
    pub groups: BTreeMap<String, GroupSpec>,
    /// Gate thresholds.
    pub gates: GateThresholds,
    /// Mirror-gap penalty parameters.
    pub mirror: MirrorParams,
    /// Bootstrap parameters.
    pub bootstrap: BootstrapParams,
    /// Pre-registration hash (sha256 hex over the canonical JSON), attached
    /// to outcomes for audit. Not part of the anchor JSON itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prereg_hash: Option<String>,
}

/// Anchor-set validation errors.
#[derive(Debug, Error)]
pub enum CompositeError {
    /// Anchor JSON failed to parse.
    #[error("anchor JSON parse: {0}")]
    Parse(#[from] serde_json::Error),
    /// Semantically invalid anchor set.
    #[error("invalid anchor set: {0}")]
    Invalid(String),
}

impl CompositeAnchors {
    /// Parse and validate an anchor set from canonical JSON.
    pub fn from_json(json: &str) -> Result<Self, CompositeError> {
        let anchors: Self = serde_json::from_str(json)?;
        anchors.validate()?;
        Ok(anchors)
    }

    /// Attach the pre-registration hash carried into every outcome.
    #[must_use]
    pub fn with_prereg_hash(mut self, hash: String) -> Self {
        self.prereg_hash = Some(hash);
        self
    }

    fn validate(&self) -> Result<(), CompositeError> {
        let mut sum = 0.0;
        for key in GROUP_KEYS {
            let Some(spec) = self.groups.get(key) else {
                return Err(CompositeError::Invalid(format!("missing group {key}")));
            };
            if spec.weight < 0.0 {
                return Err(CompositeError::Invalid(format!("{key} weight negative")));
            }
            if spec.metrics.is_empty() {
                return Err(CompositeError::Invalid(format!("{key} has no metrics")));
            }
            sum += spec.weight;
        }
        if (sum - 1.0).abs() > 1e-6 {
            return Err(CompositeError::Invalid(format!(
                "group weights sum to {sum}, expected 1"
            )));
        }
        if !(0.0..=1.0).contains(&self.gates.g3_min)
            || !(0.0..=1.0).contains(&self.gates.g8_min)
            || self.gates.ci_half_width_delta <= 0.0
        {
            return Err(CompositeError::Invalid(
                "gate thresholds out of range".to_string(),
            ));
        }
        if self.bootstrap.b == 0 || self.bootstrap.lcb_z <= 0.0 {
            return Err(CompositeError::Invalid("bad bootstrap params".to_string()));
        }
        Ok(())
    }
}

/// One metric's raw value plus its cluster-level observations.
///
/// `value` is the aggregate on the metric's natural scale; `clusters` maps
/// cluster id (domain / passage / template family) to the metric computed on
/// that cluster's items — the bootstrap resampling units (step 5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSeries {
    /// Raw aggregate value (natural scale).
    pub value: f64,
    /// Per-cluster values (bootstrap units); empty = no variance estimate.
    #[serde(default)]
    pub clusters: BTreeMap<String, f64>,
}

/// Budget facts for the fixed-recipe gates (step 3).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BudgetFacts {
    /// Trained parameter count.
    pub params: u64,
    /// Wall-clock seconds.
    pub wall_s: f64,
    /// Real training tokens consumed (recorded, not gated).
    pub tokens: u64,
}

/// A public/private mirror pair for the contamination-gap penalty (step 2.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirrorPair {
    /// Group the penalty applies to (`"g2"` or `"g4"`).
    pub group: String,
    /// Metric key whose anchors normalize both sides (e.g.
    /// `org.g2.arc_easy_acc`).
    pub metric: String,
    /// Public-anchor scores (raw scale of `metric`).
    pub public: MetricSeries,
    /// Private-mirror scores (same raw scale).
    pub mirror: MetricSeries,
}

/// Everything the composite needs for one submission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SubmissionMetrics {
    /// Raw metrics keyed `org.<group>.<name>`; only keys declared in the
    /// anchor set participate in scoring.
    #[serde(default)]
    pub metrics: BTreeMap<String, MetricSeries>,
    /// Budget facts.
    #[serde(default)]
    pub budget: BudgetFacts,
    /// Mirror pairs for the G2/G4 contamination gap.
    #[serde(default)]
    pub mirrors: Vec<MirrorPair>,
}

/// One gate failure (structured; serialized for the dashboard/audit).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum GateFailure {
    /// A whole group's metrics are absent from the submission.
    MissingGroup {
        /// Group key (`g1`..`g8`).
        group: String,
    },
    /// A declared sub-metric is absent from the submission.
    MissingMetric {
        /// Metric key (`org.<group>.<name>`).
        key: String,
    },
    /// `g3 < g3_min` (recall floor).
    G3BelowFloor {
        /// Observed g3.
        g: f64,
        /// Required floor.
        floor: f64,
    },
    /// `g8 < g8_min` (stability floor).
    G8BelowFloor {
        /// Observed g8.
        g: f64,
        /// Required floor.
        floor: f64,
    },
    /// Parameter budget exceeded.
    ParamsOverBudget {
        /// Observed params.
        params: u64,
        /// Budget.
        max: u64,
    },
    /// Wall-clock budget exceeded.
    WallClockOverBudget {
        /// Observed seconds.
        wall_s: f64,
        /// Budget seconds.
        max_s: f64,
    },
    /// An axis's clustered 95% CI half-width exceeds δ.
    CiHalfWidthTooWide {
        /// Group key.
        group: String,
        /// Observed half-width.
        half_width: f64,
        /// Pre-registered δ.
        delta: f64,
    },
}

/// Per-gate pass/fail flags (all true ⇔ eligible).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct GateReport {
    /// Every declared group/metric present.
    pub complete: bool,
    /// g3 floor passed.
    pub g3_ok: bool,
    /// g8 floor passed.
    pub g8_ok: bool,
    /// Budget gates passed.
    pub budgets_ok: bool,
    /// CI-sufficiency gate passed.
    pub ci_ok: bool,
}

impl GateReport {
    fn from_failures(failures: &[GateFailure]) -> Self {
        let has = |f: fn(&GateFailure) -> bool| failures.iter().any(f);
        Self {
            complete: !has(|f| {
                matches!(
                    f,
                    GateFailure::MissingGroup { .. } | GateFailure::MissingMetric { .. }
                )
            }),
            g3_ok: !has(|f| matches!(f, GateFailure::G3BelowFloor { .. })),
            g8_ok: !has(|f| matches!(f, GateFailure::G8BelowFloor { .. })),
            budgets_ok: !has(|f| {
                matches!(
                    f,
                    GateFailure::ParamsOverBudget { .. } | GateFailure::WallClockOverBudget { .. }
                )
            }),
            ci_ok: !has(|f| matches!(f, GateFailure::CiHalfWidthTooWide { .. })),
        }
    }
}

/// One group's score with its bootstrap CI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupScore {
    /// Group key (`g1`..`g8`).
    pub group: String,
    /// Point estimate after the mirror-gap penalty.
    pub g: f64,
    /// Mirror-gap penalty deducted (0 for non-mirrored groups).
    pub mirror_penalty: f64,
    /// Clustered-bootstrap 2.5% percentile (absent if bootstrap did not run).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_lo: Option<f64>,
    /// Clustered-bootstrap 97.5% percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_hi: Option<f64>,
}

/// Bootstrap provenance attached to scored outcomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BootstrapInfo {
    /// Resample count B actually used.
    pub b: u32,
    /// RNG seed (xorshift64*; reproducible).
    pub seed: u64,
}

/// Eligible outcome (step 6).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeScore {
    /// Per-group scores in `g1`..`g8` order (len 8).
    pub groups: Vec<GroupScore>,
    /// Weighted geometric mean C.
    pub composite: f64,
    /// Clustered-bootstrap standard error of C.
    pub se: f64,
    /// Lower confidence bound `max(0, C − z·SE)`.
    pub lcb: f64,
    /// `round(SCORE_MAX × LCB)` — the v3 lattice score.
    pub lattice: u64,
    /// Gate flags (all true).
    pub gates: GateReport,
    /// Anchor-set version.
    pub anchor_version: u16,
    /// Pre-registration hash of the anchor set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prereg_hash: Option<String>,
    /// Bootstrap provenance.
    pub bootstrap: BootstrapInfo,
}

/// Ineligible outcome (share = 0, emission burns).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ineligible {
    /// Structured gate failures, in lexicographic gate order.
    pub reasons: Vec<GateFailure>,
    /// Per-group point estimates (CIs only if the bootstrap ran).
    pub groups: Vec<GroupScore>,
    /// Composite C (computed for the dashboard; not payable).
    pub composite: f64,
    /// Gate flags.
    pub gates: GateReport,
    /// Anchor-set version.
    pub anchor_version: u16,
    /// Pre-registration hash of the anchor set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prereg_hash: Option<String>,
}

/// Result of the composite pipeline (step 6 output).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum CompositeOutcome {
    /// Eligible: full scored outcome.
    Scored(CompositeScore),
    /// Ineligible: structured reasons; lattice score is 0.
    Ineligible(Ineligible),
}

impl CompositeOutcome {
    /// Whether the submission is eligible for emissions.
    #[must_use]
    pub fn is_eligible(&self) -> bool {
        matches!(self, Self::Scored(_))
    }

    /// The v3 lattice score (`None` when ineligible).
    #[must_use]
    pub fn lattice(&self) -> Option<u64> {
        match self {
            Self::Scored(s) => Some(s.lattice),
            Self::Ineligible(_) => None,
        }
    }
}

/// xorshift64* PRNG (Vigna 2016) — tiny, seedless-env, deterministic across
/// platforms; keeps `cargo deny` green (no new dependency).
struct XorShift64Star(u64);

impl XorShift64Star {
    const fn new(seed: u64) -> Self {
        // splitmix64 finalizer: nearby seeds must decorrelate (raw xorshift
        // state from seed±1 otherwise produces correlated leading bits), and
        // seed 0 must not be a fixed point.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform draw in `0..n` (n ≥ 1) via Lemire's multiply-high map —
    /// avoids both modulo bias and xorshift*'s weak low bits.
    #[allow(clippy::cast_possible_truncation)]
    const fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

/// Point estimate vs. bootstrap replicate drawing mode.
enum Draw<'a> {
    Point,
    Bootstrap(&'a mut XorShift64Star),
}

/// A metric's value under the current draw mode: the stored aggregate for
/// the point estimate, or a with-replacement cluster resample mean.
fn value_of(series: &MetricSeries, draw: &mut Draw<'_>) -> f64 {
    match draw {
        Draw::Point => series.value,
        Draw::Bootstrap(rng) => {
            if series.clusters.is_empty() {
                return series.value;
            }
            let vals: Vec<f64> = series.clusters.values().copied().collect();
            let n = vals.len();
            let mut acc = 0.0;
            for _ in 0..n {
                acc += vals[rng.below(n)];
            }
            acc / n as f64
        }
    }
}

/// Lookup of every declared metric key → its normalization descriptor.
fn norm_lookup(anchors: &CompositeAnchors) -> BTreeMap<&str, &NormDesc> {
    anchors
        .groups
        .values()
        .flat_map(|g| g.metrics.iter().map(|(k, v)| (k.as_str(), &v.norm)))
        .collect()
}

/// Step 2.5: mirror-gap penalty for one group under the current draw mode.
fn mirror_penalty(
    sub: &SubmissionMetrics,
    anchors: &CompositeAnchors,
    norms: &BTreeMap<&str, &NormDesc>,
    group: &str,
    draw: &mut Draw<'_>,
) -> f64 {
    if !anchors.mirror.groups.iter().any(|g| g == group) {
        return 0.0;
    }
    let mut gap_sum = 0.0;
    let mut n = 0usize;
    for pair in &sub.mirrors {
        if pair.group != group {
            continue;
        }
        let Some(norm) = norms.get(pair.metric.as_str()) else {
            continue;
        };
        let public = norm.normalize(value_of(&pair.public, draw));
        let mirror = norm.normalize(value_of(&pair.mirror, draw));
        gap_sum += public - mirror;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    (gap_sum / n as f64 - anchors.mirror.tau_m).max(0.0)
}

/// Steps 1–2.5: normalize, two-level group means, mirror penalty. Returns
/// `(g, penalty, completeness_failures)`.
#[allow(clippy::type_complexity)]
fn run_groups(
    sub: &SubmissionMetrics,
    anchors: &CompositeAnchors,
    norms: &BTreeMap<&str, &NormDesc>,
    draw: &mut Draw<'_>,
) -> ([f64; N_GROUPS], [f64; N_GROUPS], Vec<GateFailure>) {
    let mut g = [0.0; N_GROUPS];
    let mut penalty = [0.0; N_GROUPS];
    let mut failures = Vec::new();
    for (idx, key) in GROUP_KEYS.iter().enumerate() {
        let Some(spec) = anchors.groups.get(*key) else {
            failures.push(GateFailure::MissingGroup {
                group: (*key).to_string(),
            });
            continue;
        };
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for (mk, mspec) in &spec.metrics {
            match sub.metrics.get(mk) {
                Some(series) => {
                    let w = if mspec.weight.is_finite() && mspec.weight > 0.0 {
                        mspec.weight
                    } else {
                        0.0
                    };
                    acc += w * mspec.norm.normalize(value_of(series, draw));
                    wsum += w;
                }
                None => failures.push(GateFailure::MissingMetric { key: mk.clone() }),
            }
        }
        if wsum <= 0.0 {
            failures.push(GateFailure::MissingGroup {
                group: (*key).to_string(),
            });
        } else {
            g[idx] = acc / wsum;
        }
        let p = mirror_penalty(sub, anchors, norms, key, draw);
        penalty[idx] = p;
        g[idx] = (g[idx] - p).max(0.0);
    }
    (g, penalty, failures)
}

/// Step 4: weighted geometric mean `∏ g_k^{w_k}` (a zero axis collapses C).
fn composite_of(g: &[f64; N_GROUPS], anchors: &CompositeAnchors) -> f64 {
    let mut log_c = 0.0;
    for (idx, key) in GROUP_KEYS.iter().enumerate() {
        let w = anchors.groups.get(*key).map_or(0.0, |s| s.weight);
        if w <= 0.0 {
            continue;
        }
        if g[idx] <= 0.0 {
            return 0.0;
        }
        log_c += w * g[idx].ln();
    }
    log_c.exp()
}

/// Linear-interpolated (type-7) percentile of a sorted slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    match sorted.len() {
        0 => f64::NAN,
        1 => sorted[0],
        n => {
            let rank = q * (n - 1) as f64;
            let lo = rank.floor() as usize;
            let hi = (lo + 1).min(n - 1);
            sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo as f64)
        }
    }
}

/// Sample standard deviation (B−1 denominator); 0 for degenerate input.
fn sample_std(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

fn group_scores(
    g: &[f64; N_GROUPS],
    penalty: &[f64; N_GROUPS],
    cis: Option<&[(f64, f64); N_GROUPS]>,
) -> Vec<GroupScore> {
    GROUP_KEYS
        .iter()
        .enumerate()
        .map(|(idx, key)| {
            let (ci_lo, ci_hi) = cis.map_or((None, None), |c| (Some(c[idx].0), Some(c[idx].1)));
            GroupScore {
                group: (*key).to_string(),
                g: g[idx],
                mirror_penalty: penalty[idx],
                ci_lo,
                ci_hi,
            }
        })
        .collect()
}

/// Run the full composite pipeline (steps 0–6) for one submission.
///
/// `seed` drives the clustered bootstrap (same seed ⇒ bit-identical SE/CIs).
#[must_use]
pub fn evaluate(
    sub: &SubmissionMetrics,
    anchors: &CompositeAnchors,
    seed: u64,
) -> CompositeOutcome {
    let norms = norm_lookup(anchors);
    let (g, penalty, mut failures) = run_groups(sub, anchors, &norms, &mut Draw::Point);

    // Step 3 — lexicographic gates on point estimates: g3, g8, budgets.
    if g[2] < anchors.gates.g3_min {
        failures.push(GateFailure::G3BelowFloor {
            g: g[2],
            floor: anchors.gates.g3_min,
        });
    }
    if g[7] < anchors.gates.g8_min {
        failures.push(GateFailure::G8BelowFloor {
            g: g[7],
            floor: anchors.gates.g8_min,
        });
    }
    if sub.budget.params > anchors.gates.max_params {
        failures.push(GateFailure::ParamsOverBudget {
            params: sub.budget.params,
            max: anchors.gates.max_params,
        });
    }
    if sub.budget.wall_s > anchors.gates.max_wall_s {
        failures.push(GateFailure::WallClockOverBudget {
            wall_s: sub.budget.wall_s,
            max_s: anchors.gates.max_wall_s,
        });
    }

    // Step 4 — composite on point estimates.
    let composite = composite_of(&g, anchors);

    if !failures.is_empty() {
        return CompositeOutcome::Ineligible(Ineligible {
            gates: GateReport::from_failures(&failures),
            reasons: failures,
            groups: group_scores(&g, &penalty, None),
            composite,
            anchor_version: anchors.version,
            prereg_hash: anchors.prereg_hash.clone(),
        });
    }

    // Step 5 — clustered bootstrap: resample clusters with replacement and
    // recompute the ENTIRE pipeline (normalization → group means → mirror
    // penalty → composite) per resample.
    let b = anchors.bootstrap.b as usize;
    let mut rng = XorShift64Star::new(seed);
    let mut composites = Vec::with_capacity(b);
    let mut per_group: Vec<Vec<f64>> = (0..N_GROUPS).map(|_| Vec::with_capacity(b)).collect();
    for _ in 0..b {
        let (rep, _, _) = run_groups(sub, anchors, &norms, &mut Draw::Bootstrap(&mut rng));
        composites.push(composite_of(&rep, anchors));
        for (idx, slot) in per_group.iter_mut().enumerate() {
            slot.push(rep[idx]);
        }
    }
    let se = sample_std(&composites);

    let mut cis = [(0.0, 0.0); N_GROUPS];
    let mut ci_failures = Vec::new();
    for (idx, values) in per_group.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        let lo = percentile(values, 0.025);
        let hi = percentile(values, 0.975);
        cis[idx] = (lo, hi);
        let half_width = (hi - lo) / 2.0;
        if half_width > anchors.gates.ci_half_width_delta {
            ci_failures.push(GateFailure::CiHalfWidthTooWide {
                group: GROUP_KEYS[idx].to_string(),
                half_width,
                delta: anchors.gates.ci_half_width_delta,
            });
        }
    }

    if !ci_failures.is_empty() {
        return CompositeOutcome::Ineligible(Ineligible {
            gates: GateReport::from_failures(&ci_failures),
            reasons: ci_failures,
            groups: group_scores(&g, &penalty, Some(&cis)),
            composite,
            anchor_version: anchors.version,
            prereg_hash: anchors.prereg_hash.clone(),
        });
    }

    // Step 6 — LCB payout ranking → integer lattice.
    let lcb = (composite - anchors.bootstrap.lcb_z * se).max(0.0);
    let lattice = (SCORE_MAX as f64 * lcb.min(1.0)).round() as u64;

    CompositeOutcome::Scored(CompositeScore {
        groups: group_scores(&g, &penalty, Some(&cis)),
        composite,
        se,
        lcb,
        lattice,
        gates: GateReport::from_failures(&[]),
        anchor_version: anchors.version,
        prereg_hash: anchors.prereg_hash.clone(),
        bootstrap: BootstrapInfo {
            b: anchors.bootstrap.b,
            seed,
        },
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Minimal 8-group anchor set: one accuracy metric per group (chance 0),
    /// single-cluster series ⇒ zero-width CIs.
    const TEST_ANCHORS_JSON: &str = r#"{
        "version": 0,
        "groups": {
            "g1": { "weight": 0.25,  "metrics": { "org.g1.bpb_code": { "kind": "accuracy", "chance": 0.0 } } },
            "g2": { "weight": 0.15,  "metrics": { "org.g2.lambada_acc": { "kind": "accuracy", "chance": 0.0 } } },
            "g3": { "weight": 0.10,  "metrics": { "org.g3.mqar_acc": { "kind": "accuracy", "chance": 0.0 } } },
            "g4": { "weight": 0.15,  "metrics": { "org.g4.arithmetic_acc": { "kind": "accuracy", "chance": 0.0 } } },
            "g5": { "weight": 0.15,  "metrics": { "org.g5.ruler_acc": { "kind": "accuracy", "chance": 0.0 } } },
            "g6": { "weight": 0.075, "metrics": { "org.g6.auc": { "kind": "accuracy", "chance": 0.0 } } },
            "g7": { "weight": 0.075, "metrics": { "org.g7.throughput": { "kind": "accuracy", "chance": 0.0 } } },
            "g8": { "weight": 0.05,  "metrics": { "org.g8.loss_spike": { "kind": "accuracy", "chance": 0.0 } } }
        },
        "gates": { "g3_min": 0.25, "g8_min": 0.5, "ci_half_width_delta": 0.05,
                   "max_params": 350000000, "max_wall_s": 21600.0 },
        "mirror": { "tau_m": 0.05, "groups": ["g2", "g4"] },
        "bootstrap": { "b": 1000, "lcb_z": 1.645 }
    }"#;

    fn test_anchors() -> CompositeAnchors {
        CompositeAnchors::from_json(TEST_ANCHORS_JSON)
            .expect("test anchors parse")
            .with_prereg_hash("deadbeef".to_string())
    }

    fn series(value: f64, clusters: &[f64]) -> MetricSeries {
        MetricSeries {
            value,
            clusters: clusters
                .iter()
                .enumerate()
                .map(|(i, v)| (format!("c{i}"), *v))
                .collect(),
        }
    }

    /// Submission with one single-cluster metric per group (deterministic).
    fn uniform_submission(v: f64) -> SubmissionMetrics {
        let mut metrics = BTreeMap::new();
        for key in [
            "org.g1.bpb_code",
            "org.g2.lambada_acc",
            "org.g3.mqar_acc",
            "org.g4.arithmetic_acc",
            "org.g5.ruler_acc",
            "org.g6.auc",
            "org.g7.throughput",
            "org.g8.loss_spike",
        ] {
            metrics.insert(key.to_string(), series(v, &[v]));
        }
        SubmissionMetrics {
            metrics,
            budget: BudgetFacts {
                params: 300_000_000,
                wall_s: 20_000.0,
                tokens: 1_000_000,
            },
            mirrors: Vec::new(),
        }
    }

    fn scored(out: &CompositeOutcome) -> &CompositeScore {
        match out {
            CompositeOutcome::Scored(s) => s,
            CompositeOutcome::Ineligible(i) => panic!("unexpected ineligible: {:?}", i.reasons),
        }
    }

    // ---- Step 1: normalization edges ----

    #[test]
    fn accuracy_normalization_edges() {
        let n = NormDesc::Accuracy { chance: 0.25 };
        assert!((n.normalize(1.0) - 1.0).abs() < f64::EPSILON);
        assert!((n.normalize(0.25)).abs() < f64::EPSILON);
        assert!((n.normalize(0.1)).abs() < f64::EPSILON, "below chance → 0");
        assert!((n.normalize(0.625) - 0.5).abs() < 1e-12);
        assert!(
            (n.normalize(1.2) - 1.0).abs() < f64::EPSILON,
            "clips above 1"
        );
    }

    #[test]
    fn bpb_log_ratio_edges() {
        let n = NormDesc::BpbLogRatio {
            chance: 3.6,
            reference: 1.0,
        };
        assert!((n.normalize(1.0) - 1.0).abs() < 1e-12, "x == ref → 1");
        assert!(n.normalize(3.6).abs() < 1e-12, "x == chance → 0");
        assert!(
            n.normalize(5.0).abs() < f64::EPSILON,
            "worse than chance → 0"
        );
        assert!(
            (n.normalize(0.8) - 1.0).abs() < f64::EPSILON,
            "beats ref → clip 1"
        );
        assert!(n.normalize(0.0).abs() < f64::EPSILON, "non-positive → 0");
        assert!(n.normalize(f64::NAN).abs() < f64::EPSILON, "NaN → 0");
    }

    #[test]
    fn efficiency_log_ratio_both_directions() {
        let higher = NormDesc::EfficiencyLogRatio {
            reference: 1000.0,
            cap: 10_000.0,
        };
        assert!(higher.normalize(1000.0).abs() < 1e-12, "at ref → 0");
        assert!(
            (higher.normalize(10_000.0) - 1.0).abs() < 1e-12,
            "at cap → 1"
        );
        assert!(
            higher.normalize(100.0).abs() < f64::EPSILON,
            "below ref → 0"
        );
        let lower = NormDesc::EfficiencyLogRatio {
            reference: 4000.0,
            cap: 200.0,
        };
        assert!(
            lower.normalize(4000.0).abs() < 1e-12,
            "lower-better: at ref → 0"
        );
        assert!(
            (lower.normalize(200.0) - 1.0).abs() < 1e-12,
            "lower-better: at cap → 1"
        );
        assert!(
            (lower.normalize(100.0) - 1.0).abs() < f64::EPSILON,
            "beyond cap clips"
        );
    }

    #[test]
    fn stability_bounded_clips() {
        let n = NormDesc::StabilityBounded;
        assert!((n.normalize(0.7) - 0.7).abs() < f64::EPSILON);
        assert!((n.normalize(1.4) - 1.0).abs() < f64::EPSILON);
        assert!(n.normalize(-0.2).abs() < f64::EPSILON);
    }

    // ---- Anchor loading / validation ----

    #[test]
    fn shipped_placeholder_anchor_set_loads() {
        // The real embedded v0 JSON (shared contract with prism-recipe).
        let json = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../prism-recipe/anchors/v0.json"
        ));
        let anchors = CompositeAnchors::from_json(json).expect("v0 anchor set parses");
        assert_eq!(anchors.version, 0);
        assert!((anchors.gates.g3_min - 0.25).abs() < f64::EPSILON);
        assert!((anchors.gates.g8_min - 0.5).abs() < f64::EPSILON);
        assert_eq!(anchors.bootstrap.b, 1000);
        assert_eq!(anchors.groups.len(), N_GROUPS);
    }

    #[test]
    fn validation_rejects_bad_weights() {
        let bad = TEST_ANCHORS_JSON.replace("\"weight\": 0.25", "\"weight\": 0.9");
        assert!(matches!(
            CompositeAnchors::from_json(&bad),
            Err(CompositeError::Invalid(_))
        ));
    }

    #[test]
    fn validation_rejects_missing_group() {
        let bad = TEST_ANCHORS_JSON.replace("\"g8\":", "\"g8x\":");
        assert!(matches!(
            CompositeAnchors::from_json(&bad),
            Err(CompositeError::Invalid(_))
        ));
    }

    // ---- Step 4: geometric mean no-compensation property ----

    #[test]
    fn geometric_mean_punishes_weak_axis_vs_arithmetic() {
        let anchors = test_anchors();
        // A: uniform 0.5 on every axis.
        let a = evaluate(&uniform_submission(0.5), &anchors, 7);
        // B: g2 tanked to 0.05, others raised so the *arithmetic* mean of the
        // group scores is identical (0.05 + 7×x = 8×0.5).
        let x = (4.0 - 0.05) / 7.0;
        let mut b_sub = uniform_submission(x);
        b_sub
            .metrics
            .insert("org.g2.lambada_acc".to_string(), series(0.05, &[0.05]));
        let b = evaluate(&b_sub, &anchors, 7);
        let sa = scored(&a);
        let sb = scored(&b);
        assert!((sa.composite - 0.5).abs() < 1e-9);
        assert!(sb.composite < 0.45, "weak axis tanks C: {}", sb.composite);
        let arith = |g: &[GroupScore]| g.iter().map(|s| s.g).sum::<f64>() / 8.0;
        assert!(
            (arith(&sa.groups) - arith(&sb.groups)).abs() < 1e-9,
            "arithmetic means equal — only the geometric mean separates them"
        );
    }

    #[test]
    fn zero_axis_collapses_composite() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        // Zero a non-gated axis (g2) while keeping g3/g8 above their floors.
        sub.metrics
            .insert("org.g2.lambada_acc".to_string(), series(0.0, &[0.0]));
        let out = evaluate(&sub, &anchors, 1);
        assert!(scored(&out).composite.abs() < f64::EPSILON);
        assert_eq!(scored(&out).lattice, 0);
    }

    // ---- Step 2.5: mirror-gap penalty ----

    #[test]
    fn mirror_gap_penalizes_g2_pre_composite() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        sub.mirrors.push(MirrorPair {
            group: "g2".to_string(),
            metric: "org.g2.lambada_acc".to_string(),
            public: series(0.9, &[0.9]),
            mirror: series(0.6, &[0.6]),
        });
        let out = evaluate(&sub, &anchors, 3);
        let s = scored(&out);
        let g2 = &s.groups[1];
        // gap = 0.9 − 0.6 = 0.3; penalty = max(0, 0.3 − 0.05) = 0.25.
        assert!((g2.mirror_penalty - 0.25).abs() < 1e-9);
        assert!((g2.g - 0.65).abs() < 1e-9, "0.9 − 0.25 = 0.65");
        assert!(
            s.groups[3].mirror_penalty.abs() < f64::EPSILON,
            "no g4 pair → no penalty"
        );
    }

    #[test]
    fn mirror_gap_within_tolerance_is_free() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        sub.mirrors.push(MirrorPair {
            group: "g2".to_string(),
            metric: "org.g2.lambada_acc".to_string(),
            public: series(0.9, &[0.9]),
            mirror: series(0.87, &[0.87]),
        });
        let out = evaluate(&sub, &anchors, 3);
        assert!(scored(&out).groups[1].mirror_penalty.abs() < f64::EPSILON);
    }

    // ---- Step 3: gates ----

    #[test]
    fn gates_are_lexicographic_and_ordered() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        // Fail g3 (recall floor) AND budgets: g3 reason must come first.
        sub.metrics
            .insert("org.g3.mqar_acc".to_string(), series(0.1, &[0.1]));
        sub.budget.params = 500_000_000;
        let out = evaluate(&sub, &anchors, 5);
        match out {
            CompositeOutcome::Ineligible(i) => {
                assert!(matches!(i.reasons[0], GateFailure::G3BelowFloor { .. }));
                assert!(i
                    .reasons
                    .iter()
                    .any(|r| matches!(r, GateFailure::ParamsOverBudget { .. })));
                assert!(!i.gates.g3_ok);
                assert!(!i.gates.budgets_ok);
                assert!(i.gates.ci_ok, "bootstrap never runs when gates fail");
                assert!(i.groups.iter().all(|g| g.ci_lo.is_none()));
            }
            CompositeOutcome::Scored(_) => panic!("must be ineligible"),
        }
    }

    #[test]
    fn g8_stability_floor_blocks() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        sub.metrics
            .insert("org.g8.loss_spike".to_string(), series(0.4, &[0.4]));
        let out = evaluate(&sub, &anchors, 5);
        match out {
            CompositeOutcome::Ineligible(i) => {
                assert!(i
                    .reasons
                    .iter()
                    .any(|r| matches!(r, GateFailure::G8BelowFloor { .. })));
            }
            CompositeOutcome::Scored(_) => panic!("must be ineligible"),
        }
    }

    #[test]
    fn missing_metric_is_ineligible() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.9);
        sub.metrics.remove("org.g5.ruler_acc");
        let out = evaluate(&sub, &anchors, 5);
        match out {
            CompositeOutcome::Ineligible(i) => {
                assert!(i
                    .reasons
                    .iter()
                    .any(|r| matches!(r, GateFailure::MissingMetric { .. })));
                assert!(!i.gates.complete);
            }
            CompositeOutcome::Scored(_) => panic!("must be ineligible"),
        }
    }

    #[test]
    fn wide_ci_is_ineligible() {
        let anchors = test_anchors();
        // Two spread clusters per metric ⇒ bootstrap CI half-width ≫ δ.
        let mut sub = SubmissionMetrics {
            metrics: BTreeMap::new(),
            budget: BudgetFacts {
                params: 300_000_000,
                wall_s: 20_000.0,
                tokens: 0,
            },
            mirrors: Vec::new(),
        };
        for key in [
            "org.g1.bpb_code",
            "org.g2.lambada_acc",
            "org.g4.arithmetic_acc",
            "org.g5.ruler_acc",
            "org.g6.auc",
            "org.g7.throughput",
        ] {
            sub.metrics
                .insert(key.to_string(), series(0.5, &[0.0, 1.0]));
        }
        sub.metrics
            .insert("org.g3.mqar_acc".to_string(), series(0.5, &[0.3, 0.7]));
        sub.metrics
            .insert("org.g8.loss_spike".to_string(), series(0.8, &[0.6, 1.0]));
        let out = evaluate(&sub, &anchors, 11);
        match out {
            CompositeOutcome::Ineligible(i) => {
                assert!(i
                    .reasons
                    .iter()
                    .any(|r| matches!(r, GateFailure::CiHalfWidthTooWide { .. })));
                assert!(!i.gates.ci_ok);
                assert!(
                    i.groups.iter().all(|g| g.ci_lo.is_some()),
                    "CIs still reported"
                );
            }
            CompositeOutcome::Scored(_) => panic!("wide CIs must be ineligible"),
        }
    }

    // ---- Steps 5–6: bootstrap determinism, LCB, lattice ----

    #[test]
    fn bootstrap_is_seeded_deterministic() {
        let mut sub = uniform_submission(0.5);
        // Give g1 real (tight) clusters so the bootstrap resamples something
        // while staying under the CI-sufficiency gate.
        sub.metrics.insert(
            "org.g1.bpb_code".to_string(),
            series(0.5, &[0.49, 0.50, 0.51, 0.505, 0.495]),
        );
        let anchors = test_anchors();
        let a = evaluate(&sub, &anchors, 42);
        let b = evaluate(&sub, &anchors, 42);
        assert_eq!(a, b, "same seed ⇒ bit-identical outcome");
        let c = evaluate(&sub, &anchors, 43);
        assert!(
            scored(&a).se.to_bits() != scored(&c).se.to_bits(),
            "different seed ⇒ different SE"
        );
    }

    #[test]
    fn lcb_bounds_and_lattice() {
        let anchors = test_anchors();
        let out = evaluate(&uniform_submission(0.8), &anchors, 9);
        let s = scored(&out);
        assert!(s.lcb <= s.composite + f64::EPSILON, "LCB ≤ C");
        assert!(s.lcb >= 0.0);
        assert!(s.lattice <= SCORE_MAX);
        let expected = (SCORE_MAX as f64 * s.lcb).round() as u64;
        assert_eq!(s.lattice, expected);
        assert!(s.se.abs() < 1e-12, "single-cluster series ⇒ zero variance");
        assert_eq!(s.lattice, (SCORE_MAX as f64 * s.composite).round() as u64);
        assert_eq!(s.groups.len(), N_GROUPS);
        assert_eq!(s.anchor_version, 0);
        assert_eq!(s.prereg_hash.as_deref(), Some("deadbeef"));
        assert_eq!(s.bootstrap.b, 1000);
        assert_eq!(s.bootstrap.seed, 9);
    }

    #[test]
    fn composite_is_weighted_geometric_mean() {
        let anchors = test_anchors();
        let out = evaluate(&uniform_submission(0.6), &anchors, 2);
        assert!((scored(&out).composite - 0.6).abs() < 1e-9);
    }

    #[test]
    fn serde_roundtrip() {
        let anchors = test_anchors();
        let out = evaluate(&uniform_submission(0.7), &anchors, 4);
        let json = serde_json::to_string(&out).expect("serialize");
        let back: CompositeOutcome = serde_json::from_str(&json).expect("parse");
        assert_eq!(out, back);
        assert!(json.contains("\"status\":\"scored\""));
    }

    #[test]
    fn unknown_metrics_are_ignored() {
        let anchors = test_anchors();
        let mut sub = uniform_submission(0.7);
        sub.metrics
            .insert("org.zoneb.custom_metric".to_string(), series(0.0, &[0.0]));
        let out = evaluate(&sub, &anchors, 4);
        assert!(out.is_eligible());
    }
}
