//! Zone B participant-reported metrics: validation lattice
//! (`docs/spikes/prism-v3/research/09-miner-metrics-leaderboards.md` §2–§3, §7).
//!
//! The contract types (envelope, verdicts, caps) live in `prism-zoneb` and
//! are re-exported here; this module is the pure validation pipeline:
//!
//! - `quarantined` — provable dishonesty or tamper evidence: organizer
//!   ground-truth contradictions (token/step/wall-clock), physics-ceiling
//!   violations (MFU), terminal-loss band breaks, broken hash chains,
//!   internal series inconsistency, cross-miner copied series, `org.*`
//!   namespace forgery. Routed to the agentic anti-cheat path as evidence.
//! - `flagged` — statistical outliers (cohort median ± 6·MAD) and checks
//!   that could not run because organizer ground truth is unavailable
//!   (e.g. Sim pods): "atypical — under review", **never** an auto-zero.
//! - `ok` — clean.
//!
//! All functions are pure; storage I/O lives in `prism-eval-store`.

use std::collections::BTreeMap;

pub use prism_zoneb::{
    is_valid_metric_name, CohortMetric, Direction, GroundTruth, IngestReject, MetricKind,
    PreparedReport, Verdict, VerdictReason, ZoneBEnvelope, ZoneBMetric, MAX_PAYLOAD_BYTES,
    MAX_SCALARS, MAX_SERIES, MAX_SERIES_POINTS,
};
use sha2::{Digest, Sha256};

/// MFU ceiling: above this fraction of the GPU's dense FP16/BF16 peak a
/// throughput claim is physically implausible (well-tuned H100 runs land
/// ~0.35–0.55; research/09 §2.2 cites PaLM 0.462).
pub const MFU_CEILING: f64 = 0.75;
/// Reported wall-clock may exceed the organizer-measured wall clock by 5%
/// before the report is quarantined (research/09 §2.1).
pub const WALL_CLOCK_SLACK: f64 = 1.05;
/// Terminal-loss band around the organizer bpb converted to nats
/// (`expected = bpb · ln 2`): the miner-reported terminal train loss must
/// lie in `[expected · LO, expected · HI]`. Wide on purpose — the band
/// catches fabricated/short-circuited curves, not honest overfitting; the
/// mapping is the recipe-declared v0 default.
pub const TERMINAL_BAND_LO: f64 = 0.1;
/// See [`TERMINAL_BAND_LO`].
pub const TERMINAL_BAND_HI: f64 = 4.0;
/// Cohort outlier band: median ± `MAD_K`·MAD (research/09 §2.4).
pub const MAD_K: f64 = 6.0;
/// Minimum cohort size before MAD outlier detection runs.
pub const MIN_COHORT: usize = 4;

/// Dense FP16/BF16 tensor peak FLOP/s per GPU class, longest-name-first so
/// substring matching never maps `A100` onto `A10`. Unknown GPUs skip the
/// MFU check (degraded to `flagged`, never a crash).
const GPU_PEAK_FLOPS: &[(&str, f64)] = &[
    ("B200", 2_250.0e12),
    ("H200", 989.0e12),
    ("H100", 989.0e12),
    ("A100", 312.0e12),
    ("RTX 5090", 419.0e12),
    ("RTX 4090", 165.0e12),
    ("A10", 125.0e12),
    ("L4", 121.0e12),
    ("RTX 3090", 71.0e12),
];

/// Envelope structural validation: names, caps, array shapes, unit charset.
/// `org.*` keys pass through here (they are a verdict, not a reject).
///
/// # Errors
/// [`IngestReject`] on any hard violation.
pub fn validate_envelope(env: &ZoneBEnvelope) -> Result<(), IngestReject> {
    let bytes = serde_json::to_vec(env).map_err(|e| IngestReject::BadEnvelope(e.to_string()))?;
    if bytes.len() > MAX_PAYLOAD_BYTES {
        return Err(IngestReject::PayloadTooLarge);
    }
    let (mut scalars, mut series) = (0usize, 0usize);
    for (name, m) in &env.metrics {
        if name.starts_with("org.") {
            continue;
        }
        if !is_valid_metric_name(name) {
            return Err(IngestReject::InvalidName(name.clone()));
        }
        if let Some(u) = &m.unit {
            if !valid_unit(u) {
                return Err(IngestReject::BadEnvelope(format!("bad unit for {name}")));
            }
        }
        match &m.kind {
            MetricKind::Scalar { .. } => scalars += 1,
            MetricKind::Series { step, value } => {
                series += 1;
                if step.len() != value.len() {
                    return Err(IngestReject::BadEnvelope(format!(
                        "{name}: step/value mismatch"
                    )));
                }
                if step.len() > MAX_SERIES_POINTS {
                    return Err(IngestReject::SeriesTooLong(name.clone()));
                }
            }
            MetricKind::Histogram { boundaries, counts } => {
                if counts.len() != boundaries.len() + 1 {
                    return Err(IngestReject::BadEnvelope(format!("{name}: bad histogram")));
                }
            }
        }
    }
    if scalars > MAX_SCALARS {
        return Err(IngestReject::TooManyScalars(scalars));
    }
    if series > MAX_SERIES {
        return Err(IngestReject::TooManySeries(series));
    }
    Ok(())
}

fn valid_unit(u: &str) -> bool {
    let ok = |c: u8| c.is_ascii_alphanumeric() || b"_{}%/.*^+-[]'".contains(&c);
    u.len() <= 32 && u.bytes().all(ok)
}

/// Validate + chain one report. Pure: the caller supplies the stored chain
/// head, organizer ground truth, and cohort observations.
///
/// # Errors
/// [`IngestReject`] when the envelope is malformed/over caps (nothing to
/// store); verdicts never error — they are stored with the report.
pub fn prepare_report(
    submission_id: &str,
    env: &ZoneBEnvelope,
    head: Option<(u64, String)>,
    gt: &GroundTruth,
    cohort: &BTreeMap<String, CohortMetric>,
) -> Result<PreparedReport, IngestReject> {
    use VerdictReason as R;
    validate_envelope(env)?;
    let (seq, prev_hash) = match head {
        Some((s, h)) => (s + 1, h),
        None => (0, submission_id.to_string()),
    };
    let mut reasons = Vec::new();
    if let Some(p) = &env.prev_hash {
        if *p != prev_hash {
            reasons.push(R::HashChainMismatch {
                expected: prev_hash.clone(),
                got: p.clone(),
            });
        }
    }
    // Strip miner-emitted org.* from the payload; each is a quarantine finding.
    let mut metrics = BTreeMap::new();
    for (name, m) in &env.metrics {
        if name.starts_with("org.") {
            reasons.push(R::OrgNamespace { key: name.clone() });
        } else {
            metrics.insert(name.clone(), m.clone());
        }
    }
    let clean = ZoneBEnvelope {
        schema_version: env.schema_version.clone(),
        prev_hash: Some(prev_hash.clone()),
        metrics,
    };
    check_internal(&clean, gt.max_steps, &mut reasons);
    check_ground_truth(&clean, gt, &mut reasons);
    check_cohort(&clean, cohort, &mut reasons);
    let verdict = if reasons.iter().any(R::is_quarantine) {
        Verdict::Quarantined
    } else if reasons.is_empty() {
        Verdict::Ok
    } else {
        Verdict::Flagged
    };
    let payload = serde_json::json!({
        "submission_id": submission_id,
        "seq": seq,
        "schema_version": clean.schema_version,
        "prev_hash": prev_hash,
        "metrics": clean.metrics,
    });
    let bytes =
        serde_json::to_vec(&payload).map_err(|e| IngestReject::BadEnvelope(e.to_string()))?;
    let report_hash = hex::encode(Sha256::digest(&bytes));
    Ok(PreparedReport {
        seq,
        prev_hash,
        report_hash,
        payload,
        verdict,
        reasons,
    })
}

/// Content hash of one series (cross-miner duplicate detection).
#[must_use]
pub fn series_hash(name: &str, step: &[u64], value: &[f64]) -> String {
    let mut h = Sha256::new();
    h.update(name.as_bytes());
    for (s, v) in step.iter().zip(value) {
        h.update(s.to_le_bytes());
        h.update(v.to_le_bytes());
    }
    hex::encode(h.finalize())
}

fn scalar(env: &ZoneBEnvelope, name: &str) -> Option<f64> {
    match &env.metrics.get(name)?.kind {
        MetricKind::Scalar { value } => Some(*value),
        _ => None,
    }
}

fn series_terminal(env: &ZoneBEnvelope, name: &str) -> Option<f64> {
    match &env.metrics.get(name)?.kind {
        MetricKind::Series { value, .. } => value.last().copied(),
        _ => None,
    }
}

fn peak_flops(gpu: &str) -> Option<f64> {
    let up = gpu.to_ascii_uppercase();
    GPU_PEAK_FLOPS
        .iter()
        .find(|(k, _)| up.contains(k))
        .map(|(_, f)| *f)
}

/// Internal series consistency (research/09 §2.3): finite values, monotonic
/// contiguous steps, recipe step cap.
fn check_internal(env: &ZoneBEnvelope, max_steps: u64, reasons: &mut Vec<VerdictReason>) {
    use VerdictReason as R;
    for (name, m) in &env.metrics {
        let nm = || name.clone();
        match &m.kind {
            MetricKind::Scalar { value } if !value.is_finite() => {
                reasons.push(R::NonFiniteValue { name: nm() });
            }
            MetricKind::Series { step, value } => {
                if value.iter().any(|v| !v.is_finite()) {
                    reasons.push(R::NonFiniteValue { name: nm() });
                }
                for w in step.windows(2) {
                    if w[1] == w[0] {
                        reasons.push(R::DuplicateSteps {
                            name: nm(),
                            step: w[0],
                        });
                        break;
                    }
                    if w[1] < w[0] {
                        reasons.push(R::NonMonotonicSteps { name: nm() });
                        break;
                    }
                }
                if let Some(&last) = step.last() {
                    if last > max_steps {
                        reasons.push(R::StepCapExceeded {
                            step: last,
                            cap: max_steps,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

/// Compare one claimed scalar against optional organizer ground truth:
/// both present → `f` may yield a finding; claim without ground truth →
/// degraded `flagged` finding; no claim → no check.
fn vs_ground_truth(
    claimed: Option<f64>,
    observed: Option<f64>,
    check: &str,
    reasons: &mut Vec<VerdictReason>,
    f: impl Fn(f64, f64) -> Option<VerdictReason>,
) {
    match (claimed, observed) {
        (Some(c), Some(o)) => reasons.extend(f(c, o)),
        (Some(_), None) => reasons.push(VerdictReason::GroundTruthUnavailable {
            check: check.to_string(),
        }),
        (None, _) => {}
    }
}

/// Ground-truth cross-checks (research/09 §2.1–2.2, §7 steps 1–3). Missing
/// organizer facts degrade the specific check to a `flagged` finding.
fn check_ground_truth(env: &ZoneBEnvelope, gt: &GroundTruth, reasons: &mut Vec<VerdictReason>) {
    use VerdictReason as R;
    let tokens = gt.tokens_seen.map(|x| x as f64);
    let seen = scalar(env, "miner.train.tokens_seen");
    vs_ground_truth(seen, tokens, "token_accounting", reasons, |c, o| {
        (c > o).then_some(R::TokenCountExceedsObserved {
            claimed: c as u64,
            observed: o as u64,
        })
    });
    if let Some(s) = scalar(env, "miner.train.steps") {
        if s > gt.max_steps as f64 {
            reasons.push(R::StepCapExceeded {
                step: s as u64,
                cap: gt.max_steps,
            });
        }
    }
    let wall = scalar(env, "miner.train.wall_clock_s");
    vs_ground_truth(wall, gt.wall_s, "wall_clock", reasons, |c, o| {
        (c > o * WALL_CLOCK_SLACK).then_some(R::WallClockExceedsObserved {
            claimed: c,
            observed: o,
        })
    });
    if let Some(tps) = scalar(env, "miner.sys.tokens_per_s") {
        let gpu = gt.gpu_type.clone().unwrap_or_default();
        match (gt.n_params, gt.gpu_type.as_deref().and_then(peak_flops)) {
            (Some(p), Some(peak)) if tps > 0.0 => {
                let mfu = tps * 6.0 * p as f64 / peak;
                if mfu > MFU_CEILING {
                    reasons.push(R::MfuAboveCeiling {
                        mfu,
                        ceiling: MFU_CEILING,
                        gpu,
                    });
                }
            }
            (Some(_), Some(_)) => {}
            _ => reasons.push(R::GroundTruthUnavailable {
                check: "mfu_ceiling".to_string(),
            }),
        }
    }
    if let Some(t) = series_terminal(env, "miner.train.loss") {
        match gt.bpb {
            Some(bpb) => {
                let e = bpb * std::f64::consts::LN_2;
                if t < e * TERMINAL_BAND_LO || t > e * TERMINAL_BAND_HI {
                    reasons.push(R::TerminalLossOutsideBand {
                        terminal: t,
                        expected: e,
                    });
                }
            }
            None => reasons.push(R::GroundTruthUnavailable {
                check: "terminal_loss".to_string(),
            }),
        }
    }
}

/// Cohort statistics (research/09 §2.4): copied-series detection
/// (quarantine) and MAD outliers (flagged only — outliers are the product
/// of an architecture competition, never an auto-zero).
fn check_cohort(
    env: &ZoneBEnvelope,
    cohort: &BTreeMap<String, CohortMetric>,
    reasons: &mut Vec<VerdictReason>,
) {
    use VerdictReason as R;
    for (name, m) in &env.metrics {
        let Some(c) = cohort.get(name) else { continue };
        let nm = name.clone();
        if let MetricKind::Series { step, value } = &m.kind {
            let h = series_hash(name, step, value);
            if c.series_hashes.contains(&h) {
                reasons.push(R::DuplicateSeries { name: nm.clone() });
            }
        }
        let terminal = match &m.kind {
            MetricKind::Scalar { value } => Some(*value),
            MetricKind::Series { value, .. } => value.last().copied(),
            MetricKind::Histogram { .. } => None,
        };
        if let Some(x) = terminal {
            if let Some((med, mad)) = median_mad(&c.terminals) {
                if (x - med).abs() > MAD_K * mad {
                    reasons.push(R::CohortOutlier {
                        name: nm,
                        value: x,
                        median: med,
                        mad,
                    });
                }
            }
        }
    }
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        f64::midpoint(v[n / 2 - 1], v[n / 2])
    }
}

fn median_mad(xs: &[f64]) -> Option<(f64, f64)> {
    if xs.len() < MIN_COHORT {
        return None;
    }
    let med = median(xs);
    let devs: Vec<f64> = xs.iter().map(|x| (x - med).abs()).collect();
    Some((med, median(&devs)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const SUB: &str = "subm-1";

    fn scalar_metric(x: f64) -> ZoneBMetric {
        ZoneBMetric {
            kind: MetricKind::Scalar { value: x },
            unit: None,
            direction: None,
        }
    }

    fn series_metric(step: Vec<u64>, value: Vec<f64>) -> ZoneBMetric {
        ZoneBMetric {
            kind: MetricKind::Series { step, value },
            unit: None,
            direction: None,
        }
    }

    fn envelope(metrics: &[(&str, ZoneBMetric)]) -> ZoneBEnvelope {
        ZoneBEnvelope {
            schema_version: "1.3.0".to_string(),
            prev_hash: None,
            metrics: metrics
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        }
    }

    fn gt() -> GroundTruth {
        GroundTruth {
            bpb: Some(1.0),
            wall_s: Some(100.0),
            tokens_seen: Some(1_000_000),
            n_params: Some(300_000_000),
            gpu_type: Some("NVIDIA H100".to_string()),
            max_steps: 20_000,
        }
    }

    fn prepare(env: &ZoneBEnvelope) -> PreparedReport {
        prepare_report(SUB, env, None, &gt(), &BTreeMap::new()).expect("prepare")
    }

    #[test]
    fn clean_report_is_ok_and_chains_to_submission() {
        let env = envelope(&[("miner.train.steps", scalar_metric(100.0))]);
        let r = prepare(&env);
        assert_eq!(r.verdict, Verdict::Ok);
        assert_eq!(r.seq, 0);
        assert_eq!(r.prev_hash, SUB);
        assert_eq!(r.report_hash.len(), 64);
        // Second report chains to the first report's hash.
        let r2 = prepare_report(
            SUB,
            &env,
            Some((r.seq, r.report_hash.clone())),
            &gt(),
            &BTreeMap::new(),
        )
        .expect("prepare");
        assert_eq!(r2.seq, 1);
        assert_eq!(r2.prev_hash, r.report_hash);
    }

    #[test]
    fn miner_prev_hash_mismatch_quarantines() {
        let mut env = envelope(&[("miner.train.steps", scalar_metric(1.0))]);
        env.prev_hash = Some("deadbeef".to_string());
        let r = prepare(&env);
        assert_eq!(r.verdict, Verdict::Quarantined);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::HashChainMismatch { .. })));
        // A matching miner chain is accepted.
        let good = prepare_report(
            SUB,
            &{
                let mut e = envelope(&[("miner.train.steps", scalar_metric(1.0))]);
                e.prev_hash = Some(r.prev_hash.clone());
                e
            },
            Some((r.seq, r.report_hash.clone())),
            &gt(),
            &BTreeMap::new(),
        );
        assert!(good.is_ok());
    }

    #[test]
    fn org_namespace_quarantined_and_stripped() {
        let env = envelope(&[
            ("org.eval.bpb", scalar_metric(0.01)),
            ("miner.train.steps", scalar_metric(10.0)),
        ]);
        let r = prepare(&env);
        assert_eq!(r.verdict, Verdict::Quarantined);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::OrgNamespace { .. })));
        let keys: Vec<&str> = r.payload["metrics"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, vec!["miner.train.steps"], "org.* never stored");
    }

    #[test]
    fn token_step_wallclock_ground_truth() {
        // Claimed tokens above the organizer counter → quarantine.
        let env = envelope(&[("miner.train.tokens_seen", scalar_metric(2_000_000.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Quarantined);
        // Within the counter → ok.
        let env = envelope(&[("miner.train.tokens_seen", scalar_metric(900_000.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Ok);
        // Steps above the recipe cap → quarantine.
        let env = envelope(&[("miner.train.steps", scalar_metric(20_001.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Quarantined);
        // Wall-clock beyond measured × 1.05 → quarantine.
        let env = envelope(&[("miner.train.wall_clock_s", scalar_metric(106.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Quarantined);
        let env = envelope(&[("miner.train.wall_clock_s", scalar_metric(104.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Ok);
    }

    #[test]
    fn missing_ground_truth_flags_never_crashes() {
        let sim = GroundTruth {
            max_steps: 20_000,
            ..GroundTruth::default()
        };
        let env = envelope(&[
            ("miner.train.tokens_seen", scalar_metric(5.0)),
            (
                "miner.train.loss",
                series_metric(vec![0, 1], vec![2.0, 1.0]),
            ),
            ("miner.sys.tokens_per_s", scalar_metric(1000.0)),
        ]);
        let r = prepare_report(SUB, &env, None, &sim, &BTreeMap::new()).expect("prepare");
        assert_eq!(r.verdict, Verdict::Flagged, "degraded, never quarantined");
        assert!(r
            .reasons
            .iter()
            .all(|x| matches!(x, VerdictReason::GroundTruthUnavailable { .. })));
        // A report with no governed metrics stays ok even without ground truth.
        let env = envelope(&[("miner.probe.custom", scalar_metric(1.0))]);
        let r = prepare_report(SUB, &env, None, &sim, &BTreeMap::new()).expect("prepare");
        assert_eq!(r.verdict, Verdict::Ok);
    }

    #[test]
    fn mfu_ceiling_per_gpu() {
        // 420k tok/s on a 300M model ≈ MFU 0.76 on H100 → quarantine.
        let env = envelope(&[("miner.sys.tokens_per_s", scalar_metric(420_000.0))]);
        let r = prepare(&env);
        assert_eq!(r.verdict, Verdict::Quarantined);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::MfuAboveCeiling { .. })));
        // 200k tok/s ≈ MFU 0.38 → ok.
        let env = envelope(&[("miner.sys.tokens_per_s", scalar_metric(200_000.0))]);
        assert_eq!(prepare(&env).verdict, Verdict::Ok);
        // Unknown GPU → degraded flag.
        let mut g = gt();
        g.gpu_type = Some("SIM".to_string());
        let r = prepare_report(SUB, &env, None, &g, &BTreeMap::new()).expect("prepare");
        assert_eq!(r.verdict, Verdict::Flagged);
        // A100 must not substring-match as A10.
        assert_eq!(peak_flops("NVIDIA A100-SXM4-80GB"), Some(312.0e12));
    }

    #[test]
    fn terminal_loss_band_against_bpb() {
        // bpb 1.0 → expected 0.693 nats; band [0.069, 2.77].
        let env = envelope(&[(
            "miner.train.loss",
            series_metric(vec![0, 1], vec![3.0, 0.7]),
        )]);
        assert_eq!(prepare(&env).verdict, Verdict::Ok);
        let env = envelope(&[(
            "miner.train.loss",
            series_metric(vec![0, 1], vec![3.0, 9.9]),
        )]);
        let r = prepare(&env);
        assert_eq!(r.verdict, Verdict::Quarantined);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::TerminalLossOutsideBand { .. })));
    }

    #[test]
    fn series_internal_consistency() {
        let env = envelope(&[(
            "miner.train.loss",
            series_metric(vec![0, 2, 1], vec![3.0, 2.0, 1.0]),
        )]);
        let r = prepare(&env);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::NonMonotonicSteps { .. })));
        let env = envelope(&[(
            "miner.train.loss",
            series_metric(vec![0, 1, 1], vec![3.0, 2.0, 1.0]),
        )]);
        let r = prepare(&env);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::DuplicateSteps { .. })));
        let env = envelope(&[(
            "miner.train.loss",
            series_metric(vec![0, 20_001], vec![3.0, 1.0]),
        )]);
        let r = prepare(&env);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::StepCapExceeded { .. })));
    }

    #[test]
    fn duplicate_series_across_miners_quarantines() {
        let series = series_metric(vec![0, 1, 2], vec![3.0, 2.0, 1.0]);
        let hash = series_hash("miner.train.loss", &[0, 1, 2], &[3.0, 2.0, 1.0]);
        let mut cohort = BTreeMap::new();
        cohort.insert(
            "miner.train.loss".to_string(),
            CohortMetric {
                terminals: vec![1.0],
                series_hashes: vec![hash],
            },
        );
        let env = envelope(&[("miner.train.loss", series)]);
        let r = prepare_report(SUB, &env, None, &gt(), &cohort).expect("prepare");
        assert_eq!(r.verdict, Verdict::Quarantined);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::DuplicateSeries { .. })));
    }

    #[test]
    fn mad_outlier_flagged_never_quarantined() {
        let mut cohort = BTreeMap::new();
        cohort.insert(
            "miner.sys.tokens_per_s".to_string(),
            CohortMetric {
                terminals: vec![100.0, 101.0, 99.0, 100.5, 99.5],
                series_hashes: vec![],
            },
        );
        // Wildly off-cohort but physically possible → flagged only.
        let env = envelope(&[("miner.sys.tokens_per_s", scalar_metric(250.0))]);
        let r = prepare_report(SUB, &env, None, &gt(), &cohort).expect("prepare");
        assert_eq!(r.verdict, Verdict::Flagged);
        assert!(r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::CohortOutlier { .. })));
        // In-cohort value → no finding.
        let env = envelope(&[("miner.sys.tokens_per_s", scalar_metric(100.2))]);
        let r = prepare_report(SUB, &env, None, &gt(), &cohort).expect("prepare");
        assert_eq!(r.verdict, Verdict::Ok);
        // Small cohort (< 4) → no statistical check at all.
        let mut small = BTreeMap::new();
        small.insert(
            "miner.sys.tokens_per_s".to_string(),
            CohortMetric {
                terminals: vec![1.0, 2.0],
                series_hashes: vec![],
            },
        );
        let env = envelope(&[("miner.sys.tokens_per_s", scalar_metric(250.0))]);
        let r = prepare_report(SUB, &env, None, &gt(), &small).expect("prepare");
        assert!(!r
            .reasons
            .iter()
            .any(|x| matches!(x, VerdictReason::CohortOutlier { .. })));
    }

    #[test]
    fn quarantine_beats_flag() {
        let mut cohort = BTreeMap::new();
        cohort.insert(
            "miner.sys.tokens_per_s".to_string(),
            CohortMetric {
                terminals: vec![1.0, 2.0, 3.0, 4.0],
                series_hashes: vec![],
            },
        );
        // Both an outlier (flag) and a token lie (quarantine) → quarantined.
        let env = envelope(&[
            ("miner.sys.tokens_per_s", scalar_metric(999.0)),
            ("miner.train.tokens_seen", scalar_metric(9_000_000.0)),
        ]);
        let r = prepare_report(SUB, &env, None, &gt(), &cohort).expect("prepare");
        assert_eq!(r.verdict, Verdict::Quarantined);
    }

    #[test]
    fn caps_reject_hard() {
        // 65 scalars > 64 cap.
        let many: Vec<(String, ZoneBMetric)> = (0..65)
            .map(|i| (format!("miner.a.k{i}"), scalar_metric(1.0)))
            .collect();
        let env = ZoneBEnvelope {
            schema_version: "1.3.0".to_string(),
            prev_hash: None,
            metrics: many.into_iter().collect(),
        };
        assert!(matches!(
            validate_envelope(&env),
            Err(IngestReject::TooManyScalars(65))
        ));
        // 10_001 points > 10k cap.
        let env = envelope(&[(
            "miner.train.loss",
            series_metric((0..10_001).collect(), vec![1.0; 10_001]),
        )]);
        assert!(matches!(
            validate_envelope(&env),
            Err(IngestReject::SeriesTooLong(_))
        ));
        // Bad name in the envelope path.
        let env = envelope(&[("train_steps", scalar_metric(1.0))]);
        assert!(matches!(
            validate_envelope(&env),
            Err(IngestReject::InvalidName(_))
        ));
        // Series length mismatch.
        let env = envelope(&[("miner.train.loss", series_metric(vec![0, 1], vec![1.0]))]);
        assert!(matches!(
            validate_envelope(&env),
            Err(IngestReject::BadEnvelope(_))
        ));
    }
}
