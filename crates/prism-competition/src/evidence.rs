//! Build significance evidence from what the eval store actually persists.
//!
//! **Why this module exists.** `paired.rs` / `frontier.rs` define the rule;
//! without a bridge from stored measurements they are unreachable types. An
//! operator flipping `PRISM_EMISSION_MODE=sig` needs a [`SigContext`] built
//! from real rows, and it must be built from data a miner cannot write.
//!
//! ## What is actually retained, and what that permits
//!
//! `prism_eval_metric(run_id, key, value, clusters)` persists, for every
//! scored run, each `org.*` metric together with its **per-cluster values**
//! — the same `{value, clusters}` series the composite bootstraps. The
//! harness records one cluster id per **item** (`g2/hellaswag#17`,
//! `mqar/…#3`, `prose#4`; G1 domains are per-document). So per-example data
//! **is** retained for every scored run, including past champions, and a
//! genuine paired test is possible without adding any new state.
//!
//! **The one real limitation, stated plainly rather than papered over.**
//! Cluster ids are *positional* (`g2/<task>#<i>` is row `i` of that task's
//! asset file). They therefore align across two runs **only if both scored
//! the same asset slice.** Under a rotating private slice, an incumbent
//! measured on round *N−1*'s slice has no cluster in common with a
//! challenger on round *N*'s — the ids may collide numerically while
//! referring to different items, which is worse than not matching at all.
//!
//! Two consequences, both enforced here rather than documented and ignored:
//!
//! 1. [`PairedInput::slice_id`] is required to be **equal on both sides**,
//!    and it is derived from the slice identity the run recorded, not from
//!    the run id. [`paired_evidence`] refuses on mismatch
//!    ([`PairedRefusal::SliceMismatch`]) instead of pairing positionally.
//! 2. That refusal is exactly what makes the **champion re-run**
//!    ([`crate::rerun`]) load-bearing rather than a nice-to-have: it
//!    re-measures the incumbent *on the challenger's slice*, which is what
//!    produces two same-slice series. Absent a re-run, the honest outcome is
//!    "no admissible evidence ⇒ the champion holds" — never a paired verdict
//!    on aggregates. Comparing two independently-bootstrapped levels is the
//!    thing the paired design exists to avoid, so this module has no path
//!    that falls back to it.
//!
//! Second limitation, smaller: `ItemRecorder` caps the per-example channel
//! at 2 000 records per metric and 30 000 overall, so a very large slice is
//! truncated. Truncation reduces `n_paired`, which the
//! [`crate::paired::MIN_DECIDED`] floor already accounts for — it makes the
//! test refuse rather than decide on thin evidence.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::frontier::{AxisScore, EliteArchive};
use crate::paired::{paired_test, Direction, ExampleSeries, PairedInput, PairedOutcome};
use crate::paired::{PairedRefusal, MIN_DECIDED};
use crate::sig::SigContext;

/// One scored run as the evidence builder sees it.
///
/// Everything here is an operator measurement or a chain fact. There is no
/// field a miner supplies.
#[derive(Debug, Clone, Default)]
pub struct RunEvidence {
    /// Submitting hotkey (hex).
    pub hotkey: String,
    /// Slice identity the run was measured on: anchor version + eval tier +
    /// asset digest. Two runs pair only when these are equal.
    pub slice_id: String,
    /// `org.*` metric key -> per-cluster values, as persisted.
    pub series: BTreeMap<String, BTreeMap<String, f64>>,
    /// Per-group normalized values (`g1`..`g8`) for the elite archive.
    pub axes: BTreeMap<String, f64>,
    /// Whether every lexicographic gate passed.
    pub gates_ok: bool,
    /// Whether the run produced live contamination evidence
    /// ([`crate::contamination`]).
    pub contamination_checked: bool,
}

impl RunEvidence {
    /// Parse one run from the persisted `{key, value, clusters}` rows.
    ///
    /// `rows` is the `prism_eval_metric` projection: each element is
    /// `(key, clusters_json)` where `clusters_json` is the object the store
    /// wrote (`None` for a metric with no per-item channel). Non-finite and
    /// non-numeric cluster values are dropped rather than propagated —
    /// a NaN in an emission input must never reach a comparison.
    #[must_use]
    pub fn from_metric_rows(
        hotkey: impl Into<String>,
        slice_id: impl Into<String>,
        rows: &[(String, Option<Value>)],
    ) -> Self {
        let mut series: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for (key, clusters) in rows {
            let Some(obj) = clusters.as_ref().and_then(Value::as_object) else {
                continue;
            };
            let parsed: BTreeMap<String, f64> = obj
                .iter()
                .filter_map(|(c, v)| {
                    let f = v.as_f64()?;
                    f.is_finite().then(|| (c.clone(), f))
                })
                .collect();
            if !parsed.is_empty() {
                series.insert(key.clone(), parsed);
            }
        }
        Self {
            hotkey: hotkey.into(),
            slice_id: slice_id.into(),
            series,
            axes: BTreeMap::new(),
            gates_ok: true,
            contamination_checked: false,
        }
    }

    /// Attach per-group normalized values (the archive's descriptor space).
    #[must_use]
    pub fn with_axes(mut self, axes: BTreeMap<String, f64>) -> Self {
        self.axes = axes;
        self
    }

    /// Attach the gate verdict.
    #[must_use]
    pub fn with_gates(mut self, gates_ok: bool) -> Self {
        self.gates_ok = gates_ok;
        self
    }

    /// Attach the contamination-check verdict.
    #[must_use]
    pub fn with_contamination_checked(mut self, checked: bool) -> Self {
        self.contamination_checked = checked;
        self
    }
}

/// Scale direction for an `org.*` metric key.
///
/// Bits/byte, latency, energy and byte-footprint metrics are lower-better;
/// everything else Prism scores is an accuracy or a throughput, which is
/// higher-better. Unknown keys default to **higher-better**, matching the
/// composite's post-normalization convention where every axis is
/// higher-is-better.
#[must_use]
pub fn direction_for(metric: &str) -> Direction {
    const LOWER_BETTER: [&str; 6] = [
        "bits_per_byte",
        "ttft_ms",
        "joules_per_token",
        "state_bytes_per_token",
        "tokens_to_threshold",
        "latency",
    ];
    if LOWER_BETTER.iter().any(|frag| metric.contains(frag)) {
        Direction::LowerBetter
    } else {
        Direction::HigherBetter
    }
}

/// The metric the displacement test runs on, in preference order.
///
/// The composite itself is **not** a valid target: it is a gated weighted
/// geometric mean of group aggregates and has no per-example decomposition,
/// so there is nothing to pair. The test therefore runs on the dominant
/// axis — G1 prose bits/byte, Prism's tokenizer-neutral primary metric —
/// falling back to the other G1 domains. Accuracy axes are deliberately not
/// used as the primary: with a 0-width dead zone on a binary item, the win
/// rate is decided by ties and the test loses its discipline.
pub const DISPLACEMENT_METRICS: [&str; 4] = [
    "org.g1.bits_per_byte_prose",
    "org.g1.bits_per_byte_code",
    "org.g1.bits_per_byte_math",
    "org.g1.bits_per_byte_fresh_crawl",
];

/// Build the paired input for one champion/challenger pair on one metric.
///
/// Returns `None` when either side lacks a per-example series for `metric`.
#[must_use]
pub fn paired_input(
    champion: &RunEvidence,
    challenger: &RunEvidence,
    metric: &str,
) -> Option<PairedInput> {
    let champ = champion.series.get(metric)?;
    let chal = challenger.series.get(metric)?;
    // Same-slice discipline: an empty `slice_id` or a mismatch must produce
    // a refusal downstream, never a positional pairing across slices.
    let slice_id = if champion.slice_id == challenger.slice_id {
        champion.slice_id.clone()
    } else {
        String::new()
    };
    Some(PairedInput {
        metric: metric.to_owned(),
        direction: direction_for(metric),
        slice_id,
        champion: ExampleSeries {
            by_cluster: champ.clone(),
        },
        challenger: ExampleSeries {
            by_cluster: chal.clone(),
        },
    })
}

/// Run the displacement test on the first metric both sides can support.
///
/// Deterministic in its inputs: [`DISPLACEMENT_METRICS`] is a fixed order,
/// and the bootstrap is fixed-seed. Returns the refusal from the last
/// attempted metric when no metric yields a verdict, so the caller can log
/// *why* the champion held.
pub fn paired_evidence(
    champion: &RunEvidence,
    challenger: &RunEvidence,
) -> Result<(String, PairedOutcome), PairedRefusal> {
    let mut last = PairedRefusal::NoOverlap;
    for metric in DISPLACEMENT_METRICS {
        let Some(input) = paired_input(champion, challenger, metric) else {
            continue;
        };
        match paired_test(&input) {
            Ok(outcome) => return Ok((metric.to_owned(), outcome)),
            Err(refusal) => last = refusal,
        }
    }
    Err(last)
}

/// Build the per-axis elite archive from the round's runs.
#[must_use]
pub fn archive_from(runs: &[RunEvidence]) -> EliteArchive {
    let mut scores: Vec<AxisScore> = Vec::new();
    for run in runs {
        for (axis, value) in &run.axes {
            scores.push(AxisScore {
                axis: axis.clone(),
                hotkey: run.hotkey.clone(),
                value: *value,
                gates_ok: run.gates_ok,
            });
        }
    }
    EliteArchive::build(&scores)
}

/// Assemble the round's [`SigContext`] from stored evidence.
///
/// `incumbent` is the current champion hotkey and `tenure_days` its title
/// age — both operator/chain facts. `runs` is every scored run in the
/// round's competition set, including the incumbent's own row (its
/// re-measured row when a champion re-run fired, which is what makes the
/// pairing same-slice).
///
/// Champion selection is *not* decided here — this only supplies evidence.
/// The leading challenger is the highest-credit non-incumbent run for which
/// a paired verdict exists; a refusal yields `challenger: None`, which
/// [`crate::sig`] reads as "the champion holds".
#[must_use]
pub fn sig_context(
    incumbent: Option<&str>,
    tenure_days: u64,
    runs: &[RunEvidence],
    credits: &BTreeMap<String, u64>,
    previous_bps: BTreeMap<String, u64>,
) -> SigContext {
    let archive = archive_from(runs);
    // Contamination evidence must hold for every run that could be paid;
    // one unchecked run in the set means the round's evidence is not clean.
    let contamination_checked = !runs.is_empty() && runs.iter().all(|r| r.contamination_checked);
    let mut ctx = SigContext {
        incumbent: incumbent.map(str::to_owned),
        tenure_days,
        challenger: None,
        archive,
        previous_bps,
        contamination_checked,
    };
    let Some(inc) = incumbent else {
        return ctx;
    };
    let Some(champ_run) = runs.iter().find(|r| r.hotkey == inc) else {
        return ctx;
    };
    // Challengers by descending credit, then lex-smallest hotkey — the same
    // tie convention the collapse uses, so the choice is deterministic.
    let mut challengers: Vec<&RunEvidence> = runs.iter().filter(|r| r.hotkey != inc).collect();
    challengers.sort_by(|a, b| {
        let ca = credits.get(&a.hotkey).copied().unwrap_or(0);
        let cb = credits.get(&b.hotkey).copied().unwrap_or(0);
        cb.cmp(&ca).then_with(|| a.hotkey.cmp(&b.hotkey))
    });
    for cand in challengers {
        if let Ok((_, outcome)) = paired_evidence(champ_run, cand) {
            ctx.challenger = Some((cand.hotkey.clone(), outcome));
            break;
        }
    }
    ctx
}

/// Whether a paired series pair is large enough to be worth attempting.
///
/// Exposed so an operator dashboard can report "this round had no
/// admissible evidence" before running the test.
#[must_use]
pub fn has_admissible_overlap(champion: &RunEvidence, challenger: &RunEvidence) -> bool {
    if champion.slice_id.is_empty() || champion.slice_id != challenger.slice_id {
        return false;
    }
    DISPLACEMENT_METRICS.iter().any(|m| {
        let (Some(a), Some(b)) = (champion.series.get(*m), challenger.series.get(*m)) else {
            return false;
        };
        a.keys().filter(|k| b.contains_key(*k)).count() >= MIN_DECIDED
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    fn run(hotkey: &str, slice: &str, prose: &[f64]) -> RunEvidence {
        let clusters: serde_json::Map<String, Value> = prose
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("prose#{i}"), json!(v)))
            .collect();
        RunEvidence::from_metric_rows(
            hotkey,
            slice,
            &[(
                "org.g1.bits_per_byte_prose".to_owned(),
                Some(Value::Object(clusters)),
            )],
        )
        .with_contamination_checked(true)
    }

    #[test]
    fn per_example_series_round_trips_from_persisted_rows() {
        let r = run("aa", "v3/private/abc", &[1.1, 1.2, 1.3]);
        let s = r.series.get("org.g1.bits_per_byte_prose").unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.get("prose#1"), Some(&1.2));
    }

    #[test]
    fn non_numeric_and_non_finite_clusters_are_dropped() {
        let rows = vec![(
            "org.g1.bits_per_byte_prose".to_owned(),
            Some(json!({"a": 1.0, "b": "x", "c": null})),
        )];
        let r = RunEvidence::from_metric_rows("aa", "s", &rows);
        let s = r.series.get("org.g1.bits_per_byte_prose").unwrap();
        assert_eq!(s.len(), 1, "only the numeric cluster survives: {s:?}");
    }

    #[test]
    fn metrics_without_a_cluster_channel_are_skipped() {
        let rows = vec![
            ("org.g2.hellaswag_acc".to_owned(), None),
            ("org.g7.joules_per_token".to_owned(), Some(json!({}))),
        ];
        let r = RunEvidence::from_metric_rows("aa", "s", &rows);
        assert!(r.series.is_empty());
    }

    #[test]
    fn direction_is_inferred_from_the_metric_key() {
        assert_eq!(
            direction_for("org.g1.bits_per_byte_prose"),
            Direction::LowerBetter
        );
        assert_eq!(direction_for("org.g7.ttft_ms_32k"), Direction::LowerBetter);
        assert_eq!(
            direction_for("org.g6.tokens_to_threshold"),
            Direction::LowerBetter
        );
        assert_eq!(
            direction_for("org.g2.hellaswag_acc"),
            Direction::HigherBetter
        );
        // Unknown keys default to higher-better (post-normalization shape).
        assert_eq!(direction_for("org.g9.whatever"), Direction::HigherBetter);
    }

    #[test]
    fn same_slice_series_produce_a_real_paired_verdict() {
        let champ: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let chal: Vec<f64> = champ.iter().map(|v| v - 0.05).collect();
        let (metric, out) = paired_evidence(
            &run("champ", "v3/private/abc", &champ),
            &run("chal", "v3/private/abc", &chal),
        )
        .unwrap();
        assert_eq!(metric, "org.g1.bits_per_byte_prose");
        assert_eq!(out.n_decided, 200);
        assert!(out.displaces, "a 0.05 bpb win on every item must displace");
    }

    #[test]
    fn cross_slice_series_are_refused_not_paired_positionally() {
        // The load-bearing safety property: positional cluster ids from two
        // different slices must NOT be treated as the same examples.
        let champ: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let chal: Vec<f64> = champ.iter().map(|v| v - 0.05).collect();
        let err = paired_evidence(
            &run("champ", "v3/private/round-1", &champ),
            &run("chal", "v3/private/round-2", &chal),
        )
        .unwrap_err();
        assert_eq!(err, PairedRefusal::SliceMismatch);
        assert!(!has_admissible_overlap(
            &run("champ", "v3/private/round-1", &champ),
            &run("chal", "v3/private/round-2", &chal)
        ));
    }

    #[test]
    fn admissible_overlap_needs_enough_shared_examples() {
        let big: Vec<f64> = vec![1.2; 200];
        let small: Vec<f64> = vec![1.1; 5];
        assert!(has_admissible_overlap(
            &run("a", "s", &big),
            &run("b", "s", &big)
        ));
        assert!(
            !has_admissible_overlap(&run("a", "s", &big), &run("b", "s", &small)),
            "5 shared examples is below the decided floor"
        );
    }

    #[test]
    fn context_holds_the_crown_when_no_paired_evidence_exists() {
        // Challenger measured on a different slice ⇒ no verdict ⇒ hold.
        let champ = run("champ", "round-1", &vec![1.2; 200]);
        let chal = run("chal", "round-2", &vec![1.0; 200]);
        let credits: BTreeMap<String, u64> = [("chal".to_owned(), 900_u64)].into_iter().collect();
        let ctx = sig_context(Some("champ"), 3, &[champ, chal], &credits, BTreeMap::new());
        assert!(
            ctx.challenger.is_none(),
            "no admissible evidence must not manufacture a displacement"
        );
        assert_eq!(ctx.incumbent.as_deref(), Some("champ"));
    }

    #[test]
    fn context_picks_the_highest_credit_challenger_with_evidence() {
        let base: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let better: Vec<f64> = base.iter().map(|v| v - 0.05).collect();
        let champ = run("champ", "s", &base);
        // `rich` has the higher credit but no same-slice series; `poor` has
        // evidence. The rich one must be tried first and skipped.
        let rich = run("rich", "other", &better);
        let poor = run("poor", "s", &better);
        let credits: BTreeMap<String, u64> = [("rich".to_owned(), 900), ("poor".to_owned(), 500)]
            .into_iter()
            .collect();
        let ctx = sig_context(
            Some("champ"),
            0,
            &[champ, rich, poor],
            &credits,
            BTreeMap::new(),
        );
        let (hk, out) = ctx.challenger.unwrap();
        assert_eq!(hk, "poor");
        assert!(out.displaces);
    }

    #[test]
    fn archive_is_built_from_stored_axis_values() {
        let a = run("aa", "s", &[1.2]).with_axes(
            [("g1".to_owned(), 0.5_f64), ("g3".to_owned(), 0.9)]
                .into_iter()
                .collect(),
        );
        let b = run("bb", "s", &[1.1]).with_axes(
            [("g1".to_owned(), 0.7_f64), ("g3".to_owned(), 0.1)]
                .into_iter()
                .collect(),
        );
        let arch = archive_from(&[a, b]);
        assert_eq!(arch.holders.get("g1").unwrap().0, "bb");
        assert_eq!(arch.holders.get("g3").unwrap().0, "aa");
    }

    #[test]
    fn gate_failures_do_not_reach_the_archive() {
        let cheat = run("cheat", "s", &[1.0])
            .with_axes([("g3".to_owned(), 1.0_f64)].into_iter().collect())
            .with_gates(false);
        let honest = run("honest", "s", &[1.0])
            .with_axes([("g3".to_owned(), 0.2_f64)].into_iter().collect());
        let arch = archive_from(&[cheat, honest]);
        assert_eq!(arch.holders.get("g3").unwrap().0, "honest");
    }

    #[test]
    fn contamination_evidence_is_all_or_nothing_for_the_round() {
        let ok = run("aa", "s", &[1.2]);
        let unchecked = run("bb", "s", &[1.1]).with_contamination_checked(false);
        let ctx = sig_context(
            None,
            0,
            std::slice::from_ref(&ok),
            &BTreeMap::new(),
            BTreeMap::new(),
        );
        assert!(ctx.contamination_checked);
        let ctx = sig_context(None, 0, &[ok, unchecked], &BTreeMap::new(), BTreeMap::new());
        assert!(
            !ctx.contamination_checked,
            "one unchecked run taints the round's evidence"
        );
        // An empty round proves nothing.
        assert!(
            !sig_context(None, 0, &[], &BTreeMap::new(), BTreeMap::new()).contamination_checked
        );
    }

    #[test]
    fn evidence_building_is_deterministic() {
        let base: Vec<f64> = (0..200).map(|i| 1.20 + f64::from(i) * 0.001).collect();
        let better: Vec<f64> = base.iter().map(|v| v - 0.05).collect();
        let runs = vec![run("champ", "s", &base), run("chal", "s", &better)];
        let credits: BTreeMap<String, u64> = [("chal".to_owned(), 900_u64)].into_iter().collect();
        let first = sig_context(Some("champ"), 1, &runs, &credits, BTreeMap::new());
        for _ in 0..5 {
            let again = sig_context(Some("champ"), 1, &runs, &credits, BTreeMap::new());
            assert_eq!(first.challenger, again.challenger);
            assert_eq!(first.archive, again.archive);
        }
    }
}
