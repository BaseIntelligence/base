//! API JSON views over eval records (additive shapes; Zone B is always
//! labelled participant-reported) plus the cohort projection used by the
//! Zone B validators.

use std::collections::BTreeMap;

use prism_store::eval::{
    AnchorSetRecord, EvalGroupRecord, EvalMetricRecord, EvalRunRecord, MetricReportRecord,
    MirrorPairRecord, PreregRecord,
};
use prism_zoneb::{series_hash, CohortMetric, MetricKind, Verdict, ZoneBMetric};
use serde_json::{json, Value};

/// `GET /v1/submissions/{id}` `eval` field: the stored `CompositeOutcome`
/// blob plus run provenance, with groups projected from the normalized
/// rows (proves the queryable projection round-trips).
#[must_use]
pub fn eval_detail(run: &EvalRunRecord, groups: &[EvalGroupRecord]) -> Value {
    let mut v = run.outcome_json.clone();
    if let Value::Object(m) = &mut v {
        m.insert("scoring_mode".into(), json!(run.scoring_mode));
        m.insert("eval_tier".into(), json!(run.eval_tier));
        m.insert(
            "groups".into(),
            groups
                .iter()
                .map(|g| {
                    json!({
                        "group": g.grp,
                        "g": g.g,
                        "ci_lo": g.ci_lo,
                        "ci_hi": g.ci_hi,
                        "mirror_penalty": g.mirror_penalty,
                    })
                })
                .collect(),
        );
    }
    v
}

/// `GET /v1/submissions/{id}/metrics?zone=a`: organizer-measured rows.
#[must_use]
pub fn zone_a_view(metrics: &[EvalMetricRecord], mirrors: &[MirrorPairRecord]) -> Value {
    json!({
        "zone": "a",
        "provenance": "organizer-measured",
        "metrics": metrics.iter().map(|m| json!({
            "key": m.key,
            "value": m.value,
            "clusters": m.clusters,
        })).collect::<Vec<_>>(),
        "mirrors": mirrors.iter().map(|p| json!({
            "group": p.grp,
            "metric": p.metric,
            "public_value": p.public_value,
            "mirror_value": p.mirror_value,
            "public_clusters": p.public_clusters,
            "mirror_clusters": p.mirror_clusters,
        })).collect::<Vec<_>>(),
    })
}

/// `GET /v1/submissions/{id}/metrics?zone=b`: the participant-reported
/// hash chain with per-report verdicts (never scored; anti-cheat evidence).
#[must_use]
pub fn zone_b_view(reports: &[MetricReportRecord]) -> Value {
    json!({
        "zone": "b",
        "provenance": "participant-reported",
        "reports": reports.iter().map(|r| json!({
            "seq": r.seq,
            "schema_version": r.schema_version,
            "prev_hash": r.prev_hash,
            "report_hash": r.report_hash,
            "verdict": r.verdict,
            "verdict_reasons": r.verdict_reasons,
            "payload": r.payload,
            "created_at": r.created_at,
        })).collect::<Vec<_>>(),
    })
}

/// `GET /v1/anchors`: every known anchor set with status.
#[must_use]
pub fn anchors_view(sets: &[AnchorSetRecord]) -> Value {
    json!({
        "anchors": sets.iter().map(|s| json!({
            "version": s.version,
            "status": s.status,
            "prereg_hash": s.prereg_hash,
            "activated_at": s.activated_at,
            "json": s.json,
        })).collect::<Vec<_>>()
    })
}

/// `GET /v1/preregistration`: anchor pre-registration hash-commits.
#[must_use]
pub fn prereg_view(entries: &[PreregRecord]) -> Value {
    json!({
        "preregistrations": entries.iter().map(|p| json!({
            "version": p.version,
            "hash": p.hash,
            "committed_at": p.committed_at,
            "notes": p.notes,
        })).collect::<Vec<_>>()
    })
}

/// Cohort observations (terminal values + series content hashes) projected
/// from stored reports of OTHER submissions.
///
/// Quarantined reports are excluded — provably dishonest series must not
/// shape the robust median/MAD bands; flagged reports stay in (MAD is
/// robust to legitimate outliers).
#[must_use]
pub fn prepare_cohort(reports: &[MetricReportRecord]) -> BTreeMap<String, CohortMetric> {
    let mut cohort: BTreeMap<String, CohortMetric> = BTreeMap::new();
    for r in reports {
        if r.verdict == Verdict::Quarantined {
            continue;
        }
        let Some(metrics) = r.payload.get("metrics").and_then(Value::as_object) else {
            continue;
        };
        for (name, raw) in metrics {
            let Ok(m) = serde_json::from_value::<ZoneBMetric>(raw.clone()) else {
                continue;
            };
            let entry = cohort.entry(name.clone()).or_default();
            match &m.kind {
                MetricKind::Scalar { value } => entry.terminals.push(*value),
                MetricKind::Series { step, value } => {
                    if let Some(last) = value.last() {
                        entry.terminals.push(*last);
                    }
                    entry.series_hashes.push(series_hash(name, step, value));
                }
                MetricKind::Histogram { .. } => {}
            }
        }
    }
    cohort
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use prism_zoneb::VerdictReason;

    fn report(verdict: Verdict, metrics: &Value) -> MetricReportRecord {
        MetricReportRecord {
            seq: 0,
            schema_version: "1.3.0".into(),
            prev_hash: "sub".into(),
            report_hash: "aa".repeat(32),
            payload: json!({"submission_id": "sub", "seq": 0, "schema_version": "1.3.0",
                "prev_hash": "sub", "metrics": metrics}),
            verdict,
            verdict_reasons: vec![],
            created_at: "0".into(),
        }
    }

    #[test]
    fn cohort_projects_terminals_and_hashes_and_skips_quarantined() {
        let clean = report(
            Verdict::Ok,
            &json!({
                "miner.train.loss": {"kind": "scalar", "value": 0.7},
                "miner.train.curve": {"kind": "series", "step": [0, 1], "value": [2.0, 1.0]},
            }),
        );
        let bad = report(
            Verdict::Quarantined,
            &json!({
                "miner.train.loss": {"kind": "scalar", "value": 0.01},
            }),
        );
        let cohort = prepare_cohort(&[clean, bad]);
        let loss = &cohort["miner.train.loss"];
        assert_eq!(loss.terminals, vec![0.7], "quarantined value excluded");
        let curve = &cohort["miner.train.curve"];
        assert_eq!(curve.terminals, vec![1.0], "series terminal");
        assert_eq!(curve.series_hashes.len(), 1);
    }

    #[test]
    fn zone_b_view_is_labelled_participant_reported() {
        let mut r = report(Verdict::Flagged, &json!({}));
        r.verdict_reasons = vec![VerdictReason::GroundTruthUnavailable {
            check: "terminal_loss".into(),
        }];
        let v = zone_b_view(&[r]);
        assert_eq!(v["zone"], "b");
        assert_eq!(v["provenance"], "participant-reported");
        assert_eq!(v["reports"][0]["verdict"], "flagged");
        assert_eq!(
            v["reports"][0]["verdict_reasons"][0]["reason"],
            "ground_truth_unavailable"
        );
    }

    #[test]
    fn zone_a_view_is_labelled_organizer_measured() {
        let v = zone_a_view(
            &[EvalMetricRecord {
                key: "org.g1.bpb_code".into(),
                value: 1.0,
                clusters: Some(json!({"c0": 1.0})),
            }],
            &[],
        );
        assert_eq!(v["zone"], "a");
        assert_eq!(v["provenance"], "organizer-measured");
        assert_eq!(v["metrics"][0]["key"], "org.g1.bpb_code");
    }
}
