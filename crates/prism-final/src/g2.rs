//! G2 public-suite lattice (scoring_version 4).
//!
//! Equal-weight mean of available downstream accuracies in `[0, 1]`, mapped to
//! `round(SCORE_MAX × mean)`. Tokenizer length cannot farm this leaf.
//! Missing every listed bench → `0` (fail-closed). Never falls back to bpb.

use prism_challenge_task::SCORE_MAX;
use serde_json::Value;

/// Canonical public G2 accuracy keys (first hit wins per task).
///
/// Prefer Zone-A `org.g2.*`; accept harness battery aliases. LAMBADA prefers
/// the strict protocol when both MC and strict keys are present.
pub const G2_PUBLIC_BENCHES: &[(&str, &[&str])] = &[
    (
        "hellaswag",
        &[
            "org.g2.hellaswag_acc",
            "g2.hellaswag.acc_norm",
            "g2.hellaswag.acc",
        ],
    ),
    (
        "arc_easy",
        &[
            "org.g2.arc_easy_acc",
            "g2.arc_easy.acc_norm",
            "g2.arc_easy.acc",
        ],
    ),
    (
        "arc_challenge",
        &[
            "org.g2.arc_challenge_acc",
            "g2.arc_challenge.acc_norm",
            "g2.arc_challenge.acc",
        ],
    ),
    (
        "piqa",
        &["org.g2.piqa_acc", "g2.piqa.acc_norm", "g2.piqa.acc"],
    ),
    (
        "winogrande",
        &[
            "org.g2.winogrande_acc",
            "g2.winogrande.acc_norm",
            "g2.winogrande.acc",
        ],
    ),
    (
        "boolq",
        &["org.g2.boolq_acc", "g2.boolq.acc_norm", "g2.boolq.acc"],
    ),
    (
        "lambada",
        &[
            "org.g2.lambada_strict_acc",
            "g2.lambada_strict.acc",
            "org.g2.lambada_acc",
            "g2.lambada.acc_norm",
            "g2.lambada.acc",
        ],
    ),
    (
        "openbookqa",
        &[
            "org.g2.obqa_acc",
            "org.g2.openbookqa_acc",
            "g2.openbookqa.acc_norm",
            "g2.openbookqa.acc",
        ],
    ),
];

/// Extract finite accuracies in `[0, 1]` from a METRICS_JSON-shaped blob.
#[must_use]
pub fn extract_g2_accuracies(metrics: &Value) -> Vec<(&'static str, f64)> {
    let mut out = Vec::new();
    for &(name, keys) in G2_PUBLIC_BENCHES {
        if let Some(v) = first_acc(metrics, keys) {
            out.push((name, v));
        }
    }
    out
}

/// Equal-weight G2 mean → lattice. Empty suite → `0`.
#[must_use]
pub fn score_from_g2_benchmarks(metrics: &Value) -> u64 {
    let accs = extract_g2_accuracies(metrics);
    if accs.is_empty() {
        return 0;
    }
    let mean = accs.iter().map(|(_, a)| *a).sum::<f64>() / accs.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return 0;
    }
    let v = (mean * (SCORE_MAX as f64)).round();
    if v >= SCORE_MAX as f64 {
        SCORE_MAX
    } else {
        v as u64
    }
}

fn first_acc(metrics: &Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = lookup_f64(metrics, key) {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                return Some(v);
            }
        }
    }
    None
}

fn lookup_f64(root: &Value, key: &str) -> Option<f64> {
    if let Some(v) = as_finite_f64(root.get(key)) {
        return Some(v);
    }
    // Nested org map (reference METRICS / some harvests): org.<key>
    if let Some(v) = as_finite_f64(root.pointer(&format!("/org/{key}"))) {
        return Some(v);
    }
    // Flattened battery map: battery.metrics.<key>
    if let Some(v) = as_finite_f64(root.pointer(&format!("/battery/metrics/{key}"))) {
        return Some(v);
    }
    // Zone-A row array: [{ "key", "value"|"point" }, ...]
    if let Some(rows) = root.get("metrics").and_then(Value::as_array) {
        for row in rows {
            if row.get("key").and_then(Value::as_str) == Some(key) {
                if let Some(v) =
                    as_finite_f64(row.get("value")).or_else(|| as_finite_f64(row.get("point")))
                {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn as_finite_f64(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    if let Some(n) = v.as_f64() {
        return n.is_finite().then_some(n);
    }
    if let Some(obj) = v.as_object() {
        return as_finite_f64(obj.get("value")).or_else(|| as_finite_f64(obj.get("point")));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equal_weight_mean_maps_to_lattice() {
        let m = json!({
            "org.g2.hellaswag_acc": 0.5,
            "org.g2.piqa_acc": 0.7,
        });
        // mean 0.6 → 600_000
        assert_eq!(score_from_g2_benchmarks(&m), 600_000);
        let names: Vec<_> = extract_g2_accuracies(&m)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert_eq!(names, vec!["hellaswag", "piqa"]);
    }

    #[test]
    fn prefers_lambada_strict_over_mc() {
        let m = json!({
            "org.g2.lambada_acc": 0.99,
            "org.g2.lambada_strict_acc": 0.25,
        });
        assert_eq!(score_from_g2_benchmarks(&m), 250_000);
    }

    #[test]
    fn empty_or_invalid_fail_closed() {
        assert_eq!(score_from_g2_benchmarks(&json!({})), 0);
        assert_eq!(
            score_from_g2_benchmarks(&json!({"org.g2.hellaswag_acc": 1.5})),
            0
        );
        assert_eq!(score_from_g2_benchmarks(&json!({"bpb": 0.1})), 0);
    }

    #[test]
    fn reads_battery_nested_and_zone_a_rows() {
        let nested = json!({"battery": {"metrics": {"org.g2.boolq_acc": {"value": 0.4}}}});
        assert_eq!(score_from_g2_benchmarks(&nested), 400_000);
        let zone = json!({"metrics": [{"key": "org.g2.arc_easy_acc", "point": 0.8}]});
        assert_eq!(score_from_g2_benchmarks(&zone), 800_000);
    }

    #[test]
    fn prefers_obqa_alias_and_org_nest() {
        let flat = json!({"org.g2.obqa_acc": 0.335});
        assert_eq!(score_from_g2_benchmarks(&flat), 335_000);
        let nested = json!({"org": {"org.g2.obqa_acc": {"value": 0.5}}});
        assert_eq!(score_from_g2_benchmarks(&nested), 500_000);
    }
}
