//! Prism list/detail enrichment: recipe era, eval groups, G2 benches, GPT-2 refs.

use serde_json::Value;

use site_types::{
    EvalGroupScore, PrismBenchmarks, PrismEvalSummary, PrismGateSummary, PrismPublicReview,
    PrismPublicSimilarity, PrismReferenceBaseline, PrismSubmissionDetail, PrismTelemetry,
    RecipeEra, Submission,
};

use site_data::map::{prism_submission, prism_telemetry};
use site_types::LeaderboardRow;

/// HuggingFace model card for GPT-2 Large reference weights.
pub const GPT2_LARGE_SOURCE_URL: &str = "https://huggingface.co/gpt2-large";
/// HuggingFace model card for GPT-2 Small (`openai-community/gpt2`).
pub const GPT2_SMALL_SOURCE_URL: &str = "https://huggingface.co/openai-community/gpt2";

/// Public disclaimer for the GPT-2 Large Prism-protocol reference row.
pub const GPT2_LARGE_DISCLAIMER: &str = "Public GPT-2 Large (774M) — Prism-protocol eval-only on 1×RTX 5090 (HF `gpt2-large` weights, public eval pack, `PRISM_TEST_EVAL_CAPS=0`). Not a miner train; G6 has no train probe curve; G8 µP width knob unsupported (floor 0).";

/// Public disclaimer for the GPT-2 Small Prism-protocol reference row.
pub const GPT2_SMALL_DISCLAIMER: &str = "Public GPT-2 Small (124M) — Prism-protocol eval-only on 1×RTX 5090 (HF `openai-community/gpt2` / `gpt2` weights, public eval pack, `PRISM_TEST_EVAL_CAPS=0`). Not a miner train; G6 has no train probe curve; G8 µP width knob unsupported (floor 0).";

/// GPT-2 Large parameter count (millions) — measured `n_params` / 1e6.
pub const GPT2_LARGE_PARAMS_M: f64 = 774.0;
/// GPT-2 Small parameter count (millions) — measured `n_params` / 1e6.
pub const GPT2_SMALL_PARAMS_M: f64 = 124.4;

/// Measured Prism validation BPB — GPT-2 Large.
pub const GPT2_LARGE_BPB: f64 = 4.163_851_322_121_356_4;
/// Measured Prism validation BPB — GPT-2 Small.
pub const GPT2_SMALL_BPB: f64 = 4.759_478_148_923_918;

/// GPT-2 Large G2 benches (Prism public pack, 1×RTX 5090).
pub const GPT2_LARGE_HELLASWAG: f64 = 0.395;
pub const GPT2_LARGE_ARC_EASY: f64 = 0.28;
pub const GPT2_LARGE_ARC_CHALLENGE: f64 = 0.28;
pub const GPT2_LARGE_PIQA: f64 = 0.69;
pub const GPT2_LARGE_WINOGRANDE: f64 = 0.545;
pub const GPT2_LARGE_BOOLQ: f64 = 0.64;
pub const GPT2_LARGE_LAMBADA: f64 = 0.985;
pub const GPT2_LARGE_OPENBOOKQA: f64 = 0.335;

/// GPT-2 Small G2 benches (Prism public pack, 1×RTX 5090).
pub const GPT2_SMALL_HELLASWAG: f64 = 0.355;
pub const GPT2_SMALL_ARC_EASY: f64 = 0.245;
pub const GPT2_SMALL_ARC_CHALLENGE: f64 = 0.24;
pub const GPT2_SMALL_PIQA: f64 = 0.585;
pub const GPT2_SMALL_WINOGRANDE: f64 = 0.515;
pub const GPT2_SMALL_BOOLQ: f64 = 0.575;
pub const GPT2_SMALL_LAMBADA: f64 = 0.97;
pub const GPT2_SMALL_OPENBOOKQA: f64 = 0.32;

/// Frozen public GPT-2 baselines for `GET …/references` (Large then Small).
#[must_use]
pub fn prism_reference_baselines() -> Vec<PrismReferenceBaseline> {
    vec![
        PrismReferenceBaseline {
            id: "gpt2-large-774m".into(),
            label: "Public GPT-2 Large (774M)".into(),
            params_m: GPT2_LARGE_PARAMS_M,
            bpb: Some(GPT2_LARGE_BPB),
            benchmarks: PrismBenchmarks {
                hellaswag: Some(GPT2_LARGE_HELLASWAG),
                arc_easy: Some(GPT2_LARGE_ARC_EASY),
                arc_challenge: Some(GPT2_LARGE_ARC_CHALLENGE),
                piqa: Some(GPT2_LARGE_PIQA),
                winogrande: Some(GPT2_LARGE_WINOGRANDE),
                boolq: Some(GPT2_LARGE_BOOLQ),
                lambada: Some(GPT2_LARGE_LAMBADA),
                openbookqa: Some(GPT2_LARGE_OPENBOOKQA),
            },
            source_url: GPT2_LARGE_SOURCE_URL.into(),
            disclaimer: GPT2_LARGE_DISCLAIMER.into(),
        },
        PrismReferenceBaseline {
            id: "gpt2-small-124m".into(),
            label: "Public GPT-2 Small (124M)".into(),
            params_m: GPT2_SMALL_PARAMS_M,
            bpb: Some(GPT2_SMALL_BPB),
            benchmarks: PrismBenchmarks {
                hellaswag: Some(GPT2_SMALL_HELLASWAG),
                arc_easy: Some(GPT2_SMALL_ARC_EASY),
                arc_challenge: Some(GPT2_SMALL_ARC_CHALLENGE),
                piqa: Some(GPT2_SMALL_PIQA),
                winogrande: Some(GPT2_SMALL_WINOGRANDE),
                boolq: Some(GPT2_SMALL_BOOLQ),
                lambada: Some(GPT2_SMALL_LAMBADA),
                openbookqa: Some(GPT2_SMALL_OPENBOOKQA),
            },
            source_url: GPT2_SMALL_SOURCE_URL.into(),
            disclaimer: GPT2_SMALL_DISCLAIMER.into(),
        },
    ]
}

/// Infer AutoModel vs legacy from a detail or list-shaped payload.
///
/// Signals (first match wins as automodel): explicit `pin_id`, AutoModel pin
/// id string, recipe major ≥ 2 in metrics / pod manifest, or `.prism`
/// automodel artifact paths. Otherwise legacy (incl. unknown / pre-2.0).
#[must_use]
pub fn infer_recipe_era(payload: &Value) -> RecipeEra {
    let sub = payload.get("submission").unwrap_or(payload);
    if pin_id_from_payload(payload).is_some() {
        return RecipeEra::Automodel;
    }
    if recipe_major(sub).is_some_and(|m| m >= 2) {
        return RecipeEra::Automodel;
    }
    // Diff / tree hints when present on a fan-in blob.
    if payload
        .pointer("/diffstat/files")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return RecipeEra::Automodel;
    }
    RecipeEra::Legacy
}

/// AutoModel pin id from detail, `/diff`, or metrics pod manifest.
#[must_use]
pub fn pin_id_from_payload(payload: &Value) -> Option<String> {
    let sub = payload.get("submission").unwrap_or(payload);
    for path in [
        "/pin_id",
        "/submission/pin_id",
        "/metrics/pod_manifest/automodel_base",
        "/metrics/pod_manifest/pin_id",
        "/submission/metrics/pod_manifest/automodel_base",
        "/submission/metrics/pod_manifest/pin_id",
    ] {
        if let Some(p) = payload
            .pointer(path)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty() && s.starts_with("automodel@"))
        {
            return Some(p.to_owned());
        }
    }
    sub.get("pin_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// Map `eval.groups` → compact `{group,g}` rows (sorted g1…g8 when possible).
#[must_use]
pub fn map_eval_groups(eval: Option<&Value>) -> Option<Vec<EvalGroupScore>> {
    let groups = eval?.get("groups")?.as_array()?;
    let mut out: Vec<EvalGroupScore> = groups
        .iter()
        .filter_map(|g| {
            let group = g.get("group")?.as_str()?.to_owned();
            let score = g.get("g").and_then(Value::as_f64)?;
            Some(EvalGroupScore { group, g: score })
        })
        .collect();
    if out.is_empty() {
        return None;
    }
    out.sort_by(|a, b| a.group.cmp(&b.group));
    Some(out)
}

/// Public G2 benches from harness `metrics` (`org.g2.*` or battery g2 keys).
#[must_use]
pub fn map_benchmarks(metrics: Option<&Value>) -> PrismBenchmarks {
    let Some(metrics) = metrics.filter(|m| !m.is_null()) else {
        return PrismBenchmarks::default();
    };
    PrismBenchmarks {
        hellaswag: metric_f64(
            metrics,
            &[
                "org.g2.hellaswag_acc",
                "g2.hellaswag.acc_norm",
                "g2.hellaswag.acc",
            ],
        ),
        arc_easy: metric_f64(
            metrics,
            &[
                "org.g2.arc_easy_acc",
                "g2.arc_easy.acc_norm",
                "g2.arc_easy.acc",
            ],
        ),
        arc_challenge: metric_f64(
            metrics,
            &[
                "org.g2.arc_challenge_acc",
                "g2.arc_challenge.acc_norm",
                "g2.arc_challenge.acc",
            ],
        ),
        piqa: metric_f64(
            metrics,
            &["org.g2.piqa_acc", "g2.piqa.acc_norm", "g2.piqa.acc"],
        ),
        winogrande: metric_f64(
            metrics,
            &[
                "org.g2.winogrande_acc",
                "g2.winogrande.acc_norm",
                "g2.winogrande.acc",
            ],
        ),
        boolq: metric_f64(
            metrics,
            &["org.g2.boolq_acc", "g2.boolq.acc_norm", "g2.boolq.acc"],
        ),
        lambada: metric_f64(
            metrics,
            &[
                "org.g2.lambada_acc",
                "g2.lambada.acc_norm",
                "g2.lambada.acc",
            ],
        ),
        openbookqa: metric_f64(
            metrics,
            &[
                "org.g2.obqa_acc",
                "org.g2.openbookqa_acc",
                "g2.openbookqa.acc_norm",
                "g2.openbookqa.acc",
            ],
        ),
    }
}

/// Flatten `GET /v1/submissions/{id}/metrics?zone=a` → object keyed by metric name.
///
/// Zone-A rows are `{ key, value | point }` arrays; challenge detail sometimes
/// omits flattened `org.g2.*` on `submission.metrics` even when the eval store
/// has them. Callers merge this into bench mapping so the FE still surfaces.
#[must_use]
pub fn metrics_object_from_zone_a(zone: &Value) -> Option<Value> {
    let rows = zone.get("metrics")?.as_array()?;
    let mut map = serde_json::Map::new();
    for row in rows {
        let key = row.get("key").and_then(Value::as_str)?;
        let val = row
            .get("value")
            .cloned()
            .or_else(|| row.get("point").cloned())
            .filter(|v| !v.is_null())?;
        map.insert(key.to_owned(), val);
    }
    if map.is_empty() {
        return None;
    }
    Some(Value::Object(map))
}

/// Resolve public benches from detail metrics and optional zone-A payload.
#[must_use]
pub fn resolve_benchmarks(detail: &Value, zone_a: Option<&Value>) -> PrismBenchmarks {
    let root = detail.get("submission").unwrap_or(detail);
    let mut benches = map_benchmarks(root.get("metrics").filter(|m| !m.is_null()));
    // Battery blob may nest under metrics without top-level org.* copies.
    if benches.is_empty() {
        if let Some(battery_metrics) = root.pointer("/metrics/battery/metrics") {
            benches = map_benchmarks(Some(battery_metrics));
        }
    }
    if let Some(zone) = zone_a {
        if let Some(obj) = metrics_object_from_zone_a(zone) {
            let from_zone = map_benchmarks(Some(&obj));
            benches.merge_missing(&from_zone);
        }
    }
    benches
}

/// Copy era / pin / eval groups / benches from a detail payload onto a board row.
pub fn enrich_leaderboard_row_from_detail_with_zone(
    row: &mut LeaderboardRow,
    detail: &Value,
    zone_a: Option<&Value>,
) {
    let root = detail.get("submission").unwrap_or(detail);
    let era = infer_recipe_era(detail);
    row.recipe_era = Some(era);
    if era == RecipeEra::Automodel {
        row.pin_id = pin_id_from_payload(detail);
    }
    if let Some(groups) = map_eval_groups(root.get("eval").filter(|e| !e.is_null())) {
        row.eval_groups = Some(groups);
    }
    let benches = resolve_benchmarks(detail, zone_a);
    if !benches.is_empty() {
        row.benchmarks = Some(benches);
    }
}

/// Apply detail fan-out fields onto a list [`Submission`].
pub fn enrich_submission_from_detail_with_zone(
    sub: &mut Submission,
    detail: &Value,
    zone_a: Option<&Value>,
) {
    let root = detail.get("submission").unwrap_or(detail);
    let era = infer_recipe_era(detail);
    sub.recipe_era = Some(era);
    let pin = pin_id_from_payload(detail);
    if era == RecipeEra::Automodel {
        sub.pin_id = pin;
    }
    let eval = root.get("eval").filter(|e| !e.is_null());
    if let Some(groups) = map_eval_groups(eval) {
        sub.eval_groups = Some(groups);
    }
    let metrics = root.get("metrics").filter(|m| !m.is_null());
    let benches = resolve_benchmarks(detail, zone_a);
    if !benches.is_empty() {
        sub.benchmarks = Some(benches);
    }
    // Prefer measured params / bpb from detail when the list row was thin.
    if sub.bpb.is_none() {
        sub.bpb = root
            .get("bpb")
            .and_then(Value::as_f64)
            .or_else(|| metrics.and_then(|m| m.get("bpb")).and_then(Value::as_f64));
    }
    if sub.params_m.is_none() {
        sub.params_m = root
            .get("n_params")
            .and_then(Value::as_u64)
            .or_else(|| {
                metrics
                    .and_then(|m| m.get("n_params"))
                    .and_then(Value::as_u64)
            })
            .map(|p| p as f64 / 1e6);
    }
}

/// Map challenge `GET /v1/submissions/{id}` (+ optional Zone-A) → public detail.
#[must_use]
pub fn prism_submission_detail_with_zone(
    detail: &Value,
    zone_a: Option<&Value>,
) -> Option<PrismSubmissionDetail> {
    let root = detail.get("submission")?;
    let mut sub = prism_submission(root)?;
    enrich_submission_from_detail_with_zone(&mut sub, detail, zone_a);
    let eval = root
        .get("eval")
        .filter(|e| !e.is_null())
        .map(map_eval_summary);
    let telemetry = prism_telemetry(detail).unwrap_or_else(|| PrismTelemetry {
        submission_id: sub.id.clone(),
        bpb: sub.bpb,
        n_params: sub.params_m.map(|m| (m * 1e6) as u64),
        val_rows: None,
        gpu_type: None,
        wall_clock_seconds: None,
        finish_reason: None,
        report_count: 0,
        points: Vec::new(),
    });
    let review = root
        .get("review")
        .filter(|r| !r.is_null())
        .map(|r| PrismPublicReview {
            quality_score: r.get("quality_score").and_then(Value::as_f64),
        });
    let similarity = root
        .get("similarity")
        .filter(|s| !s.is_null())
        .and_then(|s| {
            let kind = s.get("kind")?.as_str()?.to_owned();
            Some(PrismPublicSimilarity {
                kind,
                score: s.get("score").and_then(Value::as_f64),
            })
        });
    Some(PrismSubmissionDetail {
        submission: sub,
        eval,
        telemetry,
        review,
        similarity,
    })
}

fn map_eval_summary(eval: &Value) -> PrismEvalSummary {
    let status = eval
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let groups = map_eval_groups(Some(eval)).unwrap_or_default();
    let gates = eval.get("gates").and_then(|g| {
        Some(PrismGateSummary {
            complete: g.get("complete")?.as_bool()?,
            g3_ok: g.get("g3_ok")?.as_bool()?,
            g8_ok: g.get("g8_ok")?.as_bool()?,
            budgets_ok: g.get("budgets_ok")?.as_bool()?,
            ci_ok: g.get("ci_ok")?.as_bool()?,
        })
    });
    PrismEvalSummary {
        status,
        groups,
        gates,
        composite: eval.get("composite").and_then(Value::as_f64),
        scoring_mode: eval
            .get("scoring_mode")
            .and_then(Value::as_str)
            .map(str::to_owned),
        eval_tier: eval
            .get("eval_tier")
            .and_then(Value::as_str)
            .map(str::to_owned),
    }
}

fn recipe_major(sub: &Value) -> Option<u32> {
    let raw = sub
        .pointer("/metrics/recipe")
        .and_then(Value::as_str)
        .or_else(|| {
            sub.pointer("/metrics/pod_manifest/recipe_version")
                .and_then(Value::as_str)
        })?;
    raw.split('.').next()?.parse().ok()
}

fn metric_f64(metrics: &Value, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = lookup_metric(metrics, key) {
            return Some(v);
        }
        // Nested under battery.groups.g2.metrics / battery.metrics (METRICS_JSON v2).
        for path in [
            format!("/battery/groups/g2/metrics/{key}"),
            format!("/battery/metrics/{key}"),
            format!("/groups/g2/metrics/{key}"),
        ] {
            if let Some(v) = metrics.pointer(&path).and_then(as_metric_node) {
                return Some(v);
            }
        }
    }
    None
}

fn lookup_metric(metrics: &Value, key: &str) -> Option<f64> {
    let v = metrics.get(key)?;
    as_metric_node(v)
}

fn as_metric_node(v: &Value) -> Option<f64> {
    as_f64_val(v)
        .or_else(|| v.get("value").and_then(as_f64_val))
        .or_else(|| v.get("point").and_then(as_f64_val))
}

fn as_f64_val(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_u64().map(|u| u as f64))
        .or_else(|| v.as_i64().map(|i| i as f64))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn era_from_recipe_major_and_pin() {
        let legacy = json!({"metrics": {"recipe": "1.4.0"}});
        assert_eq!(infer_recipe_era(&legacy), RecipeEra::Legacy);
        let auto = json!({"metrics": {"recipe": "2.0.0"}});
        assert_eq!(infer_recipe_era(&auto), RecipeEra::Automodel);
        let pin = json!({"pin_id": "automodel@v0.5.0"});
        assert_eq!(infer_recipe_era(&pin), RecipeEra::Automodel);
        assert_eq!(
            pin_id_from_payload(&pin).as_deref(),
            Some("automodel@v0.5.0")
        );
    }

    #[test]
    fn benchmarks_from_org_g2_and_battery() {
        let m = json!({
            "org.g2.hellaswag_acc": {"value": 0.31},
            "org.g2.piqa_acc": {"value": 0.62},
            "org.g2.arc_easy_acc": {"value": 0.44},
            "battery": {"groups": {"g2": {"metrics": {
                "g2.arc_challenge.acc_norm": 0.22,
                "g2.winogrande.acc_norm": 0.51,
                "g2.boolq.acc": 0.60
            }}}}
        });
        let b = map_benchmarks(Some(&m));
        assert!((b.hellaswag.unwrap() - 0.31).abs() < f64::EPSILON);
        assert!((b.piqa.unwrap() - 0.62).abs() < f64::EPSILON);
        assert!((b.arc_easy.unwrap() - 0.44).abs() < f64::EPSILON);
        assert!((b.arc_challenge.unwrap() - 0.22).abs() < f64::EPSILON);
        assert!((b.winogrande.unwrap() - 0.51).abs() < f64::EPSILON);
        assert!((b.boolq.unwrap() - 0.60).abs() < f64::EPSILON);
    }

    #[test]
    fn zone_a_metrics_fill_missing_benches() {
        let detail = json!({
            "submission": {
                "id": "z1",
                "miner_hotkey": "aa".repeat(32),
                "epoch": 1,
                "status": "terminated",
                "label": "z",
                "bpb": 1.1,
                "created_at_ms": 1u64,
                "metrics": { "bpb": 1.1, "recipe": "2.0.0" }
            }
        });
        let zone = json!({
            "zone": "a",
            "metrics": [
                {"key": "org.g2.hellaswag_acc", "value": 0.33},
                {"key": "org.g2.arc_easy_acc", "point": 0.41}
            ]
        });
        let b = resolve_benchmarks(&detail, Some(&zone));
        assert!((b.hellaswag.unwrap() - 0.33).abs() < 1e-9);
        assert!((b.arc_easy.unwrap() - 0.41).abs() < 1e-9);
    }

    #[test]
    fn eval_groups_sorted() {
        let eval = json!({"groups": [
            {"group": "g2", "g": 0.5},
            {"group": "g1", "g": 0.9}
        ]});
        let g = map_eval_groups(Some(&eval)).unwrap();
        assert_eq!(g[0].group, "g1");
        assert_eq!(g[1].group, "g2");
    }

    #[test]
    fn gpt2_baselines_are_prism_protocol_large_and_small() {
        let refs = prism_reference_baselines();
        assert_eq!(refs.len(), 2);
        assert_eq!(refs[0].id, "gpt2-large-774m");
        assert_eq!(refs[1].id, "gpt2-small-124m");
        assert!(refs[0].bpb.is_some());
        assert!(refs[1].bpb.is_some());
        assert!((refs[0].bpb.unwrap() - GPT2_LARGE_BPB).abs() < 1e-9);
        assert!((refs[1].bpb.unwrap() - GPT2_SMALL_BPB).abs() < 1e-9);
        assert!((refs[0].params_m - 774.0).abs() < f64::EPSILON);
        assert!((refs[1].params_m - 124.4).abs() < 1e-9);
        assert!((refs[0].benchmarks.hellaswag.unwrap() - GPT2_LARGE_HELLASWAG).abs() < f64::EPSILON);
        assert!((refs[1].benchmarks.hellaswag.unwrap() - GPT2_SMALL_HELLASWAG).abs() < f64::EPSILON);
        assert!((refs[0].benchmarks.lambada.unwrap() - GPT2_LARGE_LAMBADA).abs() < f64::EPSILON);
        assert!((refs[1].benchmarks.openbookqa.unwrap() - GPT2_SMALL_OPENBOOKQA).abs() < f64::EPSILON);
        assert!(refs[0].disclaimer.contains("Prism-protocol"));
        assert!(refs[1].disclaimer.contains("Prism-protocol"));
        assert!(refs[0].source_url.contains("gpt2-large"));
        assert!(refs[1].source_url.contains("openai-community/gpt2"));
    }

    #[test]
    fn detail_maps_public_subset() {
        let detail = json!({
            "submission": {
                "id": "sub1",
                "miner_hotkey": "bb".repeat(32),
                "epoch": 3,
                "status": "terminated",
                "label": "base",
                "bpb": 1.25,
                "n_params": 12_000_000_u64,
                "score": {"kind":"score","value": 900},
                "created_at_ms": 1_700_000_000_000_u64,
                "pin_id": "automodel@v0.5.0",
                "metrics": {
                    "recipe": "2.0.0",
                    "bpb": 1.25,
                    "n_params": 12_000_000_u64,
                    "org.g2.hellaswag_acc": {"value": 0.4},
                    "telemetry": {
                        "finish_reason": "finish_evaluation",
                        "report_count": 1,
                        "loss_series": [{"step": 1, "loss": 2.0}]
                    }
                },
                "eval": {
                    "status": "scored",
                    "composite": 0.55,
                    "scoring_mode": "shadow",
                    "gates": {
                        "complete": true, "g3_ok": true, "g8_ok": true,
                        "budgets_ok": true, "ci_ok": true
                    },
                    "groups": [{"group":"g1","g":0.8},{"group":"g2","g":0.4}]
                },
                "review": {"quality_score": 0.9, "issues": ["x"]},
                "similarity": {"kind": "Clean", "score": 0.1, "evidence": "secret"}
            },
            "events": []
        });
        let d = prism_submission_detail_with_zone(&detail, None).unwrap();
        assert_eq!(d.submission.recipe_era, Some(RecipeEra::Automodel));
        assert_eq!(d.submission.pin_id.as_deref(), Some("automodel@v0.5.0"));
        assert!((d.submission.benchmarks.as_ref().unwrap().hellaswag.unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(d.eval.as_ref().unwrap().status, "scored");
        assert_eq!(d.eval.as_ref().unwrap().groups.len(), 2);
        assert!(d.eval.as_ref().unwrap().gates.as_ref().unwrap().complete);
        assert_eq!(d.telemetry.points.len(), 1);
        assert!((d.review.as_ref().unwrap().quality_score.unwrap() - 0.9).abs() < 1e-9);
        assert_eq!(d.similarity.as_ref().unwrap().kind, "Clean");
        // No raw patch / evidence dumps on the public contract.
        let v = serde_json::to_value(&d).unwrap();
        assert!(v.get("patch").is_none());
        assert!(v.pointer("/similarity/evidence").is_none());
    }
}
