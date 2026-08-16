//! Measurement → composite → storage glue (E4): build `SubmissionMetrics`
//! from a METRICS_JSON v2 blob, run the composite, persist the run, and
//! ingest the Zone B `train_metrics` lift.
//!
//! Battery blob contract (emitted by the G1–G8 battery harness; absent on
//! v1/v2 training-only runs): `battery.metrics` maps `org.<group>.<name>` to
//! either a bare number or `{value, clusters: {cluster_id -> value}}`;
//! `battery.mirrors` is a list of `{group, metric, public, mirror}` (each
//! side the same number/`{value, clusters}` shape); `battery.tier` is an
//! optional label. When no `org.*` metrics exist the composite is skipped
//! entirely (`Ok(None)`) and the v2 scoring path is untouched.

use std::collections::BTreeMap;
use std::sync::Arc;

use prism_pipeline::composite::{
    self, BindingCap, BudgetFacts, CompositeAnchors, CompositeError, CompositeOutcome,
    MetricSeries, MirrorPair, SubmissionMetrics,
};
use prism_pipeline::score::ScoringMode;
use prism_store::eval::{
    AnchorSetRecord, EvalGroupRecord, EvalMetricRecord, EvalRunRecord, EvalStore, MirrorPairRecord,
    PreregRecord,
};
use prism_store::StoreError;
use prism_zoneb::{
    is_valid_metric_name, prepare_report, Direction, GroundTruth, IngestReject, MetricKind,
    ZoneBEnvelope, ZoneBMetric, MAX_SCALARS,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::views::prepare_cohort;

/// Anchor-set input for finalization, decoupled from
/// `prism-recipe::anchors` (wired into the recipe crate root by the E3
/// integration pass). The pre-registration hash is sha256 over exactly
/// `canonical_json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorInput {
    /// Canonical anchor-set JSON bytes.
    pub canonical_json: String,
    /// Registry status (`placeholder` | `active`).
    pub status: String,
}

impl AnchorInput {
    /// The embedded v0 placeholder set (canonical bytes shared with
    /// `prism-recipe/anchors/v0.json`).
    #[must_use]
    pub fn v0_placeholder() -> Self {
        Self {
            canonical_json: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../prism-recipe/anchors/v0.json"
            ))
            .to_owned(),
            status: "placeholder".into(),
        }
    }

    /// The embedded v1 placeholder set (Prism v2.1 battery additions:
    /// `org.g7.reasoning_throughput` + `org.g8.mup_scaling_slope`; canonical
    /// bytes shared with `prism-recipe/anchors/v1.json`).
    #[must_use]
    pub fn v1_placeholder() -> Self {
        Self {
            canonical_json: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../prism-recipe/anchors/v1.json"
            ))
            .to_owned(),
            status: "placeholder".into(),
        }
    }

    /// The embedded v2 placeholder set (Prism v2.2 swap: saturated MC
    /// `org.g2.lambada_acc` replaced by canonical strict
    /// `org.g2.lambada_strict_acc`; canonical bytes shared with
    /// `prism-recipe/anchors/v2.json`).
    #[must_use]
    pub fn v2_placeholder() -> Self {
        Self {
            canonical_json: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../prism-recipe/anchors/v2.json"
            ))
            .to_owned(),
            status: "placeholder".into(),
        }
    }

    /// The embedded v3 placeholder set (Prism v3 **measurement**: the
    /// confounded `org.g8.mup_scaling_slope` dropped from the scored set,
    /// byte/compute-denominated G6 keys added, the inert `conf` confirmation
    /// tier added, and the dual-cap compute gates `max_flops` +
    /// `min_spend_fraction` added; canonical bytes shared with
    /// `prism-recipe/anchors/v3.json`).
    ///
    /// **Not selectable in practice yet.** v3 declares keys the battery does
    /// not emit (`org.g6.auc_log_bytes` and siblings need the byte-curve
    /// change; `org.conf.*` needs the operator confirmation tier), and a
    /// declared-but-absent key is a hard `MissingMetric` → `Ineligible`.
    /// Emit before declare.
    #[must_use]
    pub fn v3_placeholder() -> Self {
        Self {
            canonical_json: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../prism-recipe/anchors/v3.json"
            ))
            .to_owned(),
            status: "placeholder".into(),
        }
    }

    /// Anchor set selected by `PRISM_ANCHOR_VERSION` (`0` default → v0,
    /// bit-identical live behavior; `1` → the v2.1 set; `2` → the v2.2
    /// LAMBADA-strict set; `3` → the v3 measurement set). Unknown values fail
    /// safe to v0 with a warning — never a new scoring surface by accident.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("PRISM_ANCHOR_VERSION").ok().as_deref() {
            Some("1") => Self::v1_placeholder(),
            Some("2") => Self::v2_placeholder(),
            Some("3") => Self::v3_placeholder(),
            None | Some("0") => Self::v0_placeholder(),
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "unknown PRISM_ANCHOR_VERSION; falling back to v0"
                );
                Self::v0_placeholder()
            }
        }
    }

    /// Pre-registration hash: sha256 hex over the canonical bytes.
    #[must_use]
    pub fn prereg_hash(&self) -> String {
        hex::encode(Sha256::digest(self.canonical_json.as_bytes()))
    }
}

/// Finalize failures.
#[derive(Debug, Error)]
pub enum FinalizeError {
    /// Anchor JSON parse/validation.
    #[error("anchors: {0}")]
    Anchors(#[from] CompositeError),
    /// Store fault.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// Zone B envelope rejection (malformed/over caps).
    #[error("zone b ingest: {0}")]
    ZoneB(#[from] IngestReject),
}

/// Run the measurement→composite→storage pipeline for one measured
/// submission.
///
/// - Zone B: lift the sanitized `train_metrics` dict into a master-chained
///   report, validate it against organizer ground truth + the cohort, and
///   append it to the submission's hash chain. Idempotent: a re-finalize
///   whose head already carries the same metrics appends nothing.
/// - Zone A: when the blob carries battery `org.*` metrics, build
///   `SubmissionMetrics`, evaluate the composite against `anchors`, persist
///   the run (replace-wholesale) + anchor registry row + prereg commit.
///
/// Returns the outcome for the caller to attach to
/// `FinalOutcome::Measured.composite`; `Ok(None)` when no battery data
/// exists (v1/v2 harness runs — the v2 scoring path is untouched).
pub async fn finalize_composite(
    store: &Arc<dyn EvalStore>,
    submission_id: &str,
    metrics_v2: &Value,
    anchors: &AnchorInput,
) -> Result<Option<CompositeOutcome>, FinalizeError> {
    ingest_zone_b(store, submission_id, metrics_v2).await?;
    let Some(sub) = submission_metrics(metrics_v2) else {
        return Ok(None);
    };
    let parsed = CompositeAnchors::from_json(&anchors.canonical_json)?
        .with_prereg_hash(anchors.prereg_hash());
    let outcome = composite::evaluate(&sub, &parsed, seed_for(submission_id));
    persist_run(
        store,
        submission_id,
        metrics_v2,
        anchors,
        &parsed,
        &sub,
        &outcome,
    )
    .await?;
    Ok(Some(outcome))
}

/// Orchestrator-facing wrapper (E7): `None` when no store is attached (the
/// default — the v2 path stays bit-identical) or the blob is absent, and
/// warn + `None` on store/finalize faults — shadow mode is unaffected and
/// composite mode fails closed to 0 in `final_lattice` by design. The
/// anchor set follows `PRISM_ANCHOR_VERSION` (default 0 = the embedded v0
/// placeholder; `1` selects the v2.1 set) until registry-driven anchors
/// land.
pub async fn finalize_for_submission(
    store: Option<&Arc<dyn EvalStore>>,
    submission_id: &str,
    metrics_v2: Option<&Value>,
) -> Option<CompositeOutcome> {
    let (Some(store), Some(blob)) = (store, metrics_v2) else {
        return None;
    };
    match finalize_composite(store, submission_id, blob, &AnchorInput::from_env()).await {
        Ok(outcome) => outcome,
        Err(e) => {
            tracing::warn!(submission_id, error = %e, "composite finalize failed (skipped)");
            None
        }
    }
}

/// Lift the sanitized miner-returned `train_metrics` dict (METRICS_JSON v2)
/// into Zone B metric entries: flat finite scalars under `miner.*` (and
/// `org.*`, which the validator strips into a quarantine finding — forgery
/// evidence, never silently dropped). `loss` keys are hinted lower-better.
/// Returns `None` when there is nothing to report, or when the dict is over
/// the scalar cap (the validator would reject the envelope anyway).
#[must_use]
pub fn from_train_metrics(metrics_v2: &Value) -> Option<BTreeMap<String, ZoneBMetric>> {
    let obj = metrics_v2.get("train_metrics")?.as_object()?;
    let mut out = BTreeMap::new();
    for (k, v) in obj.iter().take(MAX_SCALARS + 1) {
        let valid = is_valid_metric_name(k) || k.starts_with("org.");
        let Some(x) = v.as_f64().filter(|x| x.is_finite()) else {
            continue;
        };
        if !valid {
            continue;
        }
        let direction = if k.contains("loss") {
            Direction::Lower
        } else {
            Direction::Neutral
        };
        out.insert(
            k.clone(),
            ZoneBMetric {
                kind: MetricKind::Scalar { value: x },
                unit: None,
                direction: Some(direction),
            },
        );
    }
    if out.is_empty() || out.len() > MAX_SCALARS {
        return None;
    }
    Some(out)
}

/// Zone B ingest for one submission (idempotent on unchanged metrics).
async fn ingest_zone_b(
    store: &Arc<dyn EvalStore>,
    submission_id: &str,
    metrics_v2: &Value,
) -> Result<(), FinalizeError> {
    let Some(metrics) = from_train_metrics(metrics_v2) else {
        return Ok(());
    };
    let head_row = store.latest_metric_report(submission_id).await?;
    let lifted = serde_json::to_value(&metrics).map_err(backend)?;
    if head_row.as_ref().and_then(|r| r.payload.get("metrics")) == Some(&lifted) {
        return Ok(());
    }
    let env = ZoneBEnvelope {
        schema_version: metrics_v2
            .get("recipe")
            .and_then(Value::as_str)
            .unwrap_or(prism_recipe::RECIPE_VERSION)
            .to_owned(),
        prev_hash: None, // master-chained at ingest (legacy lift)
        metrics,
    };
    let head = head_row.map(|r| (u64::try_from(r.seq).unwrap_or(0), r.report_hash));
    let cohort = prepare_cohort(&store.cohort_reports(submission_id).await?);
    let prepared = prepare_report(
        submission_id,
        &env,
        head,
        &ground_truth(metrics_v2),
        &cohort,
    )?;
    store.insert_metric_report(submission_id, &prepared).await?;
    Ok(())
}

/// Organizer ground truth from the METRICS_JSON v2 blob. Every field is
/// optional — missing values degrade the affected checks to `flagged`
/// (never crash); `tokens_seen` is authoritative only when the harness
/// counted it from the train stream. Public for the external Zone B POST
/// route (`prism-attribution`), which validates miner self-reports against
/// the same organizer facts as the internal `train_metrics` lift.
#[must_use]
pub fn ground_truth(v: &Value) -> GroundTruth {
    let train_stream = v.get("tokens_seen_source").and_then(Value::as_str) == Some("train_stream");
    GroundTruth {
        bpb: v.get("bpb").and_then(Value::as_f64),
        wall_s: v.get("wall_clock_seconds").and_then(Value::as_f64),
        tokens_seen: if train_stream {
            v.get("tokens_seen").and_then(Value::as_u64)
        } else {
            None
        },
        n_params: v.get("n_params").and_then(Value::as_u64),
        gpu_type: v.get("gpu_type").and_then(Value::as_str).map(str::to_owned),
        max_steps: u64::from(prism_recipe::MAX_TRAIN_STEPS),
    }
}

/// `SubmissionMetrics` from the battery blob; `None` when no `org.*`
/// metrics exist (v1/v2 harness run — composite skipped).
fn submission_metrics(v: &Value) -> Option<SubmissionMetrics> {
    let battery = v.get("battery")?;
    let mut metrics = BTreeMap::new();
    for (k, raw) in battery.get("metrics")?.as_object()? {
        if !k.starts_with("org.") {
            continue;
        }
        // Skip an entry that is not a number or `{value, clusters}` rather
        // than abandoning the whole submission. Aborting here used to mean a
        // single malformed `org.*` value (e.g. a string diagnostic) made
        // `submission_metrics` return `None`, which SKIPS the composite
        // silently — a scoring bypass triggered by a typo. Skipping instead
        // is fail-closed: if the key is *declared* by the anchor set its
        // absence is a hard `MissingMetric` → `Ineligible`, and if it is not
        // declared it was inert anyway (`unknown_metrics_are_ignored`).
        if let Some(series) = series_from(raw) {
            metrics.insert(k.clone(), series);
        }
    }
    if metrics.is_empty() {
        return None;
    }
    let mirrors = battery
        .get("mirrors")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(mirror_from).collect())
        .unwrap_or_default();
    Some(SubmissionMetrics {
        metrics,
        budget: budget_facts(v),
        mirrors,
    })
}

/// Bare number or `{value, clusters}` → `MetricSeries`.
fn series_from(v: &Value) -> Option<MetricSeries> {
    if let Some(x) = v.as_f64() {
        return Some(MetricSeries {
            value: x,
            clusters: BTreeMap::new(),
        });
    }
    let value = v.get("value")?.as_f64()?;
    let clusters = v
        .get("clusters")
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .filter_map(|(k, x)| Some((k.clone(), x.as_f64()?)))
                .collect()
        })
        .unwrap_or_default();
    Some(MetricSeries { value, clusters })
}

fn mirror_from(v: &Value) -> Option<MirrorPair> {
    Some(MirrorPair {
        group: v.get("group")?.as_str()?.to_owned(),
        metric: v.get("metric")?.as_str()?.to_owned(),
        public: series_from(v.get("public")?)?,
        mirror: series_from(v.get("mirror")?)?,
    })
}

/// Budget facts for the fixed-recipe gates; `tokens` is recorded (not
/// gated) and only meaningful when the harness counted the train stream.
///
/// FLOPs attestation (`budget.flops_attested`, `budget.binding_cap`) is read
/// from the harness's `budget` block, which is written by
/// `prismlib/stream.py` — never by miner code. It is deliberately gated on
/// the same `tokens_seen_source == "train_stream"` discriminator as `tokens`:
/// `flops_attested = flops_per_token × tokens_seen`, so an unauthoritative
/// token count makes the product unauthoritative too. Treating a
/// bypassed-stream run's FLOPs as attested would let a miner with their own
/// dataloader choose their own budget.
fn budget_facts(v: &Value) -> BudgetFacts {
    let stream_owned = v.get("tokens_seen_source").and_then(Value::as_str) == Some("train_stream");
    let budget = v.get("budget");
    BudgetFacts {
        params: v.get("n_params").and_then(Value::as_u64).unwrap_or(0),
        wall_s: v
            .get("wall_clock_seconds")
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        tokens: if stream_owned {
            v.get("tokens_seen").and_then(Value::as_u64).unwrap_or(0)
        } else {
            0
        },
        flops_attested: if stream_owned {
            budget
                .and_then(|b| b.get("flops_attested"))
                .and_then(Value::as_f64)
                .filter(|f| f.is_finite() && *f > 0.0)
        } else {
            None
        },
        binding_cap: budget
            .and_then(|b| b.get("binding_cap"))
            .and_then(Value::as_str)
            .map_or(BindingCap::None, BindingCap::parse),
    }
}

/// Deterministic bootstrap seed from the submission id (reproducible
/// re-scores).
fn seed_for(submission_id: &str) -> u64 {
    let d = Sha256::digest(submission_id.as_bytes());
    u64::from_le_bytes(d[..8].try_into().unwrap_or([0; 8]))
}

fn backend<E: std::fmt::Display>(e: E) -> StoreError {
    StoreError::Backend(e.to_string())
}

fn clusters_json(s: &MetricSeries) -> Option<Value> {
    if s.clusters.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&s.clusters).unwrap_or(Value::Null))
    }
}

/// Persist the run (replace-wholesale) + anchor registry + prereg commit.
async fn persist_run(
    store: &Arc<dyn EvalStore>,
    submission_id: &str,
    metrics_v2: &Value,
    anchors: &AnchorInput,
    parsed: &CompositeAnchors,
    sub: &SubmissionMetrics,
    outcome: &CompositeOutcome,
) -> Result<(), FinalizeError> {
    let prereg_hash = anchors.prereg_hash();
    let version = i32::from(parsed.version);
    store
        .upsert_anchor_set(&AnchorSetRecord {
            version,
            json: serde_json::from_str(&anchors.canonical_json).map_err(CompositeError::from)?,
            prereg_hash: prereg_hash.clone(),
            status: anchors.status.clone(),
            activated_at: None,
        })
        .await?;
    store
        .record_prereg(&PreregRecord {
            version,
            hash: prereg_hash.clone(),
            committed_at: String::new(), // store-assigned
            notes: None,
        })
        .await?;
    let outcome_groups = match outcome {
        CompositeOutcome::Scored(s) => &s.groups,
        CompositeOutcome::Ineligible(i) => &i.groups,
    };
    let scoring_mode = ScoringMode::from_env().name();
    let run = EvalRunRecord {
        run_id: String::new(),     // store-assigned
        created_at: String::new(), // store-assigned
        submission_id: submission_id.to_owned(),
        anchor_version: version,
        prereg_hash,
        scoring_mode: scoring_mode.to_owned(),
        pod_manifest: metrics_v2.get("pod_manifest").cloned(),
        harness_files_sha256: metrics_v2
            .get("harness_files_sha256")
            .and_then(Value::as_str)
            .map(str::to_owned),
        netns: metrics_v2.get("netns").and_then(Value::as_bool),
        eval_tier: metrics_v2
            .get("battery")
            .and_then(|b| b.get("tier"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        outcome_json: serde_json::to_value(outcome).map_err(backend)?,
    };
    let groups: Vec<EvalGroupRecord> = outcome_groups
        .iter()
        .map(|g| EvalGroupRecord {
            grp: g.group.clone(),
            g: g.g,
            ci_lo: g.ci_lo,
            ci_hi: g.ci_hi,
            mirror_penalty: g.mirror_penalty,
        })
        .collect();
    let metrics: Vec<EvalMetricRecord> = sub
        .metrics
        .iter()
        .map(|(k, s)| EvalMetricRecord {
            key: k.clone(),
            value: s.value,
            clusters: clusters_json(s),
        })
        .collect();
    let mirrors: Vec<MirrorPairRecord> = sub
        .mirrors
        .iter()
        .map(|m| MirrorPairRecord {
            grp: m.group.clone(),
            metric: m.metric.clone(),
            public_value: m.public.value,
            mirror_value: m.mirror.value,
            public_clusters: clusters_json(&m.public),
            mirror_clusters: clusters_json(&m.mirror),
        })
        .collect();
    store
        .save_eval_run(&run, &groups, &metrics, &mirrors)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::{eval_detail, MemoryEvalStore};
    use prism_zoneb::{Verdict, VerdictReason};
    use serde_json::json;

    fn store() -> Arc<dyn EvalStore> {
        Arc::new(MemoryEvalStore::new())
    }

    fn v2_blob() -> Value {
        json!({
            "bpb": 1.0,
            "tokens_seen": 1_000_000_000_u64,
            "tokens_seen_source": "train_stream",
            "wall_clock_seconds": 3600.0,
            "gpu_type": "H100",
            "n_params": 125_000_000_u64,
            "recipe": "1.3.0",
            "metrics_version": 2,
            "netns": true,
            "harness_files_sha256": "ab".repeat(32),
            "train_metrics": { "miner.train.loss": 0.7 },
        })
    }

    /// Battery with every anchor-declared metric at its perfect value
    /// (data-driven from the embedded v0 set) + zero-variance clusters.
    fn perfect_battery() -> Value {
        let anchors: Value =
            serde_json::from_str(&AnchorInput::v0_placeholder().canonical_json).unwrap();
        let mut metrics = serde_json::Map::new();
        for (_gk, g) in anchors["groups"].as_object().unwrap() {
            for (mk, m) in g["metrics"].as_object().unwrap() {
                let perfect = match m["kind"].as_str().unwrap() {
                    "accuracy" | "stability_bounded" => 1.0,
                    "bpb_log_ratio" => m["reference"].as_f64().unwrap(),
                    "efficiency_log_ratio" => m["cap"].as_f64().unwrap(),
                    other => panic!("unknown norm {other}"),
                };
                metrics.insert(
                    mk.clone(),
                    json!({"value": perfect, "clusters": {"c0": perfect, "c1": perfect}}),
                );
            }
        }
        json!({ "metrics": metrics, "tier": "battery-v1" })
    }

    #[test]
    fn lift_scalars_direction_and_passthrough() {
        let blob = json!({ "train_metrics": {
            "miner.train.loss": 0.7,
            "miner.sys.tokens_per_s": 42_000.0,
            "org.eval.bpb": 0.5,
            "not.a.metric": 1.0,
            "miner.train.bogus": f64::NAN,
        }});
        let m = from_train_metrics(&blob).unwrap();
        assert_eq!(m.len(), 3, "invalid name + NaN dropped: {m:?}");
        assert_eq!(
            m["miner.train.loss"].direction,
            Some(Direction::Lower),
            "loss hinted lower-better"
        );
        assert_eq!(
            m["miner.sys.tokens_per_s"].direction,
            Some(Direction::Neutral)
        );
        assert!(
            m.contains_key("org.eval.bpb"),
            "org.* lifted for quarantine"
        );
    }

    #[test]
    fn lift_absent_or_over_cap_is_none() {
        assert!(from_train_metrics(&json!({})).is_none());
        assert!(from_train_metrics(&json!({"train_metrics": {}})).is_none());
        let over: serde_json::Map<String, Value> = (0..=MAX_SCALARS)
            .map(|i| (format!("miner.train.k{i}"), json!(1.0)))
            .collect();
        assert!(from_train_metrics(&json!({"train_metrics": over})).is_none());
    }

    #[tokio::test]
    async fn skips_composite_without_battery_but_ingests_zone_b() {
        let st = store();
        let out = finalize_composite(&st, "sub-1", &v2_blob(), &AnchorInput::v0_placeholder())
            .await
            .unwrap();
        assert_eq!(out, None, "no battery -> composite skipped");
        assert!(st.eval_run("sub-1").await.unwrap().is_none());
        let reports = st.metric_reports("sub-1").await.unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].seq, 0);
        assert_eq!(reports[0].prev_hash, "sub-1");
        assert_eq!(
            reports[0].verdict,
            Verdict::Ok,
            "{:?}",
            reports[0].verdict_reasons
        );
        assert_eq!(reports[0].schema_version, "1.3.0");
    }

    /// A malformed `org.*` entry must not disable scoring. It used to: a
    /// single non-numeric value made `submission_metrics` return `None`,
    /// which skips the composite entirely — a scoring bypass reachable by a
    /// typo (or by a harness emitting a string diagnostic into the battery).
    #[tokio::test]
    async fn non_numeric_org_metric_is_skipped_not_fatal() {
        let st = store();
        let mut blob = v2_blob();
        let mut battery = perfect_battery();
        // A string where a number is contracted, plus an undeclared numeric
        // diagnostic that must simply be ignored.
        battery["metrics"]["org.diag.binding_cap"] = json!("flops");
        battery["metrics"]["org.diag.mfu_achieved"] = json!(0.23);
        blob["battery"] = battery;
        let out = finalize_composite(&st, "sub-skip", &blob, &AnchorInput::v0_placeholder())
            .await
            .unwrap()
            .expect("battery present: the composite must still run");
        let CompositeOutcome::Scored(s) = &out else {
            panic!("a stray non-numeric diagnostic must not break scoring: {out:?}");
        };
        assert!(s.composite > 0.95, "composite {}", s.composite);
    }

    /// The dual-cap attestation reaches the gates through the top-level
    /// `budget` block, and only when the token count is authoritative.
    #[tokio::test]
    async fn budget_block_feeds_the_compute_gates() {
        let mut blob = v2_blob();
        blob["budget"] = json!({
            "flops_attested": 2.9e18,
            "binding_cap": "flops",
            "spend_fraction": 0.97,
        });
        let facts = budget_facts(&blob);
        assert_eq!(facts.flops_attested, Some(2.9e18));
        assert_eq!(facts.binding_cap, BindingCap::Flops);

        // A bypassed stream makes the product unauthoritative: `flops` is
        // `flops_per_token × tokens_seen`, so an untrusted token count
        // untrusts the FLOPs too.
        blob["tokens_seen_source"] = json!("legacy");
        let bypassed = budget_facts(&blob);
        assert_eq!(bypassed.flops_attested, None, "legacy stream ⇒ unattested");
        assert_eq!(bypassed.tokens, 0);

        // Garbage never becomes a budget.
        for bad in [json!(0.0), json!(-1.0), json!("lots"), json!(null)] {
            let mut b = v2_blob();
            b["budget"] = json!({ "flops_attested": bad });
            assert_eq!(budget_facts(&b).flops_attested, None, "{bad:?}");
        }
        // Absent budget block ⇒ pre-attestation run, gates skipped.
        assert_eq!(budget_facts(&v2_blob()).flops_attested, None);
    }

    #[tokio::test]
    async fn scores_full_battery_and_persists_everything() {
        let st = store();
        let mut blob = v2_blob();
        let battery = perfect_battery();
        let n_metrics = battery["metrics"].as_object().unwrap().len();
        blob["battery"] = battery;
        let out = finalize_composite(&st, "sub-2", &blob, &AnchorInput::v0_placeholder())
            .await
            .unwrap()
            .expect("battery present");
        let CompositeOutcome::Scored(s) = &out else {
            panic!("perfect battery must score: {out:?}");
        };
        assert!(s.composite > 0.95, "composite {}", s.composite);
        assert!(s.lcb > 0.95);
        assert_eq!(s.anchor_version, 0);

        let run = st.eval_run("sub-2").await.unwrap().expect("run row");
        assert_eq!(run.anchor_version, 0);
        assert_eq!(run.scoring_mode, "benchmarks");
        assert_eq!(run.prereg_hash, AnchorInput::v0_placeholder().prereg_hash());
        assert_eq!(run.eval_tier.as_deref(), Some("battery-v1"));
        assert_eq!(run.netns, Some(true));
        assert_eq!(
            run.harness_files_sha256.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        let groups = st.eval_groups(&run.run_id).await.unwrap();
        assert_eq!(groups.len(), 8);
        assert!(groups.iter().all(|g| g.ci_lo.is_some()));
        assert_eq!(st.eval_metrics(&run.run_id).await.unwrap().len(), n_metrics);
        let anchors = st.anchor_sets().await.unwrap();
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].status, "placeholder");
        let prereg = st.preregistrations().await.unwrap();
        assert_eq!(prereg.len(), 1);
        assert_eq!(prereg[0].hash, run.prereg_hash);

        let detail = eval_detail(&run, &groups);
        assert_eq!(detail["scoring_mode"], "benchmarks");
        assert_eq!(detail["groups"].as_array().unwrap().len(), 8);
        assert!(detail["composite"].as_f64().unwrap() > 0.95);
    }

    #[tokio::test]
    async fn persists_ineligible_outcome() {
        let st = store();
        let mut blob = v2_blob();
        blob["battery"] = json!({ "metrics": { "org.g1.bits_per_byte_code": 1.0 } });
        let out = finalize_composite(&st, "sub-3", &blob, &AnchorInput::v0_placeholder())
            .await
            .unwrap()
            .expect("battery present");
        let CompositeOutcome::Ineligible(i) = &out else {
            panic!("partial battery must be ineligible: {out:?}");
        };
        assert!(!i.reasons.is_empty());
        let run = st.eval_run("sub-3").await.unwrap().expect("run row");
        assert_eq!(run.outcome_json["status"], "ineligible");
        assert_eq!(st.eval_groups(&run.run_id).await.unwrap().len(), 8);
    }

    #[tokio::test]
    async fn zone_b_idempotent_then_chains_on_new_metrics() {
        let st = store();
        let blob = v2_blob();
        for _ in 0..2 {
            finalize_composite(&st, "sub-4", &blob, &AnchorInput::v0_placeholder())
                .await
                .unwrap();
        }
        let reports = st.metric_reports("sub-4").await.unwrap();
        assert_eq!(reports.len(), 1, "re-finalize with same metrics is a no-op");

        let mut blob2 = v2_blob();
        blob2["train_metrics"] = json!({ "miner.train.loss": 0.71 });
        finalize_composite(&st, "sub-4", &blob2, &AnchorInput::v0_placeholder())
            .await
            .unwrap();
        let reports = st.metric_reports("sub-4").await.unwrap();
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[1].seq, 1);
        assert_eq!(reports[1].prev_hash, reports[0].report_hash);
    }

    #[tokio::test]
    async fn zone_b_org_namespace_quarantined_and_stripped() {
        let st = store();
        let mut blob = v2_blob();
        blob["train_metrics"] = json!({
            "miner.train.loss": 0.7,
            "org.eval.bpb": 0.5,
        });
        finalize_composite(&st, "sub-5", &blob, &AnchorInput::v0_placeholder())
            .await
            .unwrap();
        let reports = st.metric_reports("sub-5").await.unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].verdict, Verdict::Quarantined);
        assert!(reports[0]
            .verdict_reasons
            .iter()
            .any(|r| matches!(r, VerdictReason::OrgNamespace { key } if key == "org.eval.bpb")));
        let metrics = reports[0].payload["metrics"].as_object().unwrap();
        assert!(metrics.contains_key("miner.train.loss"));
        assert!(
            !metrics.contains_key("org.eval.bpb"),
            "forgery stripped from payload"
        );
    }
}
