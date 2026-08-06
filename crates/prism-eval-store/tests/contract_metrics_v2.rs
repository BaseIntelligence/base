//! Contract test: a REAL harness-emitted `METRICS_JSON` v2 blob through the
//! production ingestion → composite → persistence path.
//!
//! `fixtures/metrics_json_v2_real.json` was captured from an actual v3
//! two-phase `main.py` run (tiny caps, stub miner, CPU) — the same flow as
//! `crates/prism-recipe/harness/tests/smoke_battery.py::check_v3_flow` —
//! then trimmed of debug bulk (telemetry series, probe curve, pod manifest
//! detail, the `items` recorder dump). Everything the ingestion path reads
//! is untrimmed harness output: top-level budget/ground-truth fields and
//! the full `battery` object (`groups`, flat `org.*` `metrics`, `mirrors`,
//! `tier`). Regenerate with the harness smoke when the contract evolves.
//!
//! This is the regression net for the battery-blob contract: if the harness
//! ever stops emitting flat `org.*` metrics or mirror pairs,
//! `finalize_composite` silently returns `None` and composite mode
//! fail-closes every submission to Score(0) — this test fails first.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use prism_eval_store::{finalize_composite, AnchorInput, MemoryEvalStore};
use prism_pipeline::composite::CompositeOutcome;
use prism_store::eval::EvalStore;
use serde_json::Value;

const REAL_BLOB: &str = include_str!("fixtures/metrics_json_v2_real.json");

fn blob() -> Value {
    serde_json::from_str(REAL_BLOB).expect("fixture parses")
}

#[tokio::test]
async fn real_harness_blob_drives_composite_and_persists_everything() {
    let blob = blob();
    // The fixture is genuinely harness-produced: v3 flow, public_dev tier,
    // every battery group ok.
    assert_eq!(blob["flow"], "v3");
    assert_eq!(blob["battery"]["tier"], "public_dev");
    let groups = blob["battery"]["groups"].as_object().unwrap();
    assert_eq!(groups.len(), 8);
    assert!(groups.values().all(|g| g["status"] == "ok"));
    let flat = blob["battery"]["metrics"].as_object().unwrap();
    assert!(!flat.is_empty());
    assert!(flat.keys().all(|k| k.starts_with("org.")));
    let mirrors = blob["battery"]["mirrors"].as_array().unwrap();
    assert!(!mirrors.is_empty());

    let store: Arc<dyn EvalStore> = Arc::new(MemoryEvalStore::new());
    let outcome = finalize_composite(&store, "sub-real", &blob, &AnchorInput::v0_placeholder())
        .await
        .expect("finalize ok")
        .expect("battery present -> composite runs (never silently skipped)");

    // A public_dev CPU smoke cannot meet the full anchor gates (GPU-only
    // G7 grid points and the µP sweep are absent) — the honest outcome is
    // Ineligible with recorded reasons, and every group row still persists.
    // The fully-Scored path is covered by the synthetic perfect battery in
    // `finalize.rs` unit tests.
    let CompositeOutcome::Ineligible(i) = &outcome else {
        panic!("public_dev CPU blob must gate Ineligible: {outcome:?}");
    };
    assert!(!i.reasons.is_empty(), "gate reasons recorded");
    assert_eq!(i.groups.len(), 8, "all 8 groups evaluated");

    let run = store
        .eval_run("sub-real")
        .await
        .unwrap()
        .expect("eval run row persists");
    assert_eq!(run.outcome_json["status"], "ineligible");
    assert_eq!(run.eval_tier.as_deref(), Some("public_dev"));
    assert_eq!(run.anchor_version, 0);
    assert_eq!(run.netns, Some(true));

    // All 8 group rows (with bootstrap CIs where measurable), every org.*
    // metric row, and every mirror pair row persist.
    let rows = store.eval_groups(&run.run_id).await.unwrap();
    assert_eq!(rows.len(), 8, "{rows:?}");
    assert_eq!(
        store.eval_metrics(&run.run_id).await.unwrap().len(),
        flat.len(),
        "one eval metric row per harness org.* key"
    );
    let mirror_rows = store.eval_mirrors(&run.run_id).await.unwrap();
    assert_eq!(mirror_rows.len(), mirrors.len());
    assert!(mirror_rows.iter().all(|m| m.grp == "g2" || m.grp == "g4"));
    assert!(
        mirror_rows
            .iter()
            .all(|m| (m.public_value - m.mirror_value).abs() < f64::EPSILON),
        "public_dev tier: the run is its own mirror (gap 0, honestly labelled)"
    );

    // Anchor registry + preregistration bootstrap: rows appear exactly
    // because persist_run ran — GET /v1/anchors is non-empty after a run.
    let anchors = store.anchor_sets().await.unwrap();
    assert_eq!(anchors.len(), 1);
    assert_eq!(anchors[0].status, "placeholder");
    let prereg = store.preregistrations().await.unwrap();
    assert_eq!(prereg.len(), 1);
    assert_eq!(prereg[0].hash, run.prereg_hash);

    // The legacy train_metrics Zone B lift rides the same call (the capture
    // stub's train() returns miner.* keys, per the Zone B naming contract).
    let reports = store.metric_reports("sub-real").await.unwrap();
    assert_eq!(reports.len(), 1);
    assert!(reports[0]
        .payload
        .get("metrics")
        .and_then(Value::as_object)
        .is_some_and(|m| m.contains_key("miner.train.loss")));
}

/// The flat map covers the canonical anchor keys E8 flagged as mismatched
/// (harness-internal `gN.family.tag` names vs anchor `org.*` keys) — the
/// `eval/rollup.py` reconciliation is where they meet.
#[test]
fn real_blob_covers_flagged_anchor_keys() {
    let blob = blob();
    let flat = blob["battery"]["metrics"].as_object().unwrap();
    for key in [
        "org.g2.arc_challenge_acc", // harness: g2.arc_challenge.acc_norm
        "org.g6.auc_log_tokens",    // harness: g6.auc.log_tokens
    ] {
        assert!(
            flat.contains_key(key),
            "missing {key}: {}",
            flat.keys().len()
        );
    }
    // Cluster structure survives for the clustered bootstrap.
    let series = &flat["org.g3.mqar_acc"];
    assert!(series["value"].is_f64());
    assert!(series["clusters"]
        .as_object()
        .is_some_and(|c| !c.is_empty()));
}
