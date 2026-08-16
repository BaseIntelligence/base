//! Fail-closed contamination gate (`PRISM_EVAL_REQUIRE_PRIVATE=1`).
//!
//! Own integration binary because [`require_contamination_check`] memoizes
//! in a `OnceLock`. The default-off behavior is covered by the crate's unit
//! tests, where the env is unset.

#![allow(clippy::expect_used)]

use std::sync::Arc;

use prism_eval_store::{
    contamination_checked, finalize_composite, finalize_for_submission, AnchorInput, FinalizeError,
    MemoryEvalStore,
};
use prism_store::eval::EvalStore;
use serde_json::{json, Value};

/// A battery blob with real `org.*` metrics and a settable defence flag.
fn blob(checked: Option<bool>) -> Value {
    let mut battery = json!({
        "metrics": {
            "org.g1.bits_per_byte_prose": {"value": 1.1, "clusters": {"p#0": 1.0, "p#1": 1.2}},
            "org.g2.hellaswag_acc": 0.42,
        },
        "tier": "public_dev",
    });
    if let Some(flag) = checked {
        battery["mirror_defence"] = json!({
            "contamination_checked": flag,
            "inert": !flag,
            "pairs": 1,
            "inert_pairs": u8::from(!flag),
            "live_pairs": u8::from(flag),
        });
    }
    json!({ "battery": battery })
}

fn store() -> Arc<dyn EvalStore> {
    Arc::new(MemoryEvalStore::default())
}

#[test]
fn inert_mirror_defence_refuses_to_score_when_required() {
    std::env::set_var("PRISM_EVAL_REQUIRE_PRIVATE", "1");
    let st = store();

    // Inert defence ⇒ hard refusal, not a silent zero.
    let err = tokio_run(finalize_composite(
        &st,
        "sub-inert",
        &blob(Some(false)),
        &AnchorInput::v0_placeholder(),
    ))
    .expect_err("must refuse");
    assert!(
        matches!(err, FinalizeError::ContaminationUnchecked),
        "got {err:?}"
    );
    // Nothing persisted for a refused run.
    assert!(tokio_run(st.eval_run("sub-inert")).unwrap().is_none());
}

#[test]
fn missing_flag_is_treated_as_unchecked() {
    std::env::set_var("PRISM_EVAL_REQUIRE_PRIVATE", "1");
    let st = store();
    // An older harness cannot prove a check ran ⇒ fail closed.
    let err = tokio_run(finalize_composite(
        &st,
        "sub-legacy",
        &blob(None),
        &AnchorInput::v0_placeholder(),
    ))
    .expect_err("absent flag must fail closed");
    assert!(matches!(err, FinalizeError::ContaminationUnchecked));
}

#[test]
fn live_mirror_defence_scores_normally() {
    std::env::set_var("PRISM_EVAL_REQUIRE_PRIVATE", "1");
    let st = store();
    let out = tokio_run(finalize_composite(
        &st,
        "sub-live",
        &blob(Some(true)),
        &AnchorInput::v0_placeholder(),
    ))
    .expect("a checked run must score");
    assert!(out.is_some(), "checked run produced no outcome");
    assert!(tokio_run(st.eval_run("sub-live")).unwrap().is_some());
}

#[test]
fn orchestrator_wrapper_degrades_to_none_not_panic() {
    std::env::set_var("PRISM_EVAL_REQUIRE_PRIVATE", "1");
    let st = store();
    // `finalize_for_submission` warns and returns None on any finalize
    // fault; in composite mode `final_lattice` then fails closed to 0.
    let out = tokio_run(finalize_for_submission(
        Some(&st),
        "sub-wrapped",
        Some(&blob(Some(false))),
    ));
    assert!(out.is_none(), "refused run must not yield a scored outcome");
}

#[test]
fn flag_reader_is_strict_about_shape() {
    assert!(contamination_checked(&blob(Some(true))));
    assert!(!contamination_checked(&blob(Some(false))));
    assert!(!contamination_checked(&blob(None)));
    assert!(!contamination_checked(&json!({})));
    // A non-bool must not read as true.
    assert!(!contamination_checked(&json!({
        "battery": {"mirror_defence": {"contamination_checked": "yes"}}
    })));
}

/// Minimal blocking bridge so each test stays a plain `#[test]`.
fn tokio_run<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
        .block_on(fut)
}
