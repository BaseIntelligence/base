//! Sim e2e: baseline agent.py → run → sanitize pages → admin winners score.
//!
//! Cheat / scrape / host-sim refusal coverage lives in `cheat_fixtures.rs`.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use design_challenge::score::{score_round, ScorePlan};
use design_challenge::SCORE_MAX;
use design_harness::{harness_id, validate_bundle, HarnessBundle};
use design_sandbox::{SandboxBackend, SimSandbox};
use design_sanitize::sanitize_bundle;
use design_store::{DesignStore, HarnessRow, MemoryDesignStore, RunStage, RunState};

const BASELINE_AGENT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/agent.py");
const BASELINE_PYPROJECT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/pyproject.toml");

#[tokio::test]
async fn sim_pipeline_pages_and_admin_score() {
    let store = Arc::new(MemoryDesignStore::new());
    let bundle = HarnessBundle {
        miner_hotkey: "ab".repeat(32),
        agent_py: BASELINE_AGENT.into(),
        pyproject_toml: BASELINE_PYPROJECT.into(),
        extra_files: BTreeMap::new(),
    };
    validate_bundle(&bundle).unwrap();
    let hid = harness_id(&bundle);
    store
        .insert_harness(&HarnessRow {
            id: hid.clone(),
            miner_hotkey: bundle.miner_hotkey.clone(),
            agent_py: bundle.agent_py.clone(),
            pyproject_toml: bundle.pyproject_toml.clone(),
            extra_files: BTreeMap::new(),
            active: true,
            eliminated_until_round: 0,
        })
        .await
        .unwrap();

    store
        .insert_run(&RunState {
            id: "run-a".into(),
            round_id: 1,
            harness_id: hid.clone(),
            prompt_id: "p01".into(),
            status: RunStage::Queued,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            final_score: None,
            retry_count: 0,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .await
        .unwrap();

    let sim = SimSandbox::new();
    let out = sim
        .execute(&bundle, 1, "run-a", "brief", "http://proxy")
        .unwrap();
    let sanitized = sanitize_bundle(&out.pages).unwrap();
    assert_eq!(sanitized.pages.len(), 3);

    let plan = ScorePlan {
        miners_with_harness: vec!["aa".into(), "bb".into()],
        miners_clean: vec!["aa".into(), "bb".into()],
        winner_miners: vec!["aa".into(), "bb".into()],
        cheat_miners: vec![],
    };
    let scores = score_round(&plan);
    assert_eq!(
        scores.get("aa"),
        Some(&design_store::FinalScore::Score(SCORE_MAX / 2))
    );
    assert_eq!(
        scores.get("bb"),
        Some(&design_store::FinalScore::Score(SCORE_MAX / 2))
    );
}

#[tokio::test]
async fn admin_two_winners_half_scores() {
    let plan = ScorePlan {
        miners_with_harness: vec!["m1".into(), "m2".into(), "m3".into()],
        miners_clean: vec!["m1".into(), "m2".into(), "m3".into()],
        winner_miners: vec!["m1".into(), "m2".into()],
        cheat_miners: vec![],
    };
    let scores = score_round(&plan);
    assert_eq!(
        scores.get("m1"),
        Some(&design_store::FinalScore::Score(SCORE_MAX / 2))
    );
    assert_eq!(
        scores.get("m2"),
        Some(&design_store::FinalScore::Score(SCORE_MAX / 2))
    );
    assert_eq!(scores.get("m3"), Some(&design_store::FinalScore::Score(0)));
}
