//! Sim e2e: baseline agent.py → run → sanitize pages → admin winners score.
//!
//! Cheat / scrape / host-sim refusal coverage lives in `cheat_fixtures.rs`.

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use design_challenge::score::{round_win_delta, score_window, ScorePlan, WindowScorePlan};
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
        env_vars: BTreeMap::new(),
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
            created_at_ms: 0,
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

    // scoring_version=3: each winner banks one window point; equal points →
    // equal split of SCORE_MAX.
    let plan = ScorePlan {
        miners_with_harness: vec!["aa".into(), "bb".into()],
        miners_clean: vec!["aa".into(), "bb".into()],
        winner_miners: vec!["aa".into(), "bb".into()],
        cheat_miners: vec![],
    };
    let delta = round_win_delta(&plan);
    assert_eq!(delta.get("aa"), Some(&1));
    assert_eq!(delta.get("bb"), Some(&1));
}

#[tokio::test]
async fn window_two_winners_share_scores() {
    let scores = score_window(&WindowScorePlan {
        win_counts: [("m1".to_owned(), 2_u32), ("m2".to_owned(), 2_u32)]
            .into_iter()
            .collect(),
        miners_with_harness: ["m1".into(), "m2".into(), "m3".into()]
            .into_iter()
            .collect(),
        cheat_miners: BTreeSet::default(),
    });
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
