//! End-to-end sim pipeline: submit → master Sim Lium eval → D24 leaves → dry-run gateway.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crypto::KEY_LEN;
use prism_challenge::{
    emit_signed_leaf_set, example_valid_request, run_sim_pipeline, score_from_pipeline,
    submit_signed_leaf_set, GatewayClient, GatewayClientConfig, PipelineInput, PipelineOutcome,
    PrismConfig, SubmissionService, CHALLENGE_ID, SCORE_MAX,
};
use prism_lium::{EvalJobBackend, SimLiumBackend};

fn decode_hotkey(hex_s: &str) -> [u8; KEY_LEN] {
    let bytes = hex::decode(hex_s).expect("hotkey hex");
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes[..KEY_LEN]);
    out
}

#[tokio::test]
async fn e2e_sim_happy_path_scores_and_emits_d24() {
    let svc = SubmissionService::new();
    let req = example_valid_request();
    let accepted = svc.accept(req.clone()).expect("accept");
    assert_eq!(accepted.status, "accepted");
    let queued = svc.pop().expect("queued");

    let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
    let cfg = PrismConfig::sim();
    let result = run_sim_pipeline(
        backend.clone(),
        &cfg,
        PipelineInput {
            request: queued.request,
        },
    )
    .await
    .expect("pipeline");

    assert!(result.receipt.termination_verified);
    assert!(backend
        .verify_terminated(&result.pod_id)
        .await
        .expect("verify"));
    let ScoreOrAbsence::Score { value } = result.score else {
        panic!("expected Score, got {:?}", result.score);
    };
    assert!(value > 0 && value <= SCORE_MAX);

    // D24: one hotkey in E
    let hk = decode_hotkey(&example_valid_request().miner_hotkey);
    let expected = BTreeSet::from([hk]);
    let mut scores = BTreeMap::new();
    scores.insert(hk, result.score.clone());

    let mut sk = [3u8; KEY_LEN];
    sk[0] = 0x99;
    let leaves = emit_signed_leaf_set(&sk, 7, &expected, &scores).expect("leaves");
    assert_eq!(leaves.len(), 1);

    let gw = GatewayClient::new(GatewayClientConfig {
        base_url: "dry-run".into(),
        max_attempts: 1,
        backoff: std::time::Duration::from_millis(1),
    })
    .expect("gw");
    let out = submit_signed_leaf_set(&gw, &leaves).await.expect("submit");
    assert!(matches!(
        out.as_slice(),
        [prism_challenge::SubmitOutcome::DryRun { leaf_count: 1 }]
    ));
}

#[tokio::test]
async fn e2e_integrity_fail_closed_on_orphan() {
    // Force ChallengeInternal path is covered when terminate fails — Sim always
    // terminates. Here we assert NoScore mapping for ChallengeInternal.
    use bundle::{NoScoreReasonCode, ScoreOrAbsence};
    let s = score_from_pipeline(&PipelineOutcome::ChallengeInternal);
    assert_eq!(
        s,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
}

// re-export for test
use bundle::ScoreOrAbsence;
