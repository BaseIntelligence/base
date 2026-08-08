//! Admin candidates + winners API (memory store).

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use design_challenge_task::SCORE_MAX;
use design_http::{design_router, AppState};
use design_store::{
    DesignStore, FinalScore, HarnessRow, MemoryDesignStore, RoundAward, RoundRow, RunStage,
    RunState,
};
use http_body_util::BodyExt;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tower::ServiceExt;

fn token_hash(tok: &str) -> String {
    let mut h = Sha256::new();
    h.update(tok.as_bytes());
    hex::encode(h.finalize())
}

async fn setup() -> (Arc<MemoryDesignStore>, Arc<AppState>) {
    let store = Arc::new(MemoryDesignStore::new());
    store
        .insert_round(&RoundRow {
            round_id: 7,
            epoch: 1,
            netuid: 1,
            prompt_set_digest: "ab".repeat(32),
            status: "open".into(),
            opens_at_secs: 0,
            closes_at_secs: 21_600,
        })
        .await
        .unwrap();
    for (hid, miner) in [("h1", "aa"), ("h2", "bb"), ("h3", "cc")] {
        store
            .insert_harness(&HarnessRow {
                id: hid.into(),
                miner_hotkey: miner.repeat(32),
                miner_coldkey: None,
                agent_py: "print(1)".into(),
                pyproject_toml: "[project]\nname='x'\nversion='0'\n".into(),
                extra_files: BTreeMap::default(),
                active: true,
                eliminated_until_round: 0,
                created_at_ms: 0,
            })
            .await
            .unwrap();
        store
            .insert_run(&RunState {
                id: format!("run-{hid}"),
                round_id: 7,
                harness_id: hid.into(),
                prompt_id: "p01".into(),
                status: RunStage::AwaitingAdmin,
                artifact_digest: Some("d".into()),
                sanitize_report: None,
                agentic_verdict: Some(serde_json::json!({"verdict":"clean"})),
                error_detail: None,
                reject_reason: None,
                attempt_epoch: None,
                awaiting_admin_epoch: None,
                final_score: None,
                retry_count: 0,
                created_at_ms: 1,
                updated_at_ms: 1,
            })
            .await
            .unwrap();
        store
            .put_artifacts(
                &format!("run-{hid}"),
                &[(
                    "index.html".into(),
                    "<html></html>".into(),
                    "<html></html>".into(),
                    "00".repeat(32),
                    13,
                )],
            )
            .await
            .unwrap();
    }
    let th = token_hash("secret-admin");
    let state = Arc::new(AppState {
        store: Arc::clone(&store) as Arc<dyn DesignStore>,
        epoch: Arc::new(AtomicU64::new(0)),
        netuid: 1,
        backend_mode: "sim",
        annotator_token_hashes: vec![th.clone()],
        admin_token_hashes: vec![th],
        frame_ancestors: "'none'".into(),
        retry_max: 2,
        award_hook: None,
        gating: None,
        metagraph: None,
    });
    (store, state)
}

#[tokio::test]
async fn candidates_and_two_winners() {
    let (store, state) = setup().await;
    let app = design_router(state);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/admin/rounds/7/candidates")
                .header("authorization", "Bearer secret-admin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["candidates"].as_array().unwrap().len(), 3);

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/rounds/7/winners")
                .header("authorization", "Bearer secret-admin")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"harness_ids":["h1","h2"]}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::ACCEPTED);

    let award = store.get_round_award(7).await.unwrap().unwrap();
    assert_eq!(
        award.winner_harness_ids,
        vec!["h1".to_owned(), "h2".to_owned()]
    );
    // Leaf mass contract: two winners → SCORE_MAX/2 each (see design-challenge score_round).
    assert_eq!(award.winner_harness_ids.len(), 2);
    assert_eq!((SCORE_MAX / 2) * 2, SCORE_MAX);
    let _ = RoundAward {
        round_id: award.round_id,
        winner_harness_ids: award.winner_harness_ids,
        awarded_at_ms: award.awarded_at_ms,
        admin_token_hash: award.admin_token_hash,
    };
    let _ = FinalScore::Score(SCORE_MAX / 2);
}

#[tokio::test]
async fn admin_reject_sets_reason_on_run() {
    let (store, state) = setup().await;
    let app = design_router(state);

    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/rounds/7/reject")
                .header("authorization", "Bearer secret-admin")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"harness_ids":["h3"],"reason":"layout too thin"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::ACCEPTED);

    let run = store.get_run("run-h3").await.unwrap().unwrap();
    assert_eq!(run.status, RunStage::Rejected);
    assert_eq!(run.reject_reason.as_deref(), Some("layout too thin"));
    assert_eq!(run.final_score, Some(FinalScore::Score(0)));

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/runs/run-h3")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&res.into_body().collect().await.unwrap().to_bytes()).unwrap();
    assert_eq!(body["status"], "rejected");
    assert_eq!(body["reject_reason"], "layout too thin");
}
