//! Sim orchestrator e2e: submit → sim Lium → sim review → similarity → score
//! → exact-E leaf set (dry-run gateway) with no network.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chain::{AxonInfo, ChainError, Metagraph, WeightsTlockPayload};
use chain::{ChainClient, FakeChain, FakeChainConfig};
use challenge_agentic::SimAgent;
use crypto::KEY_LEN;
use prism_challenge::FinalScore;
use prism_challenge::{
    example_valid_request, submission_id, GatewayClient, GatewayClientConfig, MemoryPrismStore,
    Orchestrator, OrchestratorConfig, PrismStore, ScoringMode, Stage, StatePatch, SubmissionState,
};
use prism_lium::{
    CostGuardrailError, EvalJobBackend, Instance, InstanceSpec, LiumError, Offer, RemoteExecResult,
    SimLiumBackend,
};
use prism_review::{
    ReviewBackend, ReviewError, ReviewVerdict, SimReviewer, SimilarityVerdict, SourceSnippet,
};
use std::sync::Mutex;

fn fake_chain() -> FakeChain {
    let cfg = FakeChainConfig {
        owner_hotkey: vec![0xA0; 32],
        ..Default::default()
    };
    FakeChain::new(cfg)
}

/// `ChainClient` behind a `Mutex` so the orchestrator can hold a non-Sync
/// `FakeChain` under `Arc` in tests (production uses `LiveChainClient`, which
/// is `Sync`).
struct LockedFake(Mutex<FakeChain>);

macro_rules! delegate {
    (fn $name:ident(&self) -> $ret:ty) => {
        fn $name(&self) -> $ret {
            self.0.lock().expect("lock").$name()
        }
    };
    (fn $name:ident(&self, $($arg:ident : $t:ty),*) -> $ret:ty) => {
        fn $name(&self, $($arg: $t),*) -> $ret {
            self.0.lock().expect("lock").$name($($arg),*)
        }
    };
}

#[allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::missing_const_for_fn
)]
impl ChainClient for LockedFake {
    delegate!(fn current_block(&self) -> Result<u64, ChainError>);
    delegate!(fn block_hash(&self, n: u64) -> Result<[u8; 32], ChainError>);
    delegate!(fn metagraph_at(&self, block_hash: &[u8; 32]) -> Result<Metagraph, ChainError>);
    delegate!(fn subnet_owner_hotkey(&self, netuid: u16) -> Result<Vec<u8>, ChainError>);
    delegate!(fn axon(&self, netuid: u16, hotkey: &[u8]) -> Result<Option<AxonInfo>, ChainError>);
    delegate!(fn axons(&self, netuid: u16) -> Result<Vec<(Vec<u8>, AxonInfo)>, ChainError>);
    delegate!(fn commit_reveal_enabled(&self, netuid: u16) -> Result<bool, ChainError>);
    delegate!(fn commit_reveal_version(&self, netuid: u16) -> Result<u16, ChainError>);
    delegate!(fn tempo(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn reveal_period_epochs(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn block_time(&self) -> Result<u64, ChainError>);
    delegate!(fn last_epoch_block(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn pending_epoch_at(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn subnet_epoch_index(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn blocks_since_last_step(&self, netuid: u16) -> Result<u64, ChainError>);
    delegate!(fn submit_timelocked_weights(
        &self,
        mecid: u8,
        payload: WeightsTlockPayload,
        reveal_round: u64
    ) -> Result<(), ChainError>);
    delegate!(fn set_weights(
        &self,
        netuid: u16,
        uids: Vec<u16>,
        values: Vec<u16>,
        version_key: u64
    ) -> Result<(), ChainError>);
}

fn sk() -> [u8; KEY_LEN] {
    let mut s = [7u8; KEY_LEN];
    s[0] = 0x42;
    s
}

fn mk_orchestrator(
    store: &Arc<MemoryPrismStore>,
    chain: &Arc<LockedFake>,
) -> Orchestrator<LockedFake> {
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "dry-run".into(),
            max_attempts: 1,
            backoff: std::time::Duration::from_millis(1),
        })
        .unwrap(),
    );
    let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
    Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            scoring_mode: ScoringMode::Shadow,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(store) as Arc<dyn PrismStore>,
        backend,
        Arc::new(SimReviewer::new()),
        Arc::new(SimAgent::new()),
        &gateway,
        Arc::clone(chain),
        sk(),
    )
}

#[tokio::test]
async fn orchestrator_completes_one_submission_and_emits() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    let req = example_valid_request();
    let id = submission_id(&req);
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: req.miner_hotkey.clone(),
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: req.architecture_py.clone(),
            training_py: req.training_py.clone(),
            tree_blob: None,
            label: req.label,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: None,
            arch_id: None,
            review: None,
            similarity: None,
            final_score: None,
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert!(
        matches!(row.status, Stage::Terminated | Stage::Failed),
        "status={:?}",
        row.status
    );
    assert!(row.final_score.is_some(), "final score must be set");
    let events = store.events(&id).await.unwrap();
    assert!(events.len() >= 4, "events={events:?}");
    // Sim backend emits telemetry: it must land on the row blob + series.
    let metrics = row.metrics_json.as_ref().expect("metrics_json populated");
    assert!(metrics
        .get("bpb")
        .and_then(serde_json::Value::as_f64)
        .is_some());
    assert_eq!(
        metrics
            .pointer("/telemetry/finish_reason")
            .and_then(|v| v.as_str()),
        Some("finish_evaluation")
    );
    let series = store.telemetry(&id).await.unwrap();
    assert_eq!(series.len(), 5, "sim loss series, got {series:?}");
    assert!(series[0].loss > series[4].loss, "series decays");
}

#[tokio::test]
async fn submission_detail_exposes_metrics_over_http() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    let req = example_valid_request();
    let id = submission_id(&req);
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: req.miner_hotkey.clone(),
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: req.architecture_py.clone(),
            training_py: req.training_py.clone(),
            tree_blob: None,
            label: req.label,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
            bpb: None,
            arch_id: None,
            review: None,
            similarity: None,
            final_score: None,
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .await
        .unwrap();
    assert!(orch.cycle_once().await.unwrap());

    let state = Arc::new(prism_challenge::AppState {
        store: Arc::clone(&store) as Arc<dyn PrismStore>,
        eval_store: Arc::new(prism_challenge::MemoryEvalStore::new()),
        epoch: std::sync::atomic::AtomicU64::new(7),
        netuid: 541,
        backend_mode: "sim",
        retry_max: 2,
        gating: None,
        metagraph: None,
        admin_token_hashes: vec![],
        payer_vault: None,
        logs: std::sync::Arc::new(prism_orphan::LogBuffer::new()),
    });
    let app = prism_challenge::submission_router(state);
    let res = tower::ServiceExt::oneshot(
        app,
        axum::http::Request::get(format!("/v1/submissions/{id}"))
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(res.status(), axum::http::StatusCode::OK);
    let bytes = http_body_util::BodyExt::collect(res.into_body())
        .await
        .unwrap()
        .to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        v.pointer("/submission/metrics/telemetry/loss_series")
            .and_then(|s| s.as_array())
            .is_some_and(|a| !a.is_empty()),
        "detail exposes telemetry series: {v}"
    );
    assert_eq!(
        v.pointer("/submission/metrics/n_params")
            .and_then(serde_json::Value::as_u64),
        Some(12_000_000)
    );
    assert_eq!(
        v.pointer("/submission/n_params")
            .and_then(serde_json::Value::as_u64),
        Some(12_000_000),
        "list-view n_params surfaces in the detail payload"
    );
}

#[tokio::test]
async fn emit_and_submit_covers_expected_set() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    // Seed one finished score for hotkey 0xAB*32 (accepted in an earlier
    // epoch on purpose: emission is epoch-close batched, not acceptance-E).
    let hk = [0xABu8; 32];
    store.apply("seed", &StatePatch::default(), None).await.ok();
    store
        .insert_queued(&SubmissionState {
            id: "seed".into(),
            miner_hotkey: hex::encode(hk),
            miner_coldkey: None,
            epoch: 3,
            netuid: 541,
            status: Stage::Terminated,
            architecture_py: "a".into(),
            training_py: "t".into(),
            tree_blob: None,
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: Some(serde_json::json!({"recipe": "2.1.0"})),
            bpb: Some(2.0),
            arch_id: None,
            review: None,
            similarity: None,
            final_score: Some(FinalScore::Score(500_000)),
            retry_count: 0,
            error_detail: None,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .await
        .unwrap();

    let expected = challenge_common::ExpectedSet {
        participants: vec![
            challenge_common::ExpectedParticipant { hotkey: hk, uid: 0 },
            challenge_common::ExpectedParticipant {
                hotkey: [0xCDu8; 32],
                uid: 1,
            },
        ],
        block_hash: [0x77u8; 32],
    };
    let summary = orch
        .emitter()
        .emit_new(7, &expected)
        .await
        .expect("emit+submit");
    assert_eq!(summary.leaves, expected.participants.len(), "D24 coverage");
    assert_eq!(summary.batch, 1, "the seeded row is the outbox batch");
    assert!(matches!(
        summary.signed.get(&hk).map(|l| &l.score_or_absence),
        Some(prism_challenge::ScoreOrAbsence::Score { value: 500_000 })
    ));
    // Cursor advanced; same-epoch tick tip-refreshes WTA (cursor stays put).
    assert_eq!(store.emit_cursor(541).await.unwrap(), Some(7));
    let tip = orch
        .emitter()
        .tick(7, &expected)
        .await
        .unwrap()
        .expect("tip refresh");
    assert_eq!(tip.epoch, 7);
    assert_eq!(store.emit_cursor(541).await.unwrap(), Some(7));
}

/// Provision always fails with a fixed Lium error (no rent).
struct ProvisionFail(&'static str);

#[async_trait]
impl EvalJobBackend for ProvisionFail {
    async fn list_offers(&self, _: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        Ok(Vec::new())
    }
    async fn provision(&self, _: &InstanceSpec) -> Result<Instance, LiumError> {
        if self.0 == "capacity" {
            return Err(CostGuardrailError::NoCapacity.into());
        }
        Err(LiumError::Api(
            "POST /executors/x/rent -> 400 You don't have permission to rent this template.".into(),
        ))
    }
    async fn terminate(&self, _: &str) -> Result<(), LiumError> {
        Ok(())
    }
    async fn verify_terminated(&self, _: &str) -> Result<bool, LiumError> {
        Ok(true)
    }
    async fn exec_eval(
        &self,
        _: &str,
        _: &str,
        _: &str,
        _: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        Err(LiumError::Exec("unreachable".into()))
    }
}

fn row_from_example(id: &str, req: &prism_challenge::SubmissionRequest) -> SubmissionState {
    SubmissionState {
        id: id.to_owned(),
        miner_hotkey: req.miner_hotkey.clone(),
        miner_coldkey: None,
        epoch: 7,
        netuid: 541,
        status: Stage::Queued,
        architecture_py: req.architecture_py.clone(),
        training_py: req.training_py.clone(),
        tree_blob: None,
        label: req.label.clone(),
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        arch_id: None,
        review: None,
        similarity: None,
        final_score: None,
        retry_count: 0,
        error_detail: None,
        created_at_ms: 1,
        updated_at_ms: 1,
    }
}

struct CountingReviewer {
    inner: SimReviewer,
    reviews: AtomicUsize,
    sims: AtomicUsize,
}

impl CountingReviewer {
    fn new() -> Self {
        Self {
            inner: SimReviewer::new(),
            reviews: AtomicUsize::new(0),
            sims: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ReviewBackend for CountingReviewer {
    async fn review(
        &self,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<ReviewVerdict, ReviewError> {
        self.reviews.fetch_add(1, Ordering::SeqCst);
        self.inner.review(architecture_py, training_py).await
    }

    async fn similarity(
        &self,
        architecture_py: &str,
        corpus: &[SourceSnippet],
    ) -> Result<SimilarityVerdict, ReviewError> {
        self.sims.fetch_add(1, Ordering::SeqCst);
        self.inner.similarity(architecture_py, corpus).await
    }
}

fn orch_with_backend(
    store: &Arc<MemoryPrismStore>,
    chain: &Arc<LockedFake>,
    backend: Arc<dyn EvalJobBackend>,
) -> Orchestrator<LockedFake> {
    orch_with_backend_review(store, chain, backend, Arc::new(SimReviewer::new()))
}

fn orch_with_backend_review(
    store: &Arc<MemoryPrismStore>,
    chain: &Arc<LockedFake>,
    backend: Arc<dyn EvalJobBackend>,
    reviewer: Arc<dyn ReviewBackend>,
) -> Orchestrator<LockedFake> {
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "dry-run".into(),
            max_attempts: 1,
            backoff: std::time::Duration::from_millis(1),
        })
        .unwrap(),
    );
    Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            scoring_mode: ScoringMode::Shadow,
            auto_retry_max: 0,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(store) as Arc<dyn PrismStore>,
        backend,
        reviewer,
        Arc::new(SimAgent::new()),
        &gateway,
        Arc::clone(chain),
        sk(),
    )
}

#[tokio::test]
async fn no_capacity_requeues_with_b200_note() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let reviewer = Arc::new(CountingReviewer::new());
    let orch = orch_with_backend_review(
        &store,
        &chain,
        Arc::new(ProvisionFail("capacity")) as Arc<dyn EvalJobBackend>,
        Arc::clone(&reviewer) as Arc<dyn ReviewBackend>,
    );
    let req = example_valid_request();
    let id = submission_id(&req);
    store
        .insert_queued(&row_from_example(&id, &req))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Queued, "{row:?}");
    assert_eq!(row.retry_count, 0, "sold-out must not burn retry_count");
    assert!(row.final_score.is_none());
    let detail = row.error_detail.unwrap_or_default();
    assert!(
        detail.contains("B200s are currently out of capacity on Lium"),
        "{detail}"
    );
    let events = store.events(&id).await.unwrap();
    let queued_note = events.iter().any(|e| {
        e.stage == Stage::Queued
            && e.detail.as_ref().is_some_and(|d| {
                d.get("no_capacity") == Some(&serde_json::json!(true))
                    && d.get("note")
                        .and_then(|n| n.as_str())
                        .is_some_and(|n| n.contains("B200s are currently out of capacity"))
            })
    });
    assert!(queued_note, "events={events:?}");

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Queued, "next tick still queued");
    assert!(row.review.is_some(), "sold-out must keep llm_review");
    assert!(row.similarity.is_some(), "sold-out must keep similarity");
    assert_eq!(
        reviewer.reviews.load(Ordering::SeqCst),
        1,
        "llm_review must not re-run on every no_capacity tick"
    );
    assert_eq!(
        reviewer.sims.load(Ordering::SeqCst),
        1,
        "similarity must not re-run on every no_capacity tick"
    );
}

#[tokio::test]
async fn template_permission_stays_failed() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = orch_with_backend(
        &store,
        &chain,
        Arc::new(ProvisionFail("permission")) as Arc<dyn EvalJobBackend>,
    );
    let req = example_valid_request();
    let id = submission_id(&req);
    store
        .insert_queued(&row_from_example(&id, &req))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Failed, "{row:?}");
    let detail = row.error_detail.unwrap_or_default();
    assert!(detail.contains("permission"), "{detail}");
    assert!(!detail.contains("B200s are currently out of capacity"));
}
