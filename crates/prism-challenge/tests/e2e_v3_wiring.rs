//! v3 wiring e2e (E7): the harness-flagged parameter-cap breach is a
//! terminal Score(0) with no measured score / no LLM spend, and an attached
//! [`EvalStore`] leaves v1/v2 metrics runs bit-identical (composite skipped).

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use chain::{AxonInfo, ChainError, Metagraph, WeightsTlockPayload};
use chain::{ChainClient, FakeChain, FakeChainConfig};
use challenge_agentic::SimAgent;
use crypto::KEY_LEN;
use prism_challenge::{
    example_valid_request, submission_id, EvalStore, FinalScore, GatewayClient,
    GatewayClientConfig, MemoryEvalStore, MemoryPrismStore, Orchestrator, OrchestratorConfig,
    PrismStore, Stage, SubmissionState,
};
use prism_lium::{
    EvalJobBackend, Instance, InstanceSpec, LiumError, Offer, RemoteExecResult, SimLiumBackend,
};
use prism_review::SimReviewer;

fn fake_chain() -> FakeChain {
    let cfg = FakeChainConfig {
        owner_hotkey: vec![0xA0; 32],
        ..Default::default()
    };
    FakeChain::new(cfg)
}

/// `ChainClient` behind a `Mutex` (production uses the `Sync` live client).
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

/// Sim lifecycle with an `exec_eval` that mirrors the harness terminal
/// cap-exceeded payload (`CAP_EXCEEDED` + minimal `METRICS_JSON` v2).
struct CapBackend(SimLiumBackend);

#[async_trait::async_trait]
impl EvalJobBackend for CapBackend {
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        self.0.list_offers(max_price_per_hour).await
    }

    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError> {
        self.0.provision(spec).await
    }

    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError> {
        self.0.terminate(instance_id).await
    }

    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError> {
        self.0.verify_terminated(instance_id).await
    }

    async fn exec_eval(
        &self,
        _instance_id: &str,
        _architecture_py: &str,
        _training_py: &str,
        _tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        Ok(RemoteExecResult {
            bpb: 0.0,
            tokens_seen: 0,
            wall_clock_seconds: 1.0,
            gpu_type: Some("SIM".into()),
            notes: "sim cap-exceeded terminal payload".into(),
            n_params: Some(400_000_000),
            val_rows: None,
            telemetry: None,
            metrics_version: Some(2),
            tokens_seen_source: Some("train_stream".into()),
            probe_curve: None,
            pod_manifest: None,
            netns: Some(true),
            harness_files_sha256: None,
            eval_tier: None,
            extra: std::collections::BTreeMap::from([(
                "cap_exceeded".to_owned(),
                serde_json::json!(true),
            )]),
        })
    }
}

fn mk_orchestrator(
    store: &Arc<MemoryPrismStore>,
    chain: &Arc<LockedFake>,
    backend: Arc<dyn EvalJobBackend>,
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

async fn insert_queued(store: &Arc<MemoryPrismStore>, id: &str) {
    let req = example_valid_request();
    store
        .insert_queued(&SubmissionState {
            id: id.into(),
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
}

#[tokio::test]
async fn cap_exceeded_is_terminal_score_zero() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let backend: Arc<dyn EvalJobBackend> = Arc::new(CapBackend(SimLiumBackend::new()));
    let orch = mk_orchestrator(&store, &chain, backend);

    let id = submission_id(&example_valid_request());
    insert_queued(&store, &id).await;

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Rejected, "cap breach rejects terminally");
    assert_eq!(row.final_score, Some(FinalScore::Score(0)));
    // Cap is measured at build; no bpb and no post-measure agentic re-spend.
    // Pre-pod screens may already be checkpointed for resume durability.
    assert!(row.bpb.is_none(), "no measured bpb on a cap breach");
    assert_eq!(row.retry_count, 0, "miner-attributable: no auto-retry");
    let detail = row.error_detail.unwrap_or_default();
    assert!(
        detail.contains("parameter cap"),
        "error_detail explains the gate: {detail}"
    );
}

#[tokio::test]
async fn attached_eval_store_skips_composite_for_v1_metrics() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
    let eval_store: Arc<dyn EvalStore> = Arc::new(MemoryEvalStore::new());
    let orch =
        mk_orchestrator(&store, &chain, backend).with_eval_store(Some(Arc::clone(&eval_store)));

    let id = submission_id(&example_valid_request());
    insert_queued(&store, &id).await;

    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert!(
        matches!(row.status, Stage::Terminated | Stage::Failed),
        "status={:?}",
        row.status
    );
    assert!(row.final_score.is_some(), "v2 path still scores");
    // Sim metrics are v1-shaped (no battery): the composite path is a no-op
    // and persists no eval run — legacy behavior stays bit-identical.
    assert!(
        eval_store.eval_run(&id).await.unwrap().is_none(),
        "no eval run without a v2 battery payload"
    );
}
