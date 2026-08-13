//! Review-stage failures must never burn GPU:
//! - Pre-pod agentic/LLM infra fails closed **before** Lium rent.
//! - A metrics-aware agentic failure after a completed measurement must
//!   resume without re-provisioning (E12).

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chain::{AxonInfo, ChainError, Metagraph, WeightsTlockPayload};
use chain::{ChainClient, FakeChain, FakeChainConfig};
use challenge_agentic::{AgenticBackend, AgenticError, AgenticVerdict, ReviewRequest, VerdictKind};
use crypto::KEY_LEN;
use prism_challenge::{
    FinalScore, GatewayClient, GatewayClientConfig, MemoryPrismStore, Orchestrator,
    OrchestratorConfig, PrismStore, Stage, SubmissionState,
};
use prism_lium::{
    EvalJobBackend, Instance, InstanceSpec, LiumError, Offer, RemoteExecResult, SimLiumBackend,
};
use prism_review::SimReviewer;

fn fake_chain() -> FakeChain {
    FakeChain::new(FakeChainConfig {
        owner_hotkey: vec![0xA0; 32],
        ..Default::default()
    })
}

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

/// Sim backend counting measure-phase invocations (the retrain detector).
struct CountingBackend {
    inner: SimLiumBackend,
    provisions: AtomicUsize,
    exec_calls: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        Self {
            inner: SimLiumBackend::new(),
            provisions: AtomicUsize::new(0),
            exec_calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl EvalJobBackend for CountingBackend {
    async fn list_offers(&self, max_price_per_hour: Option<f64>) -> Result<Vec<Offer>, LiumError> {
        self.inner.list_offers(max_price_per_hour).await
    }

    async fn provision(&self, spec: &InstanceSpec) -> Result<Instance, LiumError> {
        self.provisions.fetch_add(1, Ordering::SeqCst);
        self.inner.provision(spec).await
    }

    async fn terminate(&self, instance_id: &str) -> Result<(), LiumError> {
        self.inner.terminate(instance_id).await
    }

    async fn verify_terminated(&self, instance_id: &str) -> Result<bool, LiumError> {
        self.inner.verify_terminated(instance_id).await
    }

    async fn exec_eval(
        &self,
        instance_id: &str,
        architecture_py: &str,
        training_py: &str,
        tree_blob: Option<&[u8]>,
    ) -> Result<RemoteExecResult, LiumError> {
        self.exec_calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .exec_eval(instance_id, architecture_py, training_py, tree_blob)
            .await
    }
}

/// Always dies the way a live `OpenRouter` budget exhaustion does.
struct BudgetDeadAgent;

#[async_trait]
impl AgenticBackend for BudgetDeadAgent {
    async fn review(&self, _req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        Err(AgenticError::NoVerdict(
            "token budget exhausted (39869)".into(),
        ))
    }
}

/// Pre-pod (no metrics) succeeds; metrics-aware pass fails with infra error.
struct MetricsPassDeadAgent;

#[async_trait]
impl AgenticBackend for MetricsPassDeadAgent {
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        if req.metrics_relpath.is_none() {
            return Ok(AgenticVerdict {
                verdict: VerdictKind::Clean,
                cheat_codes: vec![],
                nearest_id: None,
                similarity_bps: 0,
                rationale: "pre-pod structural clean".into(),
            });
        }
        Err(AgenticError::NoVerdict(
            "token budget exhausted (39869)".into(),
        ))
    }
}

fn training_with_hooks() -> &'static str {
    concat!(
        "import prism_telemetry\n",
        "def train(model, ctx):\n",
        "    prism_telemetry.report(loss=1.0, step=1)\n",
        "    prism_telemetry.finish_evaluation()\n",
        "    return {'loss': 1.0}\n",
    )
}

fn architecture_py() -> &'static str {
    "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n"
}

#[tokio::test]
async fn agentic_infra_pre_pod_never_provisions() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "dry-run".into(),
            max_attempts: 1,
            backoff: std::time::Duration::from_millis(1),
        })
        .unwrap(),
    );
    let mut sk = [7u8; KEY_LEN];
    sk[0] = 0x42;
    let backend = Arc::new(CountingBackend::new());
    let orch = Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            auto_retry_max: 1,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(&store) as Arc<dyn PrismStore>,
        Arc::clone(&backend) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(BudgetDeadAgent),
        &gateway,
        chain,
        sk,
    );

    let id = "agentic-pre-pod-no-rent".to_owned();
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: "11".repeat(32),
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: architecture_py().into(),
            training_py: training_with_hooks().into(),
            tree_blob: None,
            label: Some("agentic-pre-pod".into()),
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
    assert_eq!(row.status, Stage::Queued, "auto-retried: {row:?}");
    assert_eq!(row.retry_count, 1);
    assert!(row.receipt.is_none() && row.metrics_json.is_none());
    assert_eq!(
        backend.provisions.load(Ordering::SeqCst),
        0,
        "pre-pod agentic infra must not rent a pod"
    );

    assert!(orch.cycle_once().await.unwrap());
    assert_eq!(backend.provisions.load(Ordering::SeqCst), 0);
    assert_eq!(backend.exec_calls.load(Ordering::SeqCst), 0);
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Failed, "{row:?}");
    assert_eq!(
        row.final_score,
        Some(FinalScore::NoScore(6)),
        "review-inconclusive terminal = NoScore(ChallengeInternal), got {:?}",
        row.final_score
    );
}

#[tokio::test]
async fn agentic_infra_retry_resumes_without_remeasure() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "dry-run".into(),
            max_attempts: 1,
            backoff: std::time::Duration::from_millis(1),
        })
        .unwrap(),
    );
    let mut sk = [7u8; KEY_LEN];
    sk[0] = 0x42;
    let backend = Arc::new(CountingBackend::new());
    let orch = Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            auto_retry_max: 1,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(&store) as Arc<dyn PrismStore>,
        Arc::clone(&backend) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(MetricsPassDeadAgent),
        &gateway,
        chain,
        sk,
    );

    let id = "agentic-retry-resume".to_owned();
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: "11".repeat(32),
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: architecture_py().into(),
            training_py: training_with_hooks().into(),
            tree_blob: None,
            label: Some("agentic-retry".into()),
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

    // Cycle 1: pre-pod agentic clean → pod once → metrics agentic fails → retry.
    assert!(orch.cycle_once().await.unwrap());
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Queued, "auto-retried: {row:?}");
    assert_eq!(row.retry_count, 1);
    assert!(
        row.receipt.is_some() && row.metrics_json.is_some() && row.bpb.is_some(),
        "measurement must survive the post-run retry reset: {row:?}"
    );

    // Cycle 2: resumes measurement — no fresh pod — then fail-closed.
    assert!(orch.cycle_once().await.unwrap());
    assert_eq!(
        backend.provisions.load(Ordering::SeqCst),
        1,
        "a metrics-review retry must not provision a second pod"
    );
    assert_eq!(
        backend.exec_calls.load(Ordering::SeqCst),
        1,
        "a metrics-review retry must not re-measure"
    );
    let row = store.get(&id).await.unwrap().expect("row");
    assert_eq!(row.status, Stage::Failed, "{row:?}");
    assert_eq!(
        row.final_score,
        Some(FinalScore::NoScore(6)),
        "review-inconclusive terminal = NoScore(ChallengeInternal), got {:?}",
        row.final_score
    );
    let detail = row.error_detail.unwrap_or_default();
    assert!(detail.contains("token budget exhausted"), "{detail}");
}
