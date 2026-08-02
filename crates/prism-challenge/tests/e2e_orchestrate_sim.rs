//! Sim orchestrator e2e: submit → sim Lium → sim review → similarity → score
//! → exact-E leaf set (dry-run gateway) with no network.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use chain::{AxonInfo, ChainError, Metagraph, WeightsTlockPayload};
use chain::{ChainClient, FakeChain, FakeChainConfig};
use crypto::KEY_LEN;
use prism_challenge::FinalScore;
use prism_challenge::{
    example_valid_request, submission_id, GatewayClient, GatewayClientConfig, MemoryPrismStore,
    Orchestrator, OrchestratorConfig, PrismStore, Stage, StatePatch, SubmissionState,
};
use prism_lium::{EvalJobBackend, SimLiumBackend};
use prism_review::SimReviewer;
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
            max_retries: 0,
        })
        .unwrap(),
    );
    let backend: Arc<dyn EvalJobBackend> = Arc::new(SimLiumBackend::new());
    Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(store) as Arc<dyn PrismStore>,
        backend,
        Arc::new(SimReviewer::new()),
        gateway,
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
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: req.architecture_py.clone(),
            training_py: req.training_py.clone(),
            label: req.label,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            bpb: None,
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
}

#[tokio::test]
async fn emit_and_submit_covers_expected_set() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    // Seed one finished score for hotkey 0xAB*32.
    let hk = [0xABu8; 32];
    store.apply("seed", &StatePatch::default(), None).await.ok();
    store
        .insert_queued(&SubmissionState {
            id: "seed".into(),
            miner_hotkey: hex::encode(hk),
            epoch: 7,
            netuid: 541,
            status: Stage::Terminated,
            architecture_py: "a".into(),
            training_py: "t".into(),
            label: None,
            pod_id: None,
            pod_provider: None,
            receipt: None,
            bpb: Some(2.0),
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

    let expected = agent_challenge::ExpectedSet {
        participants: vec![
            agent_challenge::ExpectedParticipant { hotkey: hk, uid: 0 },
            agent_challenge::ExpectedParticipant {
                hotkey: [0xCDu8; 32],
                uid: 1,
            },
        ],
        block_hash: [0x77u8; 32],
    };
    let n = orch
        .emit_and_submit_at(7, &expected)
        .await
        .expect("emit+submit");
    assert_eq!(n, expected.participants.len(), "exact-E coverage");
}
