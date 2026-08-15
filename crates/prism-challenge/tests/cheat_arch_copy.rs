//! Prism: byte/AST copy of architecture+training corpus → agentic cheat → Score(0).

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;
use std::sync::Mutex;

use chain::{AxonInfo, ChainError, Metagraph, WeightsTlockPayload};
use chain::{ChainClient, FakeChain, FakeChainConfig};
use challenge_agentic::SimAgent;
use crypto::KEY_LEN;
use prism_challenge::{
    FinalScore, GatewayClient, GatewayClientConfig, MemoryPrismStore, Orchestrator,
    OrchestratorConfig, PrismStore, ScoringMode, Stage, SubmissionState,
};
use prism_lium::{EvalJobBackend, SimLiumBackend};
use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};
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

#[tokio::test]
async fn baseline_arch_train_copy_scores_zero() {
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
    let orch = Arc::new(Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            scoring_mode: ScoringMode::Shadow,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(&store) as Arc<dyn PrismStore>,
        Arc::new(SimLiumBackend::new()) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(SimAgent::new()),
        &gateway,
        Arc::clone(&chain),
        sk,
    ));

    let id = "cheat-arch-copy-fixture".to_owned();
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: "22".repeat(32),
            miner_coldkey: None,
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            // Exact corpus baseline → SimAgent byte-identical cheat.
            architecture_py: BASELINE_ARCHITECTURE_PY.into(),
            training_py: BASELINE_TRAINING_PY.into(),
            tree_blob: None,
            label: Some("cheat-arch-copy".into()),
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
    assert_eq!(
        row.status,
        Stage::Rejected,
        "baseline arch copy must fail pre-pod similarity, got {:?}",
        row.status
    );
    assert!(row.pod_id.is_none(), "arch copy must not rent a pod");
    assert_eq!(
        row.final_score,
        Some(FinalScore::Score(0)),
        "arch/train corpus copy must Score(0), got {:?}",
        row.final_score
    );
}
