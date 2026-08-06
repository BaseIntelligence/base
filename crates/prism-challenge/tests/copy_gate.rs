//! Prism pre-LLM copy gate: byte/AST copy of a strictly-earlier
//! `architecture.py` (`created_at` ordered) → terminal `rejected` + Score(0),
//! no pod, no LLM. Simultaneous/original architectures pass through.

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
    OrchestratorConfig, PrismStore, Stage, SubmissionState,
};
use prism_lium::{EvalJobBackend, SimLiumBackend};
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

const ARCH_A: &str = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(16, 16)\n";
const TRAIN_OK: &str = concat!(
    "import prism_telemetry\n",
    "def train(model, ctx):\n",
    "    prism_telemetry.report(loss=1.0, step=1)\n",
    "    prism_telemetry.finish_evaluation()\n",
    "    return {'loss': 1.0}\n",
);

fn row(
    id: &str,
    hotkey: &str,
    arch: &str,
    train: &str,
    status: Stage,
    created_ms: u64,
) -> SubmissionState {
    SubmissionState {
        id: id.into(),
        miner_hotkey: hotkey.into(),
        epoch: 7,
        netuid: 541,
        status,
        architecture_py: arch.into(),
        training_py: train.into(),
        label: None,
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
        created_at_ms: created_ms,
        updated_at_ms: created_ms,
    }
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
    Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(store) as Arc<dyn PrismStore>,
        Arc::new(SimLiumBackend::new()) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(SimAgent::new()),
        &gateway,
        Arc::clone(chain),
        [9u8; KEY_LEN],
    )
}

#[tokio::test]
async fn byte_copy_of_earlier_arch_is_rejected_without_pod() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    // Miner A's submission already terminated (prior art).
    store
        .insert_queued(&row(
            "a-prior-subm-0001",
            &"aa".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Terminated,
            1_000,
        ))
        .await
        .unwrap();
    // Miner B byte-copies A's architecture later.
    store
        .insert_queued(&row(
            "b-copy-subm-00001",
            &"bb".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Queued,
            2_000,
        ))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let b = store
        .get("b-copy-subm-00001")
        .await
        .unwrap()
        .expect("row b");
    assert_eq!(b.status, Stage::Rejected, "status={:?}", b.status);
    assert_eq!(b.final_score, Some(FinalScore::Score(0)));
    assert!(b.pod_id.is_none(), "copy gate must skip pod provisioning");
    assert!(b.review.is_none(), "copy gate must skip the LLM review");
    let sim = b.similarity.expect("similarity verdict recorded");
    assert!(matches!(sim.kind, prism_review::SimilarityKind::Copied));
    assert!(sim
        .closest
        .as_deref()
        .is_some_and(|c| c.contains("a-prior-subm-0001")));
    assert!(b.status.is_terminal());
}

#[tokio::test]
async fn ast_copy_with_renames_is_rejected() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    // Cosmetic whitespace shuffle: same AST, not byte-identical.
    let renamed = ARCH_A.replace("torch.nn.Linear(16, 16)", "torch.nn.Linear(16,16)");
    store
        .insert_queued(&row(
            "a-prior-subm-0001",
            &"aa".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Terminated,
            1_000,
        ))
        .await
        .unwrap();
    store
        .insert_queued(&row(
            "b-ast-subm-000001",
            &"bb".repeat(32),
            &renamed,
            TRAIN_OK,
            Stage::Queued,
            2_000,
        ))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let b = store
        .get("b-ast-subm-000001")
        .await
        .unwrap()
        .expect("row b");
    assert_eq!(b.status, Stage::Rejected, "status={:?}", b.status);
    assert_eq!(b.final_score, Some(FinalScore::Score(0)));
}

#[tokio::test]
async fn same_arch_same_timestamp_passes_the_gate() {
    // created_at ties cannot be ordered → the copy gate must NOT reject;
    // the row proceeds to the normal pipeline (sim: terminates with a score).
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    store
        .insert_queued(&row(
            "a-prior-subm-0001",
            &"aa".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Terminated,
            1_000,
        ))
        .await
        .unwrap();
    store
        .insert_queued(&row(
            "b-tie-subm-000001",
            &"bb".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Queued,
            1_000,
        ))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let b = store
        .get("b-tie-subm-000001")
        .await
        .unwrap()
        .expect("row b");
    assert_ne!(b.status, Stage::Rejected, "tie must not hard-reject");
    // The LLM similarity path (SimReviewer, arch-only) still judges the copy.
    assert_eq!(b.final_score, Some(FinalScore::Score(0)));
    assert!(matches!(b.status, Stage::Terminated | Stage::Failed));
}

#[tokio::test]
async fn same_training_on_different_arch_is_not_similarity_copy() {
    // training.py is exempt from similarity: miner B reuses A's exact training
    // script on a genuinely different architecture → not Copied, real score.
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    let arch_b = "import torch\ndef build_model(ctx):\n    return torch.nn.Sequential(\n        torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32))\n";
    store
        .insert_queued(&row(
            "a-prior-subm-0001",
            &"aa".repeat(32),
            ARCH_A,
            TRAIN_OK,
            Stage::Terminated,
            1_000,
        ))
        .await
        .unwrap();
    store
        .insert_queued(&row(
            "b-orig-subm-00001",
            &"bb".repeat(32),
            arch_b,
            TRAIN_OK,
            Stage::Queued,
            2_000,
        ))
        .await
        .unwrap();

    assert!(orch.cycle_once().await.unwrap());
    let b = store
        .get("b-orig-subm-00001")
        .await
        .unwrap()
        .expect("row b");
    assert_eq!(b.status, Stage::Terminated, "status={:?}", b.status);
    let sim = b.similarity.expect("similarity verdict");
    assert!(matches!(sim.kind, prism_review::SimilarityKind::Original));
    assert!(
        matches!(b.final_score, Some(FinalScore::Score(v)) if v > 0),
        "original arch with reused training script must score: {:?}",
        b.final_score
    );
}
