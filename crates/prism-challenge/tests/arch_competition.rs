//! Architecture registry + training-only competition:
//! intake materializes the registry source and gates per `(hotkey, arch_id)`;
//! training-only rows skip the copy gate / similarity; arch submissions
//! publish on first score; competition credits owner + challenger.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;
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
use prism_store::{ArchitectureRecord, EpochScoreRow, PublishArchOutcome};

const ARCH_SRC: &str = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(16, 16)\n";
const TRAIN_OK: &str = concat!(
    "import prism_telemetry\n",
    "def train(model, ctx):\n",
    "    prism_telemetry.report(loss=1.0, step=1)\n",
    "    prism_telemetry.finish_evaluation()\n",
    "    return {'loss': 1.0}\n",
);
const TRAIN_OK_V2: &str = concat!(
    "import prism_telemetry\n",
    "def train(model, ctx):\n",
    "    prism_telemetry.report(loss=0.9, step=1)\n",
    "    prism_telemetry.report(loss=0.8, step=2)\n",
    "    prism_telemetry.finish_evaluation()\n",
    "    return {'loss': 0.8}\n",
);

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

fn row(
    id: &str,
    hotkey: &str,
    arch: &str,
    train: &str,
    arch_id: Option<&str>,
    created_ms: u64,
) -> SubmissionState {
    SubmissionState {
        id: id.into(),
        miner_hotkey: hotkey.into(),
        miner_coldkey: None,
        epoch: 7,
        netuid: 541,
        status: Stage::Queued,
        architecture_py: arch.into(),
        training_py: train.into(),
        label: None,
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: None,
        arch_id: arch_id.map(str::to_owned),
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
        Arc::new(SimLiumBackend::new()) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(SimAgent::new()),
        &gateway,
        Arc::clone(chain),
        [3u8; KEY_LEN],
    )
}

#[tokio::test]
async fn arch_submission_publishes_on_first_score() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));

    store
        .insert_queued(&row(
            "arch-subm-000001",
            &"aa".repeat(32),
            ARCH_SRC,
            TRAIN_OK,
            None,
            1_000,
        ))
        .await
        .unwrap();
    assert!(orch.cycle_once().await.unwrap());

    let done = store.get("arch-subm-000001").await.unwrap().expect("row");
    assert_eq!(done.status, Stage::Terminated, "status={:?}", done.status);
    assert!(
        matches!(done.final_score, Some(FinalScore::Score(v)) if v > 0),
        "real score: {:?}",
        done.final_score
    );
    // Registry publication + back-link + best bpb.
    let expected_arch = prism_pipeline::arch_id_for(ARCH_SRC);
    assert_eq!(done.arch_id.as_deref(), Some(expected_arch.as_str()));
    let rec = store
        .get_arch(&expected_arch)
        .await
        .unwrap()
        .expect("arch registered");
    assert_eq!(rec.owner_hotkey, "aa".repeat(32));
    assert_eq!(rec.architecture_py, ARCH_SRC);
    assert_eq!(rec.best_bpb, done.bpb);
    // Idempotent: re-publish same digest is a duplicate, not an error.
    let again = store.publish_arch(&rec).await.unwrap();
    assert!(matches!(again, PublishArchOutcome::Duplicate(_)));
}

#[tokio::test]
async fn training_only_skips_copy_gate_and_similarity() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let orch = Arc::new(mk_orchestrator(&store, &chain));
    let arch = prism_pipeline::arch_id_for(ARCH_SRC);

    // Owner's arch submission terminated earlier (would trip the copy gate
    // for any later byte-identical architecture).
    let mut owner = row(
        "owner-subm-00001",
        &"aa".repeat(32),
        ARCH_SRC,
        TRAIN_OK,
        None,
        1_000,
    );
    owner.status = Stage::Terminated;
    owner.bpb = Some(2.0);
    owner.final_score = Some(FinalScore::Score(400_000));
    owner.arch_id = Some(arch.clone());
    store.insert_queued(&owner).await.unwrap();
    store
        .publish_arch(&ArchitectureRecord {
            arch_id: arch.clone(),
            owner_hotkey: "aa".repeat(32),
            arch_digest: prism_pipeline::arch_digest(ARCH_SRC),
            architecture_py: ARCH_SRC.into(),
            source_submission: "owner-subm-00001".into(),
            best_bpb: Some(2.0),
            created_at_ms: 1_000,
        })
        .await
        .unwrap();

    // Challenger: training-only on the published arch (same arch bytes!).
    store
        .insert_queued(&row(
            "challenger-0001",
            &"bb".repeat(32),
            ARCH_SRC,
            TRAIN_OK_V2,
            Some(&arch),
            2_000,
        ))
        .await
        .unwrap();
    assert!(orch.cycle_once().await.unwrap());

    let done = store.get("challenger-0001").await.unwrap().expect("row");
    assert_eq!(
        done.status,
        Stage::Terminated,
        "registry arch must not trip the copy gate: {:?}",
        done.status
    );
    let sim = done.similarity.expect("similarity recorded");
    assert!(matches!(sim.kind, prism_review::SimilarityKind::Original));
    assert!(
        matches!(done.final_score, Some(FinalScore::Score(v)) if v > 0),
        "challenger scores on its own result: {:?}",
        done.final_score
    );
}

#[tokio::test]
async fn competition_credits_owner_and_challenger() {
    let store = Arc::new(MemoryPrismStore::new());
    let arch = "arch_0123456789abcdef";
    store
        .publish_arch(&ArchitectureRecord {
            arch_id: arch.into(),
            owner_hotkey: "aa".repeat(32),
            arch_digest: "00".repeat(32),
            architecture_py: ARCH_SRC.into(),
            source_submission: "src".into(),
            best_bpb: Some(2.0),
            created_at_ms: 1_000,
        })
        .await
        .unwrap();

    let mut owner_row = row(
        "owner-subm-00001",
        &"aa".repeat(32),
        ARCH_SRC,
        TRAIN_OK,
        Some(arch),
        1_000,
    );
    owner_row.status = Stage::Terminated;
    owner_row.final_score = Some(FinalScore::Score(400_000));
    owner_row.bpb = Some(2.0);
    store.insert_queued(&owner_row).await.unwrap();
    let mut chall_row = row(
        "challenger-0001",
        &"bb".repeat(32),
        ARCH_SRC,
        TRAIN_OK_V2,
        Some(arch),
        2_000,
    );
    chall_row.status = Stage::Terminated;
    chall_row.final_score = Some(FinalScore::Score(900_000));
    chall_row.bpb = Some(1.5);
    store.insert_queued(&chall_row).await.unwrap();

    let rows = store.assign_emit_batch(541, 7).await.unwrap();
    assert_eq!(rows.len(), 2, "one entry per submission (not collapsed)");
    let owners: BTreeMap<String, String> = store.arch_owners().await.unwrap().into_iter().collect();
    let scores = prism_registry::competition_scores(&rows, &owners);
    // TEMP: owner-arch credit disabled — each hotkey keeps only own score.
    // Re-enable OWNER_ARCH_CREDIT_ENABLED to restore owner=900k / chall=900k.
    assert!(
        !prism_registry::OWNER_ARCH_CREDIT_ENABLED,
        "update expectations when restoring owner-arch credit"
    );
    assert_eq!(
        scores.get(&"aa".repeat(32)),
        Some(&FinalScore::Score(400_000))
    );
    assert_eq!(
        scores.get(&"bb".repeat(32)),
        Some(&FinalScore::Score(900_000))
    );
}

#[tokio::test]
async fn topmodel_hooks_graceful_without_publisher() {
    let store: Arc<dyn PrismStore> = Arc::new(MemoryPrismStore::new());
    let mut r = row(
        "arch-subm-000001",
        &"aa".repeat(32),
        ARCH_SRC,
        TRAIN_OK,
        None,
        1_000,
    );
    r.status = Stage::Terminated;
    r.bpb = Some(1.25);
    r.final_score = Some(FinalScore::Score(800_000));
    r.metrics_json = Some(serde_json::json!({"n_params": 12_000_000}));
    store.insert_queued(&r).await.unwrap();

    prism_registry::post_score_hooks(&store, None, &r).await;

    // Arch published even without a GitHub publisher; nothing journaled.
    assert!(store
        .get_arch(&prism_pipeline::arch_id_for(ARCH_SRC))
        .await
        .unwrap()
        .is_some());
    assert_eq!(store.last_publication_bpb().await.unwrap(), None);
}

#[tokio::test]
async fn emit_batch_feeds_competition_via_store() {
    // Smoke: EpochScoreRow shape carries arch_id through the emission outbox.
    let store = Arc::new(MemoryPrismStore::new());
    let mut r = row(
        "challenger-0001",
        &"bb".repeat(32),
        ARCH_SRC,
        TRAIN_OK_V2,
        Some("arch_x"),
        1_000,
    );
    r.status = Stage::Terminated;
    r.final_score = Some(FinalScore::Score(123));
    store.insert_queued(&r).await.unwrap();
    let rows: Vec<EpochScoreRow> = store.assign_emit_batch(541, 7).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].arch_id.as_deref(), Some("arch_x"));
    // Sticky: the assignment is re-readable and never re-assigned.
    assert_eq!(store.emit_batch(541, 7).await.unwrap().len(), 1);
    assert!(store.assign_emit_batch(541, 8).await.unwrap().is_empty());
}
