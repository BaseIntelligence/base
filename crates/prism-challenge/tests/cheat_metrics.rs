//! Malicious Prism fixtures: hardcoded `METRICS_JSON` → agentic cheat → Score(0).

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

#[tokio::test]
async fn hardcoded_metrics_json_scores_zero() {
    let store = Arc::new(MemoryPrismStore::new());
    let chain = Arc::new(LockedFake(Mutex::new(fake_chain())));
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "dry-run".into(),
            max_retries: 0,
        })
        .unwrap(),
    );
    let mut sk = [7u8; KEY_LEN];
    sk[0] = 0x42;
    let orch = Arc::new(Orchestrator::new(
        OrchestratorConfig {
            netuid: 541,
            claim_poll: std::time::Duration::from_millis(10),
            ..Default::default()
        },
        Arc::clone(&store) as Arc<dyn PrismStore>,
        Arc::new(SimLiumBackend::new()) as Arc<dyn EvalJobBackend>,
        Arc::new(SimReviewer::new()),
        Arc::new(SimAgent::new()),
        gateway,
        Arc::clone(&chain),
        sk,
    ));

    // Contract-valid sources that short-circuit by printing METRICS_JSON.
    let architecture_py = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
    let training_py = r#"
def train(model, ctx):
    print('METRICS_JSON={"bpb":0.01,"tokens_seen":0,"wall_clock_seconds":0.0}')
    return {"loss": 0.0}
"#;
    let id = "cheat-metrics-fixture".to_owned();
    store
        .insert_queued(&SubmissionState {
            id: id.clone(),
            miner_hotkey: "11".repeat(32),
            epoch: 7,
            netuid: 541,
            status: Stage::Queued,
            architecture_py: architecture_py.into(),
            training_py: training_py.into(),
            label: Some("cheat-metrics".into()),
            pod_id: None,
            pod_provider: None,
            receipt: None,
            metrics_json: None,
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
    assert_eq!(
        row.final_score,
        Some(FinalScore::Score(0)),
        "hardcoded METRICS_JSON must Score(0), got {:?}",
        row.final_score
    );
}

#[tokio::test]
async fn impossible_bpb_metrics_json_is_cheat() {
    use challenge_agentic::{AgenticBackend, CheatCode, ReviewRequest, VerdictKind};
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let arch = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
    let train = "def train(model, ctx):\n    return {\"loss\": 1.0}\n";
    std::fs::write(dir.path().join("architecture.py"), arch).unwrap();
    std::fs::write(dir.path().join("training.py"), train).unwrap();
    std::fs::write(
        dir.path().join("metrics.json"),
        r#"{"bpb":0.01,"tokens_seen":0,"wall_clock_seconds":0.0,"notes":"forged"}"#,
    )
    .unwrap();
    let req = ReviewRequest {
        workdir: dir.path().to_path_buf(),
        primary_relpaths: vec!["architecture.py".into(), "training.py".into()],
        corpus: vec![],
        metrics_relpath: Some("metrics.json".into()),
        pages_relpath: None,
        sanitize_report_relpath: None,
        domain_rules: "prism".into(),
    };
    let v = SimAgent::new().review(&req).await.unwrap();
    assert_eq!(v.verdict, VerdictKind::Cheat);
    assert!(v.cheat_codes.contains(&CheatCode::InconsistentMetrics));
}
