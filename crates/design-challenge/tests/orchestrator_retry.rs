//! Orchestrator auto-retry wiring: infra-class failures requeue up to
//! `auto_retry_max` then finalize `failed` + gating `blocked`; miner-class
//! failures are terminal on the first attempt.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::sync::Arc;

use design_challenge::{GatewayClient, GatewayClientConfig, Orchestrator, OrchestratorConfig};
use design_harness::HarnessBundle;
use design_sandbox::{SandboxBackend, SandboxError, SandboxOutput, SandboxSession};
use design_store::{DesignStore, FinalScore, HarnessRow, MemoryDesignStore, RunStage, RunState};
use submission_gating::{GatingState, GatingStore, MemoryGatingStore};

fn hk() -> String {
    "ab".repeat(32)
}

async fn seed(store: &MemoryDesignStore, hid: &str) {
    store
        .insert_harness(&HarnessRow {
            id: hid.to_owned(),
            miner_hotkey: hk(),
            agent_py: "def run(task, llm, out):\n    pass\n".into(),
            pyproject_toml: "[project]\nname='x'\nversion='0'\n".into(),
            extra_files: BTreeMap::new(),
            active: true,
            eliminated_until_round: 0,
            created_at_ms: 1,
        })
        .await
        .unwrap();
    store
        .insert_run(&RunState {
            id: "run-1".into(),
            round_id: 1,
            harness_id: hid.to_owned(),
            prompt_id: "p01".into(),
            status: RunStage::Queued,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            final_score: None,
            retry_count: 0,
            created_at_ms: 1,
            updated_at_ms: 1,
        })
        .await
        .unwrap();
}

#[derive(Debug)]
struct PhaseSandbox {
    fail_install: bool,
}

impl SandboxBackend for PhaseSandbox {
    fn install(
        &self,
        _bundle: &HarnessBundle,
        _round_id: u64,
        run_id: &str,
        _prompt: &str,
        _llm_proxy: &str,
    ) -> Result<SandboxSession, SandboxError> {
        if self.fail_install {
            return Err(SandboxError::PhaseFailed {
                phase: "install",
                status: 1,
                logs: "pip boom".into(),
            });
        }
        Ok(SandboxSession {
            work_root: std::env::temp_dir().join(format!("design-test-{run_id}")),
            install_logs: "ok".into(),
        })
    }

    fn run_session(
        &self,
        session: SandboxSession,
        _round_id: u64,
        _run_id: &str,
        _prompt: &str,
        _llm_proxy: &str,
    ) -> Result<SandboxOutput, SandboxError> {
        let _ = session;
        Err(SandboxError::PhaseFailed {
            phase: "run",
            status: 1,
            logs: "agent boom".into(),
        })
    }
}

fn orchestrator(
    store: Arc<MemoryDesignStore>,
    gating: Arc<MemoryGatingStore>,
    sandbox: Arc<dyn SandboxBackend>,
) -> Arc<Orchestrator<chain::NotImplementedChain>> {
    let gateway = Arc::new(
        GatewayClient::new(GatewayClientConfig {
            base_url: "http://127.0.0.1:1".into(),
            ..GatewayClientConfig::default()
        })
        .unwrap(),
    );
    let orch = Orchestrator::new(
        OrchestratorConfig {
            auto_retry_max: 3,
            ..OrchestratorConfig::default()
        },
        store,
        sandbox,
        Arc::new(challenge_agentic::SimAgent::new()),
        gateway,
        Arc::new(chain::NotImplementedChain),
        [0u8; 32],
    )
    .with_gating(gating);
    Arc::new(orch)
}

#[tokio::test]
async fn install_failure_auto_retries_then_blocks() {
    let store = Arc::new(MemoryDesignStore::new());
    let gating = Arc::new(MemoryGatingStore::new());
    seed(&store, "h1").await;
    gating
        .mark_registered("design", &hk(), Some(3))
        .await
        .unwrap();
    let orch = orchestrator(
        store.clone(),
        gating.clone(),
        Arc::new(PhaseSandbox { fail_install: true }),
    );

    // Attempts 1–3: auto-requeued (initial + 3 retries within budget of 3).
    for attempt in 1..=3_u32 {
        assert!(orch.cycle_once().await.unwrap());
        let run = store.get_run("run-1").await.unwrap().unwrap();
        assert_eq!(run.status, RunStage::Queued, "attempt {attempt}");
        assert_eq!(run.retry_count, attempt);
        assert_eq!(
            gating
                .get("design", &hk())
                .await
                .unwrap()
                .unwrap()
                .attempt_count,
            attempt
        );
    }
    // Attempt 4 exhausts the budget → terminal failed + gating blocked.
    assert!(orch.cycle_once().await.unwrap());
    let run = store.get_run("run-1").await.unwrap().unwrap();
    assert_eq!(run.status, RunStage::Failed);
    assert_eq!(
        run.final_score,
        Some(FinalScore::NoScore(
            bundle::NoScoreReasonCode::ChallengeInternal as u8
        ))
    );
    let row = gating.get("design", &hk()).await.unwrap().unwrap();
    assert_eq!(row.state, GatingState::Blocked);
    assert_eq!(row.last_error_class.as_deref(), Some("install"));

    let events = store.run_events("run-1").await.unwrap();
    assert_eq!(events.iter().filter(|e| e.stage == "auto_retry").count(), 3);
}

#[tokio::test]
async fn run_phase_miner_failure_is_terminal_immediately() {
    let store = Arc::new(MemoryDesignStore::new());
    let gating = Arc::new(MemoryGatingStore::new());
    seed(&store, "h1").await;
    gating
        .mark_registered("design", &hk(), Some(3))
        .await
        .unwrap();
    let orch = orchestrator(
        store.clone(),
        gating.clone(),
        Arc::new(PhaseSandbox {
            fail_install: false,
        }),
    );

    assert!(orch.cycle_once().await.unwrap());
    let run = store.get_run("run-1").await.unwrap().unwrap();
    assert_eq!(run.status, RunStage::Failed);
    assert_eq!(run.final_score, Some(FinalScore::Score(0)));
    assert_eq!(run.retry_count, 0, "miner errors never auto-retry");
    let row = gating.get("design", &hk()).await.unwrap().unwrap();
    assert_eq!(row.state, GatingState::Blocked);
    assert_eq!(row.last_error_class.as_deref(), Some("miner"));
}
