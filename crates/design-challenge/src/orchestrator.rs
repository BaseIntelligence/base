//! Round + run orchestrator: sandbox, agentic review, admin award, leaf emit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chain::ChainClient;
use challenge_agentic::{
    copy_gate, AgenticBackend, AgenticError, AgenticVerdict, ReviewRequest, VerdictKind,
};
use challenge_common::{
    emit_signed_leaf_set, expected_set_at_chain, submit_signed_leaf_set, GatewayClient,
    PinnedBlockHash,
};
use crypto::KEY_LEN;
use design_challenge_task::{
    awaiting_admin_unscored_expired, clip_logs, design_emit_plan, design_leaf_emit_enabled,
    not_attempted, now_ms, now_secs, reject_awaiting_admin_run, round_id_at, round_secs,
    score_window, to_leaf, window_start, WindowScorePlan, MAX_LOG_CHARS, UNSCORED_EPOCH_LIMIT,
};
use design_http::{enqueue_active_harnesses_for_round, mark_awaiting_admin, AdminAwardHook};
use design_prompts::{prompt_set_digest, select_prompts_for_round};
use design_sandbox::{SandboxBackend, SandboxError};
use design_sanitize::sanitize_bundle;
use design_store::{
    DesignStore, FinalScore, RatingRow, RoundRow, RunStage, StageEvent, StorePatch,
};
use serde_json::json;
use submission_gating::{GatingState, GatingStore};
use tokio::time::sleep;
use tracing::{info, warn};

use crate::corpus;
use crate::screenshot::{capture_full_page_png, png_artifact_tuple};
use crate::CHALLENGE_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    Install,
    AstInfra,
    LlmInfra,
    Miner,
}

impl ErrorClass {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::AstInfra => "ast_infra",
            Self::LlmInfra => "llm_infra",
            Self::Miner => "miner",
        }
    }

    #[must_use]
    pub const fn retryable(self) -> bool {
        !matches!(self, Self::Miner)
    }
}

/// One failed run attempt with its error class.
#[derive(Debug)]
struct RunFailure {
    class: ErrorClass,
    msg: String,
}

impl RunFailure {
    fn new(class: ErrorClass, msg: impl Into<String>) -> Self {
        Self {
            class,
            msg: msg.into(),
        }
    }
}

fn classify_sandbox(e: &SandboxError) -> RunFailure {
    match e {
        SandboxError::PhaseFailed {
            phase,
            status,
            logs,
        } => {
            let class = if *phase == "install" {
                ErrorClass::Install
            } else {
                ErrorClass::Miner
            };
            RunFailure::new(
                class,
                format!("phase {phase}: exit={status}: {}", clip_logs(logs)),
            )
        }
        SandboxError::Docker(msg) => RunFailure::new(ErrorClass::Install, format!("docker: {msg}")),
        SandboxError::Io(e) => RunFailure::new(ErrorClass::AstInfra, format!("io: {e}")),
        SandboxError::MissingOutput(m) => {
            RunFailure::new(ErrorClass::Miner, format!("missing output: {m}"))
        }
    }
}

fn classify_agentic(e: &AgenticError) -> RunFailure {
    match e {
        AgenticError::Tool(m) => RunFailure::new(ErrorClass::AstInfra, format!("agentic: {m}")),
        other => RunFailure::new(ErrorClass::LlmInfra, format!("agentic: {other}")),
    }
}

const DESIGN_AGENTIC_RULES: &str = r"
Design challenge rules:
- Allowed: PyPI deps, external APIs/MCP over egress (network ≠ cheat), Mobbin/Dribbble, image gen, UI libs.
- Corpus `baseline` is the published reference — starting from it is allowed; copying another miner is not.
- Cheat: near-identical harness copy, HTML republish, scrape-clone, sanitize bypass/JS exfil, obfuscation to hide copy.
- suspicious and cheat → Score(0), not admin-eligible.
";

#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub netuid: u16,
    pub claim_poll: Duration,
    pub stuck_grace_secs: u64,
    pub llm_proxy: String,
    pub staging_root: PathBuf,
    pub stage_delay: Duration,
    pub auto_retry_max: u32,
    pub emit_poll: Duration,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            netuid: 1,
            claim_poll: Duration::from_millis(750),
            stuck_grace_secs: 3600,
            llm_proxy: "http://design-egress-proxy:8094".into(),
            staging_root: PathBuf::from("/var/lib/design/staging"),
            stage_delay: Duration::ZERO,
            auto_retry_max: 3,
            emit_poll: Duration::from_secs(15),
        }
    }
}

pub struct Orchestrator<C: ChainClient + Send + Sync> {
    cfg: OrchestratorConfig,
    store: Arc<dyn DesignStore>,
    sandbox: Arc<dyn SandboxBackend>,
    agentic: Arc<dyn AgenticBackend>,
    gateway: Arc<GatewayClient>,
    chain: Arc<C>,
    sk: [u8; KEY_LEN],
    gating: Option<Arc<dyn GatingStore>>,
    epoch_cache: Option<Arc<AtomicU64>>,
    emitted_epoch: AtomicU64,
}

impl<C: ChainClient + Send + Sync> std::fmt::Debug for Orchestrator<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("netuid", &self.cfg.netuid)
            .finish_non_exhaustive()
    }
}

impl<C: ChainClient + Send + Sync + 'static> Orchestrator<C> {
    #[must_use]
    pub fn new(
        cfg: OrchestratorConfig,
        store: Arc<dyn DesignStore>,
        sandbox: Arc<dyn SandboxBackend>,
        agentic: Arc<dyn AgenticBackend>,
        gateway: Arc<GatewayClient>,
        chain: Arc<C>,
        sk: [u8; KEY_LEN],
    ) -> Self {
        Self {
            cfg,
            store,
            sandbox,
            agentic,
            gateway,
            chain,
            sk,
            gating: None,
            epoch_cache: None,
            emitted_epoch: AtomicU64::new(0),
        }
    }

    #[must_use]
    pub fn with_gating(mut self, gating: Arc<dyn GatingStore>) -> Self {
        self.gating = Some(gating);
        self
    }

    #[must_use]
    pub fn with_epoch_cache(mut self, epoch: Arc<AtomicU64>) -> Self {
        self.epoch_cache = Some(epoch);
        self
    }

    fn current_epoch(&self) -> u64 {
        let epoch = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map(|s| chain::current_epoch_pre_run_coinbase(&s, s.current_block))
            .unwrap_or(0);
        if let Some(cache) = &self.epoch_cache {
            cache.store(epoch, Ordering::Relaxed);
        }
        epoch
    }

    async fn pause_stage(&self) {
        if !self.cfg.stage_delay.is_zero() {
            sleep(self.cfg.stage_delay).await;
        }
    }

    pub async fn run_worker(self: Arc<Self>) {
        loop {
            match self.cycle_once().await {
                Ok(true) => {}
                Ok(false) => sleep(self.cfg.claim_poll).await,
                Err(e) => {
                    warn!(error = %e, "design worker error");
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    pub async fn run_round_loop(self: Arc<Self>) {
        let mut last_closed = 0u64;
        loop {
            sleep(Duration::from_secs(30)).await;
            let rid = round_id_at(now_secs());
            // Close previous round when we enter a new one.
            if rid > 0 {
                let prev = rid - 1;
                if prev != last_closed {
                    if let Err(e) = self.close_round(prev).await {
                        warn!(error = %e, round = prev, "close_round failed");
                    } else {
                        last_closed = prev;
                    }
                }
            }
            // Open the round, then auto-enqueue every eligible active harness
            // with this round's shared prompt (Scheduled origin; idempotent).
            let _ = self.ensure_round(rid).await;
            if let Err(e) = enqueue_active_harnesses_for_round(
                self.store.as_ref(),
                rid,
                self.cfg.netuid,
                self.current_epoch(),
            )
            .await
            {
                warn!(error = %e, round = rid, "auto-enqueue failed");
            }
        }
    }

    pub async fn run_emitter(self: Arc<Self>)
    where
        C: Sync,
    {
        loop {
            if let Err(e) = self.emitter_tick().await {
                warn!(error = %e, "design emitter tick error");
            }
            sleep(self.cfg.emit_poll).await;
        }
    }

    pub async fn emitter_tick(&self) -> Result<bool, String>
    where
        C: Sync,
    {
        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        if !design_leaf_emit_enabled() {
            return Ok(false);
        }
        let Some(plan) = design_emit_plan(
            self.emitted_epoch.load(Ordering::Relaxed),
            state.subnet_epoch_index,
            state.blocks_since_last_step,
            u64::from(state.tempo.max(1)),
            state.last_epoch_block,
        ) else {
            return Ok(false);
        };
        self.emit_leaves_at(plan.epoch, plan.pin_block).await?;
        Ok(true)
    }

    async fn sweep_unscored_timeouts(&self) -> Result<(), String> {
        let current = self.current_epoch();
        if current == 0 {
            return Ok(());
        }
        let awaiting = self
            .store
            .list_runs(Some("awaiting_admin"), 500)
            .await
            .map_err(|e| e.to_string())?;
        let gating = self.gating.as_deref();
        for run in awaiting {
            if !awaiting_admin_unscored_expired(&run, current) {
                continue;
            }
            let start = run.awaiting_admin_epoch.or(run.attempt_epoch).unwrap_or(0);
            let reason = format!(
                "unscored_timeout: no admin score within {UNSCORED_EPOCH_LIMIT} epochs (since epoch {start})"
            );
            info!(run_id = %run.id, start_epoch = start, current_epoch = current, "auto-reject unscored");
            reject_awaiting_admin_run(
                self.store.as_ref(),
                gating,
                &run,
                &reason,
                "unscored_timeout",
            )
            .await
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub async fn run_sweeper(self: Arc<Self>) {
        loop {
            sleep(Duration::from_secs(60)).await;
            if let Err(e) = self.sweep_unscored_timeouts().await {
                warn!(error = %e, "unscored timeout sweep failed");
            }
            if let Ok(stuck) = self.store.list_stuck_runs(self.cfg.stuck_grace_secs).await {
                for r in stuck {
                    let _ = self
                        .store
                        .apply_run(
                            &r.id,
                            &StorePatch {
                                status: Some(RunStage::Failed),
                                error_detail: Some("stuck timeout".into()),
                                final_score: Some(FinalScore::Score(0)),
                                ..StorePatch::default()
                            },
                            Some(&StageEvent {
                                stage: "failed".into(),
                                detail: Some(serde_json::json!({"reason":"stuck"})),
                                at_ms: now_ms(),
                            }),
                        )
                        .await;
                }
            }
        }
    }

    pub async fn award_round(&self, rid: u64) -> Result<(), String> {
        let _ = self.ensure_round(rid).await;
        let _ = self.store.set_round_status(rid, "scoring").await;

        let runs = self
            .store
            .runs_for_round(rid)
            .await
            .map_err(|e| e.to_string())?;

        let mut miners_with_harness = BTreeSet::new();
        let mut miners_clean = BTreeSet::new();
        let mut cheat_miners = BTreeSet::new();
        let mut harness_to_miner = BTreeMap::new();

        for r in &runs {
            let Some(h) = self
                .store
                .get_harness(&r.harness_id)
                .await
                .map_err(|e| e.to_string())?
            else {
                continue;
            };
            harness_to_miner.insert(r.harness_id.clone(), h.miner_hotkey.clone());
            miners_with_harness.insert(h.miner_hotkey.clone());

            let verdict = r
                .agentic_verdict
                .as_ref()
                .and_then(|v| v.get("verdict").and_then(|x| x.as_str()));
            match (r.status, verdict) {
                (RunStage::AwaitingAdmin, _) => {
                    miners_clean.insert(h.miner_hotkey.clone());
                }
                (_, Some("cheat" | "suspicious")) => {
                    cheat_miners.insert(h.miner_hotkey.clone());
                }
                (RunStage::Scored, Some(_))
                    if matches!(r.final_score, Some(FinalScore::Score(0))) =>
                {
                    cheat_miners.insert(h.miner_hotkey.clone());
                }
                _ => {}
            }
        }

        let award = self
            .store
            .get_round_award(rid)
            .await
            .map_err(|e| e.to_string())?;
        let mut winner_miners = Vec::new();
        if let Some(a) = &award {
            for hid in &a.winner_harness_ids {
                if let Some(hk) = harness_to_miner.get(hid) {
                    winner_miners.push(hk.clone());
                }
            }
            winner_miners.sort();
            winner_miners.dedup();
        }

        let _ = (miners_clean, winner_miners);

        // Aggregate rolling-window round wins → proportional SCORE_MAX share
        // (scoring_version 3, cheat excluded).
        let ws = window_start(rid);
        let window_awards = self
            .store
            .list_round_awards(ws, rid)
            .await
            .map_err(|e| e.to_string())?;
        let mut win_counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut window_miners = miners_with_harness.clone();
        let mut window_cheat = cheat_miners.clone();
        for a in &window_awards {
            for hid in &a.winner_harness_ids {
                let hk = if let Some(h) = harness_to_miner.get(hid) {
                    h.clone()
                } else if let Some(h) = self
                    .store
                    .get_harness(hid)
                    .await
                    .map_err(|e| e.to_string())?
                {
                    window_miners.insert(h.miner_hotkey.clone());
                    h.miner_hotkey
                } else {
                    continue;
                };
                *win_counts.entry(hk).or_insert(0) += 1;
            }
            // Pull miners/cheats from other window rounds for leaf projection.
            if a.round_id != rid {
                if let Ok(other_runs) = self.store.runs_for_round(a.round_id).await {
                    for r in other_runs {
                        if let Ok(Some(h)) = self.store.get_harness(&r.harness_id).await {
                            window_miners.insert(h.miner_hotkey.clone());
                            let verdict = r
                                .agentic_verdict
                                .as_ref()
                                .and_then(|v| v.get("verdict").and_then(|x| x.as_str()));
                            if matches!(verdict, Some("cheat" | "suspicious")) {
                                window_cheat.insert(h.miner_hotkey);
                            }
                        }
                    }
                }
            }
        }

        let scores = score_window(&WindowScorePlan {
            win_counts: win_counts.clone(),
            miners_with_harness: window_miners,
            cheat_miners: window_cheat,
        });

        for (hk, fs) in &scores {
            let wins = win_counts.get(hk).copied().unwrap_or(0);
            let _ = self
                .store
                .upsert_rating(&RatingRow {
                    round_id: rid,
                    miner_hotkey: hk.clone(),
                    rating: match fs {
                        FinalScore::Score(v) => i32::try_from(*v / 1000).unwrap_or(0),
                        FinalScore::NoScore(_) => 0,
                    },
                    wins,
                    losses: 0,
                    final_score: Some(fs.clone()),
                })
                .await;
        }

        // Mark awaiting_admin runs scored.
        for r in &runs {
            if r.status == RunStage::AwaitingAdmin {
                let hk = harness_to_miner.get(&r.harness_id);
                let fs = hk
                    .and_then(|h| scores.get(h))
                    .cloned()
                    .unwrap_or(FinalScore::Score(0));
                let _ = self
                    .store
                    .apply_run(
                        &r.id,
                        &StorePatch {
                            status: Some(RunStage::Scored),
                            final_score: Some(fs),
                            ..StorePatch::default()
                        },
                        Some(&StageEvent {
                            stage: "scored".into(),
                            detail: None,
                            at_ms: now_ms(),
                        }),
                    )
                    .await;
            }
        }

        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        self.emit_leaves_at(state.subnet_epoch_index, state.last_epoch_block)
            .await?;
        let _ = self.store.set_round_status(rid, "emitted").await;
        Ok(())
    }

    pub async fn cycle_once(&self) -> Result<bool, String> {
        let Some(run) = self
            .store
            .claim_next_run(round_id_at(now_secs()))
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(false);
        };
        info!(run_id = %run.id, "claimed design run");
        let _ = self
            .store
            .apply_run(
                &run.id,
                &StorePatch {
                    status: Some(RunStage::Installing),
                    ..StorePatch::default()
                },
                Some(&StageEvent {
                    stage: "installing".into(),
                    detail: Some(json!({"prompt_id": run.prompt_id})),
                    at_ms: now_ms(),
                }),
            )
            .await;
        self.pause_stage().await;
        if let Err(f) = self.execute_run(&run).await {
            self.handle_run_failure(&run, f).await;
        }
        Ok(true)
    }

    async fn handle_run_failure(&self, run: &design_store::RunState, f: RunFailure) {
        let hotkey = self
            .store
            .get_harness(&run.harness_id)
            .await
            .ok()
            .flatten()
            .map(|h| h.miner_hotkey);
        if f.class.retryable() && run.retry_count < self.cfg.auto_retry_max {
            warn!(
                run_id = %run.id,
                class = f.class.as_str(),
                attempt = run.retry_count + 1,
                error = %f.msg,
                "auto-retrying run after infra failure"
            );
            let _ = self.store.reset_run(&run.id).await;
            let _ = self
                .store
                .apply_run(
                    &run.id,
                    &StorePatch::default(),
                    Some(&StageEvent {
                        stage: "auto_retry".into(),
                        detail: Some(json!({
                            "class": f.class.as_str(),
                            "attempt": run.retry_count + 1,
                            "max": self.cfg.auto_retry_max,
                            "error": clip_logs(&f.msg),
                        })),
                        at_ms: now_ms(),
                    }),
                )
                .await;
            if let (Some(g), Some(hk)) = (&self.gating, hotkey) {
                let _ = g.bump_attempt(CHALLENGE_ID, &hk, f.class.as_str()).await;
            }
            return;
        }
        warn!(run_id = %run.id, class = f.class.as_str(), error = %f.msg, "run failed (terminal)");
        // Retryable classes that exhausted the budget are internal, not miner
        // fault: NoScore(ChallengeInternal). Miner errors stay Score(0).
        let final_score = if f.class.retryable() {
            FinalScore::NoScore(bundle::NoScoreReasonCode::ChallengeInternal as u8)
        } else {
            FinalScore::Score(0)
        };
        let _ = self
            .store
            .apply_run(
                &run.id,
                &StorePatch {
                    status: Some(RunStage::Failed),
                    error_detail: Some(f.msg.clone()),
                    final_score: Some(final_score),
                    ..StorePatch::default()
                },
                Some(&StageEvent {
                    stage: "failed".into(),
                    detail: Some(json!({"class": f.class.as_str()})),
                    at_ms: now_ms(),
                }),
            )
            .await;
        if let (Some(g), Some(hk)) = (&self.gating, hotkey) {
            let _ = g
                .set_terminal(
                    CHALLENGE_ID,
                    &hk,
                    GatingState::Blocked,
                    Some(f.class.as_str()),
                )
                .await;
        }
    }

    async fn append_log(
        &self,
        run_id: &str,
        phase: &str,
        seq: u32,
        text: &str,
    ) -> Result<(), String> {
        let clipped = clip_logs(text);
        self.store
            .apply_run(
                run_id,
                &StorePatch::default(),
                Some(&StageEvent {
                    stage: "log".into(),
                    detail: Some(json!({
                        "phase": phase,
                        "stream": "combined",
                        "seq": seq,
                        "text": clipped,
                        "bytes": text.len(),
                        "truncated": text.len() > MAX_LOG_CHARS,
                    })),
                    at_ms: now_ms(),
                }),
            )
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn execute_run(&self, run: &design_store::RunState) -> Result<(), RunFailure> {
        let harness = self
            .store
            .get_harness(&run.harness_id)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?
            .ok_or_else(|| RunFailure::new(ErrorClass::Miner, "harness missing"))?;
        if harness.eliminated_until_round > run.round_id {
            return Err(RunFailure::new(ErrorClass::Miner, "harness eliminated"));
        }
        let prompts = select_prompts_for_round(run.round_id)
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let prompt = prompts
            .iter()
            .find(|p| p.id == run.prompt_id)
            .map(|p| p.prompt.clone())
            .unwrap_or_default();
        let bundle = design_harness::bundle_from_stored(
            harness.miner_hotkey.clone(),
            harness.agent_py.clone(),
            harness.pyproject_toml.clone(),
            harness.extra_files.clone(),
        );
        let sandbox = Arc::clone(&self.sandbox);
        let llm = self.cfg.llm_proxy.clone();
        let run_id = run.id.clone();
        let round_id = run.round_id;
        let prompt_c = prompt.clone();
        let bundle_c = bundle.clone();
        let install_res = tokio::task::spawn_blocking({
            let sandbox = Arc::clone(&sandbox);
            let run_id = run_id.clone();
            move || sandbox.install(&bundle_c, round_id, &run_id, &prompt_c, &llm)
        })
        .await
        .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let session = match install_res {
            Ok(s) => s,
            Err(e) => {
                if let SandboxError::PhaseFailed { phase, logs, .. } = &e {
                    let _ = self.append_log(&run.id, phase, 0, logs).await;
                }
                return Err(classify_sandbox(&e));
            }
        };
        self.append_log(&run.id, "install", 0, &session.install_logs)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e))?;

        let _ = self
            .store
            .apply_run(
                &run.id,
                &StorePatch {
                    status: Some(RunStage::Running),
                    ..StorePatch::default()
                },
                Some(&StageEvent {
                    stage: "running".into(),
                    detail: Some(json!({"phase": "run"})),
                    at_ms: now_ms(),
                }),
            )
            .await;
        self.pause_stage().await;

        let llm = self.cfg.llm_proxy.clone();
        let run_id = run.id.clone();
        let run_res = tokio::task::spawn_blocking(move || {
            sandbox.run_session(session, round_id, &run_id, &prompt, &llm)
        })
        .await
        .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let out = match run_res {
            Ok(o) => o,
            Err(e) => {
                if let SandboxError::PhaseFailed { phase, logs, .. } = &e {
                    let _ = self.append_log(&run.id, phase, 1, logs).await;
                }
                return Err(classify_sandbox(&e));
            }
        };
        self.append_log(&run.id, "run", 1, &out.run_logs)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e))?;

        let _ = self
            .store
            .apply_run(
                &run.id,
                &StorePatch {
                    status: Some(RunStage::Sanitizing),
                    ..StorePatch::default()
                },
                Some(&StageEvent {
                    stage: "sanitizing".into(),
                    detail: Some(json!({"pages": out.pages.len()})),
                    at_ms: now_ms(),
                }),
            )
            .await;
        self.pause_stage().await;
        // Sanitize reject is the miner's fault: terminal, no auto-retry.
        let sanitized = sanitize_bundle(&out.pages)
            .map_err(|e| RunFailure::new(ErrorClass::Miner, e.to_string()))?;
        let mut pages: Vec<_> = sanitized
            .pages
            .iter()
            .map(|p| {
                (
                    p.path.clone(),
                    p.sanitized_html.clone(),
                    p.raw_html.clone(),
                    p.raw_sha256.clone(),
                    p.bytes,
                )
            })
            .collect();
        // Best-effort full-page screenshot of the styled index for the site UI
        // (replaces iframe previews). Failure never fails the run. The capture
        // spawns Chromium, so keep it off the async worker threads.
        if let Some(index) = sanitized.pages.iter().find(|p| p.path == "index.html") {
            let shot_dir = self.cfg.staging_root.join("screenshots").join(&run.id);
            let html = index.sanitized_html.clone();
            let dir = shot_dir.clone();
            let png = tokio::task::spawn_blocking(move || capture_full_page_png(&html, &dir))
                .await
                .ok()
                .flatten();
            if let Some(png) = png {
                info!(run_id = %run.id, bytes = png.len(), "captured design page screenshot");
                pages.push(png_artifact_tuple(&png));
            } else {
                warn!(run_id = %run.id, "design page screenshot unavailable");
            }
            let _ = std::fs::remove_dir_all(&shot_dir);
        }
        self.store
            .put_artifacts(&run.id, &pages)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let report = serde_json::to_value(&sanitized.report).unwrap_or_default();

        // Pre-LLM copy gate → terminal `rejected` (one fetch for gate + review).
        let recent = self
            .store
            .list_recent_harnesses(64)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let gate_corpus = corpus::gate_corpus(&harness, &recent);
        if let Some(hit) = copy_gate(&harness.agent_py, harness.created_at_ms, &gate_corpus) {
            warn!(
                run_id = %run.id,
                nearest = %hit.nearest_id,
                similarity_bps = hit.similarity_bps,
                "copy gate rejected harness (created_at ordered, LLM skipped)"
            );
            let verdict_json = json!({
                "verdict": "cheat",
                "cheat_codes": [if hit.byte_identical {
                    "near_identical_harness_copy"
                } else {
                    "ast_architecture_copy"
                }],
                "nearest_id": hit.nearest_id,
                "similarity_bps": hit.similarity_bps,
                "rationale": "pre-LLM copy gate: byte/AST copy of an earlier harness",
                "gate": "copy_created_at",
            });
            let copy_reason = if hit.byte_identical {
                "near_identical_harness_copy"
            } else {
                "ast_architecture_copy"
            };
            let _ = self
                .store
                .apply_run(
                    &run.id,
                    &StorePatch {
                        status: Some(RunStage::Rejected),
                        artifact_digest: Some(sanitized.artifact_digest.clone()),
                        sanitize_report: Some(report),
                        agentic_verdict: Some(verdict_json),
                        reject_reason: Some(copy_reason.into()),
                        final_score: Some(FinalScore::Score(0)),
                        ..StorePatch::default()
                    },
                    Some(&StageEvent {
                        stage: "rejected".into(),
                        detail: Some(json!({
                            "gate": "copy_created_at",
                            "nearest_id": hit.nearest_id,
                            "similarity_bps": hit.similarity_bps,
                            "reject_reason": copy_reason,
                        })),
                        at_ms: now_ms(),
                    }),
                )
                .await;
            if let Some(g) = &self.gating {
                let _ = g
                    .set_terminal(
                        CHALLENGE_ID,
                        &harness.miner_hotkey,
                        GatingState::Rejected,
                        None,
                    )
                    .await;
            }
            return Ok(());
        }

        // Agentic anti-cheat review.
        let _ = self
            .store
            .apply_run(
                &run.id,
                &StorePatch {
                    status: Some(RunStage::AgenticReview),
                    artifact_digest: Some(sanitized.artifact_digest.clone()),
                    sanitize_report: Some(report.clone()),
                    ..StorePatch::default()
                },
                Some(&StageEvent {
                    stage: "agentic_review".into(),
                    detail: None,
                    at_ms: now_ms(),
                }),
            )
            .await;
        self.pause_stage().await;

        let verdict = self
            .run_agentic_review(run, &harness, &recent, &pages, &report)
            .await?;
        let verdict_json = serde_json::to_value(&verdict).unwrap_or_default();
        match verdict.verdict {
            VerdictKind::Clean => {
                let epoch = self.current_epoch();
                mark_awaiting_admin(
                    self.store.as_ref(),
                    &run.id,
                    &sanitized.artifact_digest,
                    report,
                    verdict_json,
                    epoch,
                )
                .await
                .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
            }
            VerdictKind::Cheat | VerdictKind::Suspicious => {
                let _ = self
                    .store
                    .apply_run(
                        &run.id,
                        &StorePatch {
                            status: Some(RunStage::Scored),
                            artifact_digest: Some(sanitized.artifact_digest.clone()),
                            sanitize_report: Some(report),
                            agentic_verdict: Some(verdict_json.clone()),
                            final_score: Some(FinalScore::Score(0)),
                            ..StorePatch::default()
                        },
                        Some(&StageEvent {
                            stage: "scored".into(),
                            detail: Some(json!({
                                "reason": "agentic",
                                "verdict": verdict_json.get("verdict"),
                            })),
                            at_ms: now_ms(),
                        }),
                    )
                    .await;
                // Cheat / suspicious is terminal: no retry, gating rejected.
                if let Some(g) = &self.gating {
                    let _ = g
                        .set_terminal(
                            CHALLENGE_ID,
                            &harness.miner_hotkey,
                            GatingState::Rejected,
                            None,
                        )
                        .await;
                }
            }
        }
        Ok(())
    }

    async fn run_agentic_review(
        &self,
        run: &design_store::RunState,
        harness: &design_store::HarnessRow,
        recent: &[design_store::HarnessRow],
        pages: &[(String, String, String, String, u32)],
        sanitize_report: &serde_json::Value,
    ) -> Result<AgenticVerdict, RunFailure> {
        let work = self.cfg.staging_root.join("agentic").join(&run.id);
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(work.join("pages"))
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        std::fs::write(work.join("agent.py"), &harness.agent_py)
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        for (path, sanitized, _, _, _) in pages {
            let name = path.rsplit('/').next().unwrap_or(path.as_str());
            std::fs::write(work.join("pages").join(name), sanitized)
                .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        }
        std::fs::write(
            work.join("sanitize_report.json"),
            sanitize_report.to_string(),
        )
        .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;

        let req = ReviewRequest {
            workdir: work.clone(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: corpus::review_corpus(harness, recent),
            metrics_relpath: None,
            pages_relpath: Some("pages".into()),
            sanitize_report_relpath: Some("sanitize_report.json".into()),
            domain_rules: DESIGN_AGENTIC_RULES.into(),
        };

        let result = self.agentic.review(&req).await;
        let _ = std::fs::remove_dir_all(&work);
        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                let f = classify_agentic(&e);
                // Record the verdict error for audit; the retry/terminal
                // decision (status) belongs to `handle_run_failure`.
                let _ = self
                    .store
                    .apply_run(
                        &run.id,
                        &StorePatch {
                            agentic_verdict: Some(json!({"error": f.msg})),
                            ..StorePatch::default()
                        },
                        None,
                    )
                    .await;
                Err(f)
            }
        }
    }

    async fn ensure_round(&self, rid: u64) -> Result<(), String> {
        if self
            .store
            .get_round(rid)
            .await
            .map_err(|e| e.to_string())?
            .is_some()
        {
            return Ok(());
        }
        let epoch = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map(|s| chain::current_epoch_pre_run_coinbase(&s, s.current_block))
            .unwrap_or(0);
        let opens = rid * round_secs();
        self.store
            .insert_round(&RoundRow {
                round_id: rid,
                epoch,
                netuid: self.cfg.netuid,
                prompt_set_digest: prompt_set_digest(),
                status: "open".into(),
                opens_at_secs: opens,
                closes_at_secs: opens + round_secs(),
            })
            .await
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn close_round(&self, rid: u64) -> Result<(), String> {
        if let Ok(Some(r)) = self.store.get_round(rid).await {
            if r.status == "emitted" {
                return Ok(());
            }
        }
        // Timeout without admin winners → Score(0) for all with harness; still emit.
        self.award_round(rid).await
    }

    async fn emit_leaves_at(&self, epoch: u64, pin_block: u64) -> Result<(), String> {
        if epoch == 0 {
            return Err("refuse emit for epoch 0".into());
        }
        if !design_leaf_emit_enabled() {
            info!(epoch, "design leaf emit skipped (DESIGN_SKIP_LEAF_EMIT)");
            return Ok(());
        }
        let block_hash = self
            .chain
            .block_hash(pin_block)
            .map_err(|e| format!("block_hash@{pin_block}: {e}"))?;
        let expected = expected_set_at_chain(
            &trustroot::ParticipantPolicy::AllMetagraphHotkeys,
            PinnedBlockHash::new(block_hash),
            self.chain.as_ref(),
        )
        .map_err(|e| format!("expected set: {e}"))?;
        let by_miner: BTreeMap<_, _> = self
            .store
            .scores_for_epoch(self.cfg.netuid, epoch)
            .await
            .map_err(|e| e.to_string())?
            .into_iter()
            .collect();
        let mut scores = BTreeMap::new();
        let mut expected_set = BTreeSet::new();
        for p in &expected.participants {
            expected_set.insert(p.hotkey);
            scores.insert(
                p.hotkey,
                by_miner
                    .get(&hex::encode(p.hotkey))
                    .map_or_else(|| to_leaf(&not_attempted()), to_leaf),
            );
        }
        let signed = emit_signed_leaf_set(
            &self.sk,
            CHALLENGE_ID.as_bytes(),
            epoch,
            &expected_set,
            &scores,
        )
        .map_err(|e| e.to_string())?;
        submit_signed_leaf_set(self.gateway.as_ref(), &signed)
            .await
            .map_err(|e| e.to_string())?;
        self.emitted_epoch.fetch_max(epoch, Ordering::Relaxed);
        info!(
            epoch,
            participants = expected_set.len(),
            pin_block,
            "design leaf set submitted"
        );
        Ok(())
    }
}

#[async_trait]
impl<C: ChainClient + Send + Sync + 'static> AdminAwardHook for Orchestrator<C> {
    async fn on_winners(&self, round_id: u64, _harness_ids: &[String]) -> Result<(), String> {
        self.award_round(round_id).await
    }
}
