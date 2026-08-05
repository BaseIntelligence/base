//! Round + run orchestrator: sandbox, agentic review, admin award, leaf emit.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use chain::ChainClient;
use challenge_agentic::{
    copy_gate, AgenticBackend, AgenticError, AgenticVerdict, CorpusEntry, GateCorpusEntry,
    ReviewRequest, VerdictKind,
};
use challenge_common::{
    emit_signed_leaf_set, expected_set_at_chain, submit_signed_leaf_set, ExpectedSet,
    GatewayClient, PinnedBlockHash,
};
use crypto::KEY_LEN;
use design_challenge_task::{round_id_at, round_secs};
use design_http::{mark_awaiting_admin, AdminAwardHook};
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

use crate::score::{not_attempted, score_window, to_leaf, window_start, WindowScorePlan};
use crate::CHALLENGE_ID;

/// Cap harness log payload stored in stage-event detail (JSON).
const MAX_LOG_CHARS: usize = 65_536;

/// Classified run failure: retryable infra classes vs terminal miner errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Package install / Docker engine infra (auto-retryable).
    Install,
    /// AST / workdir / store infra (auto-retryable).
    AstInfra,
    /// LLM transport / provider / verdict parse (auto-retryable).
    LlmInfra,
    /// Miner-caused failure: agent crash, sanitize reject, missing output
    /// (terminal — no auto-retry).
    Miner,
}

impl ErrorClass {
    /// DB / gating label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::AstInfra => "ast_infra",
            Self::LlmInfra => "llm_infra",
            Self::Miner => "miner",
        }
    }

    /// Auto-retry budget applies (initial attempt + up to 3 retries).
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

/// Design domain rules appended to the agentic system prompt.
const DESIGN_AGENTIC_RULES: &str = r"
Design challenge rules:
- Allowed: PyPI via egress, Mobbin/Dribbble inspiration, image generation, UI libs.
- The corpus entry labeled `baseline` is the published reference miner: starting
  from it is explicitly allowed and is NOT a copy. Copying ANOTHER MINER's
  harness is what counts as a copy.
- Cheat: near-identical harness copy, trivial HTML republish, scrape-clone of a real site,
  sanitize bypass / JS exfil, obfuscation to hide a copy.
- suspicious and cheat both → Score(0), not eligible for admin winner selection.
";

/// Orchestrator config.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Netuid.
    pub netuid: u16,
    /// Claim poll.
    pub claim_poll: Duration,
    /// Stuck grace secs.
    pub stuck_grace_secs: u64,
    /// LLM proxy URL for sandbox.
    pub llm_proxy: String,
    /// Staging root for agentic workdirs.
    pub staging_root: PathBuf,
    /// Local/e2e only: pause after each published stage so mid-flight is
    /// photographable. Zero in production (default).
    pub stage_delay: Duration,
    /// Auto-retry budget for infra-class failures (initial attempt + this
    /// many retries). Default 3; `cheat` / `rejected` are always terminal.
    pub auto_retry_max: u32,
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
        }
    }
}

/// Orchestrator handle.
pub struct Orchestrator<C: ChainClient + Send + Sync> {
    cfg: OrchestratorConfig,
    store: Arc<dyn DesignStore>,
    sandbox: Arc<dyn SandboxBackend>,
    agentic: Arc<dyn AgenticBackend>,
    gateway: Arc<GatewayClient>,
    chain: Arc<C>,
    sk: [u8; KEY_LEN],
    gating: Option<Arc<dyn GatingStore>>,
}

impl<C: ChainClient + Send + Sync> std::fmt::Debug for Orchestrator<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Orchestrator")
            .field("netuid", &self.cfg.netuid)
            .finish_non_exhaustive()
    }
}

impl<C: ChainClient + Send + Sync + 'static> Orchestrator<C> {
    /// Construct.
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
        }
    }

    /// Attach the submission gating store (terminal states + retry attempts).
    #[must_use]
    pub fn with_gating(mut self, gating: Arc<dyn GatingStore>) -> Self {
        self.gating = Some(gating);
        self
    }

    /// Local/e2e: hold after publishing a stage so polls can photograph it.
    async fn pause_stage(&self) {
        if !self.cfg.stage_delay.is_zero() {
            sleep(self.cfg.stage_delay).await;
        }
    }

    /// Worker loop.
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

    /// Round closer + leaf emitter loop.
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
            // Ensure current round row exists.
            let _ = self.ensure_round(rid).await;
        }
    }

    /// Stuck sweeper.
    pub async fn run_sweeper(self: Arc<Self>) {
        loop {
            sleep(Duration::from_secs(60)).await;
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

    /// Award winners (admin API) or timeout close: score + emit leaves.
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

        self.emit_leaves().await?;
        let _ = self.store.set_round_status(rid, "emitted").await;
        Ok(())
    }

    /// One claim→execute cycle; `Ok(true)` when one run was worked.
    ///
    /// # Errors
    /// Claim fault only; per-run failures are retried or finalized inline.
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

    /// Auto-retry retryable infra classes up to `auto_retry_max`; otherwise
    /// finalize `failed` and block the hotkey in the gating store.
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
        let pages: Vec<_> = sanitized
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
        self.store
            .put_artifacts(&run.id, &pages)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let report = serde_json::to_value(&sanitized.report).unwrap_or_default();

        // Pre-LLM copy gate: byte/AST copy of an *earlier* harness → terminal
        // `rejected` without spending the LLM review.
        let gate_corpus = self.gate_corpus(&run.harness_id).await;
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
            let _ = self
                .store
                .apply_run(
                    &run.id,
                    &StorePatch {
                        status: Some(RunStage::Rejected),
                        artifact_digest: Some(sanitized.artifact_digest.clone()),
                        sanitize_report: Some(report),
                        agentic_verdict: Some(verdict_json),
                        final_score: Some(FinalScore::Score(0)),
                        ..StorePatch::default()
                    },
                    Some(&StageEvent {
                        stage: "rejected".into(),
                        detail: Some(json!({
                            "gate": "copy_created_at",
                            "nearest_id": hit.nearest_id,
                            "similarity_bps": hit.similarity_bps,
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
            .run_agentic_review(run, &harness.agent_py, &pages, &report)
            .await?;
        let verdict_json = serde_json::to_value(&verdict).unwrap_or_default();
        match verdict.verdict {
            VerdictKind::Clean => {
                mark_awaiting_admin(
                    self.store.as_ref(),
                    &run.id,
                    &sanitized.artifact_digest,
                    report,
                    verdict_json,
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

    /// Corpus for the pre-LLM copy gate (recent harnesses minus the candidate).
    async fn gate_corpus(&self, exclude_harness_id: &str) -> Vec<GateCorpusEntry> {
        self.store
            .list_recent_harnesses(64)
            .await
            .unwrap_or_default()
            .into_iter()
            .filter(|h| h.id != exclude_harness_id)
            .map(|h| GateCorpusEntry {
                id: format!("harness:{}", h.id),
                source: h.agent_py,
                created_at_ms: h.created_at_ms,
            })
            .collect()
    }

    async fn run_agentic_review(
        &self,
        run: &design_store::RunState,
        agent_py: &str,
        pages: &[(String, String, String, String, u32)],
        sanitize_report: &serde_json::Value,
    ) -> Result<AgenticVerdict, RunFailure> {
        let work = self.cfg.staging_root.join("agentic").join(&run.id);
        let _ = std::fs::remove_dir_all(&work);
        std::fs::create_dir_all(work.join("pages"))
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        std::fs::write(work.join("agent.py"), agent_py)
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

        let recent = self
            .store
            .list_recent_harnesses(64)
            .await
            .map_err(|e| RunFailure::new(ErrorClass::AstInfra, e.to_string()))?;
        let cand_created = recent
            .iter()
            .find(|h| h.id == run.harness_id)
            .map(|h| h.created_at_ms)
            .unwrap_or(0);
        let mut corpus: Vec<CorpusEntry> = vec![CorpusEntry {
            // The published baseline is always in the corpus (same as prism):
            // it anchors originality judgments and keeps an empty recent-set
            // from producing incoherent verdicts.
            id: "baseline".into(),
            source: include_str!("../../../docs/external-miner/examples/design-baseline/agent.py")
                .to_owned(),
        }];
        corpus.extend(
            recent
                .into_iter()
                .filter(|h| h.id != run.harness_id)
                // Prior art only (created_at ordered like the pre-LLM gate): a
                // later byte-copy must never poison the original's review.
                .filter(|h| {
                    cand_created == 0 || (h.created_at_ms > 0 && h.created_at_ms < cand_created)
                })
                .map(|h| CorpusEntry {
                    id: format!("harness:{}", h.id),
                    source: h.agent_py,
                }),
        );

        let req = ReviewRequest {
            workdir: work.clone(),
            primary_relpaths: vec!["agent.py".into()],
            corpus,
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

    /// Close round: if winners already awarded/emitted, no-op; else award (timeout zeros).
    async fn close_round(&self, rid: u64) -> Result<(), String> {
        if let Ok(Some(r)) = self.store.get_round(rid).await {
            if r.status == "emitted" {
                return Ok(());
            }
        }
        // Timeout without admin winners → Score(0) for all with harness; still emit.
        self.award_round(rid).await
    }

    async fn emit_leaves(&self) -> Result<(), String> {
        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        let epoch = chain::current_epoch_pre_run_coinbase(&state, state.current_block);
        let block_hash = self
            .chain
            .block_hash(state.last_epoch_block)
            .map_err(|e| format!("block_hash: {e}"))?;
        let expected: ExpectedSet = expected_set_at_chain(
            &trustroot::ParticipantPolicy::AllMetagraphHotkeys,
            PinnedBlockHash::new(block_hash),
            self.chain.as_ref(),
        )
        .map_err(|e| format!("expected set: {e}"))?;
        let stored = self
            .store
            .scores_for_epoch(self.cfg.netuid, epoch)
            .await
            .map_err(|e| e.to_string())?;
        let by_miner: BTreeMap<String, FinalScore> = stored.into_iter().collect();
        let mut scores = BTreeMap::new();
        let mut expected_set: BTreeSet<[u8; KEY_LEN]> = BTreeSet::new();
        for p in &expected.participants {
            expected_set.insert(p.hotkey);
            let leaf = match by_miner.get(&hex::encode(p.hotkey)) {
                Some(fs) => to_leaf(fs),
                None => to_leaf(&not_attempted()),
            };
            scores.insert(p.hotkey, leaf);
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
        Ok(())
    }
}

#[async_trait]
impl<C: ChainClient + Send + Sync + 'static> AdminAwardHook for Orchestrator<C> {
    async fn on_winners(&self, round_id: u64, _harness_ids: &[String]) -> Result<(), String> {
        self.award_round(round_id).await
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn clip_logs(text: &str) -> String {
    if text.len() <= MAX_LOG_CHARS {
        return text.to_owned();
    }
    let mut out = text[text.len() - MAX_LOG_CHARS..].to_owned();
    out.insert_str(0, "...[truncated]\n");
    out
}
