//! Lium orchestrator: claim→screen→review→pod→score; emitter via `run_emitter`.

use std::sync::Arc;
use std::time::Duration;

use bundle::NoScoreReasonCode;
use chain::ChainClient;
use challenge_agentic::{
    copy_gate, prism_static_screen, AgenticBackend, AgenticVerdict, VerdictKind,
};
use challenge_common::{expected_set_at_chain, GatewayClient, PinnedBlockHash};
use crypto::KEY_LEN;
use prism_emit::EpochEmitter;
use prism_lium::HARNESS_ABSENT;
use prism_lium::{EvalJobBackend, InstanceSpec, RemoteExecResult};
use prism_lium_payer::PayerBackendFactory;
use prism_orphan::{
    fail_terminal, finish_measure, reject_gating, reject_pre_pod, spawn_log_watch, ActiveJobs,
    LogBuffer, DEFAULT_LOG_POLL_SECS, DEFAULT_ORPHAN_GRACE_SECS,
};
use prism_pipeline::{
    gating_key, measurement_patch, mid_pod_resume, resume_measurement, ScoringMode,
};
use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};
use prism_review::{ReviewBackend, SimilarityVerdict, SourceSnippet};
use submission_gating::{classify_eval_fail, GatingStore};
use tokio::time::sleep;
use tracing::{info, warn};

use prism_eval_store::finalize_for_submission;
use prism_final::{build_review_request, corpus_from_rows, gate_corpus_from_rows, same_miner};
use prism_final::{combine_final, FinalOutcome};
use prism_store::eval::EvalStore;
use prism_store::{FinalScore, PrismStore, Stage, StageEvent, StatePatch, SubmissionState};

/// Worker + emitter settings.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub netuid: u16,
    pub max_price_per_hour: f64,
    pub max_lifetime_hours: f64,
    pub ssh_public_keys: Vec<String>,
    pub image_digest: Option<String>,
    pub claim_poll: Duration,
    pub emit_poll: Duration,
    pub max_attempts: u32,
    pub similarity_corpus_limit: u32,
    pub stuck_grace_secs: u64,
    pub stage_delay: Duration,
    pub auto_retry_max: u32,
    pub scoring_mode: ScoringMode,
    pub orphan_grace_secs: u64,
    /// GPUs rented per eval pod (`PRISM_POD_GPU_COUNT`, default 4).
    ///
    /// Miners may train across all of them; the eval battery stays pinned to
    /// GPU 0 so G7 timings stay comparable across submissions.
    pub pod_gpu_count: u32,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            netuid: 1,
            max_price_per_hour: 2.5,
            max_lifetime_hours: prism_recipe::POD_LIFETIME_HOURS_CAP,
            ssh_public_keys: vec![],
            image_digest: None,
            claim_poll: Duration::from_millis(750),
            emit_poll: Duration::from_secs(15),
            max_attempts: 2,
            similarity_corpus_limit: 6,
            stuck_grace_secs: 10 * 3600,
            stage_delay: Duration::ZERO,
            auto_retry_max: 3,
            scoring_mode: ScoringMode::from_env(),
            orphan_grace_secs: DEFAULT_ORPHAN_GRACE_SECS,
            pod_gpu_count: prism_lium::pod_gpu_count_from_env(),
        }
    }
}

/// Orchestrator handle (workers + sweeper + emitter + API may share one instance).
pub struct Orchestrator<C: ChainClient + Send> {
    cfg: OrchestratorConfig,
    store: Arc<dyn PrismStore>,
    backend: Arc<dyn EvalJobBackend>,
    payer: Option<PayerBackendFactory>,
    reviewer: Arc<dyn ReviewBackend>,
    agentic: Arc<dyn AgenticBackend>,
    chain: Arc<C>,
    emitter: EpochEmitter,
    gating: Option<Arc<dyn GatingStore>>,
    topmodel: Option<Arc<prism_registry::TopModelPublisher>>,
    eval_store: Option<Arc<dyn EvalStore>>,
    active: Arc<ActiveJobs>,
    logs: Arc<LogBuffer>,
}

impl<C: ChainClient + Send> Orchestrator<C> {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: OrchestratorConfig,
        store: Arc<dyn PrismStore>,
        backend: Arc<dyn EvalJobBackend>,
        reviewer: Arc<dyn ReviewBackend>,
        agentic: Arc<dyn AgenticBackend>,
        gateway: &GatewayClient,
        chain: Arc<C>,
        sk: [u8; KEY_LEN],
    ) -> Self {
        let emitter = EpochEmitter::new(Arc::clone(&store), sk, cfg.netuid, gateway.clone());
        Self {
            cfg,
            store,
            backend,
            payer: None,
            reviewer,
            agentic,
            chain,
            emitter,
            gating: None,
            topmodel: None,
            eval_store: None,
            active: Arc::new(ActiveJobs::new()),
            logs: Arc::new(LogBuffer::new()),
        }
    }

    #[must_use]
    pub fn with_runtime(mut self, active: Arc<ActiveJobs>, logs: Arc<LogBuffer>) -> Self {
        self.active = active;
        self.logs = logs;
        self
    }

    #[must_use]
    pub fn with_payer(mut self, payer: PayerBackendFactory) -> Self {
        self.payer = Some(payer);
        self
    }

    #[must_use]
    pub fn with_gating(mut self, gating: Arc<dyn GatingStore>) -> Self {
        self.gating = Some(gating);
        self
    }

    #[must_use]
    pub fn with_eval_store(mut self, eval_store: Option<Arc<dyn EvalStore>>) -> Self {
        self.eval_store = eval_store;
        self
    }

    #[must_use]
    pub fn with_topmodel(
        mut self,
        publisher: Option<Arc<prism_registry::TopModelPublisher>>,
    ) -> Self {
        self.topmodel = publisher;
        self
    }

    fn backend_for(&self, submission_id: &str) -> Result<Arc<dyn EvalJobBackend>, String> {
        match &self.payer {
            Some(p) => p.resolve(submission_id, Arc::clone(&self.backend)),
            None => Ok(Arc::clone(&self.backend)),
        }
    }

    #[must_use]
    pub const fn cfg(&self) -> &OrchestratorConfig {
        &self.cfg
    }

    #[must_use]
    pub const fn emitter(&self) -> &EpochEmitter {
        &self.emitter
    }

    pub async fn run_worker(self: Arc<Self>)
    where
        C: Sync,
    {
        loop {
            match self.cycle_once().await {
                Ok(true) => {}
                Ok(false) => sleep(self.cfg.claim_poll).await,
                Err(e) => {
                    warn!(error = %e, "orchestrator cycle error");
                    sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    pub async fn run_sweeper(self: Arc<Self>)
    where
        C: Sync,
    {
        loop {
            sleep(Duration::from_secs(
                (self.cfg.stuck_grace_secs / 2).clamp(60, 3600),
            ))
            .await;
            if let Err(e) = self.sweep_once().await {
                warn!(error = %e, "sweeper tick error");
            }
        }
    }

    pub async fn reconcile_orphans(
        &self,
        boot: bool,
    ) -> Result<prism_orphan::ReconcileReport, String> {
        let grace = if boot { 0 } else { self.cfg.orphan_grace_secs };
        prism_orphan::reconcile_once(
            self.store.as_ref(),
            &self.active,
            self.payer.as_ref(),
            Arc::clone(&self.backend),
            grace,
            boot,
            self.gating.as_ref(),
        )
        .await
    }

    async fn sweep_once(&self) -> Result<(), String> {
        let stuck = self
            .store
            .list_stuck(self.cfg.stuck_grace_secs)
            .await
            .map_err(|e| e.to_string())?;
        for row in stuck {
            if self.active.contains(&row.id) {
                continue;
            }
            let be = self
                .backend_for(&row.id)
                .unwrap_or_else(|_| Arc::clone(&self.backend));
            let harvested = if let Some(pod) = row.pod_id.as_deref() {
                be.harvest_logs(pod).await.unwrap_or_default()
            } else {
                String::new()
            };
            if let Some(pod) = row.pod_id.clone() {
                let _ = be.terminate(&pod).await;
                let _ = be.verify_terminated(&pod).await;
            }
            if let Some(p) = &self.payer {
                p.vault.remove(&row.id);
            }
            let msg = if harvested.trim().is_empty() {
                "swept: stuck beyond grace".into()
            } else {
                format!(
                    "swept: stuck beyond grace; harvested: {}",
                    prism_lium::truncate_tail(&harvested, prism_lium::HARNESS_LOG_RETAIN_BYTES)
                )
            };
            if self.maybe_auto_retry(&row, "install", &msg).await {
                continue;
            }
            fail_terminal(
                self.store.as_ref(),
                self.gating.as_ref(),
                &row,
                "install",
                &msg,
            )
            .await;
        }
        Ok(())
    }

    /// One claim→finalize cycle; `Ok(true)` when one row was worked.
    pub async fn cycle_once(&self) -> Result<bool, String> {
        let row = self.store.claim_next().await.map_err(|e| e.to_string())?;
        let Some(row) = row else { return Ok(false) };
        self.run_row(row).await?;
        Ok(true)
    }

    async fn maybe_auto_retry(&self, row: &SubmissionState, class: &str, msg: &str) -> bool {
        let rate = lium_rent_pool::is_rate_limited(msg);
        if !rate && row.retry_count >= self.cfg.auto_retry_max {
            return false;
        }
        warn!(
            submission_id = %row.id,
            class,
            rate_limited = rate,
            attempt = row.retry_count + 1,
            max = self.cfg.auto_retry_max,
            error = %msg,
            "auto-retrying submission after infra failure"
        );
        let _ = self.store.reset_for_retry(&row.id, !rate).await;
        let _ = self
            .store
            .apply(
                &row.id,
                &StatePatch::default(),
                Some(&StageEvent {
                    stage: Stage::Queued,
                    detail: Some(serde_json::json!({
                        "auto_retry": true,
                        "class": class,
                        "rate_limited": rate,
                        "attempt": row.retry_count + 1,
                        "error": msg,
                    })),
                    at_ms: 0,
                }),
            )
            .await;
        if !rate {
            if let Some(g) = &self.gating {
                let _ = g
                    .bump_attempt(
                        &gating_key(row.arch_id.as_deref()),
                        &row.miner_hotkey,
                        class,
                    )
                    .await;
            }
        }
        true
    }

    async fn cap_terminal(&self, row: &SubmissionState, m: Option<&RemoteExecResult>) -> bool {
        let Some(m) = m.filter(|m| cap_flag(m)) else {
            return false;
        };
        let n_params = m.n_params;
        warn!(submission_id = %row.id, ?n_params, "parameter cap exceeded — terminal Score(0)");
        let patch = StatePatch {
            status: Some(Stage::Rejected),
            final_score: Some(FinalScore::Score(0)),
            error_detail: Some(format!("parameter cap exceeded: n_params={n_params:?}")),
            metrics_json: serde_json::to_value(m).ok(),
            ..StatePatch::default()
        };
        let event = StageEvent {
            stage: Stage::Rejected,
            detail: Some(serde_json::json!({ "gate": "parameter_cap", "n_params": n_params })),
            at_ms: 0,
        };
        let _ = self.store.apply(&row.id, &patch, Some(&event)).await;
        reject_gating(self.gating.as_ref(), row).await;
        true
    }

    fn terminal_event(&self, status: Stage, agentic: &AgenticVerdict) -> StageEvent {
        StageEvent {
            stage: status,
            detail: Some(serde_json::json!({
                "agentic": serde_json::to_value(agentic).unwrap_or_default(),
                "scoring_mode": self.cfg.scoring_mode.name(),
                "scoring_version": self.cfg.scoring_mode.scoring_version(),
            })),
            at_ms: 0,
        }
    }

    async fn fail_or_retry_measure(
        &self,
        row: &SubmissionState,
        err: String,
    ) -> Result<(), String> {
        let msg = format!("measure: {err}");
        // Harness EVAL_FAIL is miner/model code, not Lium infra — do not burn
        // auto-retries (BYOK seal is kept on Err; see finish_measure). The
        // miner-fixable phases (`install_deps` custom-deps install,
        // `train_script` training crash) additionally fail terminal under
        // their own class, which grants unbounded resubmit. Every other
        // EVAL_FAIL phase (eval / battery / score) stays the historical
        // windowed `install` class. Non-EVAL_FAIL failures are Lium infra and
        // keep `install` + operator-paid auto-retry.
        if msg.contains("EVAL_FAIL") {
            let class = classify_eval_fail(&msg);
            fail_terminal(self.store.as_ref(), self.gating.as_ref(), row, class, &msg).await;
            return Ok(());
        }
        if self.maybe_auto_retry(row, "install", &msg).await {
            return Ok(());
        }
        fail_terminal(
            self.store.as_ref(),
            self.gating.as_ref(),
            row,
            "install",
            &msg,
        )
        .await;
        Ok(())
    }

    pub async fn run_row(&self, row: SubmissionState) -> Result<(), String> {
        let id = row.id.clone();
        let _active = self.active.enter(&id);
        info!(submission_id = %id, miner = %row.miner_hotkey, "prism eval start");

        let Some((similarity, review)) = self.screens_or_resume(&id, &row).await else {
            return Ok(());
        };

        let (measured, fresh) = match resume_measurement(&row) {
            Some(mr) => (Ok(mr), false),
            None => (self.measure(&id, &row).await, true),
        };
        let (metrics, receipt) = match measured {
            Ok((m, r)) => (Some(m), Some(r)),
            Err(e) => return self.fail_or_retry_measure(&row, e).await,
        };

        if self.cap_terminal(&row, metrics.as_ref()).await {
            return Ok(());
        }
        if let (true, Some(m), Some(r)) = (fresh, metrics.as_ref(), receipt.as_ref()) {
            let _ = self.store.apply(&id, &measurement_patch(m, r), None).await;
        }
        let bpb = metrics.as_ref().map(|m| m.bpb);

        // Metrics-aware agentic (inconsistent_metrics / forge); never rents.
        let Some(agentic) = self
            .agentic_step(&id, &row, metrics.as_ref(), receipt.as_ref())
            .await
        else {
            return Ok(());
        };

        if matches!(
            agentic.verdict,
            VerdictKind::Cheat | VerdictKind::Suspicious
        ) {
            reject_gating(self.gating.as_ref(), &row).await;
        }

        let blob = metrics
            .as_ref()
            .map(|m| serde_json::to_value(m).unwrap_or_default());
        let composite = finalize_for_submission(self.eval_store.as_ref(), &id, blob.as_ref()).await;

        let outcome = match bpb {
            Some(b) => FinalOutcome::Measured {
                bpb: b,
                quality: review.quality_score,
                similarity: similarity.kind,
                similarity_score: similarity.score,
                similarity_evidence: similarity.evidence.clone(),
                agentic: agentic.verdict,
                composite,
                metrics: blob.clone(),
            },
            None => FinalOutcome::ChallengeInternal,
        };
        let final_score = combine_final(&outcome, self.cfg.scoring_mode);
        let status = if matches!(outcome, FinalOutcome::ChallengeInternal) {
            Stage::Failed
        } else {
            Stage::Terminated
        };

        self.store
            .apply(
                &id,
                &StatePatch {
                    status: Some(status),
                    final_score: Some(final_score.clone()),
                    bpb: match &outcome {
                        FinalOutcome::Measured { bpb, .. } => Some(*bpb),
                        FinalOutcome::ChallengeInternal => None,
                    },
                    review: Some(review),
                    similarity: Some(similarity),
                    pod_id: receipt.as_ref().map(|r| r.pod_id.clone()),
                    pod_provider: receipt.as_ref().map(|r| r.provider.clone()),
                    receipt,
                    metrics_json: metrics.as_ref().and_then(|m| serde_json::to_value(m).ok()),
                    ..StatePatch::default()
                },
                Some(&self.terminal_event(status, &agentic)),
            )
            .await
            .map_err(|e| e.to_string())?;

        if let Ok(Some(scored)) = self.store.get(&id).await {
            prism_registry::post_score_hooks(&self.store, self.topmodel.as_deref(), &scored, false)
                .await;
        }
        self.logs.clear(&id);
        Ok(())
    }

    async fn pre_pod_screens(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Option<(
        SimilarityVerdict,
        prism_review::ReviewVerdict,
        AgenticVerdict,
    )> {
        if self.copy_gate_step(row).await {
            return None;
        }
        if self.static_source_step(row).await {
            return None;
        }
        let similarity = match self.similarity_step(id, row).await {
            Ok(v) => v,
            Err(e) => {
                if self.maybe_auto_retry(row, "ast_infra", &e).await {
                    return None;
                }
                fail_terminal(
                    self.store.as_ref(),
                    self.gating.as_ref(),
                    row,
                    "ast_infra",
                    &e,
                )
                .await;
                return None;
            }
        };
        if prism_review::cheap_similarity_hard_zeros(
            similarity.kind,
            similarity.score,
            &similarity.evidence,
        ) {
            let detail = format!(
                "pre-pod similarity: {:?} score={:.2}",
                similarity.kind, similarity.score
            );
            reject_pre_pod(
                self.store.as_ref(),
                self.gating.as_ref(),
                row,
                Some(similarity),
                None,
                detail,
            )
            .await;
            return None;
        }
        let review = self.review_step(id, row).await?;
        let agentic = self.agentic_step(id, row, None, None).await?;
        if matches!(
            agentic.verdict,
            VerdictKind::Cheat | VerdictKind::Suspicious
        ) {
            let detail = format!("pre-pod agentic: {:?}", agentic.verdict);
            reject_pre_pod(
                self.store.as_ref(),
                self.gating.as_ref(),
                row,
                Some(similarity),
                Some(serde_json::json!({
                    "gate": "agentic_pre_pod",
                    "agentic": serde_json::to_value(&agentic).unwrap_or_default(),
                })),
                detail,
            )
            .await;
            return None;
        }
        Some((similarity, review, agentic))
    }

    async fn copy_gate_step(&self, row: &SubmissionState) -> bool {
        if row.arch_id.is_some() {
            return false;
        }
        let recent = self
            .store
            .list_champions(self.cfg.similarity_corpus_limit.max(64))
            .await
            .unwrap_or_default();
        let corpus = gate_corpus_from_rows(row, &recent);
        let Some(hit) = copy_gate(&row.architecture_py, row.created_at_ms, &corpus) else {
            return false;
        };
        warn!(
            submission_id = %row.id,
            nearest = %hit.nearest_id,
            similarity_bps = hit.similarity_bps,
            byte_identical = hit.byte_identical,
            "copy gate rejected architecture (created_at ordered, LLM + pod skipped)"
        );
        let similarity = SimilarityVerdict {
            kind: prism_review::SimilarityKind::Copied,
            score: f64::from(hit.similarity_bps) / 10_000.0,
            closest: Some(hit.nearest_id.clone()),
            evidence: vec![if hit.byte_identical {
                "pre-LLM copy gate: byte-identical earlier architecture".into()
            } else {
                "pre-LLM copy gate: AST copy of earlier architecture".into()
            }],
            prompt_version: prism_review::SIMILARITY_PROMPT_VERSION,
        };
        reject_pre_pod(
            self.store.as_ref(),
            self.gating.as_ref(),
            row,
            Some(similarity),
            Some(serde_json::json!({
                "gate": "copy_created_at",
                "nearest_id": hit.nearest_id,
                "similarity_bps": hit.similarity_bps,
                "byte_identical": hit.byte_identical,
            })),
            format!(
                "copy gate: architecture clones {} (bps={})",
                hit.nearest_id, hit.similarity_bps
            ),
        )
        .await;
        true
    }

    async fn static_source_step(&self, row: &SubmissionState) -> bool {
        let patch = row
            .tree_blob
            .as_deref()
            .and_then(prism_automodel::patch_text_from_tree_blob);
        let Some(hit) =
            prism_static_screen(&row.architecture_py, &row.training_py, patch.as_deref())
        else {
            return false;
        };
        warn!(
            submission_id = %row.id,
            kind = ?hit.kind,
            rationale = %hit.rationale,
            "static source cheat rejected (pod skipped)"
        );
        reject_pre_pod(
            self.store.as_ref(),
            self.gating.as_ref(),
            row,
            None,
            Some(serde_json::json!({
                "gate": "static_source",
                "cheat_kind": format!("{:?}", hit.kind),
                "rationale": hit.rationale,
            })),
            hit.rationale.clone(),
        )
        .await;
        true
    }

    async fn screens_or_resume(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Option<(SimilarityVerdict, prism_review::ReviewVerdict)> {
        if mid_pod_resume(row) {
            if let (Some(s), Some(r)) = (row.similarity.clone(), row.review.clone()) {
                return Some((s, r));
            }
        }
        let (s, r, _) = self.pre_pod_screens(id, row).await?;
        if !mid_pod_resume(row) {
            let _ = self
                .store
                .apply(
                    id,
                    &StatePatch {
                        review: Some(r.clone()),
                        similarity: Some(s.clone()),
                        ..StatePatch::default()
                    },
                    Some(&StageEvent {
                        stage: Stage::Provisioning,
                        detail: Some(serde_json::json!({"checkpoint": "pre_measure"})),
                        at_ms: 0,
                    }),
                )
                .await;
        }
        Some((s, r))
    }

    async fn measure(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Result<(prism_lium::RemoteExecResult, prism_lium::EvalReceipt), String> {
        self.to_stage(id, Stage::Provisioning).await?;
        // Re-seal immediately before measure so a full train+eval wall cannot
        // race TTL; fail closed when BYOK is required and the vault is empty.
        if let Some(p) = &self.payer {
            if !p.vault.refresh(id) && !p.allow_operator_fallback {
                return Err(
                    "miner Lium API key missing for this submission — resubmit with X-Lium-Api-Key"
                        .into(),
                );
            }
        }
        let backend = self.backend_for(id)?;
        let resume = mid_pod_resume(row);
        let (pod_id, provider) = if let Some(pid) = row.pod_id.clone() {
            (
                pid,
                row.pod_provider.clone().unwrap_or_else(|| "lium".into()),
            )
        } else {
            let spec = InstanceSpec {
                name: format!("prism-{}", &id[..12.min(id.len())]),
                max_lifetime_hours: self.cfg.max_lifetime_hours,
                max_price_per_hour: self.cfg.max_price_per_hour,
                gpu_count: self.cfg.pod_gpu_count,
                image_digest: self.cfg.image_digest.clone(),
                ssh_public_keys: self.cfg.ssh_public_keys.clone(),
                ssh_key_name: Some("prism-mission-worker".into()),
                preferred_offer_id: None,
                template_id: None,
                template_name: None,
            };
            let inst = backend
                .provision(&spec)
                .await
                .map_err(|e| format!("provision: {e}"))?;
            (inst.id, inst.provider)
        };
        self.to_stage(id, Stage::Running).await?;
        let _ = self
            .store
            .apply(
                id,
                &StatePatch {
                    pod_id: Some(pod_id.clone()),
                    pod_provider: Some(provider.clone()),
                    ..StatePatch::default()
                },
                None,
            )
            .await;
        let payer_vault = self.payer.as_ref().map(|p| Arc::clone(&p.vault));
        let stop_tx = spawn_log_watch(
            Arc::clone(&self.logs),
            Arc::clone(&self.store),
            Arc::clone(&backend),
            Arc::clone(&self.active),
            payer_vault,
            id.to_owned(),
            pod_id.clone(),
            DEFAULT_LOG_POLL_SECS,
        );
        let (arch, train, tree) = (
            row.architecture_py.as_str(),
            row.training_py.as_str(),
            row.tree_blob.as_deref(),
        );
        let metrics = if resume {
            match backend.resume_eval(&pod_id).await {
                Ok(m) => Ok(m),
                Err(e) if e.to_string().contains(HARNESS_ABSENT) => {
                    backend.exec_eval(&pod_id, arch, train, tree).await
                }
                Err(e) => Err(e),
            }
        } else {
            backend.exec_eval(&pod_id, arch, train, tree).await
        };
        let _ = stop_tx.send(true);
        finish_measure(
            &backend,
            self.payer.as_ref(),
            id,
            row,
            &pod_id,
            provider,
            self.cfg.image_digest.clone().unwrap_or_default(),
            metrics,
        )
        .await
    }

    async fn review_step(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Option<prism_review::ReviewVerdict> {
        self.to_stage(id, Stage::LlmReview).await.ok()?;
        match self
            .reviewer
            .review(&row.architecture_py, &row.training_py)
            .await
        {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = format!("llm review: {e}");
                if self.maybe_auto_retry(row, "llm_infra", &msg).await {
                    return None;
                }
                fail_terminal(
                    self.store.as_ref(),
                    self.gating.as_ref(),
                    row,
                    "llm_infra",
                    &msg,
                )
                .await;
                None
            }
        }
    }

    async fn similarity_step(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Result<SimilarityVerdict, String> {
        self.to_stage(id, Stage::Similarity).await?;
        if let Some(a) = &row.arch_id {
            return Ok(SimilarityVerdict {
                kind: prism_review::SimilarityKind::Original,
                score: 0.0,
                closest: Some(a.clone()),
                evidence: vec![
                    "training-only: architecture from registry (similarity-exempt)".into(),
                ],
                prompt_version: prism_review::SIMILARITY_PROMPT_VERSION,
            });
        }
        let corpus = self.similarity_corpus(row).await;
        self.reviewer
            .similarity(&row.architecture_py, &corpus)
            .await
            .map_err(|e| format!("similarity: {e}"))
    }

    async fn agentic_step(
        &self,
        id: &str,
        row: &SubmissionState,
        metrics: Option<&prism_lium::RemoteExecResult>,
        receipt: Option<&prism_lium::EvalReceipt>,
    ) -> Option<AgenticVerdict> {
        if self.to_stage(id, Stage::Scoring).await.is_err() {
            return None;
        }
        let dir = match tempfile::tempdir() {
            Ok(d) => d,
            Err(e) => {
                self.fail_agentic(id, &format!("workdir: {e}")).await;
                return None;
            }
        };
        let recent = self
            .store
            .list_champions(self.cfg.similarity_corpus_limit)
            .await
            .unwrap_or_default();
        // Training-only rows: drop the referenced registry arch from the
        // corpus (byte-identity with it is by design, not a copy).
        let corpus = corpus_from_rows(
            row,
            &recent,
            row.arch_id
                .is_some()
                .then_some(row.architecture_py.as_str()),
        );
        let req = match build_review_request(dir.path(), row, metrics, receipt, corpus) {
            Ok(r) => r,
            Err(e) => {
                self.fail_agentic(id, &e).await;
                return None;
            }
        };
        match self.agentic.review(&req).await {
            Ok(v) => Some(v),
            Err(e) => {
                let msg = format!("agentic: {e}");
                if self.maybe_auto_retry(row, "llm_infra", &msg).await {
                    return None;
                }
                fail_terminal(
                    self.store.as_ref(),
                    self.gating.as_ref(),
                    row,
                    "llm_infra",
                    &msg,
                )
                .await;
                None
            }
        }
    }

    async fn fail_agentic(&self, id: &str, err: &str) {
        let _ = self
            .store
            .apply(
                id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    error_detail: Some(format!("agentic: {err}")),
                    final_score: Some(FinalScore::NoScore(
                        NoScoreReasonCode::ChallengeInternal as u8,
                    )),
                    ..StatePatch::default()
                },
                Some(&StageEvent {
                    stage: Stage::Failed,
                    detail: Some(serde_json::json!({"where": "agentic", "error": err})),
                    at_ms: 0,
                }),
            )
            .await;
    }

    async fn similarity_corpus(&self, candidate: &SubmissionState) -> Vec<SourceSnippet> {
        let recent = self
            .store
            .list_champions(self.cfg.similarity_corpus_limit)
            .await
            .unwrap_or_default();
        let mut v = vec![SourceSnippet {
            label: "baseline".into(),
            architecture_py: BASELINE_ARCHITECTURE_PY.into(),
            training_py: BASELINE_TRAINING_PY.into(),
        }];
        v.extend(
            recent
                .into_iter()
                .filter(|r| r.id != candidate.id && !same_miner(candidate, r))
                .map(|r| SourceSnippet {
                    label: format!("subm:{}", &r.id[..r.id.len().min(8)]),
                    architecture_py: r.architecture_py,
                    training_py: r.training_py,
                }),
        );
        v
    }

    async fn to_stage(&self, id: &str, stage: Stage) -> Result<(), String> {
        self.store
            .apply(
                id,
                &StatePatch {
                    status: Some(stage),
                    ..StatePatch::default()
                },
                Some(&StageEvent {
                    stage,
                    detail: None,
                    at_ms: 0,
                }),
            )
            .await
            .map_err(|e| format!("stage {stage:?}: {e}"))?;
        // Hold after publish so local evidence can photograph mid-flight stages.
        if !self.cfg.stage_delay.is_zero() {
            sleep(self.cfg.stage_delay).await;
        }
        Ok(())
    }

    pub async fn run_emitter(self: Arc<Self>)
    where
        C: Sync,
    {
        loop {
            if let Err(e) = self.emitter_tick().await {
                warn!(error = %e, "emitter tick error");
            }
            sleep(self.cfg.emit_poll).await;
        }
    }

    pub async fn emitter_tick(&self) -> Result<Option<prism_emit::EmitSummary>, String> {
        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        // Pin expected set at `last_epoch_block` (current epoch), not +1.
        let epoch = state.subnet_epoch_index;
        let block_hash = self
            .chain
            .block_hash(state.last_epoch_block)
            .map_err(|e| format!("block_hash: {e}"))?;
        let expected = expected_set_at_chain(
            &trustroot::ParticipantPolicy::AllMetagraphHotkeys,
            PinnedBlockHash::new(block_hash),
            self.chain.as_ref(),
        )
        .map_err(|e| format!("expected set: {e}"))?;
        let summary = self
            .emitter
            .tick(epoch, &expected)
            .await
            .map_err(|e| e.to_string())?;
        if let Some(s) = &summary {
            #[rustfmt::skip]
            info!(epoch = s.epoch, leaves = s.leaves, batch = s.batch, "epoch leaf set emitted");
        }
        Ok(summary)
    }
}

fn cap_flag(m: &RemoteExecResult) -> bool {
    m.extra
        .get("cap_exceeded")
        .and_then(serde_json::Value::as_bool)
        == Some(true)
}
