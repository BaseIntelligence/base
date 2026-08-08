//! Lium job orchestrator: DB-backed state machine, recovery, epoch emitter.
//!
//! Workers claim `queued` rows, run cheap source screens (copy gate, static
//! cheat patterns, AST similarity) **before** renting a Lium pod, then run
//! the recipe + master-side LLM review + agentic anti-cheat, and compute the
//! chain-facing score. Leaf emission is decoupled from finalizes: the
//! epoch-close emitter ([`prism_emit::EpochEmitter`], driven by
//! [`Orchestrator::run_emitter`]) assigns every newly-finalized row to the
//! next chain-epoch boundary's D24 set via the emission outbox
//! (`emitted_epoch` watermark + emit cursor), so independent same-epoch
//! scorers all land and each scoring run is assigned exactly once. Positive
//! scores then carry into later epochs' competition sets until superseded;
//! leaf emission applies WTA so only the single best hotkey gets Prism's
//! share. All state lives in the store, so the API is a pure projection and
//! restarts sweep orphans.

use std::sync::Arc;
use std::time::Duration;

use bundle::NoScoreReasonCode;
use chain::ChainClient;
use challenge_agentic::{
    copy_gate, static_source_cheat, AgenticBackend, AgenticVerdict, VerdictKind,
};
use challenge_common::{expected_set_at_chain, PinnedBlockHash};
use crypto::KEY_LEN;
use prism_emit::EpochEmitter;
use prism_lium::{EvalJobBackend, InstanceSpec};
use prism_pipeline::gating_key;
use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};
use prism_review::{ReviewBackend, SimilarityVerdict, SourceSnippet};
use submission_gating::{GatingState, GatingStore};
use tokio::time::sleep;
use tracing::{info, warn};

use crate::agentic::{build_review_request, corpus_from_rows, gate_corpus_from_rows, same_miner};
use crate::score::{combine_final, FinalOutcome};
use crate::submit::GatewayClient;
use prism_store::{FinalScore, PrismStore, Stage, StageEvent, StatePatch, SubmissionState};

/// Worker + emitter settings.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Netuid.
    pub netuid: u16,
    /// Pod price cap.
    pub max_price_per_hour: f64,
    /// Pod lifetime hours cap (train cap + margin).
    pub max_lifetime_hours: f64,
    /// SSH public keys for rent.
    pub ssh_public_keys: Vec<String>,
    /// Optional image digest pin.
    pub image_digest: Option<String>,
    /// Queue polling cadence.
    pub claim_poll: Duration,
    /// Emitter tick cadence (chain-epoch boundary detection lag).
    pub emit_poll: Duration,
    /// Rent attempt budget before `failed`.
    pub max_attempts: u32,
    /// Similarity corpus size (recent submissions + baseline).
    pub similarity_corpus_limit: u32,
    /// Stuck sweep grace (seconds).
    pub stuck_grace_secs: u64,
    /// Local/e2e only: pause after each published stage so mid-flight is
    /// photographable. Zero in production (default).
    pub stage_delay: Duration,
    /// Auto-retry budget for infra-class failures (Lium install / AST / LLM).
    /// Default 3; cheat / rejected verdicts are always terminal.
    pub auto_retry_max: u32,
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
            stuck_grace_secs: 7 * 3600,
            stage_delay: Duration::ZERO,
            auto_retry_max: 3,
        }
    }
}

/// Orchestrator handle (workers + sweeper + emitter + API may share one instance).
pub struct Orchestrator<C: ChainClient + Send> {
    cfg: OrchestratorConfig,
    store: Arc<dyn PrismStore>,
    backend: Arc<dyn EvalJobBackend>,
    reviewer: Arc<dyn ReviewBackend>,
    agentic: Arc<dyn AgenticBackend>,
    chain: Arc<C>,
    emitter: EpochEmitter,
    gating: Option<Arc<dyn GatingStore>>,
    topmodel: Option<Arc<prism_registry::TopModelPublisher>>,
}

impl<C: ChainClient + Send> Orchestrator<C> {
    /// Construct.
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
        let emitter = EpochEmitter::new(Arc::clone(&store), sk, cfg.netuid, gateway.common());
        Self {
            cfg,
            store,
            backend,
            reviewer,
            agentic,
            chain,
            emitter,
            gating: None,
            topmodel: None,
        }
    }

    /// Attach the submission gating store (terminal states + retry attempts).
    #[must_use]
    pub fn with_gating(mut self, gating: Arc<dyn GatingStore>) -> Self {
        self.gating = Some(gating);
        self
    }

    /// Attach the top-model GitHub publisher (absent = publish step no-ops).
    #[must_use]
    pub fn with_topmodel(
        mut self,
        publisher: Option<Arc<prism_registry::TopModelPublisher>>,
    ) -> Self {
        self.topmodel = publisher;
        self
    }

    /// Config getter (API views).
    #[must_use]
    pub const fn cfg(&self) -> &OrchestratorConfig {
        &self.cfg
    }

    /// Emitter accessor (tests / diagnostics).
    #[must_use]
    pub const fn emitter(&self) -> &EpochEmitter {
        &self.emitter
    }

    /// Claim loop (spawn one per concurrency permit).
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

    /// Stuck-row sweeper loop.
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

    async fn sweep_once(&self) -> Result<(), String> {
        let stuck = self
            .store
            .list_stuck(self.cfg.stuck_grace_secs)
            .await
            .map_err(|e| e.to_string())?;
        for row in stuck {
            if let Some(pod) = row.pod_id.clone() {
                let _ = self.backend.terminate(&pod).await;
                let _ = self.backend.verify_terminated(&pod).await;
            }
            let id = row.id.clone();
            let _ = self
                .store
                .apply(
                    &id,
                    &StatePatch {
                        status: Some(Stage::Failed),
                        error_detail: Some("swept: stuck beyond grace".into()),
                        retry_bump: 1,
                        ..StatePatch::default()
                    },
                    Some(&StageEvent {
                        stage: Stage::Failed,
                        detail: Some(serde_json::json!({"reason": "stuck-sweep"})),
                        at_ms: 0,
                    }),
                )
                .await;
        }
        Ok(())
    }

    /// One claim→finalize cycle; `Ok(true)` when one row was worked.
    ///
    /// # Errors
    /// Claim fault only; per-row business errors become `failed` rows.
    pub async fn cycle_once(&self) -> Result<bool, String> {
        let row = self.store.claim_next().await.map_err(|e| e.to_string())?;
        let Some(row) = row else { return Ok(false) };
        self.run_row(row).await?;
        Ok(true)
    }

    /// Requeue on infra-class failures while the auto-retry budget lasts
    /// (`false` → caller finalizes terminal). Records the gating attempt.
    async fn maybe_auto_retry(&self, row: &SubmissionState, class: &str, msg: &str) -> bool {
        if row.retry_count >= self.cfg.auto_retry_max {
            return false;
        }
        warn!(
            submission_id = %row.id,
            class,
            attempt = row.retry_count + 1,
            max = self.cfg.auto_retry_max,
            error = %msg,
            "auto-retrying submission after infra failure"
        );
        let _ = self.store.reset_for_retry(&row.id).await;
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
                        "attempt": row.retry_count + 1,
                        "error": msg,
                    })),
                    at_ms: 0,
                }),
            )
            .await;
        if let Some(g) = &self.gating {
            let _ = g
                .bump_attempt(
                    &gating_key(row.arch_id.as_deref()),
                    &row.miner_hotkey,
                    class,
                )
                .await;
        }
        true
    }

    /// Gating `rejected` for cheat-class terminals (composite key for
    /// training-only rows, `prism` otherwise).
    async fn reject_gating(&self, row: &SubmissionState) {
        if let Some(g) = &self.gating {
            let _ = g
                .set_terminal(
                    &gating_key(row.arch_id.as_deref()),
                    &row.miner_hotkey,
                    GatingState::Rejected,
                    None,
                )
                .await;
        }
    }

    /// Terminal failure: `failed` row + gating `blocked`. The
    /// `NoScore(ChallengeInternal)` enters the emission outbox and lands in
    /// the next epoch-boundary leaf set (`run_emitter`).
    async fn fail_terminal(&self, row: &SubmissionState, class: &str, msg: &str) {
        let _ = self
            .store
            .apply(
                &row.id,
                &StatePatch {
                    status: Some(Stage::Failed),
                    error_detail: Some(msg.to_owned()),
                    final_score: Some(FinalScore::NoScore(
                        NoScoreReasonCode::ChallengeInternal as u8,
                    )),
                    ..StatePatch::default()
                },
                Some(&StageEvent {
                    stage: Stage::Failed,
                    detail: Some(serde_json::json!({"class": class, "error": msg})),
                    at_ms: 0,
                }),
            )
            .await;
        if let Some(g) = &self.gating {
            let _ = g
                .set_terminal(
                    &gating_key(row.arch_id.as_deref()),
                    &row.miner_hotkey,
                    GatingState::Blocked,
                    Some(class),
                )
                .await;
        }
    }

    /// Process one submission end-to-end.
    pub async fn run_row(&self, row: SubmissionState) -> Result<(), String> {
        let id = row.id.clone();
        info!(submission_id = %id, miner = %row.miner_hotkey, "prism eval start");

        // Phase 0: pre-pod cheap screens (no GPU, no private eval assets).
        // Copy gate → static cheat patterns → AST similarity. Fail-fast with
        // Score(0) so a bad submission never rents a Lium pod (~6h waste).
        if self.copy_gate_step(&row).await {
            return Ok(());
        }
        if self.static_source_step(&row).await {
            return Ok(());
        }
        let similarity = match self.similarity_step(&id, &row).await {
            Ok(v) => v,
            Err(e) => {
                if self.maybe_auto_retry(&row, "ast_infra", &e).await {
                    return Ok(());
                }
                self.fail_terminal(&row, "ast_infra", &e).await;
                return Ok(());
            }
        };
        if matches!(
            similarity.kind,
            prism_review::SimilarityKind::Copied | prism_review::SimilarityKind::Suspicious
        ) {
            let detail = format!("pre-pod similarity: {:?}", similarity.kind);
            self.reject_pre_pod(&row, Some(similarity), None, detail)
                .await;
            return Ok(());
        }

        // Phase 1: provision + recipe exec + terminate (always verified).
        // Lium/infra failures auto-retry (install class); budget exhaustion is
        // terminal with NoScore(ChallengeInternal), never a miner zero.
        let measured = self.measure(&id, &row).await;
        let (metrics, receipt) = match measured {
            Ok((m, r)) => (Some(m), Some(r)),
            Err(e) => {
                let msg = format!("measure: {e}");
                if self.maybe_auto_retry(&row, "install", &msg).await {
                    return Ok(());
                }
                self.fail_terminal(&row, "install", &msg).await;
                return Ok(());
            }
        };
        let bpb = metrics.as_ref().map(|m| m.bpb);

        // Phase 2: cheap review (LLM infra → auto-retry, then terminal).
        let Some(review) = self.review_step(&id, &row).await else {
            return Ok(());
        };

        // Phase 3: agentic anti-cheat (needs metrics/receipt; post-pod).
        // Source-only screens already ran pre-pod; this catches metrics forge.
        let Some(agentic) = self
            .agentic_step(&id, &row, metrics.as_ref(), receipt.as_ref())
            .await
        else {
            return Ok(());
        };

        // Cheat / suspicious is terminal: no retry, gating rejected.
        if matches!(
            agentic.verdict,
            VerdictKind::Cheat | VerdictKind::Suspicious
        ) {
            self.reject_gating(&row).await;
        }

        let outcome = match bpb {
            Some(b) => FinalOutcome::Measured {
                bpb: b,
                quality: review.quality_score,
                similarity: similarity.kind,
                agentic: agentic.verdict,
            },
            None => FinalOutcome::ChallengeInternal,
        };
        let final_score = combine_final(&outcome);
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
                    bpb: outcome.bpb().copied(),
                    review: Some(review),
                    similarity: Some(similarity),
                    pod_id: receipt.as_ref().map(|r| r.pod_id.clone()),
                    pod_provider: receipt.as_ref().map(|r| r.provider.clone()),
                    receipt,
                    metrics_json: metrics.as_ref().and_then(|m| serde_json::to_value(m).ok()),
                    ..StatePatch::default()
                },
                Some(&StageEvent {
                    stage: status,
                    detail: Some(serde_json::json!({
                        "agentic": serde_json::to_value(&agentic).unwrap_or_default()
                    })),
                    at_ms: 0,
                }),
            )
            .await
            .map_err(|e| e.to_string())?;

        // Leaf emission is decoupled: the epoch-close emitter batches this
        // score into the next boundary's D24 set (exactly-once outbox), so
        // the arch publish below lands well before the row's leaf epoch.
        if let Ok(Some(scored)) = self.store.get(&id).await {
            prism_registry::post_score_hooks(&self.store, self.topmodel.as_deref(), &scored).await;
        }
        let _ = final_score;
        Ok(())
    }

    /// Pre-LLM copy gate on `architecture.py`. Returns `true` when the row was
    /// finalized terminal `rejected` (caller must stop processing).
    ///
    /// The corpus is recent submissions (any status — prior art is prior art)
    /// ordered by store `created_at`; the published baseline is exempt by id
    /// prefix inside [`copy_gate`]. Ties / unknown timestamps fall through to
    /// the LLM similarity review. Training-only rows (`arch_id` set) skip the
    /// gate entirely: their architecture is registry-identical by design.
    async fn copy_gate_step(&self, row: &SubmissionState) -> bool {
        if row.arch_id.is_some() {
            return false;
        }
        let recent = self.store.list(None, None, 64).await.unwrap_or_default();
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
        self.reject_pre_pod(
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

    /// Static source cheat screen (METRICS_JSON / telemetry hooks). Pre-pod.
    /// Returns `true` when the row was finalized terminal `rejected`.
    async fn static_source_step(&self, row: &SubmissionState) -> bool {
        let Some(hit) = static_source_cheat(&row.architecture_py, &row.training_py) else {
            return false;
        };
        warn!(
            submission_id = %row.id,
            code = ?hit.code,
            rationale = %hit.rationale,
            "static source cheat rejected (pod skipped)"
        );
        self.reject_pre_pod(
            row,
            None,
            Some(serde_json::json!({
                "gate": "static_source",
                "cheat_code": format!("{:?}", hit.code),
                "rationale": hit.rationale,
            })),
            hit.rationale.clone(),
        )
        .await;
        true
    }

    /// Terminal Score(0) reject before any Lium rent. Shared by copy gate,
    /// static screens, and pre-pod similarity.
    async fn reject_pre_pod(
        &self,
        row: &SubmissionState,
        similarity: Option<SimilarityVerdict>,
        detail: Option<serde_json::Value>,
        error_detail: String,
    ) {
        let _ = self
            .store
            .apply(
                &row.id,
                &StatePatch {
                    status: Some(Stage::Rejected),
                    final_score: Some(FinalScore::Score(0)),
                    similarity,
                    error_detail: Some(error_detail),
                    ..StatePatch::default()
                },
                Some(&StageEvent {
                    stage: Stage::Rejected,
                    detail,
                    at_ms: 0,
                }),
            )
            .await;
        if let Some(g) = &self.gating {
            let _ = g
                .set_terminal(
                    &gating_key(row.arch_id.as_deref()),
                    &row.miner_hotkey,
                    GatingState::Rejected,
                    None,
                )
                .await;
        }
    }

    /// Pod phase. Returns `(bpb, receipt)` on full success.
    async fn measure(
        &self,
        id: &str,
        row: &SubmissionState,
    ) -> Result<(prism_lium::RemoteExecResult, prism_lium::EvalReceipt), String> {
        self.to_stage(id, Stage::Provisioning).await?;

        let spec = InstanceSpec {
            name: format!("prism-{}", &id[..12]),
            max_lifetime_hours: self.cfg.max_lifetime_hours,
            max_price_per_hour: self.cfg.max_price_per_hour,
            gpu_count: 1,
            image_digest: self.cfg.image_digest.clone(),
            ssh_public_keys: self.cfg.ssh_public_keys.clone(),
            ssh_key_name: Some("prism-mission-worker".into()),
            preferred_offer_id: None,
            template_id: None,
            template_name: None, // default recipe template (prism-recipe-v2 w/ sshd)
        };
        self.to_stage(id, Stage::Running).await?;

        let inst = self
            .backend
            .provision(&spec)
            .await
            .map_err(|e| format!("provision: {e}"))?;
        let pod_id = inst.id.clone();
        let _ = self
            .store
            .apply(
                id,
                &StatePatch {
                    pod_id: Some(pod_id.clone()),
                    pod_provider: Some(inst.provider.clone()),
                    ..StatePatch::default()
                },
                None,
            )
            .await;

        let metrics = self
            .backend
            .exec_eval(&pod_id, &row.architecture_py, &row.training_py)
            .await;

        // Always terminate + verify (billing guard, receipt gate).
        if let Err(e) = self.backend.terminate(&pod_id).await {
            warn!(error = %e, %pod_id, "terminate failed");
        }
        let mut termination_verified = self
            .backend
            .verify_terminated(&pod_id)
            .await
            .unwrap_or(false);
        if !termination_verified {
            tokio::time::sleep(Duration::from_secs(5)).await;
            termination_verified = self
                .backend
                .verify_terminated(&pod_id)
                .await
                .unwrap_or(false);
        }

        let receipt = prism_lium::EvalReceipt {
            provider: inst.provider.clone(),
            pod_id: pod_id.clone(),
            image_digest: self.cfg.image_digest.clone().unwrap_or_default(),
            submission_hash: prism_lium::EvalReceipt::hash_submission(
                &row.architecture_py,
                &row.training_py,
            ),
            metrics_hash: metrics.as_ref().ok().map_or_else(
                || "none".into(),
                |m| {
                    prism_lium::EvalReceipt::hash_metrics_bytes(
                        &serde_json::to_vec(m).unwrap_or_default(),
                    )
                },
            ),
            termination_verified,
        };

        let metrics = metrics.map_err(|e| format!("exec: {e}"))?;
        Ok((metrics, receipt))
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
                self.fail_terminal(row, "llm_infra", &msg).await;
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
        // Training-only rows train a registry architecture: similarity is
        // exempt by definition (the arch copy judgment happened when the
        // owner's architecture submission was reviewed).
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

    /// Agentic anti-cheat on sources + metrics/receipt. Fail-closed on error.
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
            .list(Some("terminated"), None, self.cfg.similarity_corpus_limit)
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
                self.fail_agentic(id, &msg).await;
                if let Some(g) = &self.gating {
                    let _ = g
                        .set_terminal(
                            &gating_key(row.arch_id.as_deref()),
                            &row.miner_hotkey,
                            GatingState::Blocked,
                            Some("llm_infra"),
                        )
                        .await;
                }
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
            .list(Some("terminated"), None, self.cfg.similarity_corpus_limit)
            .await
            .unwrap_or_default();
        let mut v = vec![SourceSnippet {
            label: "baseline".into(),
            architecture_py: BASELINE_ARCHITECTURE_PY.into(),
            training_py: BASELINE_TRAINING_PY.into(),
        }];
        for r in recent {
            if r.id == candidate.id || same_miner(candidate, &r) {
                continue;
            }
            let label = if r.id.len() >= 8 {
                format!("subm:{}", &r.id[..8])
            } else {
                format!("subm:{}", r.id)
            };
            v.push(SourceSnippet {
                label,
                architecture_py: r.architecture_py.clone(),
                training_py: r.training_py.clone(),
            });
        }
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

    /// Emitter loop: one D24-complete leaf set per chain epoch (epoch-close
    /// batching, exactly-once outbox — see `prism-emit` docs).
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

    /// One emitter tick: read the live chain epoch + expected set, then let
    /// the outbox recover/emit. `Ok(None)` = this epoch already emitted.
    ///
    /// # Errors
    /// Chain / store / sign / submit failures (retried next tick).
    pub async fn emitter_tick(&self) -> Result<Option<prism_emit::EmitSummary>, String> {
        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        // Label with the *current* chain epoch, not the pre-coinbase +1: the
        // expected set below is pinned at `last_epoch_block` (the current
        // epoch's start block), so a pre-run +1 label attaches the *previous*
        // boundary's metagraph to the new epoch number. The other >0-bps
        // challenge pins the same way, and D24 requires both same-label sets
        // to cover the seal block's metagraph exactly — the +1 skew made
        // every boundary churn a permanent 409 (`IncompleteParticipantSet`).
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
            info!(
                epoch = s.epoch,
                leaves = s.leaves,
                batch = s.batch,
                "epoch leaf set emitted"
            );
        }
        Ok(summary)
    }
}

impl FinalOutcome {
    fn bpb(&self) -> Option<&f64> {
        match self {
            FinalOutcome::Measured { bpb, .. } => Some(bpb),
            FinalOutcome::ChallengeInternal => None,
        }
    }
}
