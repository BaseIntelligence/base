//! Lium job orchestrator: DB-backed state machine, recovery, close-loop leaves.
//!
//! Workers claim `queued` rows, rent + run the recipe, run master-side LLM
//! review + cheap similarity + agentic anti-cheat, compute the chain-facing
//! score, and emit + submit the exact-E leaf set for the current chain epoch.
//! All state lives in the store, so the API is a pure projection and restarts
//! sweep orphans.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use chain::ChainClient;
use challenge_agentic::{AgenticBackend, AgenticVerdict};
use challenge_common::{expected_set_at_chain, ExpectedSet, PinnedBlockHash};
use crypto::KEY_LEN;
use prism_lium::{EvalJobBackend, InstanceSpec};
use prism_recipe::{BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY};
use prism_review::{ReviewBackend, SimilarityVerdict, SourceSnippet};
use tokio::time::sleep;
use tracing::{info, warn};

use crate::agentic::{build_review_request, corpus_from_rows};
use crate::leaf_emit::emit_signed_leaf_set;
use crate::score::{combine_final, FinalOutcome};
use crate::submit::{submit_signed_leaf_set, GatewayClient};
use crate::CHALLENGE_ID;
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
    /// Rent attempt budget before `failed`.
    pub max_attempts: u32,
    /// Similarity corpus size (recent submissions + baseline).
    pub similarity_corpus_limit: u32,
    /// Stuck sweep grace (seconds).
    pub stuck_grace_secs: u64,
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
            max_attempts: 2,
            similarity_corpus_limit: 6,
            stuck_grace_secs: 7 * 3600,
        }
    }
}

/// Orchestrator handle (workers + sweeper + API may share one instance).
pub struct Orchestrator<C: ChainClient + Send> {
    cfg: OrchestratorConfig,
    store: Arc<dyn PrismStore>,
    backend: Arc<dyn EvalJobBackend>,
    reviewer: Arc<dyn ReviewBackend>,
    agentic: Arc<dyn AgenticBackend>,
    gateway: Arc<GatewayClient>,
    chain: Arc<C>,
    sk: [u8; KEY_LEN],
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
        gateway: Arc<GatewayClient>,
        chain: Arc<C>,
        sk: [u8; KEY_LEN],
    ) -> Self {
        Self {
            cfg,
            store,
            backend,
            reviewer,
            agentic,
            gateway,
            chain,
            sk,
        }
    }

    /// Config getter (API views).
    #[must_use]
    pub const fn cfg(&self) -> &OrchestratorConfig {
        &self.cfg
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

    /// Process one submission end-to-end.
    pub async fn run_row(&self, row: SubmissionState) -> Result<(), String> {
        let id = row.id.clone();
        info!(submission_id = %id, miner = %row.miner_hotkey, "prism eval start");

        // Phase 1: provision + recipe exec + terminate (always verified).
        let measured = self.measure(&id, &row).await;
        let (metrics, receipt) = match &measured {
            Ok((m, r)) => (Some(m.clone()), Some(r.clone())),
            Err(e) => {
                warn!(submission_id = %id, error = %e, "measure phase failed");
                let _ = self
                    .store
                    .apply(
                        &id,
                        &StatePatch {
                            error_detail: Some(format!("measure: {e}")),
                            ..StatePatch::default()
                        },
                        None,
                    )
                    .await;
                (None, None)
            }
        };
        let bpb = metrics.as_ref().map(|m| m.bpb);

        // Phase 2: cheap review (fail → row becomes failed with ChallengeInternal).
        let review = match self.review_step(&id, &row).await {
            Some(v) => Some(v),
            None => return Ok(()), // already failed+finalized
        };

        // Phase 3: cheap similarity (first filter; agentic is primary judge).
        let similarity = self.similarity_step(&id, &row).await?;

        // Phase 4: agentic anti-cheat (NoVerdict → ChallengeInternal).
        let Some(agentic) = self
            .agentic_step(&id, &row, metrics.as_ref(), receipt.as_ref())
            .await
        else {
            return Ok(());
        };

        let outcome = match (bpb, &review) {
            (Some(b), Some(r)) => FinalOutcome::Measured {
                bpb: b,
                quality: r.quality_score,
                similarity: similarity.kind,
                agentic: agentic.verdict,
            },
            _ => FinalOutcome::ChallengeInternal,
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
                    review,
                    similarity: Some(similarity),
                    pod_id: receipt.as_ref().map(|r| r.pod_id.clone()),
                    pod_provider: receipt.as_ref().map(|r| r.provider.clone()),
                    receipt,
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

        // Close the loop: exact-E leaf set for the current chain epoch (D24),
        // idempotent against the gateway's append-only ledger.
        match self.emit_and_submit_epoch().await {
            Ok(n) => info!(submission_id = %id, leaves = n, stage = ?status, "leaf set submitted"),
            Err(e) => warn!(submission_id = %id, error = %e, "leaf submission deferred"),
        }
        let _ = final_score;
        Ok(())
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
                let _ = self
                    .store
                    .apply(
                        id,
                        &StatePatch {
                            status: Some(Stage::Failed),
                            error_detail: Some(format!("llm review: {e}")),
                            final_score: Some(FinalScore::NoScore(
                                NoScoreReasonCode::ChallengeInternal as u8,
                            )),
                            ..StatePatch::default()
                        },
                        Some(&StageEvent {
                            stage: Stage::Failed,
                            detail: Some(
                                serde_json::json!({"where": "review", "error": e.to_string()}),
                            ),
                            at_ms: 0,
                        }),
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
        let corpus = self.similarity_corpus(id).await;
        self.reviewer
            .similarity(&row.architecture_py, &row.training_py, &corpus)
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
            .list(Some("terminated"), self.cfg.similarity_corpus_limit)
            .await
            .unwrap_or_default();
        let corpus = corpus_from_rows(id, &recent);
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
                self.fail_agentic(id, &e.to_string()).await;
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

    async fn similarity_corpus(&self, current_id: &str) -> Vec<SourceSnippet> {
        let recent = self
            .store
            .list(Some("terminated"), self.cfg.similarity_corpus_limit)
            .await
            .unwrap_or_default();
        let mut v = vec![SourceSnippet {
            label: "baseline".into(),
            architecture_py: BASELINE_ARCHITECTURE_PY.into(),
            training_py: BASELINE_TRAINING_PY.into(),
        }];
        for r in recent {
            if r.id == current_id {
                continue;
            }
            v.push(SourceSnippet {
                label: format!("subm:{}", &r.id[..8]),
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
            .map(|_| ())
            .map_err(|e| format!("stage {stage:?}: {e}"))
    }

    /// Emit + POST the exact-E leaf set for the chain's current epoch.
    ///
    /// # Errors
    /// Chain/read/sign/submit failures (retried on next finalize).
    pub async fn emit_and_submit_epoch(&self) -> Result<usize, String> {
        let state = chain::gather_schedule_state(self.chain.as_ref(), self.cfg.netuid)
            .map_err(|e| format!("schedule: {e}"))?;
        let epoch = chain::current_epoch_pre_run_coinbase(&state, state.current_block);
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
        self.emit_and_submit_at(epoch, &expected).await
    }

    /// Same with an explicit `(epoch, E)` (tests).
    pub async fn emit_and_submit_at(
        &self,
        epoch: u64,
        expected: &ExpectedSet,
    ) -> Result<usize, String> {
        let rows = self
            .store
            .scores_for_epoch(self.cfg.netuid, epoch)
            .await
            .map_err(|e| e.to_string())?;
        let by_miner: BTreeMap<String, FinalScore> = rows.into_iter().collect();

        let mut scores: BTreeMap<[u8; KEY_LEN], ScoreOrAbsence> = BTreeMap::new();
        let mut expected_set: BTreeSet<[u8; KEY_LEN]> = BTreeSet::new();
        for p in &expected.participants {
            expected_set.insert(p.hotkey);
            let soa = match by_miner.get(&hex::encode(p.hotkey)) {
                Some(FinalScore::Score(v)) => ScoreOrAbsence::Score { value: *v },
                Some(FinalScore::NoScore(r)) => ScoreOrAbsence::NoScore {
                    reason: reason_from_u8(*r),
                },
                None => ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::NotAttempted,
                },
            };
            scores.insert(p.hotkey, soa);
        }
        let leaves = emit_signed_leaf_set(&self.sk, epoch, &expected_set, &scores)
            .map_err(|e| format!("emit: {e}"))?;
        submit_signed_leaf_set(&self.gateway, CHALLENGE_ID, epoch, &leaves)
            .await
            .map_err(|e| format!("submit: {e}"))?;
        Ok(leaves.len())
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

fn reason_from_u8(v: u8) -> NoScoreReasonCode {
    match v {
        0 => NoScoreReasonCode::NotAttempted,
        1 => NoScoreReasonCode::Timeout,
        2 => NoScoreReasonCode::InvalidResponse,
        3 => NoScoreReasonCode::AttestationNotVerified,
        4 => NoScoreReasonCode::MinerError,
        5 => NoScoreReasonCode::RateLimited,
        7 => NoScoreReasonCode::PolicySkip,
        _ => NoScoreReasonCode::ChallengeInternal,
    }
}
