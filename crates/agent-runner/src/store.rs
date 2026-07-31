//! In-memory task store: dispatch auth + pack executor + receipt signing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_dispatch::{
    patch_sha256, sign_work_receipt, TaskDescriptorV1, TaskResultV1, TaskStatusV1,
    WorkReceiptBodyV1, DISPATCH_PROTOCOL,
};
use crypto::{MemoryNonceStore, KEY_LEN};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use uuid::Uuid;

use crate::api::CapacityResponse;
use crate::auth::{
    verify_and_consume_dispatch, DispatchAuthError, SignedDispatchRequest,
    DEFAULT_DISPATCH_NONCE_TTL,
};
use crate::egress::{AgentEgressPosture, DEFAULT_AGENT_EGRESS_POSTURE};
use crate::executor::{execute_pack, ExecutionBackend};
use crate::receipt_key::ReceiptKey;

/// Inclusive lower bound for miner-declared concurrency (deepagent `--n-concurrent`).
pub const MIN_CONCURRENCY: u32 = 1;
/// Inclusive upper bound for miner-declared concurrency.
pub const MAX_CONCURRENCY_BOUND: u32 = 5;

/// Clamp a miner-declared concurrency into the upstream eval window `1..=5`.
#[must_use]
pub fn clamp_concurrency(n: u32) -> u32 {
    n.clamp(MIN_CONCURRENCY, MAX_CONCURRENCY_BOUND)
}

/// Runner process configuration (env-driven at the binary boundary).
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Miner-declared max concurrent tasks (clamped to `1..=5` at construction).
    pub max_concurrency: u32,
    /// When true (default), `POST /v1/task` requires a signed dispatch envelope.
    pub auth_enabled: bool,
    /// Trusted challenge public key for dispatch auth (required when auth on).
    pub trusted_challenge_pubkey: Option<[u8; KEY_LEN]>,
    /// Max dispatch-auth TTL (must stay below one epoch).
    pub dispatch_nonce_ttl: Duration,
    /// CVM-local work-receipt key. Required to complete tasks with a real sig.
    pub receipt_key: Option<ReceiptKey>,
    /// Pack execution backend (stub or Docker).
    pub execution: ExecutionBackend,
    /// Documented egress posture (default OPEN).
    pub egress_posture: AgentEgressPosture,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 1,
            auth_enabled: true,
            trusted_challenge_pubkey: None,
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: None,
            execution: ExecutionBackend::default(),
            egress_posture: DEFAULT_AGENT_EGRESS_POSTURE,
        }
    }
}

/// Lifecycle status exposed on `GET /v1/task/{id}` (broader than wire terminal status).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskLifecycle {
    /// Accepted, not yet started.
    Pending,
    /// Executor holds a slot.
    Running,
    /// Finished successfully (result present).
    Completed,
    /// Hard deadline exceeded (result present, typically no patch).
    TimedOut,
    /// Finished with failure (result present).
    Failed,
}

#[derive(Debug, Clone)]
struct TaskRecord {
    lifecycle: TaskLifecycle,
    result: Option<TaskResultV1>,
}

/// Shared runner state (cloneable axum `State`).
#[derive(Clone)]
pub struct RunnerState {
    config: RunnerConfig,
    /// Effective concurrency after [`clamp_concurrency`].
    effective_max: u32,
    /// Permits = effective max; acquired at accept, released when the task finishes.
    slots: Arc<Semaphore>,
    inner: Arc<Mutex<StoreInner>>,
}

struct StoreInner {
    tasks: HashMap<String, TaskRecord>,
    /// Occupied concurrency slots (accepted and not yet finished).
    running: u32,
    /// Single-use dispatch nonces (todo 18).
    dispatch_nonces: MemoryNonceStore,
}

/// Accept refused because every concurrency slot is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityExhausted;

impl RunnerState {
    /// New empty store with the given configuration.
    ///
    /// `max_concurrency` is clamped to `1..=5`. The semaphore is sized to the
    /// **effective** value so capacity advertisement matches enforcement.
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        let effective_max = clamp_concurrency(config.max_concurrency);
        Self {
            config,
            effective_max,
            slots: Arc::new(Semaphore::new(effective_max as usize)),
            inner: Arc::new(Mutex::new(StoreInner {
                tasks: HashMap::new(),
                running: 0,
                dispatch_nonces: MemoryNonceStore::new(),
            })),
        }
    }

    /// Effective max concurrency after clamp (`1..=5`).
    #[must_use]
    pub fn effective_max_concurrency(&self) -> u32 {
        self.effective_max
    }

    /// Configured egress posture (default OPEN).
    #[must_use]
    pub fn egress_posture(&self) -> AgentEgressPosture {
        self.config.egress_posture
    }

    /// Whether dispatch auth is enforced.
    #[must_use]
    pub fn auth_enabled(&self) -> bool {
        self.config.auth_enabled
    }

    /// Verify a signed dispatch envelope and consume its nonce (fail closed).
    ///
    /// # Errors
    ///
    /// Propagates [`DispatchAuthError`] from the auth module.
    pub async fn verify_dispatch_auth(
        &self,
        req: &SignedDispatchRequest,
        now_unix_ms: u64,
        now_instant: Instant,
    ) -> Result<(), DispatchAuthError> {
        let trusted = self
            .config
            .trusted_challenge_pubkey
            .as_ref()
            .ok_or(DispatchAuthError::Unauthorized)?;
        let mut g = self.inner.lock().await;
        verify_and_consume_dispatch(
            trusted,
            self.config.dispatch_nonce_ttl,
            &mut g.dispatch_nonces,
            req,
            now_unix_ms,
            now_instant,
        )
    }

    /// Number of known tasks (tests / metrics).
    pub async fn task_count(&self) -> usize {
        let g = self.inner.lock().await;
        g.tasks.len()
    }

    /// Snapshot capacity for `GET /v1/capacity`.
    #[must_use]
    pub fn capacity(&self) -> CapacityResponse {
        let load = self.inner.try_lock().map_or(0, |g| g.running);
        CapacityResponse {
            max_concurrency: self.effective_max,
            current_load: load,
        }
    }

    /// Async capacity (accurate under load).
    pub async fn capacity_async(&self) -> CapacityResponse {
        let g = self.inner.lock().await;
        CapacityResponse {
            max_concurrency: self.effective_max,
            current_load: g.running,
        }
    }

    /// Try to accept a descriptor under the concurrency semaphore.
    ///
    /// Acquires a permit **before** inserting the task. On exhaustion returns
    /// [`CapacityExhausted`] without creating a task or queueing it.
    ///
    /// # Errors
    ///
    /// [`CapacityExhausted`] when all slots are held.
    pub async fn accept_task(
        &self,
        descriptor: TaskDescriptorV1,
    ) -> Result<String, CapacityExhausted> {
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| CapacityExhausted)?;

        let task_id = Uuid::new_v4().to_string();
        {
            let mut g = self.inner.lock().await;
            g.tasks.insert(
                task_id.clone(),
                TaskRecord {
                    lifecycle: TaskLifecycle::Pending,
                    result: None,
                },
            );
            g.running = g.running.saturating_add(1);
        }
        let state = self.clone();
        let id = task_id.clone();
        tokio::spawn(async move {
            state.run_task(id, descriptor, permit).await;
        });
        Ok(task_id)
    }

    /// Lookup task view fields.
    pub async fn get_task(&self, id: &str) -> Option<(TaskLifecycle, Option<TaskResultV1>)> {
        let g = self.inner.lock().await;
        g.tasks.get(id).map(|r| (r.lifecycle, r.result.clone()))
    }

    async fn run_task(
        &self,
        task_id: String,
        descriptor: TaskDescriptorV1,
        permit: OwnedSemaphorePermit,
    ) {
        {
            let mut g = self.inner.lock().await;
            if let Some(rec) = g.tasks.get_mut(&task_id) {
                rec.lifecycle = TaskLifecycle::Running;
            }
        }

        let backend = self.config.execution.clone();
        let pack_id = descriptor.pack_id.clone();
        let deadline = descriptor.deadline_unix_ms;
        let outcome =
            tokio::task::spawn_blocking(move || execute_pack(&backend, &pack_id, deadline))
                .await
                .unwrap_or(crate::executor::ExecOutcome {
                    status: TaskStatusV1::Failed,
                    model_patch: None,
                });

        let (lifecycle, result) = match finalize_result(&self.config, &descriptor, outcome) {
            Ok((lc, result)) => (lc, result),
            Err(msg) => {
                tracing::error!(
                    event = "receipt_sign_failed",
                    error = %msg,
                    task_id = %task_id,
                    "fail-closed: receipt key missing or sign failed"
                );
                (TaskLifecycle::Failed, failed_result(&descriptor))
            }
        };

        {
            let mut g = self.inner.lock().await;
            g.running = g.running.saturating_sub(1);
            if let Some(rec) = g.tasks.get_mut(&task_id) {
                rec.lifecycle = lifecycle;
                rec.result = Some(result);
            }
        }
        drop(permit);
    }
}

fn finalize_result(
    config: &RunnerConfig,
    descriptor: &TaskDescriptorV1,
    outcome: crate::executor::ExecOutcome,
) -> Result<(TaskLifecycle, TaskResultV1), String> {
    let key = config
        .receipt_key
        .as_ref()
        .ok_or_else(|| "receipt signing key missing".to_owned())?;

    let patch_bytes: &[u8] = outcome.model_patch.as_deref().map_or(b"", str::as_bytes);
    let digest = patch_sha256(patch_bytes);
    let miner_hotkey = parse_hotkey_hex(&descriptor.miner_hotkey_hex)?;

    let body = WorkReceiptBodyV1 {
        challenge_id: descriptor.challenge_id.as_bytes().to_vec(),
        scoring_version: descriptor.scoring_version,
        epoch: descriptor.epoch,
        miner_hotkey,
        pack_id: descriptor.pack_id.as_bytes().to_vec(),
        patch_sha256: digest,
    };
    let signed = sign_work_receipt(key.secret(), body).map_err(|e| e.to_string())?;

    let lifecycle = match outcome.status {
        TaskStatusV1::Completed => TaskLifecycle::Completed,
        TaskStatusV1::TimedOut => TaskLifecycle::TimedOut,
        TaskStatusV1::Failed => TaskLifecycle::Failed,
    };

    // Timeout / failure: no patch body on the wire (even if digest is zero).
    let model_patch = match outcome.status {
        TaskStatusV1::Completed => outcome.model_patch,
        TaskStatusV1::TimedOut | TaskStatusV1::Failed => None,
    };

    Ok((
        lifecycle,
        TaskResultV1 {
            protocol: DISPATCH_PROTOCOL.into(),
            challenge_id: descriptor.challenge_id.clone(),
            scoring_version: descriptor.scoring_version,
            epoch: descriptor.epoch,
            miner_hotkey_hex: descriptor.miner_hotkey_hex.clone(),
            pack_id: descriptor.pack_id.clone(),
            status: outcome.status,
            model_patch,
            patch_sha256_hex: hex::encode(digest),
            receipt_sig_hex: hex::encode(signed.signature),
        },
    ))
}

fn failed_result(descriptor: &TaskDescriptorV1) -> TaskResultV1 {
    let empty = patch_sha256(b"");
    TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: descriptor.challenge_id.clone(),
        scoring_version: descriptor.scoring_version,
        epoch: descriptor.epoch,
        miner_hotkey_hex: descriptor.miner_hotkey_hex.clone(),
        pack_id: descriptor.pack_id.clone(),
        status: TaskStatusV1::Failed,
        model_patch: None,
        patch_sha256_hex: hex::encode(empty),
        receipt_sig_hex: String::new(),
    }
}

fn parse_hotkey_hex(hex_s: &str) -> Result<[u8; KEY_LEN], String> {
    let bytes = hex::decode(hex_s).map_err(|e| format!("miner_hotkey_hex: {e}"))?;
    if bytes.len() != KEY_LEN {
        return Err(format!(
            "miner_hotkey_hex must be {KEY_LEN} bytes, got {}",
            bytes.len()
        ));
    }
    let mut out = [0u8; KEY_LEN];
    out.copy_from_slice(&bytes);
    Ok(out)
}
