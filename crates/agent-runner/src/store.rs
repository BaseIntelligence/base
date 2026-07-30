//! In-memory task store: dispatch auth state + stub executor with receipt signing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_dispatch::{
    patch_sha256, sign_work_receipt, TaskDescriptorV1, TaskResultV1, TaskStatusV1,
    WorkReceiptBodyV1, DISPATCH_PROTOCOL,
};
use crypto::{MemoryNonceStore, KEY_LEN};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::api::CapacityResponse;
use crate::auth::{
    verify_and_consume_dispatch, DispatchAuthError, SignedDispatchRequest,
    DEFAULT_DISPATCH_NONCE_TTL,
};
use crate::receipt_key::ReceiptKey;

/// Runner process configuration (env-driven at the binary boundary).
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Advertised max concurrent tasks (todo 19 will clamp/enforce).
    pub max_concurrency: u32,
    /// When true (default), `POST /v1/task` requires a signed dispatch envelope.
    pub auth_enabled: bool,
    /// Trusted challenge public key for dispatch auth (required when auth on).
    pub trusted_challenge_pubkey: Option<[u8; KEY_LEN]>,
    /// Max dispatch-auth TTL (must stay below one epoch).
    pub dispatch_nonce_ttl: Duration,
    /// CVM-local work-receipt key. Required to complete tasks with a real sig.
    pub receipt_key: Option<ReceiptKey>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 1,
            auth_enabled: true,
            trusted_challenge_pubkey: None,
            dispatch_nonce_ttl: DEFAULT_DISPATCH_NONCE_TTL,
            receipt_key: None,
        }
    }
}

/// Lifecycle status exposed on `GET /v1/task/{id}` (broader than wire terminal status).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskLifecycle {
    /// Accepted, not yet started.
    Pending,
    /// Stub/real executor holds a slot.
    Running,
    /// Finished successfully (result present).
    Completed,
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
    inner: Arc<Mutex<StoreInner>>,
}

struct StoreInner {
    tasks: HashMap<String, TaskRecord>,
    /// Tasks currently in [`TaskLifecycle::Running`].
    running: u32,
    /// Single-use dispatch nonces (todo 18).
    dispatch_nonces: MemoryNonceStore,
}

impl RunnerState {
    /// New empty store with the given configuration.
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self {
            config,
            inner: Arc::new(Mutex::new(StoreInner {
                tasks: HashMap::new(),
                running: 0,
                dispatch_nonces: MemoryNonceStore::new(),
            })),
        }
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
        let load = self
            .inner
            .try_lock()
            .map(|g| g.running)
            .unwrap_or(0);
        CapacityResponse {
            max_concurrency: self.config.max_concurrency,
            current_load: load,
        }
    }

    /// Async capacity (accurate under load).
    pub async fn capacity_async(&self) -> CapacityResponse {
        let g = self.inner.lock().await;
        CapacityResponse {
            max_concurrency: self.config.max_concurrency,
            current_load: g.running,
        }
    }

    /// Enqueue descriptor, spawn stub executor, return assigned task id.
    pub async fn accept_task(&self, descriptor: TaskDescriptorV1) -> String {
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
        }
        let state = self.clone();
        let id = task_id.clone();
        tokio::spawn(async move {
            state.run_stub(id, descriptor).await;
        });
        task_id
    }

    /// Lookup task view fields.
    pub async fn get_task(
        &self,
        id: &str,
    ) -> Option<(TaskLifecycle, Option<TaskResultV1>)> {
        let g = self.inner.lock().await;
        g.tasks
            .get(id)
            .map(|r| (r.lifecycle, r.result.clone()))
    }

    async fn run_stub(&self, task_id: String, descriptor: TaskDescriptorV1) {
        {
            let mut g = self.inner.lock().await;
            if let Some(rec) = g.tasks.get_mut(&task_id) {
                rec.lifecycle = TaskLifecycle::Running;
                g.running = g.running.saturating_add(1);
            }
        }

        tokio::task::yield_now().await;

        let (lifecycle, result) = match build_signed_result(&self.config, &descriptor) {
            Ok(result) => (TaskLifecycle::Completed, result),
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

        let mut g = self.inner.lock().await;
        let was_running = g
            .tasks
            .get(&task_id)
            .is_some_and(|r| r.lifecycle == TaskLifecycle::Running);
        if was_running {
            g.running = g.running.saturating_sub(1);
        }
        if let Some(rec) = g.tasks.get_mut(&task_id) {
            rec.lifecycle = lifecycle;
            rec.result = Some(result);
        }
    }
}

fn build_signed_result(
    config: &RunnerConfig,
    descriptor: &TaskDescriptorV1,
) -> Result<TaskResultV1, String> {
    let key = config
        .receipt_key
        .as_ref()
        .ok_or_else(|| "receipt signing key missing".to_owned())?;

    let patch = stub_model_patch(&descriptor.pack_id);
    let digest = patch_sha256(patch.as_bytes());
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

    Ok(TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: descriptor.challenge_id.clone(),
        scoring_version: descriptor.scoring_version,
        epoch: descriptor.epoch,
        miner_hotkey_hex: descriptor.miner_hotkey_hex.clone(),
        pack_id: descriptor.pack_id.clone(),
        status: TaskStatusV1::Completed,
        model_patch: Some(patch),
        patch_sha256_hex: hex::encode(digest),
        receipt_sig_hex: hex::encode(signed.signature),
    })
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

fn stub_model_patch(pack_id: &str) -> String {
    format!(
        "diff --git a/README.md b/README.md\n\
         --- a/README.md\n\
         +++ b/README.md\n\
         @@ -1 +1,2 @@\n\
          # pack\n\
         +stub patch for {pack_id}\n"
    )
}
