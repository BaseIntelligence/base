//! Miner-funded Lium keys (BYOK): vault + live-client factory.
//!
//! Keys are held in process memory and optionally in a TTL-bounded
//! encrypted seal directory (`PRISM_PAYER_VAULT_DIR` + key file). Default TTL
//! is ≥36h (train wall + eval + skew) and heartbeats re-seal so long GPU runs
//! survive control-plane restarts. Keys are never written to the submission
//! store and never logged. Master still SSHs with the operator keypair; the
//! miner key pays for rent/terminate only.

#![forbid(unsafe_code)]

mod sealed;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use prism_lium::{EvalJobBackend, LiumClient, LiumError, LiumSshConfig, LIUM_API_BASE_URL};

pub use sealed::{
    expiry_secs, recommended_ttl_secs, SealedVaultConfig, DEFAULT_TTL_SECS, DIR_ENV,
    EVAL_BUDGET_SECS, KEY_ENV, SEAL_SKEW_SECS, TRAIN_WALL_SECS, TTL_ENV,
};

/// In-memory map `submission_id → Lium API key`, with optional sealed persist.
pub struct PayerKeyVault {
    inner: Mutex<HashMap<String, String>>,
    sealed: Option<SealedVaultConfig>,
}

impl Default for PayerKeyVault {
    fn default() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            sealed: None,
        }
    }
}

impl fmt::Debug for PayerKeyVault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.inner.lock().map_or(0, |g| g.len());
        f.debug_struct("PayerKeyVault")
            .field("entries", &n)
            .field("sealed", &self.sealed.is_some())
            .finish()
    }
}

impl PayerKeyVault {
    /// Empty vault (memory only).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach sealed-disk backend (call before hydrate/insert).
    #[must_use]
    pub fn with_sealed(mut self, cfg: SealedVaultConfig) -> Self {
        self.sealed = Some(cfg);
        self
    }

    /// Build from env: sealed when dir+key present, else memory-only.
    #[must_use]
    pub fn from_env() -> Self {
        match SealedVaultConfig::from_env() {
            Some(cfg) => {
                let v = Self::new().with_sealed(cfg);
                let n = v.hydrate();
                tracing::info!(hydrated = n, "payer vault sealed backend enabled");
                v
            }
            None => Self::new(),
        }
    }

    /// Load unexpired seals into memory. Returns count hydrated.
    pub fn hydrate(&self) -> usize {
        let Some(cfg) = &self.sealed else {
            return 0;
        };
        let pairs = sealed::hydrate_all(cfg);
        let n = pairs.len();
        if let Ok(mut g) = self.inner.lock() {
            for (id, key) in pairs {
                g.insert(id, key);
            }
        }
        n
    }

    /// Insert or replace the payer key for `submission_id`.
    pub fn insert(&self, submission_id: impl Into<String>, api_key: impl Into<String>) {
        let id = submission_id.into();
        let key = api_key.into();
        if key.trim().is_empty() {
            return;
        }
        if let Some(cfg) = &self.sealed {
            if let Err(e) = sealed::persist(cfg, &id, &key) {
                tracing::warn!(error = %e, "payer seal persist failed");
            }
        }
        if let Ok(mut g) = self.inner.lock() {
            g.insert(id, key);
        }
    }

    /// Clone of the stored key, if any (memory, then sealed file).
    ///
    /// Measure / resume paths rely on this hydrate-from-seal fallback when the
    /// in-memory map is empty after a restart (as long as the seal is unexpired).
    #[must_use]
    pub fn get(&self, submission_id: &str) -> Option<String> {
        if let Ok(g) = self.inner.lock() {
            if let Some(k) = g.get(submission_id) {
                return Some(k.clone());
            }
        }
        let cfg = self.sealed.as_ref()?;
        let key = sealed::load(cfg, submission_id)?;
        if let Ok(mut g) = self.inner.lock() {
            g.insert(submission_id.to_owned(), key.clone());
        }
        Some(key)
    }

    /// Re-persist the seal with a fresh TTL window (memory and/or disk).
    ///
    /// Call before measure and on heartbeats so full-budget trains cannot
    /// outlive the on-disk seal. Returns `false` when no key is available.
    pub fn refresh(&self, submission_id: &str) -> bool {
        let Some(key) = self.get(submission_id) else {
            return false;
        };
        if let Some(cfg) = &self.sealed {
            if let Err(e) = sealed::persist(cfg, submission_id, &key) {
                tracing::warn!(error = %e, "payer seal refresh failed");
                return false;
            }
        }
        if let Ok(mut g) = self.inner.lock() {
            g.insert(submission_id.to_owned(), key);
        }
        true
    }

    /// Drop the key after the eval finishes (memory + seal).
    pub fn remove(&self, submission_id: &str) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(submission_id);
        }
        if let Some(cfg) = &self.sealed {
            sealed::remove(cfg, submission_id);
        }
    }
}

/// Builds a per-submission [`LiumClient`] from the vault (miner pays).
#[derive(Clone)]
pub struct PayerBackendFactory {
    /// Shared vault filled at intake.
    pub vault: Arc<PayerKeyVault>,
    /// Operator SSH config (private key path, retries) reused for every client.
    pub ssh: LiumSshConfig,
    /// Lium API base URL.
    pub base_url: String,
    /// When true and the vault misses, fall back to the process operator backend.
    pub allow_operator_fallback: bool,
}

impl fmt::Debug for PayerBackendFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PayerBackendFactory")
            .field("vault", &self.vault)
            .field("base_url", &self.base_url)
            .field("allow_operator_fallback", &self.allow_operator_fallback)
            .finish_non_exhaustive()
    }
}

impl PayerBackendFactory {
    /// Live factory (default Lium base URL).
    #[must_use]
    pub fn new(
        vault: Arc<PayerKeyVault>,
        ssh: LiumSshConfig,
        allow_operator_fallback: bool,
    ) -> Self {
        Self {
            vault,
            ssh,
            base_url: LIUM_API_BASE_URL.to_owned(),
            allow_operator_fallback,
        }
    }

    /// Resolve the backend that will bill this submission.
    ///
    /// # Errors
    /// Missing miner key (when fallback disabled) or client build failure.
    pub fn resolve(
        &self,
        submission_id: &str,
        operator: Arc<dyn EvalJobBackend>,
    ) -> Result<Arc<dyn EvalJobBackend>, String> {
        if let Some(key) = self.vault.get(submission_id) {
            let client = LiumClient::with_config(key, self.base_url.clone(), self.ssh.clone())
                .map_err(|e: LiumError| e.to_string())?;
            return Ok(Arc::new(client));
        }
        if self.allow_operator_fallback {
            return Ok(operator);
        }
        Err("miner Lium API key missing for this submission — resubmit with X-Lium-Api-Key".into())
    }
}

/// Header name miners use to fund their own pod.
pub const LIUM_API_KEY_HEADER: &str = "x-lium-api-key";

/// Extract `X-Lium-Api-Key` (case-insensitive header map lookup done by caller).
#[must_use]
pub fn normalize_lium_api_key(raw: &str) -> Option<String> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    Some(t.to_owned())
}

/// `PRISM_ALLOW_OPERATOR_LIUM=1` → operator key may bill when vault misses.
#[must_use]
pub fn allow_operator_lium_fallback() -> bool {
    matches!(
        std::env::var("PRISM_ALLOW_OPERATOR_LIUM").as_deref(),
        Ok("1" | "true" | "TRUE")
    )
}

/// Live Lium intake requires a miner key unless explicitly disabled.
#[must_use]
pub fn require_miner_lium(backend_mode: &str) -> bool {
    if !backend_mode.starts_with("lium") {
        return false;
    }
    !matches!(
        std::env::var("PRISM_REQUIRE_MINER_LIUM").as_deref(),
        Ok("0" | "false" | "FALSE")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn vault_roundtrip_and_redacted_debug() {
        let v = PayerKeyVault::new();
        v.insert("abc", "sk_test_secret");
        assert_eq!(v.get("abc").as_deref(), Some("sk_test_secret"));
        let dbg = format!("{v:?}");
        assert!(dbg.contains("entries"));
        assert!(!dbg.contains("sk_test_secret"));
        v.remove("abc");
        assert!(v.get("abc").is_none());
    }

    #[test]
    fn require_miner_lium_policy() {
        assert!(!require_miner_lium("sim"));
        assert!(!require_miner_lium("sim/openrouter"));
    }

    #[test]
    fn default_ttl_covers_train_wall() {
        const {
            assert!(DEFAULT_TTL_SECS >= TRAIN_WALL_SECS + EVAL_BUDGET_SECS + SEAL_SKEW_SECS);
        }
        assert!(recommended_ttl_secs() >= TRAIN_WALL_SECS);
        assert!(recommended_ttl_secs() >= 36 * 3600);
    }

    #[test]
    fn refresh_extends_seal_expiry() {
        let dir = tempdir().unwrap();
        let cfg = SealedVaultConfig {
            dir: dir.path().to_path_buf(),
            key: [3u8; 32],
            ttl_secs: 120,
        };
        let v = PayerKeyVault::new().with_sealed(cfg.clone());
        v.insert("job1", "sk_miner");
        let first = sealed::expiry_secs(&cfg, "job1").expect("sealed");
        // Advance perceived lifetime by re-sealing after a short sleep so the
        // unix expiry second is guaranteed to move forward under CI load.
        std::thread::sleep(std::time::Duration::from_secs(1));
        assert!(v.refresh("job1"));
        let second = sealed::expiry_secs(&cfg, "job1").expect("refreshed");
        assert!(
            second > first,
            "refresh must extend expiry (first={first} second={second})"
        );
        assert_eq!(v.get("job1").as_deref(), Some("sk_miner"));
    }

    #[test]
    fn measure_hydrates_from_seal_without_ram() {
        let dir = tempdir().unwrap();
        let cfg = SealedVaultConfig {
            dir: dir.path().to_path_buf(),
            key: [5u8; 32],
            ttl_secs: 3600,
        };
        sealed::persist(&cfg, "onlyseal", "sk_from_disk").unwrap();
        // Fresh vault: empty RAM, sealed backend only — same as post-restart
        // before hydrate_all, or when get() mid-run must reload from disk.
        let v = PayerKeyVault::new().with_sealed(cfg);
        assert_eq!(v.get("onlyseal").as_deref(), Some("sk_from_disk"));
        assert!(v.refresh("onlyseal"));
        // Clear RAM and prove seal-only hydrate still works.
        if let Ok(mut g) = v.inner.lock() {
            g.clear();
        }
        assert_eq!(v.get("onlyseal").as_deref(), Some("sk_from_disk"));
    }
}
