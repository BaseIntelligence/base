//! Miner-funded Lium keys (BYOK): vault + live-client factory.
//!
//! Keys are held only in process memory, never logged, never written to the
//! submission store. Master still SSHs with the operator keypair; the miner
//! key pays for rent/terminate only.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{EvalJobBackend, LiumClient, LiumError, LiumSshConfig, LIUM_API_BASE_URL};

/// In-memory map `submission_id → Lium API key`.
#[derive(Default)]
pub struct PayerKeyVault {
    inner: Mutex<HashMap<String, String>>,
}

impl fmt::Debug for PayerKeyVault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.inner.lock().map(|g| g.len()).unwrap_or(0);
        f.debug_struct("PayerKeyVault")
            .field("entries", &n)
            .finish()
    }
}

impl PayerKeyVault {
    /// Empty vault.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the payer key for `submission_id`.
    pub fn insert(&self, submission_id: impl Into<String>, api_key: impl Into<String>) {
        let key = api_key.into();
        if key.trim().is_empty() {
            return;
        }
        if let Ok(mut g) = self.inner.lock() {
            g.insert(submission_id.into(), key);
        }
    }

    /// Clone of the stored key, if any.
    #[must_use]
    pub fn get(&self, submission_id: &str) -> Option<String> {
        self.inner
            .lock()
            .ok()
            .and_then(|g| g.get(submission_id).cloned())
    }

    /// Drop the key after the eval finishes (best-effort hygiene).
    pub fn remove(&self, submission_id: &str) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(submission_id);
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
        Ok("1") | Ok("true") | Ok("TRUE")
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
        Ok("0") | Ok("false") | Ok("FALSE")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
