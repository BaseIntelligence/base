//! Encrypted-at-rest BYOK seals (not the submission DB).
//!
//! ChaCha20-Poly1305 with a process key file. Plaintext never lands in Postgres.
//! Files are mode-0600 under `PRISM_PAYER_VAULT_DIR`.
//!
//! TTL must outlast a full train wall + eval + control-plane skew. Heartbeats
//! re-seal so mid-flight restarts never hydrate an expired file after a long
//! GPU run.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;
use sha2::{Digest, Sha256};

/// Recipe train wall, **derived from the recipe** rather than duplicated.
///
/// This constant used to be a hardcoded 6 h next to a hardcoded 2 h eval,
/// which is how the payer came to model an 8 h pod while
/// `prism_recipe::POD_LIFETIME_HOURS_CAP` said 7 h. Deriving it means the
/// dual-cap reconciliation (train 5 h, pod 7.5 h) cannot leave the payer
/// behind. Overridden by `PRISM_TRAIN_HOURS_CAP` when set.
// The recipe caps are small positive hour counts, so the second counts are
// exact in u64; the casts cannot truncate or lose a sign in practice.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const TRAIN_WALL_SECS: u64 = (prism_recipe::TRAIN_HOURS_CAP * 3600.0) as u64;

/// Post-train eval / harvest / terminate budget: everything the pod cap
/// reserves beyond the train wall, so the seal outlives whatever the
/// orchestrator is still allowed to do after training ends.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub const EVAL_BUDGET_SECS: u64 =
    ((prism_recipe::POD_LIFETIME_HOURS_CAP - prism_recipe::TRAIN_HOURS_CAP) * 3600.0) as u64;
/// Queue, pre-pod screens, restart skew, and clock margin.
pub const SEAL_SKEW_SECS: u64 = 4 * 3600;
/// Default TTL floor (≥36h): covers full-budget runs with substantial queue wait.
pub const DEFAULT_TTL_SECS: u64 = 36 * 3600;
/// Env: directory for `*.seal` files.
pub const DIR_ENV: &str = "PRISM_PAYER_VAULT_DIR";
/// Env: 32-byte key file (raw or 64-hex).
pub const KEY_ENV: &str = "PRISM_PAYER_VAULT_KEY_FILE";
/// Env: soft TTL seconds (floored by [`recommended_ttl_secs`] when unset).
pub const TTL_ENV: &str = "PRISM_PAYER_VAULT_TTL_SECS";

/// Soft TTL: `max(default_floor, train_wall + eval + skew)`.
///
/// `PRISM_TRAIN_HOURS_CAP` (hours, float) raises the train component when set.
#[must_use]
pub fn recommended_ttl_secs() -> u64 {
    let train = std::env::var("PRISM_TRAIN_HOURS_CAP")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|h| h.is_finite() && *h > 0.0)
        .map_or(TRAIN_WALL_SECS, |h| {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                (h * 3600.0).ceil() as u64
            }
        });
    let computed = train
        .saturating_add(EVAL_BUDGET_SECS)
        .saturating_add(SEAL_SKEW_SECS);
    DEFAULT_TTL_SECS.max(computed)
}

/// On-disk sealed vault settings.
#[derive(Debug, Clone)]
pub struct SealedVaultConfig {
    /// Directory for per-submission seal files.
    pub dir: PathBuf,
    /// 32-byte AEAD key.
    pub key: [u8; 32],
    /// Soft TTL seconds.
    pub ttl_secs: u64,
}

impl SealedVaultConfig {
    /// From env; `None` when dir or key file unset/unreadable (memory-only).
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var(DIR_ENV)
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        let key_path = std::env::var(KEY_ENV)
            .ok()
            .filter(|s| !s.trim().is_empty())?;
        let key = load_key(Path::new(&key_path))?;
        let floor = recommended_ttl_secs();
        let ttl_secs = std::env::var(TTL_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .map_or(floor, |configured: u64| configured.max(floor));
        let dir = PathBuf::from(dir);
        let _ = fs::create_dir_all(&dir);
        Some(Self { dir, key, ttl_secs })
    }
}

fn load_key(path: &Path) -> Option<[u8; 32]> {
    let raw = fs::read(path).ok()?;
    let t = String::from_utf8_lossy(&raw);
    let hex = t.trim();
    if hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        return Some(out);
    }
    if raw.len() == 32 {
        let mut out = [0u8; 32];
        out.copy_from_slice(&raw);
        return Some(out);
    }
    // Derive a 32-byte key from whatever was mounted (still not plaintext at rest).
    let mut out = [0u8; 32];
    out.copy_from_slice(&Sha256::digest(&raw));
    Some(out)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn seal_path(dir: &Path, submission_id: &str) -> PathBuf {
    // submission ids are hex; keep filename boring.
    let safe: String = submission_id
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(64)
        .collect();
    dir.join(format!("{safe}.seal"))
}

/// Encrypt + write seal file. Never logs key material.
pub fn persist(cfg: &SealedVaultConfig, submission_id: &str, api_key: &str) -> Result<(), String> {
    let expires = now_secs().saturating_add(cfg.ttl_secs);
    let payload = format!("{expires}\n{api_key}");
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let cipher = ChaCha20Poly1305::new_from_slice(&cfg.key).map_err(|e| e.to_string())?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher
        .encrypt(nonce, payload.as_bytes())
        .map_err(|_| "seal encrypt failed".to_string())?;
    let mut out = Vec::with_capacity(12 + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    let path = seal_path(&cfg.dir, submission_id);
    let tmp = path.with_extension("seal.tmp");
    {
        let mut f = fs::File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&out).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = f.set_permissions(fs::Permissions::from_mode(0o600));
        }
    }
    fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    Ok(())
}

/// Absolute unix expiry embedded in the seal, if present and decryptable.
#[must_use]
pub fn expiry_secs(cfg: &SealedVaultConfig, submission_id: &str) -> Option<u64> {
    let path = seal_path(&cfg.dir, submission_id);
    let bytes = fs::read(&path).ok()?;
    if bytes.len() < 13 {
        return None;
    }
    let (nonce_bytes, ct) = bytes.split_at(12);
    let cipher = ChaCha20Poly1305::new_from_slice(&cfg.key).ok()?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let pt = cipher.decrypt(nonce, ct).ok()?;
    let text = String::from_utf8(pt).ok()?;
    let (exp_s, _) = text.split_once('\n')?;
    exp_s.parse().ok()
}

/// Decrypt seal when present and unexpired.
pub fn load(cfg: &SealedVaultConfig, submission_id: &str) -> Option<String> {
    let path = seal_path(&cfg.dir, submission_id);
    let bytes = fs::read(&path).ok()?;
    if bytes.len() < 13 {
        return None;
    }
    let (nonce_bytes, ct) = bytes.split_at(12);
    let cipher = ChaCha20Poly1305::new_from_slice(&cfg.key).ok()?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let pt = cipher.decrypt(nonce, ct).ok()?;
    let text = String::from_utf8(pt).ok()?;
    let (exp_s, key) = text.split_once('\n')?;
    let exp: u64 = exp_s.parse().ok()?;
    if now_secs() > exp {
        let _ = fs::remove_file(&path);
        return None;
    }
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some(key.to_owned())
}

/// Delete seal file (best-effort).
pub fn remove(cfg: &SealedVaultConfig, submission_id: &str) {
    let _ = fs::remove_file(seal_path(&cfg.dir, submission_id));
}

/// Load every unexpired seal into `(submission_id, key)` pairs.
pub fn hydrate_all(cfg: &SealedVaultConfig) -> Vec<(String, String)> {
    let Ok(rd) = fs::read_dir(&cfg.dir) else {
        return vec![];
    };
    let mut out = Vec::new();
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("seal") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(key) = load(cfg, stem) {
            out.push((stem.to_owned(), key));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn recommended_ttl_covers_train_wall() {
        let ttl = recommended_ttl_secs();
        assert!(
            ttl >= TRAIN_WALL_SECS + EVAL_BUDGET_SECS + SEAL_SKEW_SECS,
            "ttl {ttl} must cover train+eval+skew"
        );
        assert!(
            ttl >= DEFAULT_TTL_SECS,
            "ttl {ttl} must be at least the 36h floor"
        );
        assert!(ttl >= 36 * 3600);
    }

    /// The payer's pod model must equal the recipe's, not merely resemble it.
    /// A silent disagreement here is how a seal expires mid-run (or a pod is
    /// killed mid-eval) without anything failing a test.
    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn payer_model_matches_the_recipe_pod_cap() {
        let recipe_pod_s = (prism_recipe::POD_LIFETIME_HOURS_CAP * 3600.0) as u64;
        assert_eq!(
            TRAIN_WALL_SECS + EVAL_BUDGET_SECS,
            recipe_pod_s,
            "payer train+eval must reconstruct POD_LIFETIME_HOURS_CAP exactly"
        );
        assert_eq!(
            TRAIN_WALL_SECS,
            (prism_recipe::TRAIN_HOURS_CAP * 3600.0) as u64
        );
        // The post-train reserve must still contain the eval phase ceiling.
        assert!(
            EVAL_BUDGET_SECS as f64 >= prism_recipe::HARNESS_EVAL_TIMEOUT_S,
            "post-train reserve {EVAL_BUDGET_SECS}s must cover the eval phase \
             ceiling {}s",
            prism_recipe::HARNESS_EVAL_TIMEOUT_S
        );
    }

    #[test]
    fn seal_roundtrip_and_ttl_expiry() {
        let dir = tempdir().unwrap();
        let cfg = SealedVaultConfig {
            dir: dir.path().to_path_buf(),
            key: [7u8; 32],
            ttl_secs: 3600,
        };
        persist(&cfg, "abc123", "sk_test_secret").unwrap();
        assert_eq!(load(&cfg, "abc123").as_deref(), Some("sk_test_secret"));
        let exp = expiry_secs(&cfg, "abc123").unwrap();
        assert!(exp > now_secs());
        let all = hydrate_all(&cfg);
        assert_eq!(all.len(), 1);
        remove(&cfg, "abc123");
        assert!(load(&cfg, "abc123").is_none());
    }

    #[test]
    fn refresh_extends_expiry() {
        let dir = tempdir().unwrap();
        let mut cfg = SealedVaultConfig {
            dir: dir.path().to_path_buf(),
            key: [9u8; 32],
            ttl_secs: 60,
        };
        persist(&cfg, "sub1", "sk_live").unwrap();
        let first = expiry_secs(&cfg, "sub1").unwrap();
        // Simulate a later refresh with a longer TTL window.
        cfg.ttl_secs = 3600;
        persist(&cfg, "sub1", "sk_live").unwrap();
        let second = expiry_secs(&cfg, "sub1").unwrap();
        assert!(
            second >= first + 3000,
            "refresh must push expiry forward (first={first} second={second})"
        );
        assert_eq!(load(&cfg, "sub1").as_deref(), Some("sk_live"));
    }
}
