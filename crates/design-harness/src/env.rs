//! Miner-provided launch environment (sandbox run phase only).

use std::collections::BTreeMap;

use thiserror::Error;

/// Reserved `extra_files` key used to persist env without a schema migration.
pub const ENV_EXTRA_KEY: &str = "__miner_env_json__";

/// Max env keys per harness.
pub const MAX_ENV_KEYS: usize = 32;

/// Max key length.
pub const MAX_ENV_KEY_LEN: usize = 64;

/// Max value bytes per key.
pub const MAX_ENV_VALUE_LEN: usize = 8 * 1024;

/// Max total env payload bytes.
pub const MAX_ENV_TOTAL_BYTES: usize = 64 * 1024;

/// Prefixes / names miners may not override.
const BLOCKED_PREFIXES: &[&str] = &[
    "DESIGN_",
    "HTTP_",
    "HTTPS_",
    "ALL_PROXY",
    "NO_PROXY",
    "PYTHON",
    "LD_",
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "TEMP",
    "TMP",
    "DOCKER",
    "BASE_",
];

/// Env validation errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EnvError {
    /// Structural validation failed.
    #[error("invalid env: {0}")]
    Invalid(String),
}

/// Validate miner env map (never log values — caller responsibility).
///
/// # Errors
/// Shape / allowlist failures.
pub fn validate_env_vars(env: &BTreeMap<String, String>) -> Result<(), EnvError> {
    if env.len() > MAX_ENV_KEYS {
        return Err(EnvError::Invalid(format!(
            "too many env keys (max {MAX_ENV_KEYS})"
        )));
    }
    let mut total = 0usize;
    for (k, v) in env {
        if k.is_empty() || k.len() > MAX_ENV_KEY_LEN {
            return Err(EnvError::Invalid(format!("bad env key length: {k}")));
        }
        if !k
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            || !k.chars().next().is_some_and(|c| c.is_ascii_uppercase())
        {
            return Err(EnvError::Invalid(format!(
                "env key must match [A-Z][A-Z0-9_]*: {k}"
            )));
        }
        let ku = k.to_ascii_uppercase();
        if BLOCKED_PREFIXES
            .iter()
            .any(|p| ku == *p || ku.starts_with(p))
        {
            return Err(EnvError::Invalid(format!("env key blocked: {k}")));
        }
        if v.len() > MAX_ENV_VALUE_LEN {
            return Err(EnvError::Invalid(format!("env value too large: {k}")));
        }
        if v.bytes().any(|b| b == 0) {
            return Err(EnvError::Invalid(format!("env value has NUL: {k}")));
        }
        total = total.saturating_add(k.len()).saturating_add(v.len());
    }
    if total > MAX_ENV_TOTAL_BYTES {
        return Err(EnvError::Invalid("env payload exceeds 64 KiB".into()));
    }
    Ok(())
}

/// Serialize env into reserved `extra_files` slot (values never logged here).
pub fn encode_env_into_extras(
    extras: &mut BTreeMap<String, String>,
    env: &BTreeMap<String, String>,
) {
    extras.remove(ENV_EXTRA_KEY);
    if env.is_empty() {
        return;
    }
    if let Ok(s) = serde_json::to_string(env) {
        extras.insert(ENV_EXTRA_KEY.into(), s);
    }
}

/// Extract + remove reserved env slot from extras.
#[must_use]
pub fn decode_env_from_extras(extras: &mut BTreeMap<String, String>) -> BTreeMap<String, String> {
    let Some(raw) = extras.remove(ENV_EXTRA_KEY) else {
        return BTreeMap::new();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

/// Peek env without mutating extras.
#[must_use]
pub fn peek_env_from_extras(extras: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    extras
        .get(ENV_EXTRA_KEY)
        .and_then(|raw| serde_json::from_str(raw).ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_api_key_style() {
        let mut m = BTreeMap::new();
        m.insert("OPENAI_API_KEY".into(), "sk-test".into());
        validate_env_vars(&m).unwrap();
    }

    #[test]
    fn rejects_design_prefix() {
        let mut m = BTreeMap::new();
        m.insert("DESIGN_PROMPT".into(), "x".into());
        assert!(validate_env_vars(&m).is_err());
    }

    #[test]
    fn roundtrip_extras() {
        let mut env = BTreeMap::new();
        env.insert("FOO".into(), "bar".into());
        let mut extras = BTreeMap::new();
        encode_env_into_extras(&mut extras, &env);
        assert_eq!(decode_env_from_extras(&mut extras), env);
        assert!(!extras.contains_key(ENV_EXTRA_KEY));
    }
}
