//! Bind the miner's work-receipt public key to the TDX measurement.
//!
//! The `report_data` binding (D10) has five frozen fields and carries no key,
//! so the only way a validator can learn a miner's receipt key without trusting
//! the miner is to read it out of the **measured** `app-compose.json`. RTMR3
//! only measures the compose *digest*, so the preimage the miner ships must be
//! re-hashed and compared against that digest before a single byte of it is
//! read.

use compose_hash::{compose_hash, ComposeHash};
use crypto::KEY_LEN;
use serde_json::Value;
use thiserror::Error;

/// Env var published in the measured compose that carries the receipt key.
pub const RECEIPT_PUBLIC_KEY_ENV: &str = "BASE_RECEIPT_PUBLIC_KEY";

/// Failures while binding a compose preimage to a measurement.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ComposeBindError {
    /// The supplied bytes do not hash to the measured compose hash.
    #[error("app_compose does not hash to the measured compose hash")]
    PreimageMismatch,
    /// The measured compose publishes no receipt key.
    #[error("app_compose publishes no {RECEIPT_PUBLIC_KEY_ENV}")]
    ReceiptKeyMissing,
    /// The published key is not one 32-byte hex value.
    #[error("{RECEIPT_PUBLIC_KEY_ENV} is not a single 32-byte hex key")]
    ReceiptKeyMalformed,
}

/// Whether `app_compose` is the preimage of `measured`.
///
/// Bytes that are not valid JSON can never be a compose preimage, so they fail
/// the same way a wrong document does.
///
/// # Errors
///
/// [`ComposeBindError::PreimageMismatch`] when the digests differ.
pub fn verify_compose_preimage(
    app_compose: &[u8],
    measured: &ComposeHash,
) -> Result<(), ComposeBindError> {
    let actual = compose_hash(app_compose).map_err(|_| ComposeBindError::PreimageMismatch)?;
    if actual == *measured {
        Ok(())
    } else {
        Err(ComposeBindError::PreimageMismatch)
    }
}

/// Recover [`RECEIPT_PUBLIC_KEY_ENV`] from a compose preimage that has already
/// been checked with [`verify_compose_preimage`].
///
/// The key is set on the agent service inside the `docker_compose_file` string;
/// `allowed_envs` only lists the name, so scanning that string is what binds a
/// value. Every occurrence must agree: two services publishing different keys
/// would leave the receipt binding ambiguous.
///
/// # Errors
///
/// [`ComposeBindError::ReceiptKeyMissing`] when absent, or
/// [`ComposeBindError::ReceiptKeyMalformed`] when it is not a single 32-byte
/// hex value.
pub fn receipt_key_from_compose(app_compose: &[u8]) -> Result<[u8; KEY_LEN], ComposeBindError> {
    let value: Value =
        serde_json::from_slice(app_compose).map_err(|_| ComposeBindError::ReceiptKeyMissing)?;
    let yaml = value
        .get("docker_compose_file")
        .and_then(Value::as_str)
        .ok_or(ComposeBindError::ReceiptKeyMissing)?;

    let mut found: Option<[u8; KEY_LEN]> = None;
    for line in yaml.lines() {
        let Some(rest) = line.trim_start().strip_prefix(RECEIPT_PUBLIC_KEY_ENV) else {
            continue;
        };
        let Some(raw) = rest.strip_prefix(':') else {
            continue;
        };
        let key = decode_key(raw.trim().trim_matches(|c| c == '"' || c == '\''))
            .ok_or(ComposeBindError::ReceiptKeyMalformed)?;
        if found.is_some_and(|prev| prev != key) {
            return Err(ComposeBindError::ReceiptKeyMalformed);
        }
        found = Some(key);
    }
    found.ok_or(ComposeBindError::ReceiptKeyMissing)
}

fn decode_key(s: &str) -> Option<[u8; KEY_LEN]> {
    let bytes = hex::decode(s).ok()?;
    let mut out = [0_u8; KEY_LEN];
    if bytes.len() != KEY_LEN {
        return None;
    }
    out.copy_from_slice(&bytes);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compose_with(env_line: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "manifest_version": 2,
            "name": "base-miner",
            "docker_compose_file": format!("services:\n  agent:\n    environment:\n      {env_line}\n"),
        }))
        .expect("json")
    }

    #[test]
    fn extracts_quoted_key() {
        let pk = [0x5a_u8; KEY_LEN];
        let bytes = compose_with(&format!(
            "{RECEIPT_PUBLIC_KEY_ENV}: \"{}\"",
            hex::encode(pk)
        ));
        assert_eq!(receipt_key_from_compose(&bytes), Ok(pk));
    }

    #[test]
    fn missing_key_is_missing_not_malformed() {
        let bytes = compose_with("BASE_NETUID: \"1\"");
        assert_eq!(
            receipt_key_from_compose(&bytes),
            Err(ComposeBindError::ReceiptKeyMissing)
        );
    }

    #[test]
    fn short_key_is_malformed() {
        let bytes = compose_with(&format!("{RECEIPT_PUBLIC_KEY_ENV}: \"deadbeef\""));
        assert_eq!(
            receipt_key_from_compose(&bytes),
            Err(ComposeBindError::ReceiptKeyMalformed)
        );
    }

    #[test]
    fn two_disagreeing_keys_are_malformed() {
        let bytes = serde_json::to_vec(&serde_json::json!({
            "docker_compose_file": format!(
                "services:\n  a:\n    environment:\n      {RECEIPT_PUBLIC_KEY_ENV}: \"{}\"\n  b:\n    environment:\n      {RECEIPT_PUBLIC_KEY_ENV}: \"{}\"\n",
                hex::encode([1_u8; KEY_LEN]),
                hex::encode([2_u8; KEY_LEN]),
            ),
        }))
        .expect("json");
        assert_eq!(
            receipt_key_from_compose(&bytes),
            Err(ComposeBindError::ReceiptKeyMalformed)
        );
    }

    #[test]
    fn preimage_check_rejects_a_flipped_byte() {
        let pk = [0x11_u8; KEY_LEN];
        let good = compose_with(&format!(
            "{RECEIPT_PUBLIC_KEY_ENV}: \"{}\"",
            hex::encode(pk)
        ));
        let measured = compose_hash(&good).expect("hash");
        assert_eq!(verify_compose_preimage(&good, &measured), Ok(()));

        let attacker = compose_with(&format!(
            "{RECEIPT_PUBLIC_KEY_ENV}: \"{}\"",
            hex::encode([0xee_u8; KEY_LEN])
        ));
        assert_eq!(
            verify_compose_preimage(&attacker, &measured),
            Err(ComposeBindError::PreimageMismatch)
        );
    }

    #[test]
    fn non_json_is_a_preimage_mismatch() {
        let measured = compose_hash(&compose_with("BASE_NETUID: \"1\"")).expect("hash");
        assert_eq!(
            verify_compose_preimage(b"not json", &measured),
            Err(ComposeBindError::PreimageMismatch)
        );
    }
}
