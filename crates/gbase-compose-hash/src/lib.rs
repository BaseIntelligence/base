//! Canonical `app-compose.json` hashing and TDX `mr_config_id` (v1).
//!
//! Mirrors dstack / Phala `get_compose_hash` and gm-miner `compose_hash`:
//! - parse JSON
//! - recursively lexicographically sort object keys
//! - compact separators (`,` / `:`, no spaces)
//! - UTF-8 non-ASCII left unescaped (`ensure_ascii=false`)
//! - no trailing newline
//! - `SHA-256` over those exact UTF-8 bytes
//!
//! `mr_config_id` v1 (48 bytes): `0x01 ‖ compose_hash (32) ‖ 0x00×15`.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha256};
use thiserror::Error;

/// SHA-256 digest width / compose hash length.
pub const COMPOSE_HASH_LEN: usize = 32;
/// TDX `MRCONFIGID` width.
pub const MR_CONFIG_ID_LEN: usize = 48;
/// Version byte for compose-hash-only `mr_config_id`.
pub const MR_CONFIG_ID_V1: u8 = 0x01;
/// Padding after version + hash in v1 layout.
pub const MR_CONFIG_ID_V1_PAD_LEN: usize = 15;

/// 32-byte compose hash (SHA-256 of canonical app-compose JSON).
pub type ComposeHash = [u8; COMPOSE_HASH_LEN];
/// 48-byte TDX `mr_config_id`.
pub type MrConfigId = [u8; MR_CONFIG_ID_LEN];

/// Failures while hashing compose JSON.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ComposeHashError {
    /// Input is not valid JSON.
    #[error("invalid app-compose JSON: {0}")]
    InvalidJson(String),
}

/// Canonicalize `app-compose.json` bytes and return `SHA-256` as 32 raw bytes.
///
/// Key order and insignificant whitespace in the input do not affect the digest.
///
/// # Errors
/// Returns [`ComposeHashError::InvalidJson`] when `bytes` is not valid JSON.
pub fn compose_hash(bytes: &[u8]) -> Result<ComposeHash, ComposeHashError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| ComposeHashError::InvalidJson(e.to_string()))?;
    let canonical = canonical_json(&value);
    let digest = Sha256::digest(canonical.as_bytes());
    Ok(digest.into())
}

/// Lowercase hex encoding of [`compose_hash`] (64 hex chars).
///
/// # Errors
/// Same as [`compose_hash`].
pub fn compose_hash_hex(bytes: &[u8]) -> Result<String, ComposeHashError> {
    Ok(hex_encode(&compose_hash(bytes)?))
}

/// Build TDX `mr_config_id` v1 from a compose hash: `0x01 ‖ hash ‖ 15×0x00`.
#[must_use]
pub fn mr_config_id(hash: &ComposeHash) -> MrConfigId {
    let mut out = [0_u8; MR_CONFIG_ID_LEN];
    out[0] = MR_CONFIG_ID_V1;
    out[1..=COMPOSE_HASH_LEN].copy_from_slice(hash);
    out
}

/// Convenience: `compose_hash` then [`mr_config_id`].
///
/// # Errors
/// Same as [`compose_hash`].
pub fn mr_config_id_from_compose(bytes: &[u8]) -> Result<MrConfigId, ComposeHashError> {
    Ok(mr_config_id(&compose_hash(bytes)?))
}

/// Deterministic JSON string matching dstack cross-language rules.
#[must_use]
pub fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical(&mut out, value);
    out
}

fn write_canonical(out: &mut String, value: &serde_json::Value) {
    match value {
        serde_json::Value::Null => out.push_str("null"),
        serde_json::Value::Bool(b) => {
            if *b {
                out.push_str("true");
            } else {
                out.push_str("false");
            }
        }
        serde_json::Value::Number(n) => out.push_str(&n.to_string()),
        serde_json::Value::String(s) => write_json_string(out, s),
        serde_json::Value::Array(arr) => {
            out.push('[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(out, item);
            }
            out.push(']');
        }
        // serde_json::Map is a BTreeMap — keys already lexicographically sorted.
        serde_json::Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(out, k);
                out.push(':');
                write_canonical(out, v);
            }
            out.push('}');
        }
    }
}

/// JSON string with control escapes; non-ASCII left as UTF-8 (`ensure_ascii=false`).
fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python/dstack fixture: `{"a":1,"b":2}` → known SHA-256.
    const SIMPLE_HEX: &str = "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777";

    /// Nested + UTF-8 fixture from Python `json.dumps(..., ensure_ascii=False)`.
    const NESTED_UTF8_HEX: &str =
        "ab25a1b1ffa8354d6c12efa29d1c689c3caf09d2710e21776b652c6fb894ca65";

    #[test]
    fn compose_hash_matches_known_simple_digest() {
        let h = compose_hash(br#"{"a":1,"b":2}"#).expect("valid json");
        assert_eq!(hex_encode(&h), SIMPLE_HEX);
        assert_eq!(
            compose_hash_hex(br#"{"a":1,"b":2}"#).expect("hex"),
            SIMPLE_HEX
        );
    }

    #[test]
    fn key_reordering_does_not_change_hash() {
        let a = compose_hash(br#"{"b":2,"a":1}"#).expect("a");
        let b = compose_hash(br#"{"a":1,"b":2}"#).expect("b");
        assert_eq!(a, b, "lexicographic key sort must stabilize hash");
        assert_eq!(hex_encode(&a), SIMPLE_HEX);
    }

    #[test]
    fn whitespace_does_not_change_hash() {
        let compact = compose_hash(br#"{"a":1,"b":2}"#).expect("compact");
        let pretty = compose_hash(
            br#"{
  "a": 1,
  "b": 2
}"#,
        )
        .expect("pretty");
        let spaced = compose_hash(br#"{ "a" : 1 , "b" : 2 }"#).expect("spaced");
        assert_eq!(compact, pretty);
        assert_eq!(compact, spaced);
    }

    #[test]
    fn nested_object_keys_are_sorted_recursively() {
        // Canonical: {"a":{"b":3,"y":2},"m":"你好","z":1}
        let reordered = r#"{"z":1,"m":"你好","a":{"y":2,"b":3}}"#.as_bytes();
        let pretty = r#"{
  "z": 1,
  "a": { "y": 2, "b": 3 },
  "m": "你好"
}"#
        .as_bytes();
        let h1 = compose_hash(reordered).expect("reordered");
        let h2 = compose_hash(pretty).expect("pretty");
        assert_eq!(h1, h2);
        assert_eq!(hex_encode(&h1), NESTED_UTF8_HEX);
    }

    #[test]
    fn canonical_json_is_compact_sorted_utf8_no_trailing_newline() {
        let v: serde_json::Value =
            serde_json::from_str(r#"{"z":1,"a":{"y":2,"b":3},"m":"你好"}"#).unwrap();
        let c = canonical_json(&v);
        assert_eq!(c, r#"{"a":{"b":3,"y":2},"m":"你好","z":1}"#);
        assert!(!c.ends_with('\n'));
        assert!(!c.contains(": "));
        assert!(!c.contains(", "));
        assert!(c.contains("你好"), "non-ASCII must not be \\u-escaped");
    }

    #[test]
    fn invalid_json_returns_error() {
        let err = compose_hash(br#"{"a":"#).expect_err("must fail");
        assert!(matches!(err, ComposeHashError::InvalidJson(_)));
    }

    #[test]
    fn mr_config_id_v1_layout() {
        let hash = compose_hash(br#"{"a":1,"b":2}"#).expect("hash");
        let mr = mr_config_id(&hash);
        assert_eq!(mr.len(), MR_CONFIG_ID_LEN);
        assert_eq!(mr[0], MR_CONFIG_ID_V1);
        assert_eq!(&mr[1..33], &hash);
        assert_eq!(&mr[33..], &[0_u8; MR_CONFIG_ID_V1_PAD_LEN]);
        let via = mr_config_id_from_compose(br#"{"b":2,"a":1}"#).expect("via");
        assert_eq!(mr, via);
    }

    #[test]
    fn empty_object_hashes_stably() {
        let h = compose_hash(b"{}").expect("empty");
        // sha256 of "{}"
        assert_eq!(
            hex_encode(&h),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        );
    }
}
