//! Parse + fingerprint types.

use std::collections::{BTreeMap, BTreeSet};

use rustpython_parser::{ast::Suite, Parse};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::walk::collect_features;

/// Parse / fingerprint failure.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AstError {
    /// Source is not valid Python 3.
    #[error("parse: {0}")]
    Parse(String),
}

/// Structural fingerprint of one Python source unit.
///
/// Integer-friendly: bags/sets only. No floats. Stable across whitespace /
/// comment changes (parser strips those).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// Counts of AST node type names (`FunctionDef`, `Call`, …).
    pub node_counts: BTreeMap<String, u32>,
    /// Import module paths (`os.path`, `typing`, …).
    pub imports: BTreeSet<String>,
    /// Defined function / class names (as written).
    pub defs: BTreeSet<String>,
    /// Called names (`print`, `Model`, `self.forward` → `forward`).
    pub calls: BTreeSet<String>,
    /// SHA-256 of the ordered structural token stream (node type names).
    pub structure_digest: [u8; 32],
}

impl Fingerprint {
    /// Hex digest of the structure stream (audit / store).
    #[must_use]
    pub fn structure_hex(&self) -> String {
        hex_encode(&self.structure_digest)
    }
}

/// Fingerprint Python source. Comments/whitespace-insensitive.
///
/// # Errors
/// Parse failure.
pub fn fingerprint_source(source: &str) -> Result<Fingerprint, AstError> {
    let suite =
        Suite::parse(source, "<challenge-ast>").map_err(|e| AstError::Parse(e.to_string()))?;
    let feat = collect_features(&suite);
    let mut hasher = Sha256::new();
    for tok in &feat.structure_tokens {
        hasher.update(tok.as_bytes());
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    let mut structure_digest = [0u8; 32];
    structure_digest.copy_from_slice(&digest);
    Ok(Fingerprint {
        node_counts: feat.node_counts,
        imports: feat.imports,
        defs: feat.defs,
        calls: feat.calls,
        structure_digest,
    })
}

fn hex_encode(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn whitespace_insensitive() {
        let a = fingerprint_source("def f(x):\n    return x + 1\n").unwrap();
        let b = fingerprint_source("def f(x):\n\n  return x+1\n").unwrap();
        assert_eq!(a.structure_digest, b.structure_digest);
        assert_eq!(a.node_counts, b.node_counts);
    }

    #[test]
    fn parse_error() {
        assert!(matches!(
            fingerprint_source("def (\n"),
            Err(AstError::Parse(_))
        ));
    }
}
