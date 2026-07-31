//! Sealed-surface admission for the hypertraining challenge.
//!
//! Miners may only change allowlisted training-code paths. Denylisted paths are
//! frozen by content hash. Selected accounting symbols are frozen by a simplified
//! AST fingerprint (SHA-256 of normalized function-body source).
//!
//! # AST hash approach (simplified)
//!
//! Full Python AST parsing is not required for unit admission. For each sealed
//! symbol name `S` in file `P`:
//!
//! 1. Locate `def S(` (or `def S `) at the start of a line (after optional indent).
//! 2. Take the function body as all subsequent lines until a line at the same or
//!    lower indentation that starts a new `def`/`class` (or EOF).
//! 3. Normalize: drop full-line `#` comments, trim each line, collapse internal
//!    runs of whitespace to a single space, join lines with `\n`, trim ends.
//! 4. `ast_hash = lowercase_hex(SHA-256(normalized_utf8))`.
//!
//! Pattern-style seals (loop condition, sample counter) use the same pipeline on
//! a synthetic one-line "body" extracted by a fixed marker string unique to that
//! seal (see [`crate::symbols::SEALED_SYMBOL_MARKERS`]).
//!
//! Normative pins: `te_version = 2.18.0+e7c550c5`,
//! `mlm_commit = cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54`.

#![forbid(unsafe_code)]

mod admit;
mod error;
mod hash;
mod manifest;
mod paths;
mod symbols;

pub use admit::{admit, AdmitInput};
pub use error::AdmitError;
pub use hash::{sha256_hex, verify_hex_sha256};
pub use manifest::{
    DatasetPin, SealedSurfaceV1, SegmentPin, DEFAULT_MLM_COMMIT, DEFAULT_TE_VERSION, MANIFEST_KIND,
};
pub use paths::{
    is_allowlisted, is_denylisted, normalize_path, DEFAULT_ALLOWLIST_GLOBS, DEFAULT_DENYLIST_GLOBS,
    DEFAULT_DENYLIST_PATHS,
};
pub use symbols::{
    extract_sealed_body, hash_default_symbols, sealed_symbol_ast_hash, DEFAULT_SEALED_SYMBOL_KEYS,
    SEALED_SYMBOL_MARKERS,
};
