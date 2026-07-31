//! Admission entrypoint.

use std::collections::BTreeMap;

use crate::error::AdmitError;
use crate::hash::verify_hex_sha256;
use crate::manifest::{SealedSurfaceV1, DEFAULT_MLM_COMMIT, DEFAULT_TE_VERSION, MANIFEST_KIND};
use crate::paths::{is_allowlisted, is_denylisted, normalize_path};
use crate::symbols::{sealed_symbol_ast_hash, split_symbol_key};

/// Inputs to [`admit`].
///
/// `changed_paths` are repo-relative paths that differ from `base_commit`.
/// `file_contents` must include every denylist path in the manifest and every
/// file referenced by sealed-symbol keys (typically the full tree slice needed
/// for those checks). Values are raw file bytes (UTF-8 source for Python).
#[derive(Debug, Clone)]
pub struct AdmitInput<'a> {
    /// Paths changed relative to the sealed base.
    pub changed_paths: &'a [String],
    /// Path → file bytes for hash / symbol checks.
    pub file_contents: &'a BTreeMap<String, Vec<u8>>,
    /// Sealed surface manifest.
    pub manifest: &'a SealedSurfaceV1,
}

/// Admit a miner fork against the sealed surface.
///
/// Rejects when:
/// - a changed path is denylisted
/// - a changed path is outside the allowlist
/// - any denylist content hash mismatches the manifest
/// - any sealed-symbol simplified AST hash mismatches the manifest
///
/// # Errors
///
/// Returns [`AdmitError`] variants describing the first failure found.
pub fn admit(input: &AdmitInput<'_>) -> Result<(), AdmitError> {
    validate_manifest_pins(input.manifest)?;

    for raw in input.changed_paths {
        let path = normalize_path(raw);
        if is_denylisted(&path) {
            return Err(AdmitError::DenylistPathTouched { path });
        }
        if !is_allowlisted(&path) {
            return Err(AdmitError::PathNotAllowlisted { path });
        }
    }

    check_denylist_hashes(input.manifest, input.file_contents)?;
    check_sealed_symbols(input.manifest, input.file_contents)?;
    Ok(())
}

fn validate_manifest_pins(m: &SealedSurfaceV1) -> Result<(), AdmitError> {
    if m.kind != MANIFEST_KIND {
        return Err(AdmitError::UnsupportedManifestKind {
            kind: m.kind.clone(),
        });
    }
    if m.mlm_commit != DEFAULT_MLM_COMMIT {
        return Err(AdmitError::PinMismatch {
            field: "mlm_commit",
            expected: DEFAULT_MLM_COMMIT.to_owned(),
            got: m.mlm_commit.clone(),
        });
    }
    if m.te_version != DEFAULT_TE_VERSION {
        return Err(AdmitError::PinMismatch {
            field: "te_version",
            expected: DEFAULT_TE_VERSION.to_owned(),
            got: m.te_version.clone(),
        });
    }
    Ok(())
}

fn check_denylist_hashes(
    m: &SealedSurfaceV1,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<(), AdmitError> {
    for (raw_path, expected) in &m.denylist_hashes {
        let path = normalize_path(raw_path);
        let bytes = files
            .get(&path)
            .ok_or_else(|| AdmitError::MissingFileContent { path: path.clone() })?;
        verify_hex_sha256(bytes, expected, &path)?;
    }
    Ok(())
}

fn check_sealed_symbols(
    m: &SealedSurfaceV1,
    files: &BTreeMap<String, Vec<u8>>,
) -> Result<(), AdmitError> {
    for (key, expected) in &m.sealed_symbols {
        let (path, _symbol) = split_symbol_key(key)?;
        let bytes = files
            .get(&path)
            .ok_or_else(|| AdmitError::MissingFileContent { path: path.clone() })?;
        let source = std::str::from_utf8(bytes)
            .map_err(|_| AdmitError::SealedSymbolNotFound { key: key.clone() })?;
        let actual = sealed_symbol_ast_hash(key, source)?;
        if !hex_eq(&actual, expected) {
            return Err(AdmitError::SealedSymbolMismatch { key: key.clone() });
        }
    }
    Ok(())
}

fn hex_eq(actual: &str, expected: &str) -> bool {
    let exp = expected.trim();
    let exp = exp
        .strip_prefix("0x")
        .or_else(|| exp.strip_prefix("0X"))
        .unwrap_or(exp);
    actual.eq_ignore_ascii_case(exp)
}
