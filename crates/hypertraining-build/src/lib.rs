//! Hermetic offline build for hypertraining (brief §8 step 2).
//!
//! Builds use the **validator** lock and wheelhouse only — never the miner's lock —
//! and produce an immutable image digest. [`FixtureBuilder`] is a pure offline
//! backend: no network, no registry, digest = SHA-256 of canonical contents.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Domain tag for build digests (distinct from agent / other hypertraining domains).
pub const BUILD_DIGEST_DOMAIN: &[u8] = b"base-hypertraining-build-v1";

/// Prefix for OCI-style digest strings produced by this crate.
pub const DIGEST_PREFIX: &str = "sha256:";

/// Admitted source tree after sealed-surface admission (path → bytes).
///
/// Paths are stored in a [`BTreeMap`] so iteration order is canonical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedSource {
    files: BTreeMap<String, Vec<u8>>,
}

impl AdmittedSource {
    /// Build from path/content pairs. Empty path keys are rejected.
    ///
    /// # Errors
    /// [`BuildError::EmptySource`] when `files` is empty.
    /// [`BuildError::InvalidPath`] when any path is empty.
    pub fn new(files: impl IntoIterator<Item = (String, Vec<u8>)>) -> Result<Self, BuildError> {
        let mut map = BTreeMap::new();
        for (path, bytes) in files {
            if path.is_empty() {
                return Err(BuildError::InvalidPath);
            }
            map.insert(path, bytes);
        }
        if map.is_empty() {
            return Err(BuildError::EmptySource);
        }
        Ok(Self { files: map })
    }

    /// Number of files in the admitted tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the admitted tree has no files (should not occur after [`Self::new`]).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Sorted path keys.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }
}

/// Validator-owned dependency lock bytes (required for hermetic build).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatorLock {
    contents: Vec<u8>,
}

impl ValidatorLock {
    /// Construct from non-empty lockfile bytes.
    ///
    /// # Errors
    /// [`BuildError::MissingValidatorLock`] when `contents` is empty.
    pub fn new(contents: impl Into<Vec<u8>>) -> Result<Self, BuildError> {
        let contents = contents.into();
        if contents.is_empty() {
            return Err(BuildError::MissingValidatorLock);
        }
        Ok(Self { contents })
    }

    /// Raw lock bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.contents
    }
}

/// Optional validator wheelhouse index (package name → offline artifact bytes).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Wheelhouse {
    entries: BTreeMap<String, Vec<u8>>,
}

impl Wheelhouse {
    /// Empty wheelhouse (still offline; lock alone pins deps in the fixture).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Build from name/blob pairs. Empty names rejected.
    ///
    /// # Errors
    /// [`BuildError::InvalidWheelhouseEntry`] when a name is empty.
    pub fn new(entries: impl IntoIterator<Item = (String, Vec<u8>)>) -> Result<Self, BuildError> {
        let mut map = BTreeMap::new();
        for (name, bytes) in entries {
            if name.is_empty() {
                return Err(BuildError::InvalidWheelhouseEntry);
            }
            map.insert(name, bytes);
        }
        Ok(Self { entries: map })
    }
}

/// Lock material offered to a builder. Only [`LockMaterial::Validator`] is legal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockMaterial {
    /// Validator lock / wheelhouse pin (required path).
    Validator(ValidatorLock),
    /// Miner-supplied lock — always rejected at build time.
    Miner {
        /// Raw miner lock bytes (never used for digest).
        contents: Vec<u8>,
    },
}

impl From<ValidatorLock> for LockMaterial {
    fn from(lock: ValidatorLock) -> Self {
        Self::Validator(lock)
    }
}

/// Full hermetic build request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildRequest {
    /// Post-admission source tree.
    pub source: AdmittedSource,
    /// Lock material (must be validator).
    pub lock: LockMaterial,
    /// Optional validator wheelhouse.
    pub wheelhouse: Wheelhouse,
}

/// Immutable build product: content-addressed image digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildArtifact {
    /// OCI-style digest `sha256:` + 64 lowercase hex chars.
    pub image_digest: String,
    /// Backend id that produced the artifact (`fixture`, later `docker`, …).
    pub builder_id: &'static str,
}

/// Hermetic offline build failures.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// No files in admitted source.
    #[error("admitted source is empty")]
    EmptySource,
    /// Empty path in source tree.
    #[error("admitted source path must be non-empty")]
    InvalidPath,
    /// Validator lock missing or empty.
    #[error("validator lock required for hermetic build")]
    MissingValidatorLock,
    /// Miner lock must never drive the build (brief §8 step 2).
    #[error("miner lock forbidden; use validator lock only")]
    MinerLockForbidden,
    /// Empty wheelhouse package name.
    #[error("wheelhouse entry name must be non-empty")]
    InvalidWheelhouseEntry,
}

/// Offline hermetic builder: admitted source + validator lock → [`BuildArtifact`].
pub trait HermeticBuilder {
    /// Build an immutable image digest without network access.
    ///
    /// # Errors
    /// Returns [`BuildError`] when lock policy fails or inputs are invalid.
    fn build(&self, request: &BuildRequest) -> Result<BuildArtifact, BuildError>;
}

/// Fixture backend: pure hash of canonical contents; never touches the network.
#[derive(Debug, Default, Clone, Copy)]
pub struct FixtureBuilder;

impl FixtureBuilder {
    /// Construct the fixture builder.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl HermeticBuilder for FixtureBuilder {
    fn build(&self, request: &BuildRequest) -> Result<BuildArtifact, BuildError> {
        let lock = match &request.lock {
            LockMaterial::Miner { .. } => return Err(BuildError::MinerLockForbidden),
            LockMaterial::Validator(lock) => lock,
        };
        if lock.as_bytes().is_empty() {
            return Err(BuildError::MissingValidatorLock);
        }
        if request.source.is_empty() {
            return Err(BuildError::EmptySource);
        }

        let digest_hex = canonical_image_digest_hex(&request.source, lock, &request.wheelhouse);
        Ok(BuildArtifact {
            image_digest: format!("{DIGEST_PREFIX}{digest_hex}"),
            builder_id: "fixture",
        })
    }
}

/// SHA-256 hex of the canonical build payload (no `sha256:` prefix).
fn canonical_image_digest_hex(
    source: &AdmittedSource,
    lock: &ValidatorLock,
    wheelhouse: &Wheelhouse,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(BUILD_DIGEST_DOMAIN);
    hasher.update(b"\0source\0");
    for (path, bytes) in &source.files {
        write_len_prefixed(&mut hasher, path.as_bytes());
        write_len_prefixed(&mut hasher, bytes);
    }
    hasher.update(b"\0lock\0");
    write_len_prefixed(&mut hasher, lock.as_bytes());
    hasher.update(b"\0wheelhouse\0");
    for (name, bytes) in &wheelhouse.entries {
        write_len_prefixed(&mut hasher, name.as_bytes());
        write_len_prefixed(&mut hasher, bytes);
    }
    hex_encode(&hasher.finalize())
}

fn write_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    #[allow(clippy::cast_possible_truncation)] // path/blob sizes fit u64 on all targets we ship
    let len = bytes.len() as u64;
    hasher.update(len.to_le_bytes());
    hasher.update(bytes);
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}
