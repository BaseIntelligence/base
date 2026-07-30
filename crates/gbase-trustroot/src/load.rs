//! Local-path loaders: fail closed, owner-verify, D21 dual-accept window.

use std::fs;
use std::path::{Path, PathBuf};

use gbase_crypto::{KEY_LEN, SIGNATURE_LEN};

use crate::error::TrustRootError;
use crate::hexutil::decode_hex_array;
use crate::sign::{
    encode_challenges_body, encode_measurements_body, measurements_digest, parse_signature_bytes,
    verify_trust_root_raw,
};
use crate::types::{
    ChallengesBody, ChallengesToml, MeasurementsBody, MeasurementsToml, TrustRootKind,
};

/// A single verified trust-root document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRoot<B> {
    /// Path the TOML was loaded from.
    pub path: PathBuf,
    /// Document version.
    pub version: u32,
    /// Epoch when this version becomes eligible.
    pub introduced_epoch: u64,
    /// Typed body.
    pub body: B,
    /// Detached owner signature (64 bytes).
    pub signature: [u8; SIGNATURE_LEN],
}

/// Active set after D21 filtering (one or two roots during rotation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveRoots<B> {
    /// Verified roots accepted at `epoch` (newest last).
    pub roots: Vec<VerifiedRoot<B>>,
    /// Epoch used for the filter.
    pub epoch: u64,
    /// Dual-accept window length.
    pub rotation_epochs: u32,
}

impl ActiveRoots<ChallengesBody> {
    /// Prefer the newest active root (primary).
    ///
    /// # Errors
    ///
    /// [`TrustRootError::NoActiveRoot`] if empty.
    pub fn primary(&self) -> Result<&VerifiedRoot<ChallengesBody>, TrustRootError> {
        self.roots.last().ok_or(TrustRootError::NoActiveRoot {
            epoch: self.epoch,
        })
    }

    /// Whether any active root contains `(id, public_key)`.
    #[must_use]
    pub fn accepts_challenge_key(&self, id: &[u8], public_key: &[u8; KEY_LEN]) -> bool {
        self.roots.iter().any(|r| {
            r.body
                .get(id)
                .is_some_and(|c| c.public_key == *public_key)
        })
    }

    /// Whether `shares` byte-equals the emission map of any active root.
    #[must_use]
    pub fn matches_emission_shares(&self, shares: &[(Vec<u8>, u16)]) -> bool {
        self.roots.iter().any(|r| r.body.emission_shares() == shares)
    }
}

impl ActiveRoots<MeasurementsBody> {
    /// Prefer the newest active root.
    ///
    /// # Errors
    ///
    /// [`TrustRootError::NoActiveRoot`] if empty.
    pub fn primary(&self) -> Result<&VerifiedRoot<MeasurementsBody>, TrustRootError> {
        self.roots.last().ok_or(TrustRootError::NoActiveRoot {
            epoch: self.epoch,
        })
    }

    /// Digest of the primary (newest) measurements body.
    ///
    /// # Errors
    ///
    /// No active root.
    pub fn primary_digest(&self) -> Result<[u8; 32], TrustRootError> {
        Ok(measurements_digest(&self.primary()?.body))
    }

    /// Quote allowed if **any** active root's body allows it.
    ///
    /// Empty bodies never allow (fail closed).
    #[must_use]
    pub fn allows_quote(
        &self,
        mr_td: &[u8; 48],
        rtmr0: &[u8; 48],
        rtmr1: &[u8; 48],
        rtmr2: &[u8; 48],
        rtmr3: &[u8; 48],
        compose_hash: &[u8; 32],
    ) -> bool {
        self.roots.iter().any(|r| {
            r.body
                .allows_quote(mr_td, rtmr0, rtmr1, rtmr2, rtmr3, compose_hash)
        })
    }
}

/// Load owner public key from a hex file (32 bytes).
///
/// # Errors
///
/// Missing file or bad encoding.
pub fn load_owner_public_key(path: &Path) -> Result<[u8; KEY_LEN], TrustRootError> {
    let text = read_required(path)?;
    let text = text.trim();
    decode_hex_array(text)
}

/// Load and verify a single challenges.toml + adjacent `.sig`.
///
/// Signature path defaults to `path` with `.sig` appended (`foo.toml.sig`).
///
/// # Errors
///
/// Missing / unsigned / bad sig / invalid body.
pub fn load_challenges_file(
    path: &Path,
    owner_public: &[u8; KEY_LEN],
) -> Result<VerifiedRoot<ChallengesBody>, TrustRootError> {
    let sig_path = sig_path_for(path);
    load_challenges_file_with_sig(path, &sig_path, owner_public)
}

/// Load challenges with an explicit signature path.
///
/// # Errors
///
/// Same as [`load_challenges_file`].
pub fn load_challenges_file_with_sig(
    path: &Path,
    sig_path: &Path,
    owner_public: &[u8; KEY_LEN],
) -> Result<VerifiedRoot<ChallengesBody>, TrustRootError> {
    let text = read_required(path)?;
    let doc: ChallengesToml = toml::from_str(&text).map_err(|e| TrustRootError::InvalidToml {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let body = doc.to_body()?;
    let body_scale = encode_challenges_body(&body);
    let signature = read_signature(sig_path)?;
    verify_owner(owner_public, doc.version, doc.introduced_epoch, &body_scale, &signature)?;
    Ok(VerifiedRoot {
        path: path.to_path_buf(),
        version: doc.version,
        introduced_epoch: doc.introduced_epoch,
        body,
        signature,
    })
}

/// Load and verify a single measurements.toml + `.sig`.
///
/// # Errors
///
/// Missing / unsigned / bad sig / invalid body.
pub fn load_measurements_file(
    path: &Path,
    owner_public: &[u8; KEY_LEN],
) -> Result<VerifiedRoot<MeasurementsBody>, TrustRootError> {
    let sig_path = sig_path_for(path);
    load_measurements_file_with_sig(path, &sig_path, owner_public)
}

/// Load measurements with an explicit signature path.
///
/// # Errors
///
/// Same as [`load_measurements_file`].
pub fn load_measurements_file_with_sig(
    path: &Path,
    sig_path: &Path,
    owner_public: &[u8; KEY_LEN],
) -> Result<VerifiedRoot<MeasurementsBody>, TrustRootError> {
    let text = read_required(path)?;
    let doc: MeasurementsToml = toml::from_str(&text).map_err(|e| TrustRootError::InvalidToml {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let body = doc.to_body()?;
    let body_scale = encode_measurements_body(&body);
    let signature = read_signature(sig_path)?;
    verify_owner(owner_public, doc.version, doc.introduced_epoch, &body_scale, &signature)?;
    Ok(VerifiedRoot {
        path: path.to_path_buf(),
        version: doc.version,
        introduced_epoch: doc.introduced_epoch,
        body,
        signature,
    })
}

/// Load all `*.toml` (non-`.sig`) challenges documents from a directory, verify, apply D21.
///
/// # Errors
///
/// I/O, verify failures, or no active root at `epoch`.
pub fn load_challenges_dir(
    dir: &Path,
    owner_public: &[u8; KEY_LEN],
    epoch: u64,
    rotation_epochs: u32,
) -> Result<ActiveRoots<ChallengesBody>, TrustRootError> {
    let mut verified = Vec::new();
    for path in list_toml_files(dir)? {
        verified.push(load_challenges_file(&path, owner_public)?);
    }
    if verified.is_empty() {
        return Err(TrustRootError::MissingFile {
            path: dir.to_path_buf(),
        });
    }
    verified.sort_by_key(|r| r.version);
    let roots = filter_active(verified, epoch, rotation_epochs);
    if roots.is_empty() {
        return Err(TrustRootError::NoActiveRoot { epoch });
    }
    Ok(ActiveRoots {
        roots,
        epoch,
        rotation_epochs,
    })
}

/// Load all measurements TOML files from a directory, verify, apply D21.
///
/// # Errors
///
/// I/O, verify failures, or no active root at `epoch`.
pub fn load_measurements_dir(
    dir: &Path,
    owner_public: &[u8; KEY_LEN],
    epoch: u64,
    rotation_epochs: u32,
) -> Result<ActiveRoots<MeasurementsBody>, TrustRootError> {
    let mut verified = Vec::new();
    for path in list_toml_files(dir)? {
        verified.push(load_measurements_file(&path, owner_public)?);
    }
    if verified.is_empty() {
        return Err(TrustRootError::MissingFile {
            path: dir.to_path_buf(),
        });
    }
    verified.sort_by_key(|r| r.version);
    let roots = filter_active(verified, epoch, rotation_epochs);
    if roots.is_empty() {
        return Err(TrustRootError::NoActiveRoot { epoch });
    }
    Ok(ActiveRoots {
        roots,
        epoch,
        rotation_epochs,
    })
}

/// Convenience: load the repo-layout pair under `config_dir`:
/// `owner.pubkey`, `challenges.toml`, `measurements.toml`.
///
/// Uses a single version each (no multi-file rotation). `epoch` / `rotation_epochs`
/// still gate `introduced_epoch`.
///
/// # Errors
///
/// Any load/verify failure.
pub fn load_config_dir(
    config_dir: &Path,
    epoch: u64,
    rotation_epochs: u32,
) -> Result<(ActiveRoots<ChallengesBody>, ActiveRoots<MeasurementsBody>), TrustRootError> {
    let owner_path = config_dir.join("owner.pubkey");
    let owner = load_owner_public_key(&owner_path)?;
    let challenges_path = config_dir.join("challenges.toml");
    let measurements_path = config_dir.join("measurements.toml");
    let ch = load_challenges_file(&challenges_path, &owner)?;
    let ms = load_measurements_file(&measurements_path, &owner)?;
    let ch_roots = filter_active(vec![ch], epoch, rotation_epochs);
    let ms_roots = filter_active(vec![ms], epoch, rotation_epochs);
    if ch_roots.is_empty() {
        return Err(TrustRootError::NoActiveRoot { epoch });
    }
    if ms_roots.is_empty() {
        return Err(TrustRootError::NoActiveRoot { epoch });
    }
    Ok((
        ActiveRoots {
            roots: ch_roots,
            epoch,
            rotation_epochs,
        },
        ActiveRoots {
            roots: ms_roots,
            epoch,
            rotation_epochs,
        },
    ))
}

/// D21: accept newest version with `introduced_epoch <= epoch`, plus the previous
/// version while `epoch < newer.introduced_epoch + rotation_epochs`.
#[must_use]
pub fn filter_active<B: Clone>(
    mut verified: Vec<VerifiedRoot<B>>,
    epoch: u64,
    rotation_epochs: u32,
) -> Vec<VerifiedRoot<B>> {
    verified.sort_by_key(|r| r.version);
    let eligible: Vec<VerifiedRoot<B>> = verified
        .into_iter()
        .filter(|r| r.introduced_epoch <= epoch)
        .collect();
    if eligible.is_empty() {
        return Vec::new();
    }
    let newest = eligible.last().cloned();
    let Some(newest) = newest else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // Previous version during dual-accept window.
    if eligible.len() >= 2 {
        let prev = &eligible[eligible.len() - 2];
        let window_end = newest
            .introduced_epoch
            .saturating_add(u64::from(rotation_epochs));
        if epoch < window_end {
            out.push(prev.clone());
        }
    }
    out.push(newest);
    out
}

/// Default signature path: `file.toml` → `file.toml.sig`.
#[must_use]
pub fn sig_path_for(toml_path: &Path) -> PathBuf {
    let mut s = toml_path.as_os_str().to_os_string();
    s.push(".sig");
    PathBuf::from(s)
}

/// Kind helper for ceremony tooling.
#[must_use]
pub const fn kind_name(kind: TrustRootKind) -> &'static str {
    match kind {
        TrustRootKind::Challenges => "challenges",
        TrustRootKind::Measurements => "measurements",
    }
}

fn verify_owner(
    owner_public: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body_scale: &[u8],
    signature: &[u8; SIGNATURE_LEN],
) -> Result<(), TrustRootError> {
    match verify_trust_root_raw(
        owner_public,
        version,
        introduced_epoch,
        body_scale,
        signature,
    ) {
        Ok(()) => Ok(()),
        Err(TrustRootError::SignatureRejected { .. }) => {
            // Distinguish wrong key vs garbage: try is still NonOwner when sig is
            // structurally valid under a different key — callers that sign with a
            // non-owner produce VerificationFailed → map to NonOwner.
            Err(TrustRootError::NonOwner)
        }
        Err(e) => Err(e),
    }
}

fn read_required(path: &Path) -> Result<String, TrustRootError> {
    if !path.is_file() {
        return Err(TrustRootError::MissingFile {
            path: path.to_path_buf(),
        });
    }
    fs::read_to_string(path).map_err(|e| TrustRootError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

fn read_signature(path: &Path) -> Result<[u8; SIGNATURE_LEN], TrustRootError> {
    if !path.is_file() {
        return Err(TrustRootError::Unsigned {
            path: path.to_path_buf(),
        });
    }
    let raw = fs::read(path).map_err(|e| TrustRootError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    if raw.iter().all(u8::is_ascii_whitespace) {
        return Err(TrustRootError::Unsigned {
            path: path.to_path_buf(),
        });
    }
    parse_signature_bytes(&raw).map_err(|e| match e {
        TrustRootError::Unsigned { .. } => TrustRootError::Unsigned {
            path: path.to_path_buf(),
        },
        other => other,
    })
}

fn list_toml_files(dir: &Path) -> Result<Vec<PathBuf>, TrustRootError> {
    if !dir.is_dir() {
        return Err(TrustRootError::MissingFile {
            path: dir.to_path_buf(),
        });
    }
    let rd = fs::read_dir(dir).map_err(|e| TrustRootError::Io {
        path: dir.to_path_buf(),
        message: e.to_string(),
    })?;
    let mut out = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| TrustRootError::Io {
            path: dir.to_path_buf(),
            message: e.to_string(),
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_toml = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"));
        let is_sig = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("sig"));
        if is_toml && !is_sig {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}
