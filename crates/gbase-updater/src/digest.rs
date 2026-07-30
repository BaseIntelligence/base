//! Digest-only image references (`repo@sha256:<64-hex>`).

use std::fmt;

use crate::error::UpdaterError;

/// Compiled once: `sha256:` + exactly 64 lowercase hex digits.
const DIGEST_LEN: usize = "sha256:".len() + 64;

/// A fully digest-pinned image reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinnedImage {
    /// Repository path without digest (may include registry and tag).
    pub repository: String,
    /// Canonical `sha256:<64-hex>` digest.
    pub digest: String,
}

impl PinnedImage {
    /// `repository@digest` form suitable for Docker pull/create.
    #[must_use]
    pub fn as_ref_string(&self) -> String {
        format!("{}@{}", self.repository, self.digest)
    }
}

impl fmt::Display for PinnedImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.repository, self.digest)
    }
}

/// True when `s` is exactly `sha256:` + 64 hex chars (case-insensitive input normalized).
#[must_use]
pub fn is_pinned_digest(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    if lower.len() != DIGEST_LEN || !lower.starts_with("sha256:") {
        return false;
    }
    lower.as_bytes()[7..].iter().all(u8::is_ascii_hexdigit)
}

/// Extract `sha256:…` from an image string if present after `@`.
#[must_use]
pub fn extract_digest(image: &str) -> Option<String> {
    let (_, digest) = image.split_once('@')?;
    let lower = digest.to_ascii_lowercase();
    if is_pinned_digest(&lower) {
        Some(lower)
    } else {
        None
    }
}

/// Parse a digest-pinned image. Rejects bare tags and `:latest`.
///
/// # Errors
/// [`UpdaterError::NotDigestPinned`] when the reference is not `…@sha256:<64-hex>`.
pub fn parse_pinned_image(image: &str) -> Result<PinnedImage, UpdaterError> {
    let image = image.trim();
    if image.is_empty() {
        return Err(UpdaterError::NotDigestPinned {
            image: image.to_owned(),
        });
    }
    let Some((repo, digest_raw)) = image.split_once('@') else {
        return Err(UpdaterError::NotDigestPinned {
            image: image.to_owned(),
        });
    };
    if repo.is_empty() {
        return Err(UpdaterError::NotDigestPinned {
            image: image.to_owned(),
        });
    }
    // Reject floating tags used as the only pin (policy: digests only).
    let name_part = repo.rsplit('/').next().unwrap_or(repo);
    if name_part == "latest" || name_part.ends_with(":latest") {
        return Err(UpdaterError::NotDigestPinned {
            image: image.to_owned(),
        });
    }
    let digest = digest_raw.to_ascii_lowercase();
    if !is_pinned_digest(&digest) {
        return Err(UpdaterError::NotDigestPinned {
            image: image.to_owned(),
        });
    }
    Ok(PinnedImage {
        repository: repo.to_owned(),
        digest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_pin() {
        let d = "sha256:".to_owned() + &"ab".repeat(32);
        let img = format!("ghcr.io/org/app:1.2.3@{d}");
        let p = parse_pinned_image(&img).expect("pin");
        assert_eq!(p.digest, d);
        assert_eq!(p.repository, "ghcr.io/org/app:1.2.3");
    }

    #[test]
    fn rejects_latest_and_untagged() {
        assert!(parse_pinned_image("ghcr.io/org/app:latest").is_err());
        assert!(parse_pinned_image("ghcr.io/org/app").is_err());
        let bad = format!("ghcr.io/org/app:latest@sha256:{}", "0".repeat(64));
        // repo ends with :latest — still digest-pinned form; policy allows
        // repo@digest even if tag segment says latest, as long as @digest present.
        // Explicit product rule from plan: "Digests only never :latest" on the
        // *reference used for pull* — we reject name_part ending with :latest.
        assert!(parse_pinned_image(&bad).is_err());
    }
}
