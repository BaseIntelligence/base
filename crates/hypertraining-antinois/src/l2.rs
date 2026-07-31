//! L2 — fixture "compiled" blob compare (brief §12.3 level 2).

use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

use crate::normalize::normalize_source;

/// Domain tag for binary fingerprints (distinct from build digests).
pub const BINARY_FP_DOMAIN: &[u8] = b"base-hypertraining-binary-fp-v1";

/// Normalize a fixture compiled blob (PTX-like text or opaque bytes).
///
/// Text path: strip comments / debug directives / collapse whitespace, then UTF-8.
/// Binary path: drop trailing NULs only (fixture blobs stay byte-stable).
#[must_use]
pub fn normalize_compiled_blob(bytes: &[u8]) -> Vec<u8> {
    if let Ok(text) = std::str::from_utf8(bytes) {
        if looks_like_text_ir(text) {
            return normalize_compiled_text(text).into_bytes();
        }
    }
    let end = bytes
        .iter()
        .rposition(|&b| b != 0)
        .map_or(0, |i| i.saturating_add(1));
    bytes[..end].to_vec()
}

fn looks_like_text_ir(text: &str) -> bool {
    text.contains(".version")
        || text.contains(".target")
        || text.contains("ld.")
        || text.contains("st.")
        || text.contains("//")
        || text.lines().count() > 1 && text.is_ascii()
}

fn normalize_compiled_text(text: &str) -> String {
    let mut lines = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with(".debug") {
            continue;
        }
        // Drop symbol-name noise: `.global .align 4 .u32 foo` → keep opcodes-ish tokens.
        let stripped = strip_c_comment(t);
        let collapsed = normalize_source(stripped); // reuse ws collapse on single line
        if !collapsed.is_empty() {
            lines.push(collapsed);
        }
    }
    lines.join("\n")
}

fn strip_c_comment(line: &str) -> &str {
    line.split("//").next().unwrap_or(line).trim()
}

/// Binary similarity in `[0.0, 1.0]` after normalization.
///
/// Exact match → 1.0. Otherwise Jaccard on 4-byte shingles (or byte multiset if short).
#[must_use]
pub fn l2_binary_similarity(a: &[u8], b: &[u8]) -> f64 {
    let na = normalize_compiled_blob(a);
    let nb = normalize_compiled_blob(b);
    if na.is_empty() && nb.is_empty() {
        return 1.0;
    }
    if na.is_empty() || nb.is_empty() {
        return 0.0;
    }
    if na == nb {
        return 1.0;
    }
    shingle_jaccard(&na, &nb, 4)
}

/// SHA-256 fingerprint of the normalized compiled blob (32 bytes).
#[must_use]
pub fn binary_fingerprint(compiled: &[u8]) -> [u8; 32] {
    let norm = normalize_compiled_blob(compiled);
    let mut hasher = Sha256::new();
    hasher.update(BINARY_FP_DOMAIN);
    hasher.update(&norm);
    let dig = hasher.finalize();
    let mut out = [0_u8; 32];
    out.copy_from_slice(&dig);
    out
}

/// Lowercase hex encoding of [`binary_fingerprint`].
#[must_use]
pub fn binary_fingerprint_hex(compiled: &[u8]) -> String {
    hex_encode(&binary_fingerprint(compiled))
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

#[allow(clippy::cast_precision_loss)]
fn shingle_jaccard(a: &[u8], b: &[u8], k: usize) -> f64 {
    if a.len() < k || b.len() < k {
        // Fall back to byte multiset Jaccard.
        return byte_multiset_jaccard(a, b);
    }
    let sa = shingles(a, k);
    let sb = shingles(b, k);
    let inter = sa.intersection(&sb).count() as f64;
    let union = sa.union(&sb).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn shingles(bytes: &[u8], k: usize) -> BTreeSet<Vec<u8>> {
    let mut set = BTreeSet::new();
    if bytes.len() < k {
        set.insert(bytes.to_vec());
        return set;
    }
    for i in 0..=bytes.len().saturating_sub(k) {
        set.insert(bytes[i..i + k].to_vec());
    }
    set
}

#[allow(clippy::cast_precision_loss)]
fn byte_multiset_jaccard(a: &[u8], b: &[u8]) -> f64 {
    let mut ca = [0_u32; 256];
    let mut cb = [0_u32; 256];
    for &x in a {
        ca[x as usize] = ca[x as usize].saturating_add(1);
    }
    for &x in b {
        cb[x as usize] = cb[x as usize].saturating_add(1);
    }
    let mut inter = 0_u64;
    let mut union = 0_u64;
    for i in 0..256 {
        let va = u64::from(ca[i]);
        let vb = u64::from(cb[i]);
        inter += va.min(vb);
        union += va.max(vb);
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_blobs_score_one_and_same_fp() {
        let blob = b".version 7.0\nadd.u32 %r1, %r2, %r3;\n";
        assert!((l2_binary_similarity(blob, blob) - 1.0).abs() < 1e-9);
        assert_eq!(binary_fingerprint(blob), binary_fingerprint(blob));
    }

    #[test]
    fn comment_only_change_still_identical_after_norm() {
        let a = b".version 7.0\n// debug\nadd.u32 %r1, %r2, %r3;\n";
        let b = b".version 7.0\nadd.u32 %r1, %r2, %r3;\n";
        assert!((l2_binary_similarity(a, b) - 1.0).abs() < 1e-9);
        assert_eq!(binary_fingerprint(a), binary_fingerprint(b));
    }

    #[test]
    fn different_opcodes_lower_similarity() {
        let a = b".version 7.0\nadd.u32 %r1, %r2, %r3;\nmul.f32 %f1, %f2, %f3;\n";
        let b = b".version 7.0\nld.global.f32 %f1, [%rd1];\nst.global.f32 [%rd2], %f1;\n";
        let s = l2_binary_similarity(a, b);
        assert!(s < 0.85, "got {s}");
    }
}
