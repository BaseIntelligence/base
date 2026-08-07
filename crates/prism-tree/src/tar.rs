//! Deterministic USTAR serialization.
//!
//! Byte-for-byte reproducible: fixed mode, zero timestamps, no owner names,
//! entries sorted by path. Two runs over the same file set produce the same
//! blob, which is what makes the tar usable as a content-addressed artifact and
//! what lets the pod-side extraction be audited.
//!
//! Deliberately hand-rolled rather than pulling in a tar crate: the writer is
//! ~40 lines, the reader has no compression or PAX surface to attack, and the
//! pod side only ever needs `tar -x` from busybox/GNU tar over SSH stdin.

use std::collections::BTreeMap;

use crate::TreeError;

const BLOCK: usize = 512;
/// Longest path the USTAR name field can hold without a PAX extension.
pub const MAX_TAR_PATH_BYTES: usize = 100;

/// Pack `files` into a deterministic USTAR archive.
///
/// Entries are sorted by path, so callers need not pre-sort. Paths must be
/// relative, `/`-separated, free of `..`, and fit [`MAX_TAR_PATH_BYTES`].
///
/// # Errors
/// Unsafe or over-long path, or a body too large for the USTAR size field.
pub fn ustar<P: AsRef<str>, B: AsRef<[u8]>>(files: &[(P, B)]) -> Result<Vec<u8>, TreeError> {
    let mut sorted: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_ref(), b.as_ref()))
        .collect();
    sorted.sort_unstable_by_key(|(p, _)| *p);
    let total: usize = sorted
        .iter()
        .map(|(_, b)| BLOCK + b.len().next_multiple_of(BLOCK))
        .sum();
    let mut out = Vec::with_capacity(total + BLOCK * 2);
    for (path, body) in sorted {
        check_path(path)?;
        out.extend_from_slice(&header(path, body.len())?);
        out.extend_from_slice(body);
        out.resize(out.len() + (BLOCK - body.len() % BLOCK) % BLOCK, 0);
    }
    out.resize(out.len() + BLOCK * 2, 0);
    Ok(out)
}

/// Unpack an archive produced by [`ustar`].
///
/// Rejects anything the writer would not emit — non-regular entries, bad
/// checksums, unsafe paths — rather than silently skipping it, so a corrupted
/// or tampered blob fails loudly instead of yielding a partial tree.
///
/// # Errors
/// Mis-framed, truncated, or tampered archive.
pub fn untar(blob: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, TreeError> {
    if !blob.len().is_multiple_of(BLOCK) {
        return Err(TreeError::Archive("length is not a multiple of 512"));
    }
    let mut files = BTreeMap::new();
    let mut at = 0usize;
    while at + BLOCK <= blob.len() {
        let head = &blob[at..at + BLOCK];
        at += BLOCK;
        if head.iter().all(|b| *b == 0) {
            break;
        }
        if &head[257..262] != b"ustar" {
            return Err(TreeError::Archive("entry is not ustar"));
        }
        if checksum(head) != parse_octal(&head[148..156])? {
            return Err(TreeError::Archive("header checksum mismatch"));
        }
        if head[156] != b'0' && head[156] != 0 {
            return Err(TreeError::Archive("non-regular entry"));
        }
        let path = field_str(&head[..MAX_TAR_PATH_BYTES])?;
        check_path(&path)?;
        let size = parse_octal(&head[124..136])?;
        let end = at
            .checked_add(size)
            .ok_or(TreeError::Archive("size overflow"))?;
        if end > blob.len() {
            return Err(TreeError::Archive("truncated body"));
        }
        files.insert(path, blob[at..end].to_vec());
        at = end + (BLOCK - size % BLOCK) % BLOCK;
    }
    Ok(files)
}

fn header(path: &str, size: usize) -> Result<[u8; BLOCK], TreeError> {
    let mut h = [0u8; BLOCK];
    h[..path.len()].copy_from_slice(path.as_bytes());
    h[100..107].copy_from_slice(b"0000644");
    let size_field = format!("{size:011o}");
    if size_field.len() != 11 {
        return Err(TreeError::TooLarge(format!("{path}: {size} bytes")));
    }
    h[124..135].copy_from_slice(size_field.as_bytes());
    h[156] = b'0';
    h[257..262].copy_from_slice(b"ustar");
    h[263..265].copy_from_slice(b"00");
    h[148..156].fill(b' ');
    let sum = checksum(&h);
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    Ok(h)
}

/// USTAR header checksum: unsigned sum of all bytes with the checksum field
/// itself read as spaces.
fn checksum(head: &[u8]) -> usize {
    head.iter()
        .enumerate()
        .map(|(i, b)| usize::from(if (148..156).contains(&i) { b' ' } else { *b }))
        .sum()
}

fn parse_octal(field: &[u8]) -> Result<usize, TreeError> {
    let mut value = 0usize;
    let mut seen = false;
    for byte in field {
        match byte {
            b'0'..=b'7' => {
                value = value
                    .checked_mul(8)
                    .and_then(|v| v.checked_add(usize::from(byte - b'0')))
                    .ok_or(TreeError::Archive("octal field overflow"))?;
                seen = true;
            }
            0 | b' ' => break,
            _ => return Err(TreeError::Archive("non-octal digit in numeric field")),
        }
    }
    if seen {
        Ok(value)
    } else {
        Err(TreeError::Archive("empty numeric field"))
    }
}

fn field_str(field: &[u8]) -> Result<String, TreeError> {
    let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
    String::from_utf8(field[..end].to_vec()).map_err(|_| TreeError::Archive("name is not utf-8"))
}

/// Reject paths that are unsafe to extract or unrepresentable in USTAR.
pub(crate) fn check_path(path: &str) -> Result<(), TreeError> {
    let bad = |why: &'static str| {
        Err(TreeError::Path {
            path: path.to_string(),
            why,
        })
    };
    if path.is_empty() {
        return bad("empty");
    }
    if path.len() > MAX_TAR_PATH_BYTES {
        return bad("longer than the ustar name field");
    }
    if path.starts_with('/') {
        return bad("absolute");
    }
    if path.ends_with('/') {
        return bad("directory entry");
    }
    if path
        .split('/')
        .any(|p| p.is_empty() || p == "." || p == "..")
    {
        return bad("traversal or empty component");
    }
    if !path.bytes().all(|b| b.is_ascii_graphic() && b != b'\\') {
        return bad("non-printable-ascii or backslash");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_arbitrary_sizes() {
        let cases: Vec<(String, Vec<u8>)> = vec![
            ("empty.txt".into(), Vec::new()),
            ("exactly_block".into(), vec![1u8; BLOCK]),
            ("a/b/c/deep.py".into(), vec![2u8; BLOCK + 1]),
            (
                "big.json".into(),
                (0..70_000u32)
                    .map(|i| u8::try_from(i % 251).unwrap_or(0))
                    .collect(),
            ),
        ];
        let blob = ustar(&cases).expect("pack");
        assert_eq!(blob.len() % BLOCK, 0);
        let back = untar(&blob).expect("unpack");
        assert_eq!(back.len(), cases.len());
        for (path, body) in &cases {
            assert_eq!(back.get(path.as_str()), Some(body), "{path}");
        }
    }

    #[test]
    fn output_is_independent_of_input_order() {
        let a = ustar(&[("b.py", b"two".as_slice()), ("a.py", b"one".as_slice())]).expect("a");
        let b = ustar(&[("a.py", b"one".as_slice()), ("b.py", b"two".as_slice())]).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn ends_with_two_zero_blocks() {
        let blob = ustar(&[("a.py", b"x".as_slice())]).expect("pack");
        assert!(blob[blob.len() - BLOCK * 2..].iter().all(|b| *b == 0));
    }

    #[test]
    fn rejects_unsafe_paths() {
        for bad in [
            "/abs.py",
            "../up.py",
            "a/../b.py",
            "dir/",
            "",
            "back\\slash.py",
            "sp ace.py",
        ] {
            let err = ustar(&[(bad, b"x".as_slice())]).expect_err(bad);
            assert!(matches!(err, TreeError::Path { .. }), "{bad}: {err}");
        }
        let long = "a".repeat(MAX_TAR_PATH_BYTES + 1);
        assert!(ustar(&[(long.as_str(), b"x".as_slice())]).is_err());
    }

    #[test]
    fn detects_corruption() {
        let mut blob = ustar(&[("a.py", b"hello".as_slice())]).expect("pack");
        blob[300] ^= 0xff;
        assert!(matches!(
            untar(&blob).expect_err("corrupt"),
            TreeError::Archive(_)
        ));
        let short = &ustar(&[("a.py", b"hello".as_slice())]).expect("pack")[..BLOCK + 3];
        assert!(untar(short).is_err());
    }

    #[test]
    fn rejects_symlink_and_directory_entries() {
        let mut blob = ustar(&[("a.py", b"hello".as_slice())]).expect("pack");
        blob[156] = b'2';
        blob[148..156].fill(b' ');
        let sum = checksum(&blob[..BLOCK]);
        blob[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
        assert!(matches!(
            untar(&blob).expect_err("symlink"),
            TreeError::Archive("non-regular entry")
        ));
    }
}
